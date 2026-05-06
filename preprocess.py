"""
preprocess.py
=============

Dataset loading and preprocessing for the Hutsul ASR reproduction.

Responsibilities
----------------

1. Load ``KSE-RESEARCH-Group/Dido-Yvanchyk-Audio-Dataset-v2`` (or any
   compatible HF dataset) via :func:`datasets.load_dataset`.
2. Auto-detect the audio and transcription columns (the upstream
   dataset has been renamed several times; we accept any of the
   candidates listed in :data:`config.AUDIO_COLUMN_CANDIDATES` /
   :data:`config.TEXT_COLUMN_CANDIDATES`).
3. Resample audio to 16 kHz via the modern ``Audio`` feature.
4. Apply project text normalisation.
5. Split into 80/10/10 train/validation/test (deterministically).
6. Cache the prepared :class:`datasets.DatasetDict` to disk so
   subsequent runs skip the heavy work.

Also exposed:

* :func:`prepare_for_model` — one-shot helper that does all of the
  above *and* runs the supplied feature-extractor / tokenizer to
  emit ``input_features`` (or ``input_values``) and ``labels`` ready
  for the Trainer.

CLI usage
---------

::

    python preprocess.py --dataset_name KSE-RESEARCH-Group/Dido-Yvanchyk-Audio-Dataset-v2 \\
                        --output_dir .cache/preprocessed
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from datasets import (
    Audio,
    Dataset,
    DatasetDict,
    load_dataset,
    load_from_disk,
)

from config import (
    AUDIO_COLUMN_CANDIDATES,
    DEFAULT_DATASET_NAME,
    DEFAULT_PREPROCESSED_DIR,
    DEFAULT_SEED,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    TARGET_SAMPLE_RATE,
    TEXT_COLUMN_CANDIDATES,
    ProjectConfig,
    configure_logging,
    resolve_storage_layout,
)
from utils.text_normalization import TextNormalizer, build_default_normalizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------


def detect_audio_column(
    columns: List[str], explicit: Optional[str] = None
) -> str:
    """Return the audio column name from ``columns``."""
    if explicit:
        if explicit not in columns:
            raise KeyError(
                f"Requested audio column {explicit!r} not in dataset "
                f"(available: {columns})"
            )
        return explicit

    lowered = {c.lower(): c for c in columns}
    for candidate in AUDIO_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]

    raise KeyError(
        "Could not auto-detect audio column. Available columns: "
        f"{columns}.  Pass --audio_column explicitly."
    )


def detect_text_column(
    columns: List[str], explicit: Optional[str] = None
) -> str:
    """Return the text/transcription column name from ``columns``."""
    if explicit:
        if explicit not in columns:
            raise KeyError(
                f"Requested text column {explicit!r} not in dataset "
                f"(available: {columns})"
            )
        return explicit

    lowered = {c.lower(): c for c in columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]

    raise KeyError(
        "Could not auto-detect text column. Available columns: "
        f"{columns}.  Pass --text_column explicitly."
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_raw_dataset(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_config: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    trust_remote_code: bool = False,
) -> Union[DatasetDict, Dataset]:
    """Load the dataset from the Hugging Face Hub.

    The function accepts both gated (``token=…``) and public datasets
    and forwards the remaining arguments verbatim to
    :func:`datasets.load_dataset`.
    """
    logger.info("Loading dataset %s (config=%s)…", dataset_name, dataset_config)
    ds = load_dataset(
        dataset_name,
        dataset_config,
        cache_dir=str(cache_dir) if cache_dir else None,
        revision=revision,
        token=token,
        trust_remote_code=trust_remote_code,
    )
    if isinstance(ds, Dataset):
        logger.info("Loaded a single split with %d rows", len(ds))
    elif isinstance(ds, DatasetDict):
        sizes = {k: len(v) for k, v in ds.items()}
        logger.info("Loaded splits: %s", sizes)
    return ds


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def split_dataset(
    ds: Union[Dataset, DatasetDict],
    *,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SEED,
) -> DatasetDict:
    """Return an 80/10/10 :class:`DatasetDict`.

    If ``ds`` already has the three canonical splits we pass it through
    unchanged.  Otherwise we concatenate any existing splits and
    re-split deterministically using the given seed.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"split ratios must sum to 1.0, got {total} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )

    if isinstance(ds, DatasetDict):
        canonical = {"train", "validation", "test"}
        if canonical.issubset(ds.keys()):
            logger.info("Dataset already has canonical splits — keeping them")
            return DatasetDict(
                {k: ds[k] for k in ("train", "validation", "test")}
            )

        # Concatenate every available split into one before re-splitting.
        from datasets import concatenate_datasets

        logger.info(
            "Re-splitting dataset (input splits: %s)", list(ds.keys())
        )
        ds = concatenate_datasets([ds[k] for k in ds.keys()])

    assert isinstance(ds, Dataset)

    n = len(ds)
    if n < 10:
        raise ValueError(
            f"Dataset has only {n} rows — cannot meaningfully split."
        )

    # Two-step split: train vs. (val+test), then val vs. test.
    first_split = ds.train_test_split(
        test_size=val_ratio + test_ratio,
        seed=seed,
        shuffle=True,
    )
    train_ds = first_split["train"]
    rest = first_split["test"]

    rel_test = test_ratio / (val_ratio + test_ratio)
    second_split = rest.train_test_split(
        test_size=rel_test,
        seed=seed,
        shuffle=True,
    )
    val_ds = second_split["train"]
    test_ds = second_split["test"]

    out = DatasetDict(
        {"train": train_ds, "validation": val_ds, "test": test_ds}
    )
    logger.info(
        "Split sizes: train=%d, validation=%d, test=%d",
        len(out["train"]),
        len(out["validation"]),
        len(out["test"]),
    )
    return out


# ---------------------------------------------------------------------------
# Normalisation pass over a DatasetDict
# ---------------------------------------------------------------------------


def normalize_dataset(
    ds: DatasetDict,
    *,
    audio_column: str,
    text_column: str,
    sample_rate: int = TARGET_SAMPLE_RATE,
    normalizer: Optional[TextNormalizer] = None,
    drop_empty: bool = True,
    num_proc: Optional[int] = None,
) -> DatasetDict:
    """Resample audio + normalise transcripts in-place."""
    normalizer = normalizer or build_default_normalizer()

    # Cast the audio column.  ``cast_column`` rebinds the column to the
    # ``Audio`` feature so ``ds[i][audio_column]`` returns
    # ``{"array": np.ndarray, "sampling_rate": int, ...}``.
    for split in ds:
        if audio_column not in ds[split].column_names:
            raise KeyError(
                f"Split {split!r} is missing audio column {audio_column!r}"
            )
        if text_column not in ds[split].column_names:
            raise KeyError(
                f"Split {split!r} is missing text column {text_column!r}"
            )
        ds[split] = ds[split].cast_column(
            audio_column, Audio(sampling_rate=sample_rate)
        )

    def _normalize_text_row(batch: Dict[str, Any]) -> Dict[str, Any]:
        # ``map`` with ``batched=True`` gives us a column-of-lists dict.
        batch[text_column] = [normalizer(t) for t in batch[text_column]]
        return batch

    ds = ds.map(
        _normalize_text_row,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        desc="Normalising transcripts",
    )

    if drop_empty:
        before = {split: len(ds[split]) for split in ds}
        ds = ds.filter(
            lambda b: [bool(t and t.strip()) for t in b[text_column]],
            batched=True,
            batch_size=1000,
            num_proc=num_proc,
            desc="Dropping empty transcripts",
        )
        after = {split: len(ds[split]) for split in ds}
        for split in before:
            dropped = before[split] - after[split]
            if dropped:
                logger.info(
                    "Dropped %d empty transcripts from split %s",
                    dropped,
                    split,
                )

    return ds


# ---------------------------------------------------------------------------
# End-to-end loader
# ---------------------------------------------------------------------------


def load_and_prepare(
    config: Optional[ProjectConfig] = None,
    *,
    dataset_name: Optional[str] = None,
    dataset_config: Optional[str] = None,
    audio_column: Optional[str] = None,
    text_column: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    use_disk_cache: bool = True,
    overwrite_cache: bool = False,
    token: Optional[str] = None,
    trust_remote_code: bool = False,
    num_proc: Optional[int] = None,
) -> Tuple[DatasetDict, str, str]:
    """Load → split → normalise → cache.

    Returns
    -------
    dataset
        A :class:`DatasetDict` with ``train`` / ``validation`` /
        ``test`` splits.
    audio_column
        Resolved audio column name.
    text_column
        Resolved text column name.
    """
    cfg = config or ProjectConfig()
    cfg.ensure_dirs()

    dataset_name = dataset_name or cfg.dataset_name

    # ---- Cache path ------------------------------------------------------
    # Priority order:
    #   1. explicit ``cache_dir`` argument
    #   2. ``cfg.preprocessed_dir`` (typically points at
    #      ``<storage_root>/preprocessed/<model_type>/`` under Colab)
    #   3. fallback to the layout-resolved preprocessed root
    cache_root = (
        Path(cache_dir)
        if cache_dir is not None
        else cfg.preprocessed_dir
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    safe_name = dataset_name.replace("/", "__")
    cache_path = cache_root / f"{safe_name}__{cfg.sample_rate}"

    if use_disk_cache and cache_path.exists() and not overwrite_cache:
        logger.info("Loading preprocessed dataset from %s", cache_path)
        try:
            ds = load_from_disk(str(cache_path))
            cols = ds["train"].column_names
            ac = detect_audio_column(cols, audio_column or cfg.audio_column)
            tc = detect_text_column(cols, text_column or cfg.text_column)
            return ds, ac, tc
        except Exception as exc:
            logger.warning(
                "Failed to load preprocessed cache (%s) — regenerating", exc
            )

    # ---- Load + split + normalise ---------------------------------------
    raw = load_raw_dataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config or cfg.dataset_config,
        cache_dir=cfg.dataset_cache_dir,
        token=token,
        trust_remote_code=trust_remote_code,
    )

    # Determine columns from the first available split.
    first_split = (
        raw if isinstance(raw, Dataset) else raw[next(iter(raw.keys()))]
    )
    cols = first_split.column_names
    ac = detect_audio_column(cols, audio_column or cfg.audio_column)
    tc = detect_text_column(cols, text_column or cfg.text_column)
    logger.info("Resolved columns: audio=%s, text=%s", ac, tc)

    ds = split_dataset(
        raw,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed,
    )

    ds = normalize_dataset(
        ds,
        audio_column=ac,
        text_column=tc,
        sample_rate=cfg.sample_rate,
        num_proc=num_proc,
    )

    # ---- Persist cache --------------------------------------------------
    if use_disk_cache:
        try:
            ds.save_to_disk(str(cache_path))
            logger.info("Saved preprocessed dataset to %s", cache_path)
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Failed to save preprocessed cache to %s: %s", cache_path, exc
            )

    return ds, ac, tc


# ---------------------------------------------------------------------------
# Per-model feature extraction
# ---------------------------------------------------------------------------


def prepare_for_model(
    dataset: DatasetDict,
    *,
    audio_column: str,
    text_column: str,
    feature_extractor: Any,
    tokenizer: Any,
    sample_rate: int = TARGET_SAMPLE_RATE,
    waveform_augmenter: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
    augment_splits: Tuple[str, ...] = ("train",),
    num_proc: Optional[int] = None,
    remove_original_columns: bool = True,
    feature_kwargs: Optional[Dict[str, Any]] = None,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
) -> DatasetDict:
    """Run ``feature_extractor`` and ``tokenizer`` over every split.

    The result has two columns per row: ``input_features`` (or
    ``input_values`` — whichever the feature extractor returns) and
    ``labels``.

    Parameters
    ----------
    waveform_augmenter
        Optional callable applied to the raw waveform before feature
        extraction.  Only fires for splits listed in ``augment_splits``.
    """
    feature_kwargs = dict(feature_kwargs or {})
    tokenizer_kwargs = dict(tokenizer_kwargs or {})

    def _build_processor_fn(split_name: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        do_augment = waveform_augmenter is not None and split_name in augment_splits

        def _process(example: Dict[str, Any]) -> Dict[str, Any]:
            audio = example[audio_column]
            samples = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio.get("sampling_rate", sample_rate))

            if do_augment:
                try:
                    samples = waveform_augmenter(samples, sr)
                except Exception as exc:  # pragma: no cover
                    logger.warning(
                        "Waveform augmentation failed (%s) — using original audio",
                        exc,
                    )

            features = feature_extractor(
                samples,
                sampling_rate=sr,
                **feature_kwargs,
            )

            # The feature extractor returns a ``BatchFeature`` with
            # one or both of ``input_features`` / ``input_values``.
            out: Dict[str, Any] = {}
            if "input_features" in features:
                out["input_features"] = features["input_features"][0]
            if "input_values" in features:
                out["input_values"] = features["input_values"][0]

            text = example[text_column]
            tokenised = tokenizer(text, **tokenizer_kwargs)
            out["labels"] = tokenised["input_ids"]
            return out

        return _process

    new_splits: Dict[str, Dataset] = {}
    for split_name, split_ds in dataset.items():
        process_fn = _build_processor_fn(split_name)
        cols_to_remove = (
            split_ds.column_names if remove_original_columns else None
        )
        new_splits[split_name] = split_ds.map(
            process_fn,
            remove_columns=cols_to_remove,
            num_proc=num_proc,
            desc=f"Featurising {split_name}",
        )

    return DatasetDict(new_splits)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Load, split and normalise the Hutsul ASR dataset.  Saves a "
            "DatasetDict to disk so subsequent training runs are fast."
        )
    )
    p.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    p.add_argument("--dataset_config", default=None)
    p.add_argument("--audio_column", default=None)
    p.add_argument("--text_column", default=None)
    p.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Where to save the preprocessed DatasetDict.  When omitted, "
            "uses ``<storage_root>/preprocessed/shared/`` "
            "(i.e. Drive when running in Colab, ./outputs/ otherwise)."
        ),
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--overwrite_cache", action="store_true", help="Ignore on-disk cache."
    )
    p.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="Number of dataset.map worker processes (default: dataset default).",
    )
    p.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token (overrides env $HF_TOKEN).",
    )
    p.add_argument("--trust_remote_code", action="store_true")
    return p


def main() -> None:  # pragma: no cover — CLI entry-point
    configure_logging()
    args = _build_argparser().parse_args()

    layout = resolve_storage_layout()
    layout.ensure()

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else layout.preprocessed / "shared"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = ProjectConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        audio_column=args.audio_column,
        text_column=args.text_column,
        seed=args.seed,
        preprocessed_dir=output_dir,
        dataset_cache_dir=layout.datasets_cache,
        cache_dir=layout.cache,
    )

    ds, ac, tc = load_and_prepare(
        cfg,
        cache_dir=output_dir,
        overwrite_cache=args.overwrite_cache,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
        num_proc=args.num_proc,
    )

    logger.info(
        "Done. Splits: train=%d, val=%d, test=%d (audio=%s, text=%s)",
        len(ds["train"]),
        len(ds["validation"]),
        len(ds["test"]),
        ac,
        tc,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
