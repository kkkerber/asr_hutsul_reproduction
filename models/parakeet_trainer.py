"""Parakeet (NVIDIA NeMo) CTC fine-tuning entry point.

Kept isolated from the transformers-based trainers because Parakeet
checkpoints ship as ``.nemo`` archives and require the NeMo toolkit
to load — they cannot be consumed by ``AutoModelForCTC``.

Install (heavy; not pinned in requirements.txt):

    pip install 'nemo_toolkit[asr]>=1.23' pytorch_lightning

Usage:

    python train.py --model_type parakeet --variant parakeet-ctc-0.6b
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
import numpy as np

from config import (
    APOSTROPHE,
    PROJECT_ROOT,
    UKRAINIAN_ALPHABET,
    ProjectConfig,
    configure_logging,
    resolve_storage_layout,
    set_global_seed,
)
from metrics import MetricCalculator, analyze_substitutions
from preprocess import decode_audio_entry, load_and_prepare
from utils.text_normalization import build_default_normalizer

logger = logging.getLogger(__name__)


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "parakeet.yaml"


@dataclass
class ParakeetTrainArgs:
    model_name_or_path: str
    variant: str
    dataset_name: str
    dataset_config: Optional[str] = None
    audio_column: Optional[str] = None
    text_column: Optional[str] = None

    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    run_name: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None

    sample_rate: int = 16000

    learning_rate: float = 1e-4
    max_steps: int = 20000
    warmup_steps: int = 1000
    weight_decay: float = 1e-3
    optimizer: str = "adamw"

    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1

    eval_every_n_steps: int = 500
    save_every_n_steps: int = 500
    logging_steps: int = 50
    save_top_k: int = 3

    precision: str = "16-mixed"
    gradient_clip_val: float = 1.0
    num_workers: int = 2

    use_augmentation: bool = False
    seed: int = 42
    deterministic: bool = False

    min_train_audio_duration_sec: float = 1.0
    tokenizer_type: str = "char"   # "char" | "bpe"

    hf_token: Optional[str] = None
    trust_remote_code: bool = False


# ---------------------------------------------------------------------------
# Lazy NeMo import
# ---------------------------------------------------------------------------


def _load_nemo():
    try:
        import nemo  # noqa: F401
        import nemo.collections.asr as nemo_asr
        import pytorch_lightning as pl
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise ImportError(
            "Parakeet training requires NeMo and PyTorch Lightning:\n"
            "    pip install 'nemo_toolkit[asr]>=1.23' pytorch_lightning\n"
            "These are intentionally not pinned in requirements.txt "
            "because of NeMo's size (~5 GB) and its dependency overlap "
            "with the rest of the stack.  Install them in a separate "
            "virtualenv if you want to keep the transformers pipeline "
            "isolated.\n"
            f"Original ImportError: {exc}"
        )
    return nemo_asr, pl, OmegaConf


# ---------------------------------------------------------------------------
# YAML -> args
# ---------------------------------------------------------------------------


def load_yaml_config(
    path: Union[str, Path] = DEFAULT_CONFIG_PATH,
    *,
    variant: Optional[str] = None,
) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    defaults = dict(raw.get("defaults", {}))
    variants = raw.get("variants", {})
    if not variants:
        raise ValueError(f"{path} declares no variants")
    if variant is None:
        variant = next(iter(variants.keys()))
        logger.info("No --variant given, defaulting to %s", variant)
    if variant not in variants:
        raise KeyError(
            f"Variant {variant!r} not in {path}. "
            f"Available: {sorted(variants)}"
        )
    merged = {**defaults, **variants[variant]}
    merged["variant"] = variant
    return merged


def args_from_yaml(
    yaml_path: Union[str, Path],
    variant: Optional[str] = None,
    *,
    overrides: Optional[Dict[str, Any]] = None,
) -> ParakeetTrainArgs:
    cfg = load_yaml_config(yaml_path, variant=variant)
    for k, v in dict(overrides or {}).items():
        if v is None:
            continue
        cfg[k] = v
    valid = set(ParakeetTrainArgs.__dataclass_fields__.keys())
    filtered = {k: v for k, v in cfg.items() if k in valid}
    for required in ("model_name_or_path", "variant", "dataset_name"):
        if required not in filtered:
            raise ValueError(
                f"Missing required Parakeet config field: {required}"
            )
    return ParakeetTrainArgs(**filtered)


# ---------------------------------------------------------------------------
# HF dataset -> NeMo manifests
# ---------------------------------------------------------------------------


def _materialize_manifest(
    split: Any,
    *,
    audio_column: str,
    text_column: str,
    out_dir: Path,
    sample_rate: int = 16000,
    overwrite: bool = False,
) -> Path:
    """Write per-sample WAVs and a NeMo JSONL manifest.

    Cached: subsequent runs reuse the manifest when its line count
    matches the split size.
    """
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    if manifest_path.exists() and not overwrite:
        try:
            n = sum(1 for _ in open(manifest_path, "r", encoding="utf-8"))
            if n == len(split):
                logger.info(
                    "Reusing manifest %s (%d entries)", manifest_path, n
                )
                return manifest_path
        except Exception:
            pass

    n_written = 0
    with open(manifest_path, "w", encoding="utf-8") as fh:
        for i, example in enumerate(split):
            audio = example[audio_column]
            # decode_audio_entry handles both the legacy decoded form
            # ``{"array": ..., "sampling_rate": ...}`` and the raw
            # ``Audio(decode=False)`` form ``{"path": ..., "bytes": ...}``;
            # it always returns mono float32 at ``sample_rate``.
            samples, sr = decode_audio_entry(audio, sample_rate)
            wav_path = (wav_dir / f"{i:07d}.wav").resolve()
            if not wav_path.exists() or overwrite:
                sf.write(str(wav_path), samples, sr, subtype="PCM_16")
            text = example[text_column] or ""
            duration = float(len(samples)) / float(sr)
            entry = {
                "audio_filepath": str(wav_path),
                "duration": duration,
                "text": text,
            }
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            n_written += 1

    logger.info(
        "Wrote manifest %s (%d entries, %s)",
        manifest_path,
        n_written,
        wav_dir,
    )
    return manifest_path


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


def _build_char_vocab() -> List[str]:
    """Ukrainian char-level vocab consistent with the rest of the project."""
    return list(UKRAINIAN_ALPHABET) + [APOSTROPHE, " "]


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------


_LATEST_SENTINEL = "LATEST"


def _resolve_resume_ckpt(
    requested: Optional[str], output_dir: Path
) -> Optional[str]:
    if requested is None:
        return None
    if requested == _LATEST_SENTINEL:
        ckpts = sorted(Path(output_dir).glob("*.ckpt"))
        if not ckpts:
            logger.warning(
                "No *.ckpt under %s — starting fresh.", output_dir
            )
            return None
        latest = str(ckpts[-1])
        logger.info("Auto-resuming from %s", latest)
        return latest
    if not Path(requested).exists():
        raise FileNotFoundError(
            f"resume_from_checkpoint not found: {requested}"
        )
    return requested


# ---------------------------------------------------------------------------
# Final-pass evaluation -> project-standard outputs
# ---------------------------------------------------------------------------


def _final_test_evaluation(
    model: Any,
    test_split: Any,
    *,
    audio_column: str,
    text_column: str,
    layout: Any,
    variant: str,
    batch_size: int,
) -> Dict[str, float]:
    """Decode the test split and write predictions.csv / test_results.json /
    error_analysis.json in the layout used by ``evaluate.py``."""
    normalizer = build_default_normalizer()
    mc = MetricCalculator()

    audio_paths: List[str] = []
    refs: List[str] = []
    for example in test_split:
        audio = example[audio_column]
        # If load_and_prepare cast to Audio, the array is already decoded;
        # NeMo's transcribe() works fastest from disk, so rely on the
        # manifest materialization for paths.  We just need refs here.
        refs.append(normalizer(example[text_column] or ""))
        audio_paths.append(audio.get("path") if isinstance(audio, dict) else None)

    # NeMo's transcribe accepts file paths only — reuse the manifest WAVs.
    test_wav_dir = layout.preprocessed_dir("parakeet") / "test" / "wav"
    audio_paths = sorted(test_wav_dir.glob("*.wav"))
    if len(audio_paths) != len(refs):
        raise RuntimeError(
            f"Test manifest size mismatch: refs={len(refs)} wavs={len(audio_paths)}"
        )

    logger.info("Transcribing %d test samples...", len(audio_paths))
    hyps_raw = model.transcribe(
        [str(p) for p in audio_paths], batch_size=batch_size
    )
    # NeMo returns either List[str] (older) or List[Hypothesis] (newer).
    preds: List[str] = []
    for h in hyps_raw:
        text = h.text if hasattr(h, "text") else str(h)
        preds.append(normalizer(text))

    overall_wer = mc.compute_wer(preds, refs)
    overall_cer = mc.compute_cer(preds, refs)
    logger.info(
        "Test: WER=%.4f  CER=%.4f  (n=%d)",
        overall_wer, overall_cer, len(preds),
    )

    csv_dir = layout.evaluations_csv / variant
    json_dir = layout.evaluations_json / variant
    pred_dir = layout.evaluations_predictions / variant
    for d in (csv_dir, json_dir, pred_dir):
        d.mkdir(parents=True, exist_ok=True)

    with open(csv_dir / "predictions.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["reference", "prediction", "wer", "cer"])
        for ref, pred in zip(refs, preds):
            w = mc.compute_wer([pred], [ref])
            c = mc.compute_cer([pred], [ref])
            writer.writerow([ref, pred, f"{w:.6f}", f"{c:.6f}"])

    payload = {
        "model_family": "parakeet",
        "variant": variant,
        "split": "test",
        "num_samples": len(preds),
        "wer": overall_wer,
        "cer": overall_cer,
        "outputs": {
            "csv": str(csv_dir / "predictions.csv"),
            "json": str(json_dir / "test_results.json"),
            "error_analysis": str(json_dir / "error_analysis.json"),
            "predictions": str(pred_dir / "predictions.txt"),
        },
    }
    (json_dir / "test_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    err = analyze_substitutions(preds, refs)
    err.to_json(json_dir / "error_analysis.json", top_k=50)

    with open(pred_dir / "predictions.txt", "w", encoding="utf-8") as fh:
        for ref, pred in zip(refs, preds):
            fh.write(f"REF: {ref}\nHYP: {pred}\n\n")

    return {"eval_final_wer": overall_wer, "eval_final_cer": overall_cer}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def train_parakeet(args: ParakeetTrainArgs) -> Dict[str, float]:
    configure_logging()
    set_global_seed(args.seed, deterministic=args.deterministic)

    if args.hf_token is None:
        args.hf_token = os.environ.get("HF_TOKEN")

    layout = resolve_storage_layout()
    layout.ensure()

    if not args.output_dir or Path(args.output_dir) == Path("outputs"):
        args.output_dir = layout.checkpoint_dir(args.variant)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    final_dir = layout.final_model_dir(args.variant)
    tensorboard_dir = layout.tensorboard_dir(args.variant)
    final_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    project_cfg = ProjectConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        audio_column=args.audio_column,
        text_column=args.text_column,
        sample_rate=args.sample_rate,
        seed=args.seed,
        deterministic=args.deterministic,
        use_augmentation=args.use_augmentation,
        preprocessed_dir=layout.preprocessed_dir("parakeet"),
        dataset_cache_dir=layout.datasets_cache,
        cache_dir=layout.cache,
        min_train_audio_duration_sec=args.min_train_audio_duration_sec,
    )
    project_cfg.ensure_dirs()

    logger.info("=" * 70)
    logger.info("Parakeet (NeMo) fine-tuning — variant=%s", args.variant)
    logger.info("Model:        %s", args.model_name_or_path)
    logger.info("Checkpoints:  %s", args.output_dir)
    logger.info("Final model:  %s", final_dir)
    logger.info("TensorBoard:  %s", tensorboard_dir)
    logger.info("=" * 70)

    # ---- 1) Dataset -> manifests -------------------------------------------
    dataset, audio_col, text_col = load_and_prepare(
        project_cfg,
        token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
    )

    manifests_root = layout.preprocessed_dir("parakeet")
    train_manifest = _materialize_manifest(
        dataset["train"],
        audio_column=audio_col, text_column=text_col,
        out_dir=manifests_root / "train",
        sample_rate=args.sample_rate,
    )
    val_manifest = _materialize_manifest(
        dataset["validation"],
        audio_column=audio_col, text_column=text_col,
        out_dir=manifests_root / "validation",
        sample_rate=args.sample_rate,
    )
    test_manifest = _materialize_manifest(
        dataset["test"],
        audio_column=audio_col, text_column=text_col,
        out_dir=manifests_root / "test",
        sample_rate=args.sample_rate,
    )

    # ---- 2) NeMo model load ------------------------------------------------
    nemo_asr, pl, OmegaConf = _load_nemo()

    logger.info("Loading Parakeet from %s ...", args.model_name_or_path)
    model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
        args.model_name_or_path,
        map_location="cpu",
    )

    # ---- 3) Vocabulary swap (canonical low-resource path) ------------------
    if args.tokenizer_type == "char":
        char_vocab = _build_char_vocab()
        try:
            model.change_vocabulary(new_vocabulary=char_vocab)
            logger.info(
                "Switched vocabulary to char-level (%d tokens)",
                len(char_vocab),
            )
        except Exception as exc:
            logger.warning(
                "change_vocabulary failed (%s); keeping base BPE tokenizer.",
                exc,
            )

    # ---- 4) Data wiring ----------------------------------------------------
    def _ds_cfg(manifest: Path, shuffle: bool, batch_size: int) -> Any:
        return OmegaConf.create({
            "manifest_filepath": str(manifest),
            "sample_rate": args.sample_rate,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": args.num_workers,
            "pin_memory": True,
            "use_start_end_token": False,
            "trim_silence": False,
            "max_duration": None,
            "min_duration": args.min_train_audio_duration_sec,
        })

    model.setup_training_data(
        _ds_cfg(train_manifest, True, args.per_device_train_batch_size)
    )
    model.setup_validation_data(
        _ds_cfg(val_manifest, False, args.per_device_eval_batch_size)
    )
    model.setup_test_data(
        _ds_cfg(test_manifest, False, args.per_device_eval_batch_size)
    )

    # ---- 5) Optimization ---------------------------------------------------
    optim_cfg = OmegaConf.create({
        "name": args.optimizer,
        "lr": args.learning_rate,
        "weight_decay": args.weight_decay,
        "betas": [0.9, 0.98],
        "sched": {
            "name": "CosineAnnealing",
            "warmup_steps": args.warmup_steps,
            "min_lr": 1.0e-6,
            "max_steps": args.max_steps,
        },
    })
    model.setup_optimization(optim_cfg)

    # ---- 6) PyTorch-Lightning trainer --------------------------------------
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    ckpt_cb = ModelCheckpoint(
        dirpath=str(args.output_dir),
        filename="parakeet-{step:07d}-{val_wer:.4f}",
        monitor="val_wer",
        mode="min",
        save_top_k=args.save_top_k,
        save_last=True,
        auto_insert_metric_name=False,
    )
    tb_logger = TensorBoardLogger(
        save_dir=str(tensorboard_dir),
        name="",
        version="",
    )

    trainer = pl.Trainer(
        max_steps=args.max_steps,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        val_check_interval=args.eval_every_n_steps,
        check_val_every_n_epoch=None,
        log_every_n_steps=args.logging_steps,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=[ckpt_cb],
        logger=tb_logger,
        accelerator="auto",
        devices="auto",
        default_root_dir=str(args.output_dir),
        enable_checkpointing=True,
        deterministic=False,
    )

    # ---- 7) Resume ---------------------------------------------------------
    resume_ckpt = _resolve_resume_ckpt(
        args.resume_from_checkpoint, args.output_dir
    )

    # ---- 8) Train ----------------------------------------------------------
    trainer.fit(model, ckpt_path=resume_ckpt)

    # ---- 9) Final .nemo archive -------------------------------------------
    final_path = final_dir / f"{args.variant}.nemo"
    try:
        model.save_to(str(final_path))
        logger.info("Saved final NeMo archive to %s", final_path)
    except Exception as exc:
        logger.warning("save_to(%s) failed: %s", final_path, exc)

    # ---- 10) Final test pass — project-standard outputs --------------------
    try:
        eval_metrics = _final_test_evaluation(
            model,
            dataset["test"],
            audio_column=audio_col,
            text_column=text_col,
            layout=layout,
            variant=args.variant,
            batch_size=args.per_device_eval_batch_size,
        )
    except Exception as exc:
        logger.exception("Final test pass failed (%s); skipping outputs", exc)
        eval_metrics = {}

    logger.info("Training complete. Final metrics: %s", eval_metrics)
    return eval_metrics


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "ParakeetTrainArgs",
    "args_from_yaml",
    "load_yaml_config",
    "train_parakeet",
]
