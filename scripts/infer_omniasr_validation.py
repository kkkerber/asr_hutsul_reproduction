"""Run OmniASR (fairseq2) inference on the project's validation split and
write the SAME output artefacts every HF model produces:

    evaluations/csv/<variant>/predictions.csv        (per-utterance WER/CER)
    evaluations/json/<variant>/test_results.json     (overall WER/CER + meta)
    evaluations/json/<variant>/error_analysis.json   (Levenshtein analysis)
    evaluations/predictions/<variant>/predictions.txt  (REF/HYP dump)
    checkpoints/<variant>/best_metric.json           (val-vs-test joiner input)

Why this script exists
----------------------

OmniASR fine-tunes save fairseq2-native checkpoints (a single ``.pt``
archive under ``final_models/<variant>/step_<N>/``).  ``evaluate.py``
cannot load them — its ``_load_omniasr`` path calls
``AutoModelForCTC.from_pretrained`` which requires a HuggingFace
``config.json``.  As a result the project's primary metric source for
OmniASR has been the TensorBoard scrape in
``scripts/evaluate_omniasr.py``, which:

    * only knows about WER (the recipe's ``score_metric``);
    * has no per-utterance breakdown;
    * cannot produce error-analysis or predictions.csv.

This script complements the TB scrape by running a real fairseq2
inference pass on the validation split.  It re-uses the project's
existing pipeline:

    * ``preprocess.load_and_prepare()`` — same split / cache / filter
    * ``preprocess.decode_audio_entry()`` — same torchcodec-free audio path
    * ``utils.text_normalization.build_default_normalizer()`` — same text norm
    * ``metrics.MetricCalculator`` + ``analyze_substitutions`` — same metrics

so the OmniASR numbers are directly comparable to the HF models.

Fallback behaviour
------------------

If ``fairseq2`` / ``omnilingual_asr`` are not installed, OR if the model
cannot be loaded, the script exits cleanly with a non-zero rc and a
clear message.  The TB scrape (``scripts/evaluate_omniasr.py``) remains
the safety net and is still invoked by the Colab workflow.

Usage
-----

    export HUTSUL_ASR_ROOT=/content/drive/MyDrive/hutsul_asr
    python scripts/infer_omniasr_validation.py --variant omniasr-ctc-300m

Optional ``--split test`` is supported but produces a JSON with
``split=test`` only if the project's test-split manifest has been
materialised separately (see ``scripts/convert_to_omniasr_manifest.py
--include_test``); by default the project uses the validation split
because that is what fairseq2 selected the best checkpoint on.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from config import (  # noqa: E402
    DEFAULT_DATASET_NAME,
    DEFAULT_SEED,
    ProjectConfig,
    configure_logging,
    resolve_storage_layout,
    set_global_seed,
)
from metrics import (  # noqa: E402
    MetricCalculator,
    analyze_substitutions,
)
from preprocess import decode_audio_entry, load_and_prepare  # noqa: E402
from utils.text_normalization import build_default_normalizer  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class _InferenceContext:
    variant: str
    split: str
    batch_size: int
    device: str
    dtype: str
    checkpoint_path: Optional[Path]
    storage_root: Path


# ---------------------------------------------------------------------------
# fairseq2 / omnilingual-asr model loader
# ---------------------------------------------------------------------------


def _find_step_dir(layout, variant: str) -> Optional[Path]:
    """Pick the OmniASR step_<N>/ checkpoint dir.

    Mirrors ``models/omniasr_trainer.py:_find_omniasr_checkpoint`` —
    final_models/<variant>/ contains one or more step_<N>/ subdirs
    (the trainer copies the best step there post-training); we pick
    the largest N.
    """
    final_dir = layout.final_model_dir(variant)
    if not final_dir.exists():
        return None
    step_dirs = [p for p in final_dir.iterdir()
                 if p.is_dir() and p.name.startswith("step_")]
    if not step_dirs:
        # The trainer may also leave the checkpoint under checkpoints/<v>/
        ckpt_root = layout.checkpoint_dir(variant) / "checkpoints"
        if ckpt_root.exists():
            step_dirs = [p for p in ckpt_root.iterdir()
                         if p.is_dir() and p.name.startswith("step_")]
    if not step_dirs:
        return None

    def _step_num(p: Path) -> int:
        try:
            return int(p.name.split("_", 1)[1])
        except (IndexError, ValueError):
            return -1

    return sorted(step_dirs, key=_step_num)[-1]


def _try_load_fairseq2_model(
    step_dir: Path,
    device: str,
) -> Tuple[Any, Any]:
    """Return (model, tokenizer).  Raises if fairseq2/omnilingual-asr
    is not installed or the checkpoint cannot be loaded.

    The loader uses the public ``omnilingual_asr`` Python API where
    available, falling back to lower-level fairseq2 primitives.  All
    imports are kept inside this function so the module can be loaded
    on machines without fairseq2 installed.
    """
    # Public, high-level API (preferred)
    try:
        from omnilingual_asr.api import (  # type: ignore
            OmnilingualAsrInference,
        )
    except Exception:  # pragma: no cover — newer/older versions
        OmnilingualAsrInference = None  # type: ignore

    if OmnilingualAsrInference is not None:
        logger.info("Using omnilingual_asr.api.OmnilingualAsrInference")
        engine = OmnilingualAsrInference(
            checkpoint_dir=str(step_dir),
            device=device,
        )
        return engine, engine.tokenizer

    # Low-level fairseq2 fallback
    logger.info("Falling back to fairseq2 primitives for OmniASR loading")
    import torch  # noqa: WPS433
    from fairseq2.models.wav2vec2.asr import (  # type: ignore
        load_wav2vec2_asr_model,
    )
    from fairseq2.data.text import load_text_tokenizer  # type: ignore

    model = load_wav2vec2_asr_model(
        "omniASR_CTC_300M",
        device=torch.device(device),
        dtype=torch.float16 if device == "cuda" else torch.float32,
        checkpoint=str(step_dir),
    )
    model.eval()
    tokenizer = load_text_tokenizer("omniASR_tokenizer_v1")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------


def _decode_one(
    engine_or_model: Any,
    tokenizer: Any,
    samples: np.ndarray,
    sample_rate: int,
) -> str:
    """Return the decoded string for a single utterance.

    Two execution modes are supported transparently:

    1. ``OmnilingualAsrInference`` (the high-level API) exposes
       ``.transcribe(waveform, sample_rate, language="ukr_Cyrl")``.
    2. The low-level fairseq2 path: forward the audio through the model,
       greedy CTC over the logits, then ``tokenizer.create_decoder()``.
    """
    transcribe = getattr(engine_or_model, "transcribe", None)
    if callable(transcribe):
        return transcribe(
            waveform=samples,
            sample_rate=sample_rate,
            language="ukr_Cyrl",
        )

    import torch
    with torch.no_grad():
        x = torch.as_tensor(samples, dtype=torch.float16).unsqueeze(0)
        if torch.cuda.is_available():
            x = x.cuda()
        out = engine_or_model(x)
        # Defensive: handle either an object with .logits or a raw tensor
        logits = getattr(out, "logits", out)
        pred_ids = logits.argmax(dim=-1).squeeze(0).tolist()
    # Collapse CTC repeats + remove blanks
    blank_id = getattr(tokenizer, "blank_id", 0)
    collapsed: List[int] = []
    prev = None
    for tid in pred_ids:
        if tid == blank_id:
            prev = None
            continue
        if tid != prev:
            collapsed.append(tid)
            prev = tid
    decoder = tokenizer.create_decoder() if hasattr(tokenizer, "create_decoder") \
        else tokenizer
    return decoder.decode(collapsed)


def run_inference(
    engine_or_model: Any,
    tokenizer: Any,
    split: Any,
    *,
    audio_column: str,
    text_column: str,
    sample_rate: int,
    normalizer: Any,
) -> Tuple[List[str], List[str]]:
    predictions: List[str] = []
    references: List[str] = []
    n = len(split)
    logger.info("Running OmniASR inference on %d utterances", n)

    for i in range(n):
        row = split[i]
        audio = row[audio_column]
        samples, sr = decode_audio_entry(audio, sample_rate)
        assert sr == sample_rate, f"resample mismatch: {sr} vs {sample_rate}"

        try:
            raw_hyp = _decode_one(
                engine_or_model, tokenizer, samples, sample_rate
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Decode failed for utterance %d (%s); treating as empty.",
                i, exc,
            )
            raw_hyp = ""

        ref = row[text_column] or ""
        predictions.append(normalizer(raw_hyp))
        references.append(normalizer(ref))

        if (i + 1) % 100 == 0:
            logger.info("  ... %d / %d", i + 1, n)

    return predictions, references


# ---------------------------------------------------------------------------
# Output writers — mirror evaluate.py's contract exactly
# ---------------------------------------------------------------------------


def _write_predictions_csv(
    out_path: Path,
    predictions: List[str],
    references: List[str],
    mc: MetricCalculator,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["reference", "prediction", "wer", "cer"])
        for ref, pred in zip(references, predictions):
            writer.writerow([
                ref,
                pred,
                f"{mc.compute_wer([pred], [ref]):.6f}",
                f"{mc.compute_cer([pred], [ref]):.6f}",
            ])
    logger.info("Wrote %s", out_path)


def _write_predictions_txt(
    out_path: Path,
    predictions: List[str],
    references: List[str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for ref, pred in zip(references, predictions):
            fh.write(f"REF: {ref}\nHYP: {pred}\n\n")
    logger.info("Wrote %s", out_path)


def _write_test_results(
    out_path: Path,
    *,
    ctx: _InferenceContext,
    wer: float,
    cer: float,
    n_samples: int,
    source_note: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _json_safe(v: float) -> Optional[float]:
        if v is None:  # type: ignore[redundant-expr]
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if math.isnan(f) or math.isinf(f):
            return None
        return f

    payload: Dict[str, Any] = {
        "checkpoint": str(ctx.checkpoint_path) if ctx.checkpoint_path else "",
        "run_name": ctx.variant,
        "model_family": "omniasr",
        "is_peft": False,
        "base_model_id": "omniASR_CTC_300M",
        "split": ctx.split,
        "num_samples": n_samples,
        "wer": _json_safe(wer),
        "cer": _json_safe(cer),
        "inference": {
            "batch_size": ctx.batch_size,
            "max_new_tokens": None,
            "num_beams": 1,
            "language": "uk",
            "task": "transcribe",
        },
        "device": ctx.device,
        "dtype": ctx.dtype,
        "storage_root": str(ctx.storage_root),
        "source": source_note,
    }
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Wrote %s", out_path)


def _write_best_metric(
    out_path: Path,
    *,
    best_step: int,
    val_cer: Optional[float],
    val_wer: Optional[float],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _clean(v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if math.isnan(f) or math.isinf(f):
            return None
        return f

    cer_clean = _clean(val_cer)
    wer_clean = _clean(val_wer)
    if cer_clean is not None:
        best_metric, best_value = "eval_cer", cer_clean
        secondary_metric, secondary_value = "eval_wer", wer_clean
    else:
        best_metric, best_value = "eval_wer", wer_clean
        secondary_metric, secondary_value = "eval_cer", cer_clean

    payload = {
        "best_step": int(best_step),
        "best_metric": best_metric,
        "best_value": best_value,
        "secondary_metric": secondary_metric,
        "secondary_value": secondary_value,
        "source": (
            "scripts/infer_omniasr_validation.py — overall metrics computed "
            "from a real fairseq2 inference pass on the project's "
            "validation split (or test if --split test was passed)."
        ),
    }
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run OmniASR (fairseq2) inference on the project's validation "
            "split and emit the same evaluation artefacts the HF models "
            "produce, with WER/CER computed by the project's metrics "
            "pipeline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--variant", default="omniasr-ctc-300m")
    p.add_argument(
        "--split",
        default="validation",
        choices=["validation", "test"],
        help=(
            "Which split to evaluate.  Defaults to validation because the "
            "fairseq2 recipe selects the best checkpoint on validation."
        ),
    )
    p.add_argument("--storage_root", default=os.environ.get("HUTSUL_ASR_ROOT"))
    p.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    p.add_argument("--dataset_config", default=None)
    p.add_argument("--audio_column", default=None)
    p.add_argument("--text_column", default=None)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--batch_size", type=int, default=1,
                   help="Per-utterance loop; batching is delegated to the "
                        "OmniASR loader.")
    p.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    configure_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    set_global_seed(args.seed, deterministic=False)

    layout = resolve_storage_layout(
        Path(args.storage_root) if args.storage_root else None,
        refresh=bool(args.storage_root),
    )
    layout.ensure()
    logger.info("Storage root: %s", layout.root)

    step_dir = _find_step_dir(layout, args.variant)
    if step_dir is None:
        logger.error(
            "No fairseq2 step_<N> checkpoint found for variant %r under "
            "%s.  Train OmniASR first, or check the path.",
            args.variant, layout.final_model_dir(args.variant),
        )
        return 1
    logger.info("Loading OmniASR checkpoint: %s", step_dir)

    try:
        engine, tokenizer = _try_load_fairseq2_model(step_dir, device="cuda")
    except ImportError as exc:
        logger.error(
            "fairseq2 / omnilingual_asr is not installed in this runtime. "
            "Install them as described in the README, then re-run.  "
            "Underlying error: %s",
            exc,
        )
        return 2
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Could not load OmniASR model from %s — falling back to the "
            "TensorBoard scrape (scripts/evaluate_omniasr.py).  Underlying "
            "error: %s",
            step_dir, exc,
        )
        return 3

    # Dataset
    project_cfg = ProjectConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        audio_column=args.audio_column,
        text_column=args.text_column,
        sample_rate=16000,
        seed=args.seed,
        preprocessed_dir=layout.preprocessed_dir("shared"),
        dataset_cache_dir=layout.datasets_cache,
        cache_dir=layout.cache,
    )
    project_cfg.ensure_dirs()
    dataset, audio_col, text_col = load_and_prepare(
        project_cfg, token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
    )

    if args.split not in dataset:
        logger.error(
            "Split %r missing from dataset (have %s)",
            args.split, list(dataset.keys()),
        )
        return 1
    split_ds = dataset[args.split]
    logger.info("Evaluating %d %s utterances", len(split_ds), args.split)

    normalizer = build_default_normalizer()
    predictions, references = run_inference(
        engine, tokenizer, split_ds,
        audio_column=audio_col, text_column=text_col,
        sample_rate=16000, normalizer=normalizer,
    )

    mc = MetricCalculator()
    overall_wer = mc.compute_wer(predictions, references)
    overall_cer = mc.compute_cer(predictions, references)
    error_report = analyze_substitutions(predictions, references)
    logger.info(
        "OmniASR %s: WER=%.4f CER=%.4f (n=%d)",
        args.split, overall_wer, overall_cer, len(predictions),
    )

    ctx = _InferenceContext(
        variant=args.variant,
        split=args.split,
        batch_size=args.batch_size,
        device="cuda",
        dtype="torch.float16",
        checkpoint_path=step_dir,
        storage_root=layout.root,
    )

    # Output dirs follow the HF convention exactly so the aggregator /
    # charts / val-vs-test joiner pick them up without changes.
    csv_path = layout.evaluations_csv / args.variant / "predictions.csv"
    json_path = layout.evaluations_json / args.variant / "test_results.json"
    err_path = layout.evaluations_json / args.variant / "error_analysis.json"
    txt_path = layout.evaluations_predictions / args.variant / "predictions.txt"
    bm_path = layout.checkpoint_dir(args.variant) / "best_metric.json"

    _write_predictions_csv(csv_path, predictions, references, mc)
    _write_predictions_txt(txt_path, predictions, references)
    _write_test_results(
        json_path, ctx=ctx,
        wer=overall_wer, cer=overall_cer,
        n_samples=len(predictions),
        source_note=(
            "scripts/infer_omniasr_validation.py — fairseq2 inference on "
            f"the project's {args.split} split; WER/CER computed by "
            "metrics.MetricCalculator (same path as the HF models)."
        ),
    )
    error_report.to_json(err_path, top_k=50)

    # best_metric.json: step number is taken from the checkpoint dir name
    # if possible, otherwise 0.
    try:
        best_step = int(step_dir.name.split("_", 1)[1])
    except (IndexError, ValueError):
        best_step = 0
    _write_best_metric(
        bm_path,
        best_step=best_step,
        val_cer=overall_cer if args.split == "validation" else None,
        val_wer=overall_wer if args.split == "validation" else None,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
