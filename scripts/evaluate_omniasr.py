"""Minimal OmniASR (fairseq2) evaluation helper.

The OmniASR variants in this project are trained through the official
Meta fairseq2 recipe (``models/omniasr_trainer.py``).  Their final
checkpoints land at::

    <storage_root>/final_models/omniasr-ctc-300m/step_<N>/

The directory contains a fairseq2-native ``.pt`` archive and no HF
``config.json`` — therefore ``evaluate.py:_load_omniasr`` (which calls
``AutoModelForCTC.from_pretrained``) cannot load it.

Wiring full fairseq2 inference into the HF-flavoured ``evaluate.py``
would be a major architecture change.  Instead, this helper takes the
*minimal* path that keeps OmniASR visible inside the project's
aggregation / charts / val-vs-test tables:

    1.  Read the OmniASR TensorBoard event files at
        ``<storage_root>/tensorboard/omniasr-ctc-300m/`` and recover the
        best validation WER (and CER if logged) — i.e. exactly the
        number fairseq2's ``score_metric: "wer"`` selects the best
        checkpoint on.

    2.  Write that number into
        ``<storage_root>/evaluations/json/<variant>/test_results.json``
        in the same shape every other run uses, with ``split`` set to
        ``"validation"`` so the aggregator / charts pick it up
        transparently.

    3.  Drop a sibling ``best_metric.json`` under
        ``<storage_root>/checkpoints/<variant>/`` so the val-vs-test
        table joins the OmniASR row correctly.

How to obtain true test-set WER for OmniASR
-------------------------------------------

True test-set numbers require running the fairseq2 evaluation recipe.
The script prints the exact command at the end — it does NOT run it,
because that requires the omnilingual-asr repo + fairseq2 to be
installed in the runtime, and the diploma's primary OmniASR figure (best
validation WER, ≈ 10%) is already obtainable from TensorBoard.

Usage
-----

    export HUTSUL_ASR_ROOT=/content/drive/MyDrive/hutsul_asr
    python scripts/evaluate_omniasr.py --variant omniasr-ctc-300m
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    StorageLayout,
    configure_logging,
    resolve_storage_layout,
)

logger = logging.getLogger(__name__)


# fairseq2 / omnilingual-asr log these tag families under TensorBoard.
# Multiple candidates are tried in order — different fairseq2 versions
# emit slightly different prefixes (``valid/*`` vs ``Validation/*`` vs
# ``validate/*``).  Lower-cased + trailing slash semantics keeps lookup
# tolerant.
_WER_TAGS: Tuple[str, ...] = (
    "validate/wer", "valid/wer", "Validation/wer", "valid_wer", "validate_wer",
    "eval/wer", "Eval/wer",
)
_CER_TAGS: Tuple[str, ...] = (
    "validate/cer", "valid/cer", "Validation/cer", "valid_cer", "validate_cer",
    "eval/cer", "Eval/cer",
)
_TRAIN_LOSS_TAGS: Tuple[str, ...] = (
    "train/loss", "Train/loss", "loss", "train_loss",
)


@dataclass(frozen=True)
class OmniMetric:
    step: int
    value: float


# ---------------------------------------------------------------------------
# TensorBoard scraping
# ---------------------------------------------------------------------------


def _load_event_accumulator(tb_dir: Path):
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError as exc:
        raise SystemExit(
            "tensorboard is required to scrape OmniASR metrics. "
            "Install it via `pip install tensorboard`."
        ) from exc

    candidates: List[Path] = []
    if any(tb_dir.glob("events.out.tfevents.*")):
        candidates.append(tb_dir)
    for event in sorted(tb_dir.rglob("events.out.tfevents.*")):
        candidates.append(event.parent)
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        return None

    preferred = None

    for c in candidates:
        if "valid" in str(c).lower():
            preferred = c
            break

    if preferred is None:
        preferred = candidates[0]

    acc = EventAccumulator(
        str(preferred),
        size_guidance={"scalars": 0}
    )
    acc.Reload()
    return acc


def _series(acc, candidate_tags: Iterable[str]
            ) -> List[OmniMetric]:
    available = set(acc.Tags().get("scalars", []))
    for tag in candidate_tags:
        if tag in available:
            events = acc.Scalars(tag)
            return [OmniMetric(step=e.step, value=float(e.value))
                    for e in events]
    return []


def _best_min(series: List[OmniMetric]) -> Optional[OmniMetric]:
    if not series:
        return None
    valid = [m for m in series if not math.isnan(m.value)]
    if not valid:
        return None
    return min(valid, key=lambda m: m.value)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_test_results(
    out_path: Path,
    *,
    variant: str,
    wer: float,
    cer: Optional[float],
    best_step: Optional[int],
    checkpoint_path: Optional[Path],
    storage_root: Path,
) -> None:
    """Emit a JSON file with the same shape as ``evaluate.py`` produces.

    Critical fields consumed downstream:

        * ``model_family``  — anchors aggregation / chart legend.
        * ``split``         — ``"validation"`` so aggregator labels it
          honestly; the diploma can footnote the difference.
        * ``wer`` / ``cer`` — overall percentages as fractions [0, 1].
        * ``num_samples``   — fairseq2 does not expose this through the
          event file; left at 0 with an explanatory note.

    A ``source`` field is added (not used by other tools) to make the
    provenance obvious in the diploma's appendices.
    """
    payload: Dict[str, Any] = {
        "checkpoint": str(checkpoint_path) if checkpoint_path else "",
        "run_name": variant,
        "model_family": "omniasr",
        "is_peft": False,
        "base_model_id": "omniASR_CTC_300M",
        "split": "validation",
        "num_samples": 0,
        "wer": float(wer),
        "cer": float(cer) if cer is not None else float("nan"),
        "inference": {
            "batch_size": None,
            "max_new_tokens": None,
            "num_beams": None,
            "language": "uk",
            "task": "transcribe",
        },
        "device": "fairseq2-recipe",
        "dtype": "float16",
        "storage_root": str(storage_root),
        "source": (
            "scripts/evaluate_omniasr.py — best validation WER scraped "
            "from fairseq2 TensorBoard events (no held-out test pass "
            "performed; OmniASR fine-tunes save fairseq2-native "
            "checkpoints incompatible with evaluate.py:_load_omniasr)."
        ),
        "best_step": best_step,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    val_wer: float,
) -> None:
    """Emit a ``best_metric.json`` matching the schema written by
    :class:`utils.callbacks.BestCERTrackerCallback`.

    Even though OmniASR is selected on WER (not CER) inside fairseq2 we
    populate the file so the val-vs-test joiner picks it up — ``best_value``
    holds whichever of WER/CER is available, and ``secondary_value`` holds
    the other.
    """
    payload: Dict[str, Any] = {
        "best_step": int(best_step),
        "best_metric": "eval_wer" if val_cer is None else "eval_cer",
        "best_value": float(val_wer) if val_cer is None else float(val_cer),
        "secondary_metric": "eval_wer" if val_cer is not None else "eval_cer",
        "secondary_value": (
            float(val_wer) if val_cer is not None else None
        ),
        "source": (
            "scripts/evaluate_omniasr.py — recovered from fairseq2 "
            "TensorBoard events because the OmniASR trainer does not "
            "register BestCERTrackerCallback."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Optional: fairseq2 test-set evaluation command
# ---------------------------------------------------------------------------


def _print_fairseq2_test_eval_command(
    layout: StorageLayout, variant: str
) -> None:
    print()
    print("-" * 72)
    print("Optional — to obtain TRUE test-set WER for OmniASR, run the")
    print("fairseq2 evaluation recipe yourself.  It is not invoked here")
    print("because it requires the omnilingual-asr repo + fairseq2 in the")
    print("runtime.  Suggested command (adjust manifest split as needed):")
    print()
    print(f"  python scripts/convert_to_omniasr_manifest.py \\")
    print(f"      --out_dir {layout.preprocessed / 'omniasr' / 'manifest'} \\")
    print(f"      --include_test")
    print()
    print(f"  # then re-run the recipe with regime.num_steps=0 against the")
    print(f"  # test split as valid_split (see configs/omniasr/ctc-finetune.yaml).")
    print(f"  # The recipe will emit validate/wer for the test manifest.")
    print("-" * 72)
    print()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _find_omniasr_checkpoint(layout: StorageLayout, variant: str
                             ) -> Optional[Path]:
    final_dir = layout.final_model_dir(variant)
    if final_dir.exists():
        # The OmniASR trainer copies the best step_<N>/ directory in here.
        step_dirs = [p for p in final_dir.iterdir()
                     if p.is_dir() and p.name.startswith("step_")]
        if step_dirs:
            return sorted(step_dirs, key=lambda p: p.name)[-1]
    return final_dir if final_dir.exists() else None


def evaluate_omniasr(
    layout: StorageLayout,
    variant: str,
) -> int:
    tb_dir = layout.checkpoint_dir(variant) / "tb"
    if not tb_dir.exists():
        logger.error(
            "TensorBoard directory %s does not exist — was %s trained "
            "and were event files written?",
            tb_dir, variant,
        )
        return 1

    acc = _load_event_accumulator(tb_dir)
    if acc is None:
        logger.error("No TensorBoard event files found under %s", tb_dir)
        return 1

    wer_series = _series(acc, _WER_TAGS)
    cer_series = _series(acc, _CER_TAGS)
    train_loss = _series(acc, _TRAIN_LOSS_TAGS)

    best_wer = _best_min(wer_series)
    best_cer = _best_min(cer_series)

    if best_wer is None and best_cer is None:
        logger.error(
            "Found event files under %s but neither WER nor CER scalars "
            "were logged (looked for tags: %s / %s).  Available scalar "
            "tags were: %s",
            tb_dir, _WER_TAGS, _CER_TAGS,
            sorted(acc.Tags().get("scalars", [])),
        )
        return 1

    if best_wer is None and best_cer is not None:
        logger.warning(
            "No validate/wer logged for %s; falling back to validate/cer. "
            "The diploma table should label this value as CER, not WER.",
            variant,
        )
    if best_wer is not None:
        logger.info(
            "Best validation WER for %s: %.4f at step %d",
            variant, best_wer.value, best_wer.step,
        )
    if best_cer is not None:
        logger.info(
            "Best validation CER for %s: %.4f at step %d",
            variant, best_cer.value, best_cer.step,
        )

    primary = best_wer if best_wer is not None else best_cer
    assert primary is not None  # ruled out above

    wer_value = best_wer.value if best_wer is not None else float("nan")
    cer_value = best_cer.value if best_cer is not None else None
    best_step = primary.step

    checkpoint_path = _find_omniasr_checkpoint(layout, variant)

    _write_test_results(
        layout.evaluations_json / variant / "test_results.json",
        variant=variant,
        wer=wer_value if not math.isnan(wer_value) else 0.0,
        cer=cer_value,
        best_step=best_step,
        checkpoint_path=checkpoint_path,
        storage_root=layout.root,
    )

    _write_best_metric(
        layout.checkpoint_dir(variant) / "best_metric.json",
        best_step=best_step,
        val_cer=cer_value,
        val_wer=wer_value if not math.isnan(wer_value) else 0.0,
    )

    # Also dump the train-loss series for the diploma's training-curve
    # figure — useful even though plot_results.py reads TB directly.
    if train_loss:
        n = len(train_loss)
        logger.info(
            "Recovered %d train-loss events (last value=%.4f at step %d).",
            n, train_loss[-1].value, train_loss[-1].step,
        )

    _print_fairseq2_test_eval_command(layout, variant)
    return 0


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Minimal OmniASR evaluation: scrape best validation WER/CER "
            "from fairseq2 TensorBoard events and emit "
            "test_results.json + best_metric.json so OmniASR is visible "
            "in the project's aggregation / charts / val-vs-test tools."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--variant",
        default="omniasr-ctc-300m",
        help="OmniASR variant directory name under tensorboard/.",
    )
    p.add_argument(
        "--storage_root",
        default=os.environ.get("HUTSUL_ASR_ROOT"),
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    configure_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    layout = resolve_storage_layout(
        Path(args.storage_root) if args.storage_root else None,
        refresh=bool(args.storage_root),
    )
    layout.ensure()
    return evaluate_omniasr(layout, args.variant)


if __name__ == "__main__":
    sys.exit(main())
