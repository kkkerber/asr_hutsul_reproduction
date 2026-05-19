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
import re
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


# ---------------------------------------------------------------------------
# Tag matching
# ---------------------------------------------------------------------------
#
# fairseq2's ``TensorBoardRecorder`` writes scalar names from the recipe's
# ``MetricBag``.  In the omnilingual-asr wav2vec2 ASR recipe these names are
# the metric-class ``display_name`` values, which can be human-readable
# phrases — e.g. ``"Word Error Rate (WER)"``, ``"Loss/Train"``, ``"CTC
# Loss/Valid"``, ``"Score"``.  fairseq2 also splits metrics across SIBLING
# event-file subdirs ``<tb_dir>/train/`` and ``<tb_dir>/valid/``, so a
# single ``EventAccumulator`` only sees one half.
#
# Strategy:
#   1. Aggregate scalars across EVERY event-file subdir under ``tb_dir``.
#   2. Match tags by normalised substring on the metric tokens (``wer``,
#      ``cer``, ``loss``) instead of exact string equality.
#   3. Infer phase (train vs valid) from BOTH the tag string AND the
#      enclosing subdir name — fairseq2 omits the phase from the tag
#      when the subdir name is the phase.
#
# A relaxed fallback pass kicks in when phase-aware matching produces
# nothing, so e.g. CER logged only inside ``valid/`` is still recovered.

# Short metric tokens (``wer``, ``cer``, ``loss``) need WORD-BOUNDARY
# matching against the original lowercased tag string to avoid false
# positives like ``"wer"`` inside ``"power"`` / ``"newer"``.  Long
# human-readable forms only appear after non-alphabetic stripping (because
# the original has whitespace between the words) and can safely be matched
# by substring against the normalised tag.
_METRIC_SHORT_TOKENS: Dict[str, Tuple[str, ...]] = {
    "wer":  ("wer",),
    "cer":  ("cer",),
    "loss": ("loss",),
}
_METRIC_LONG_TOKENS: Dict[str, Tuple[str, ...]] = {
    "wer":  ("worderrorrate",),
    "cer":  ("charactererrorrate",),
    "loss": ("ctcloss",),
}

# Tokens that identify the validation phase.  ``dev`` is included for
# datasets that label the held-out split as ``dev`` rather than ``valid``.
_VALID_TOKENS: Tuple[str, ...] = (
    "valid", "validation", "eval", "evaluation", "dev",
)
_TRAIN_TOKENS: Tuple[str, ...] = ("train", "training")

_NORM_RE = re.compile(r"[^a-z0-9]+")

# Compiled word-boundary patterns per short token, cached on first use.
_WORD_BOUND_CACHE: Dict[str, "re.Pattern[str]"] = {}


def _norm(s: str) -> str:
    """Lowercase + strip every non-alphanumeric character.

    Examples:
        ``"Word Error Rate (WER)"`` -> ``"worderrorratewer"``
        ``"Loss/Train"``            -> ``"losstrain"``
        ``"valid/wer"``             -> ``"validwer"``
    """
    return _NORM_RE.sub("", s.lower())


def _word_bound_pattern(token: str) -> "re.Pattern[str]":
    pat = _WORD_BOUND_CACHE.get(token)
    if pat is None:
        # ``(?<![a-z])`` / ``(?![a-z])`` are letter-only word boundaries
        # — digits and underscores count as "non-letter" so tags like
        # ``"valid_wer1"`` still match.
        pat = re.compile(rf"(?<![a-z]){re.escape(token)}(?![a-z])")
        _WORD_BOUND_CACHE[token] = pat
    return pat


def _has_metric(metric: str, original: str, normalised: str) -> bool:
    """Return True if ``original`` (the raw tag string) or ``normalised``
    contains the given metric in a way that is not a substring accident.

    Strategy:
        1. Short form (``wer`` / ``cer`` / ``loss``) — match against the
           lowercased original with letter-only word boundaries.  Catches
           ``"WER"``, ``"valid/wer"``, ``"Word Error Rate (WER)"``,
           ``"Loss/Train"``, ``"Train Loss"``, etc.
        2. Long form (``worderrorrate``, ``charactererrorrate``,
           ``ctcloss``) — substring match on the normalised string.
           Catches tags whose human-readable phrase has no parenthesised
           short form (e.g. just ``"Word Error Rate"``).
    """
    lo = original.lower()
    for tok in _METRIC_SHORT_TOKENS[metric]:
        if _word_bound_pattern(tok).search(lo):
            return True
    for tok in _METRIC_LONG_TOKENS[metric]:
        if tok in normalised:
            return True
    return False


def _has_any(needle_tokens: Iterable[str], haystack_norm: str) -> bool:
    """Substring match on the normalised string.  Used only for phase
    detection (``valid`` / ``train`` / ``dev`` / ``eval``) — these never
    suffer from false positives in realistic fairseq2 tag vocabularies."""
    return any(tok in haystack_norm for tok in needle_tokens)


@dataclass(frozen=True)
class OmniMetric:
    step: int
    value: float


@dataclass(frozen=True)
class _TaggedSeries:
    """Resolved (event-dir, tag, series) bundle for diagnostics + decoding."""

    event_dir: Path
    tag: str
    series: List[OmniMetric]


# ---------------------------------------------------------------------------
# TensorBoard discovery & loading
# ---------------------------------------------------------------------------


def _resolve_tb_dir(layout: StorageLayout, variant: str) -> Optional[Path]:
    """Find the directory where TensorBoard events actually live.

    fairseq2's output layout is not standardised in this project: depending
    on the run, events can land in any of these locations.  We try the
    most-likely path first and accept the first one that contains
    ``events.out.tfevents.*`` files anywhere in its tree.
    """
    candidates: List[Path] = [
        layout.checkpoint_dir(variant) / "tb",
        layout.tensorboard_dir(variant),
        layout.checkpoint_dir(variant) / "tensorboard",
        layout.checkpoint_dir(variant),
    ]
    for c in candidates:
        try:
            if c.exists() and any(c.rglob("events.out.tfevents.*")):
                return c
        except OSError:
            continue
    return None


def _list_event_dirs(tb_dir: Path) -> List[Path]:
    """Return every subdir under ``tb_dir`` that holds at least one event
    file (the parent of each ``events.out.tfevents.*``).  Preserves
    discovery order; deduplicates."""
    dirs: List[Path] = []
    if any(tb_dir.glob("events.out.tfevents.*")):
        dirs.append(tb_dir)
    for event in sorted(tb_dir.rglob("events.out.tfevents.*")):
        if event.parent not in dirs:
            dirs.append(event.parent)
    # ``dict.fromkeys`` is the canonical order-preserving de-dupe.
    return list(dict.fromkeys(dirs))


def _load_all_accumulators(
    event_dirs: List[Path],
) -> List[Tuple[Path, Any]]:
    """Load an :class:`EventAccumulator` per event subdir.

    Returns a list of (event_dir, accumulator) pairs.  Defensive: any
    individual accumulator that fails to reload is skipped with a warning.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError as exc:
        raise SystemExit(
            "tensorboard is required to scrape OmniASR metrics. "
            "Install it via `pip install tensorboard`."
        ) from exc

    out: List[Tuple[Path, Any]] = []
    for d in event_dirs:
        try:
            acc = EventAccumulator(str(d), size_guidance={"scalars": 0})
            acc.Reload()
        except Exception as exc:  # noqa: BLE001 — tensorboard raises subclasses
            logger.warning("Could not load events under %s: %s", d, exc)
            continue
        out.append((d, acc))
    return out


def _all_scalar_tags(accumulators: List[Tuple[Path, Any]]
                     ) -> List[Tuple[Path, str]]:
    """Flatten (event_dir, tag) across every accumulator for logging."""
    flat: List[Tuple[Path, str]] = []
    for d, acc in accumulators:
        for tag in acc.Tags().get("scalars", []):
            flat.append((d, tag))
    return flat


# ---------------------------------------------------------------------------
# Fuzzy metric resolution
# ---------------------------------------------------------------------------


def _find_metric(
    accumulators: List[Tuple[Path, Any]],
    metric: str,
    phase: str,
) -> Optional[_TaggedSeries]:
    """Locate the (event_dir, tag, series) triple best matching ``metric``
    in ``phase`` ("train" or "valid").

    Matching algorithm:

    1. Strict pass: tag normalises to contain a metric token AND either the
       tag itself or its enclosing event-dir name contains a phase token.
    2. Relaxed pass (run only when strict fails): drop the phase
       constraint and match by metric token alone — this rescues metrics
       that fairseq2 only logs in one phase (e.g. CER in ``valid/`` with
       no phase token in the tag).
    3. Tie-break: shorter normalised tag wins (less noise).  Within
       equally-short tags, prefer event dirs whose name contains the
       phase token.
    """
    phase_tokens = _VALID_TOKENS if phase == "valid" else _TRAIN_TOKENS

    def _score(event_dir: Path, tag: str) -> Optional[Tuple[int, int]]:
        tag_norm = _norm(tag)
        if not _has_metric(metric, tag, tag_norm):
            return None
        # When the metric we're resolving is "loss" but the tag refers to
        # a different loss-shaped metric (e.g. a WER tag tied into a CTC
        # loss family), the metric matcher would still claim the tag.
        # Filter out tags that ALSO match a different (more specific)
        # metric so we never confuse WER with loss or vice versa.
        for other in ("wer", "cer", "loss"):
            if other == metric:
                continue
            if _has_metric(other, tag, tag_norm):
                return None
        dir_norm = _norm(event_dir.name)
        phase_in_tag = _has_any(phase_tokens, tag_norm)
        phase_in_dir = _has_any(phase_tokens, dir_norm)
        return (len(tag_norm), 0 if (phase_in_tag or phase_in_dir) else 1)

    # Pass 1 — phase-aware.
    strict_candidates: List[Tuple[Tuple[int, int], Path, Any, str]] = []
    for d, acc in accumulators:
        for tag in acc.Tags().get("scalars", []):
            sc = _score(d, tag)
            if sc is None:
                continue
            if sc[1] == 0:  # phase matches
                strict_candidates.append((sc, d, acc, tag))

    chosen = strict_candidates

    # Pass 2 — relaxed (metric token only).
    if not chosen:
        for d, acc in accumulators:
            for tag in acc.Tags().get("scalars", []):
                sc = _score(d, tag)
                if sc is None:
                    continue
                chosen.append((sc, d, acc, tag))

    if not chosen:
        return None

    chosen.sort(key=lambda c: c[0])
    _, event_dir, acc, tag = chosen[0]
    try:
        events = acc.Scalars(tag)
    except KeyError:
        return None
    series = [OmniMetric(step=int(e.step), value=float(e.value))
              for e in events]
    return _TaggedSeries(event_dir=event_dir, tag=tag, series=series)


def _best_min(series: List[OmniMetric]) -> Optional[OmniMetric]:
    """Return the step with the lowest non-NaN value, or ``None`` if the
    series is empty or fully NaN.

    Both WER and CER are minimisation targets — lower is better — so the
    same helper covers both.  ``train/loss`` is also minimised, but the
    script only uses it for diagnostic logging, not for best-step
    selection.
    """
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
    wer: Optional[float],
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
    # ``wer`` / ``cer`` may be NaN or None depending on what the recipe
    # actually logged.  Emit ``null`` rather than NaN so the resulting
    # JSON validates against strict parsers; aggregate_results.py and
    # val_test_table.py both tolerate null via their ``_maybe_float``
    # helpers.
    def _json_safe(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(v) or math.isinf(v):
            return None
        return v

    payload: Dict[str, Any] = {
        "checkpoint": str(checkpoint_path) if checkpoint_path else "",
        "run_name": variant,
        "model_family": "omniasr",
        "is_peft": False,
        "base_model_id": "omniASR_CTC_300M",
        "split": "validation",
        "num_samples": 0,
        "wer": _json_safe(wer),
        "cer": _json_safe(cer),
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
    val_wer: Optional[float],
) -> None:
    """Emit a ``best_metric.json`` matching the schema written by
    :class:`utils.callbacks.BestCERTrackerCallback`.

    Even though OmniASR is selected on WER (not CER) inside fairseq2 we
    populate the file so the val-vs-test joiner picks it up — ``best_value``
    holds whichever of WER/CER is available, and ``secondary_value`` holds
    the other.  Either ``val_cer`` or ``val_wer`` may be ``None`` (or NaN),
    in which case ``best_metric`` defaults to whichever exists.
    """
    def _clean(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(v) or math.isinf(v):
            return None
        return v

    cer_clean = _clean(val_cer)
    wer_clean = _clean(val_wer)

    # Prefer CER as best_value to match BestCERTrackerCallback's contract;
    # fall back to WER if CER is unavailable.  When neither is available
    # the file is still written so downstream tooling sees an explicit
    # null pair rather than a missing file.
    if cer_clean is not None:
        best_metric, best_value = "eval_cer", cer_clean
        secondary_metric, secondary_value = "eval_wer", wer_clean
    else:
        best_metric, best_value = "eval_wer", wer_clean
        secondary_metric, secondary_value = "eval_cer", cer_clean

    payload: Dict[str, Any] = {
        "best_step": int(best_step),
        "best_metric": best_metric,
        "best_value": best_value,
        "secondary_metric": secondary_metric,
        "secondary_value": secondary_value,
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
    """Scrape best validation WER (and CER if logged) for one OmniASR
    variant from TensorBoard events; persist as test_results.json +
    best_metric.json so OmniASR is visible in the project's aggregation /
    charts / val-vs-test tooling.

    Returns 0 on success, 1 on any unrecoverable error (missing tb_dir,
    no event files, no WER and no CER tag matched).  Every failure path
    logs the candidate locations / tags it tried so the operator can fix
    the run without re-reading the source.
    """
    # 1) Locate the TensorBoard root — try every plausible location used
    #    across this project's run history.
    tb_dir = _resolve_tb_dir(layout, variant)
    if tb_dir is None:
        logger.error(
            "Could not find any TensorBoard events for variant %r. "
            "Looked under: %s, %s, %s, %s",
            variant,
            layout.checkpoint_dir(variant) / "tb",
            layout.tensorboard_dir(variant),
            layout.checkpoint_dir(variant) / "tensorboard",
            layout.checkpoint_dir(variant),
        )
        return 1
    logger.info("Resolved TensorBoard dir: %s", tb_dir)

    # 2) Load every event-file subdir into its own accumulator — fairseq2
    #    typically logs train and valid metrics into sibling subdirs.
    event_dirs = _list_event_dirs(tb_dir)
    if not event_dirs:
        logger.error("No events.out.tfevents.* files under %s", tb_dir)
        return 1
    logger.info("Loaded %d event subdir(s): %s",
                len(event_dirs), [str(d) for d in event_dirs])

    accumulators = _load_all_accumulators(event_dirs)
    if not accumulators:
        logger.error(
            "Found event files under %s but none could be loaded by the "
            "TensorBoard EventAccumulator.",
            tb_dir,
        )
        return 1

    all_tags = _all_scalar_tags(accumulators)
    logger.info("Total scalar tags across all event dirs: %d", len(all_tags))
    if logger.isEnabledFor(logging.DEBUG):
        for d, tag in all_tags:
            logger.debug("  [%s] %s", d.name or d, tag)

    # 3) Fuzzy-resolve WER + CER + train_loss.
    wer_hit = _find_metric(accumulators, "wer", phase="valid")
    cer_hit = _find_metric(accumulators, "cer", phase="valid")
    train_hit = _find_metric(accumulators, "loss", phase="train")

    if wer_hit is not None:
        logger.info(
            "WER tag matched: %r (in %s, %d points)",
            wer_hit.tag, wer_hit.event_dir.name or wer_hit.event_dir,
            len(wer_hit.series),
        )
    if cer_hit is not None:
        logger.info(
            "CER tag matched: %r (in %s, %d points)",
            cer_hit.tag, cer_hit.event_dir.name or cer_hit.event_dir,
            len(cer_hit.series),
        )
    if train_hit is not None:
        logger.info(
            "Train-loss tag matched: %r (in %s, %d points)",
            train_hit.tag,
            train_hit.event_dir.name or train_hit.event_dir,
            len(train_hit.series),
        )

    best_wer = _best_min(wer_hit.series) if wer_hit is not None else None
    best_cer = _best_min(cer_hit.series) if cer_hit is not None else None

    # 4) Hard failure ONLY when both metrics are missing.  CER alone is
    #    enough to still produce a usable diploma row; WER alone is fine
    #    too (the headline metric for OmniASR is WER anyway).
    if best_wer is None and best_cer is None:
        tag_dump = "\n  ".join(
            f"[{d.name or d}] {t}" for d, t in all_tags
        ) or "(none)"
        logger.error(
            "No WER or CER scalar could be resolved for %r in %s. "
            "Looked for normalised tokens %s / %s in any of:\n  %s",
            variant, tb_dir,
            _METRIC_TOKENS["wer"], _METRIC_TOKENS["cer"],
            tag_dump,
        )
        return 1

    if best_wer is None:
        logger.warning(
            "No WER series resolved for %s; only CER was found. "
            "test_results.json.wer will be set to NaN and the diploma "
            "row should be labelled as CER-only.",
            variant,
        )
    if best_cer is None:
        logger.info(
            "No CER series resolved for %s (CER is optional in the "
            "fairseq2 recipe).  Reporting WER only.",
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

    # 5) Pick best_step from whichever primary metric is available.  Both
    #    paths are safe — the function never returns None here because we
    #    already returned 1 if both are None.
    primary = best_wer if best_wer is not None else best_cer
    # ``primary`` is non-None by the guard above; mypy needs the local.
    assert primary is not None
    best_step = primary.step

    wer_value: Optional[float] = best_wer.value if best_wer is not None else None
    cer_value: Optional[float] = best_cer.value if best_cer is not None else None

    checkpoint_path = _find_omniasr_checkpoint(layout, variant)

    # ``wer_value`` / ``cer_value`` may be None.  The writers below emit
    # JSON ``null`` rather than NaN so downstream parsers see a clean
    # type; aggregator + val_test_table tolerate None via their existing
    # ``_maybe_float`` helpers.
    _write_test_results(
        layout.evaluations_json / variant / "test_results.json",
        variant=variant,
        wer=wer_value,
        cer=cer_value,
        best_step=best_step,
        checkpoint_path=checkpoint_path,
        storage_root=layout.root,
    )

    _write_best_metric(
        layout.checkpoint_dir(variant) / "best_metric.json",
        best_step=best_step,
        val_cer=cer_value,
        val_wer=wer_value,
    )

    # Diagnostic: surface train-loss progression if available.  Not used
    # downstream — plot_results.py reads TensorBoard directly.
    if train_hit is not None and train_hit.series:
        last = train_hit.series[-1]
        logger.info(
            "Recovered %d train-loss events (last value=%.4f at step %d).",
            len(train_hit.series), last.value, last.step,
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
