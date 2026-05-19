"""Validation-vs-test comparison table — one row per trained variant.

Joins two existing artefacts that the project already writes:

    1. ``checkpoints/<variant>/best_metric.json``
        Written by ``utils/callbacks.py:BestCERTrackerCallback`` during
        training.  Contains ``best_step``, ``best_value`` (= validation
        CER), and ``secondary_value`` (= validation WER at that step).

    2. ``evaluations/json/<variant>/test_results.json``
        Written by ``evaluate.py``.  Contains overall test-split WER + CER.

Output formats
--------------

The same table is emitted in three forms (all under
``<storage_root>/evaluations/summary/``):

    * Console print — for sanity-checking after evaluation finishes.
    * ``val_test_table.csv`` — for spreadsheets / further processing.
    * ``val_test_table.md``  — Markdown table ready to paste into the
      diploma's Section 3.5.

Variants are discovered automatically — any subdirectory under
``<storage_root>/checkpoints/`` that has a ``best_metric.json`` is
included.  Missing files are reported as ``--`` rather than crashing.

Usage
-----

    export HUTSUL_ASR_ROOT=/content/drive/MyDrive/hutsul_asr
    python scripts/val_test_table.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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
# Row model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JoinedRow:
    variant: str
    best_step: Optional[int]
    val_cer: Optional[float]
    val_wer: Optional[float]
    test_cer: Optional[float]
    test_wer: Optional[float]
    n_test: Optional[int]
    notes: str = ""

    @staticmethod
    def _fmt_pct(v: Optional[float]) -> str:
        return "--" if v is None else f"{v * 100:.2f}"

    @staticmethod
    def _fmt_step(s: Optional[int]) -> str:
        return "--" if s is None else str(s)

    @staticmethod
    def _fmt_int(s: Optional[int]) -> str:
        return "--" if s is None else str(s)

    def to_console(self) -> str:
        return (
            f"{self.variant:<24} "
            f"{self._fmt_step(self.best_step):>10} "
            f"{self._fmt_pct(self.val_cer):>9} "
            f"{self._fmt_pct(self.val_wer):>9} "
            f"{self._fmt_pct(self.test_cer):>10} "
            f"{self._fmt_pct(self.test_wer):>10} "
            f"{self._fmt_int(self.n_test):>6} "
            f"{self.notes}"
        )

    def to_csv_row(self) -> Dict[str, Any]:
        return {
            "variant":   self.variant,
            "best_step": self.best_step if self.best_step is not None else "",
            "val_cer":   "" if self.val_cer is None else f"{self.val_cer:.6f}",
            "val_wer":   "" if self.val_wer is None else f"{self.val_wer:.6f}",
            "test_cer":  "" if self.test_cer is None else f"{self.test_cer:.6f}",
            "test_wer":  "" if self.test_wer is None else f"{self.test_wer:.6f}",
            "val_cer_pct":  self._fmt_pct(self.val_cer),
            "val_wer_pct":  self._fmt_pct(self.val_wer),
            "test_cer_pct": self._fmt_pct(self.test_cer),
            "test_wer_pct": self._fmt_pct(self.test_wer),
            "n_test":    self.n_test if self.n_test is not None else "",
            "notes":     self.notes,
        }

    def to_markdown_row(self) -> str:
        return (
            f"| `{self.variant}` "
            f"| {self._fmt_step(self.best_step)} "
            f"| {self._fmt_pct(self.val_cer)} "
            f"| {self._fmt_pct(self.val_wer)} "
            f"| {self._fmt_pct(self.test_cer)} "
            f"| {self._fmt_pct(self.test_wer)} "
            f"| {self._fmt_int(self.n_test)} "
            f"| {self.notes or ''} |"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


def _discover_variants(layout: StorageLayout) -> List[str]:
    """Discover variants from BOTH the checkpoints/ and evaluations/json/
    trees — so the table also includes evaluation-only runs (e.g. OmniASR)
    that have no transformers-style ``best_metric.json``."""
    names: set = set()
    if layout.checkpoints.exists():
        for p in layout.checkpoints.iterdir():
            if p.is_dir():
                names.add(p.name)
    if layout.evaluations_json.exists():
        for p in layout.evaluations_json.iterdir():
            if p.is_dir():
                names.add(p.name)
    return sorted(names)


def _row_for_variant(layout: StorageLayout, variant: str) -> JoinedRow:
    bm_path = layout.checkpoint_dir(variant) / "best_metric.json"
    tr_path = layout.evaluations_json / variant / "test_results.json"

    bm = _read_json(bm_path) or {}
    tr = _read_json(tr_path) or {}

    notes_parts: List[str] = []
    if not bm:
        notes_parts.append("no best_metric.json")
    if not tr:
        notes_parts.append("no test_results.json")

    best_step = bm.get("best_step")
    if isinstance(best_step, str) and best_step.isdigit():
        best_step = int(best_step)
    elif not isinstance(best_step, int):
        best_step = None

    # ``best_metric.json`` (transformers callback) stores the *best CER* in
    # ``best_value`` and the matching *WER* in ``secondary_value``.  The
    # OmniASR fallback (``evaluate_omniasr.py``) preserves the same schema.
    bm_best_metric = str(bm.get("best_metric") or "").lower()
    bm_sec_metric = str(bm.get("secondary_metric") or "").lower()

    def _maybe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            v = float(value)
            return v if not (v != v) else None  # filter NaN
        except (TypeError, ValueError):
            return None

    val_cer: Optional[float] = None
    val_wer: Optional[float] = None
    # Map (best_metric, best_value) and (secondary_metric, secondary_value)
    # onto val_cer/val_wer regardless of which one was the optimisation
    # target — OmniASR optimises WER, transformers trainers optimise CER.
    for metric_name, raw in (
        (bm_best_metric, bm.get("best_value")),
        (bm_sec_metric, bm.get("secondary_value")),
    ):
        if "cer" in metric_name and val_cer is None:
            val_cer = _maybe_float(raw)
        elif "wer" in metric_name and val_wer is None:
            val_wer = _maybe_float(raw)

    # ``evaluate.py`` always sets ``test_results.json:split``; the
    # OmniASR fallback writes ``split="validation"`` because it has no
    # held-out test pass.  Route the numbers into the correct columns
    # so OmniASR is reported under val_* only, not test_*.
    tr_split = str(tr.get("split") or "").lower()
    tr_cer = _maybe_float(tr.get("cer"))
    tr_wer = _maybe_float(tr.get("wer"))
    test_cer: Optional[float] = None
    test_wer: Optional[float] = None
    if tr_split == "test":
        test_cer = tr_cer
        test_wer = tr_wer
    elif tr_split == "validation":
        # Prefer best_metric.json for val numbers; fall back to the JSON
        # the OmniASR helper writes if best_metric.json was missing.
        if val_cer is None:
            val_cer = tr_cer
        if val_wer is None:
            val_wer = tr_wer
        notes_parts.append("val-only (no test pass available)")

    n_test = int(tr["num_samples"]) if (
        tr_split == "test" and isinstance(tr.get("num_samples"), int)
    ) else None

    return JoinedRow(
        variant=variant,
        best_step=best_step,
        val_cer=val_cer,
        val_wer=val_wer,
        test_cer=test_cer,
        test_wer=test_wer,
        n_test=n_test,
        notes="; ".join(notes_parts),
    )


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _write_csv(rows: List[JoinedRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flat = [r.to_csv_row() for r in rows]
    fields = list(flat[0].keys()) if flat else [
        "variant", "best_step", "val_cer", "val_wer",
        "test_cer", "test_wer", "n_test", "notes",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in flat:
            writer.writerow(r)
    logger.info("Wrote %s", out_path)


def _write_markdown(rows: List[JoinedRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Validation vs Test — per variant",
        "",
        "All values are percentages. ``--`` = source file missing.",
        "",
        "| Variant | Best step | val CER% | val WER% | test CER% | test WER% | N test | Notes |",
        "|---------|-----------|----------|----------|-----------|-----------|--------|-------|",
    ]
    for r in rows:
        lines.append(r.to_markdown_row())
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build(layout: StorageLayout) -> List[JoinedRow]:
    variants = _discover_variants(layout)
    if not variants:
        logger.warning(
            "No variants found under %s or %s",
            layout.checkpoints, layout.evaluations_json,
        )
        return []
    return [_row_for_variant(layout, v) for v in variants]


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Print and persist the val-vs-test comparison table from "
            "best_metric.json + test_results.json."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--storage_root",
        default=os.environ.get("HUTSUL_ASR_ROOT"),
    )
    p.add_argument("--output_dir", default=None)
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

    rows = build(layout)
    if not rows:
        return 1

    print()
    print(
        f"{'variant':<24} {'best_step':>10} {'val_CER%':>9} "
        f"{'val_WER%':>9} {'test_CER%':>10} {'test_WER%':>10} {'N':>6} notes"
    )
    print("-" * 90)
    for r in rows:
        print(r.to_console())
    print()

    out_dir = Path(args.output_dir) if args.output_dir \
        else (layout.evaluations / "summary")
    _write_csv(rows, out_dir / "val_test_table.csv")
    _write_markdown(rows, out_dir / "val_test_table.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
