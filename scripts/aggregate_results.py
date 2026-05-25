"""Aggregate per-variant evaluation JSONs into a single comparison table.

For every directory under ``<storage_root>/evaluations/json/<run>/`` this
script reads:

    * ``test_results.json``   (overall WER/CER for one (variant, split) pair)
    * ``error_analysis.json`` (top substitutions / insertions / deletions,
                               and dialect-pair counts) — optional, used to
                               enrich the summary with dialect signal.

It produces two artefacts under
``<storage_root>/evaluations/summary/``:

    * ``summary.csv``  — long-format machine-readable table.
    * ``summary.md``   — Markdown table directly pasteable into the diploma.

Both files cover every (variant, split) pair found on disk; the script is
robust against partially-completed evaluation runs (it skips any run that
is missing ``test_results.json``).

Usage
-----

    export HUTSUL_ASR_ROOT=/content/drive/MyDrive/hutsul_asr
    python scripts/aggregate_results.py

Optional CLI:

    --storage_root /alt/root      override the storage root
    --output_dir   /alt/summary   override the output directory
    --verbose                     INFO logging

The script reuses the project's existing :class:`StorageLayout` so paths
stay consistent with the rest of the codebase.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make ``import config`` work regardless of CWD.
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
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalRow:
    """One row of the aggregated table, exactly one (variant, split) pair."""

    run: str
    family: str
    split: str
    n_samples: int
    wer: float
    cer: float
    checkpoint: str
    base_model_id: Optional[str]
    is_peft: bool
    dtype: str
    device: str
    dialect_pairs: Dict[str, int] = field(default_factory=dict)
    n_top_substitutions: int = 0
    n_top_insertions: int = 0
    n_top_deletions: int = 0

    def as_csv_dict(self) -> Dict[str, Any]:
        flat: Dict[str, Any] = {
            "run": self.run,
            "family": self.family,
            "split": self.split,
            "n_samples": self.n_samples,
            "wer": f"{self.wer:.6f}",
            "cer": f"{self.cer:.6f}",
            "wer_pct": f"{self.wer * 100:.2f}",
            "cer_pct": f"{self.cer * 100:.2f}",
            "checkpoint": self.checkpoint,
            "base_model_id": self.base_model_id or "",
            "is_peft": int(self.is_peft),
            "dtype": self.dtype,
            "device": self.device,
            "top_substitutions": self.n_top_substitutions,
            "top_insertions": self.n_top_insertions,
            "top_deletions": self.n_top_deletions,
        }
        # One column per tracked dialect pair, prefixed for readability.
        for key, count in self.dialect_pairs.items():
            flat[f"dialect[{key}]"] = count
        return flat


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read a JSON file, logging — but never raising — on failure."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Skipping malformed JSON %s: %s", path, exc)
        return None
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


# Variants that have been retired from the project but whose stale
# directories may still exist on Drive.  Any run/checkpoint/tensorboard
# directory whose name matches one of these prefixes is filtered out by
# the discovery functions in aggregate_results.py / plot_results.py /
# val_test_table.py, so no phantom rows or charts re-appear.
DEPRECATED_VARIANT_PREFIXES: tuple = ("parakeet",)


def _is_deprecated(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in DEPRECATED_VARIANT_PREFIXES)


def _collect_run_dirs(json_root: Path) -> List[Path]:
    """Return every direct subdirectory of ``json_root`` sorted by name,
    skipping any retired variants (see ``DEPRECATED_VARIANT_PREFIXES``)."""
    if not json_root.exists():
        logger.error("Evaluations JSON root does not exist: %s", json_root)
        return []
    return sorted(
        p for p in json_root.iterdir()
        if p.is_dir() and not _is_deprecated(p.name)
    )


def _build_row(run_dir: Path) -> Optional[EvalRow]:
    """Construct an :class:`EvalRow` from one ``evaluations/json/<run>/`` dir.

    Returns ``None`` when the run is missing the mandatory
    ``test_results.json`` file.
    """
    test_results = _read_json(run_dir / "test_results.json")
    if test_results is None:
        logger.warning(
            "Run %s missing test_results.json — skipping",
            run_dir.name,
        )
        return None

    error_analysis = _read_json(run_dir / "error_analysis.json") or {}
    dialect_pairs: Dict[str, int] = dict(
        error_analysis.get("dialect_pairs", {}) or {}
    )

    def _safe_float(value: Any) -> float:
        """Coerce to float, defaulting to NaN on None / invalid / inf.

        Required because ``scripts/evaluate_omniasr.py`` emits JSON
        ``null`` for missing WER/CER (correct per RFC 8259 — fairseq2's
        CER is not always logged), and a plain ``float(None)`` raises.
        """
        if value is None:
            return float("nan")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    return EvalRow(
        run=run_dir.name,
        family=str(test_results.get("model_family", "") or ""),
        split=str(test_results.get("split", "") or ""),
        n_samples=int(test_results.get("num_samples", 0) or 0),
        wer=_safe_float(test_results.get("wer")),
        cer=_safe_float(test_results.get("cer")),
        checkpoint=str(test_results.get("checkpoint", "") or ""),
        base_model_id=test_results.get("base_model_id"),
        is_peft=bool(test_results.get("is_peft", False)),
        dtype=str(test_results.get("dtype", "") or ""),
        device=str(test_results.get("device", "") or ""),
        dialect_pairs=dialect_pairs,
        n_top_substitutions=len(error_analysis.get("top_substitutions") or []),
        n_top_insertions=len(error_analysis.get("top_insertions") or []),
        n_top_deletions=len(error_analysis.get("top_deletions") or []),
    )


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_csv(rows: List[EvalRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    flat_rows = [r.as_csv_dict() for r in rows]
    fieldnames: List[str] = []
    seen = set()
    for row in flat_rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in flat_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    logger.info("Wrote %d rows to %s", len(rows), out_path)


def _write_markdown(rows: List[EvalRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Group by split for readability — the diploma uses test as the headline
    # table and validation as a secondary table.
    by_split: Dict[str, List[EvalRow]] = {}
    for row in rows:
        by_split.setdefault(row.split or "unknown", []).append(row)

    lines: List[str] = []
    lines.append("# ASR Hutsul — Evaluation summary")
    lines.append("")
    lines.append(
        "Auto-generated by ``scripts/aggregate_results.py``. "
        "Source: ``evaluations/json/<run>/test_results.json``."
    )
    lines.append("")

    # A consistent split ordering: test first, then validation, then anything
    # else alphabetically.
    split_order = ["test", "validation"] + sorted(
        s for s in by_split if s not in ("test", "validation")
    )

    for split in split_order:
        if split not in by_split:
            continue
        split_rows = sorted(by_split[split], key=lambda r: r.cer)
        lines.append(f"## Split: ``{split}`` ({len(split_rows)} runs)")
        lines.append("")
        lines.append(
            "| Variant | Family | N | WER, % | CER, % | Checkpoint |"
        )
        lines.append(
            "|---------|--------|---|--------|--------|------------|"
        )
        for r in split_rows:
            ckpt_disp = r.checkpoint.rsplit("/", 1)[-1] or r.checkpoint
            lines.append(
                f"| `{r.run}` | {r.family} | {r.n_samples} | "
                f"{r.wer * 100:.2f} | {r.cer * 100:.2f} | `{ckpt_disp}` |"
            )
        lines.append("")

    # Dialect-pair table — same set of pairs across rows guarantees columns
    # are aligned even if some pairs were not seen for a particular model.
    all_pairs: List[str] = []
    seen: set = set()
    for r in rows:
        for key in r.dialect_pairs:
            if key not in seen:
                seen.add(key)
                all_pairs.append(key)
    if all_pairs:
        lines.append("## Dialect-pair substitutions (Hutsul ↔ standard Uk.)")
        lines.append("")
        header = "| Variant | Split | " + " | ".join(all_pairs) + " |"
        sep = "|---------|-------|" + "|".join(["---"] * len(all_pairs)) + "|"
        lines.append(header)
        lines.append(sep)
        for r in sorted(rows, key=lambda x: (x.split, x.run)):
            cells = [str(r.dialect_pairs.get(p, 0)) for p in all_pairs]
            lines.append(
                f"| `{r.run}` | {r.split} | " + " | ".join(cells) + " |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %d-byte Markdown summary to %s",
                out_path.stat().st_size, out_path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def aggregate(
    layout: StorageLayout,
    output_dir: Optional[Path] = None,
) -> List[EvalRow]:
    """Walk the layout and return the aggregated rows.

    Side effect: writes ``summary.csv`` and ``summary.md`` under
    ``output_dir`` (defaults to ``layout.evaluations / "summary"``).
    """
    json_root = layout.evaluations_json
    out_dir = output_dir or (layout.evaluations / "summary")

    logger.info("Storage layout : %s", layout.root)
    logger.info("Reading from   : %s", json_root)
    logger.info("Writing to     : %s", out_dir)

    rows: List[EvalRow] = []
    for run_dir in _collect_run_dirs(json_root):
        row = _build_row(run_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        logger.warning(
            "No test_results.json files found under %s. "
            "Run ``evaluate.py`` for at least one variant first.",
            json_root,
        )
        return rows

    _write_csv(rows, out_dir / "summary.csv")
    _write_markdown(rows, out_dir / "summary.md")
    return rows


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Aggregate per-variant evaluation JSONs into a single "
            "comparison CSV + Markdown table."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--storage_root",
        default=os.environ.get("HUTSUL_ASR_ROOT"),
        help="Override the canonical storage root.",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Override the summary output directory.",
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

    rows = aggregate(
        layout,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    if not rows:
        return 1

    # Console summary so the operator immediately sees the numbers.
    print()
    print(f"{'variant':<32} {'split':<11} {'WER%':>8} {'CER%':>8} {'N':>6}")
    print("-" * 70)
    for r in sorted(rows, key=lambda x: (x.split, x.cer)):
        print(
            f"{r.run:<32} {r.split:<11} "
            f"{r.wer * 100:>8.2f} {r.cer * 100:>8.2f} {r.n_samples:>6}"
        )
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
