"""Publication-quality charts for the ASR Hutsul diploma project.

The script reads everything from the standard project layout — no
manual paths required — and emits matplotlib PNGs under
``<storage_root>/evaluations/summary/charts/``.

Charts produced
---------------

For each (variant, split) pair found in ``evaluations/json/<run>/``:

    * ``wer_cer_bar__<split>.png``
        Global side-by-side bar chart of WER and CER, one bar pair per
        variant for the given split.  Headline figure for the diploma.

    * ``error_breakdown__<run>.png``
        Top-10 substitutions / insertions / deletions for the given run.

    * ``dialect_pairs__<run>.png``
        Bar chart of the project's predefined Hutsul-vs-standard
        substitution pairs.

For each variant with TensorBoard event files under
``tensorboard/<variant>/``:

    * ``training_curves__<variant>.png``
        4-panel plot: train loss, eval loss, eval WER, eval CER.

Design choices
--------------

* matplotlib only (the diploma rules out seaborn / plotly).
* Fixed colour palette so figures look consistent across the report.
* 300 DPI, A4-friendly aspect ratios.
* Vector-quality text via the default DejaVu Sans font (ships with mpl,
  no system dependency).
* Each figure is fully self-described (title + legend + axis labels) so
  it stands on its own when extracted from the LaTeX/Word source.

Usage
-----

    export HUTSUL_ASR_ROOT=/content/drive/MyDrive/hutsul_asr
    python scripts/plot_results.py
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Headless backend — required when running under Colab without a display.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
# Style
# ---------------------------------------------------------------------------


CHART_DPI = 300
COLOR_WER = "#1f4e79"      # deep blue
COLOR_CER = "#c0504d"      # muted red
COLOR_SUB = "#2e75b6"
COLOR_INS = "#7f7f7f"
COLOR_DEL = "#548235"
COLOR_DIALECT = "#9e480e"
COLOR_CURVE = "#1f4e79"


def _apply_global_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": CHART_DPI,
            "savefig.dpi": CHART_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunResult:
    run: str
    family: str
    split: str
    n_samples: int
    wer: float
    cer: float


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Skipping %s: %s", path, exc)
        return None


def _safe_float(value: Any) -> float:
    """Coerce to float; default NaN on None / invalid.  Required for the
    OmniASR JSON path which emits ``null`` for unlogged metrics."""
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


# Variants that have been retired from the project but whose stale
# directories may still exist on Drive.  Filtering here keeps the
# generated charts free of phantom Parakeet (etc.) artefacts.
DEPRECATED_VARIANT_PREFIXES: tuple = ("parakeet",)


def _is_deprecated(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in DEPRECATED_VARIANT_PREFIXES)


def _collect_results(json_root: Path) -> List[RunResult]:
    out: List[RunResult] = []
    if not json_root.exists():
        logger.error("Missing evaluations JSON root: %s", json_root)
        return out
    for run_dir in sorted(p for p in json_root.iterdir() if p.is_dir()):
        if _is_deprecated(run_dir.name):
            logger.info("Skipping deprecated variant directory: %s", run_dir)
            continue
        data = _read_json(run_dir / "test_results.json")
        if data is None:
            continue
        try:
            out.append(
                RunResult(
                    run=run_dir.name,
                    family=str(data.get("model_family", "")),
                    split=str(data.get("split", "")),
                    n_samples=int(data.get("num_samples", 0) or 0),
                    wer=_safe_float(data.get("wer")),
                    cer=_safe_float(data.get("cer")),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Skipping malformed run %s: %s", run_dir, exc)
    return out


def _group_by_split(rows: Sequence[RunResult]) -> Dict[str, List[RunResult]]:
    out: Dict[str, List[RunResult]] = {}
    for r in rows:
        out.setdefault(r.split or "unknown", []).append(r)
    return out


# ---------------------------------------------------------------------------
# Chart 1 — global WER/CER bar chart
# ---------------------------------------------------------------------------


def plot_wer_cer_bar(rows: Sequence[RunResult], out_path: Path) -> None:
    if not rows:
        return

    # Sort by CER (treating NaN as +inf so CER-missing rows sort last).
    rows = sorted(rows, key=lambda r: (math.isnan(r.cer), r.cer))
    variants = [r.run for r in rows]
    wer_pct = np.array([r.wer * 100 for r in rows], dtype=float)
    cer_pct = np.array([r.cer * 100 for r in rows], dtype=float)
    x = np.arange(len(variants))
    width = 0.40

    # Width scales with the number of variants so 7 fits the column.
    fig_width = max(7.0, 1.2 * len(variants))
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))

    bars_w = ax.bar(
        x - width / 2, wer_pct, width,
        color=COLOR_WER, label="WER (%)",
        edgecolor="white", linewidth=0.6,
    )
    bars_c = ax.bar(
        x + width / 2, cer_pct, width,
        color=COLOR_CER, label="CER (%)",
        edgecolor="white", linewidth=0.6,
    )

    # NaN-safe y-axis bound — matplotlib refuses NaN/Inf limits.
    finite = np.concatenate([
        wer_pct[np.isfinite(wer_pct)],
        cer_pct[np.isfinite(cer_pct)],
    ])
    ymax = float(finite.max()) if finite.size else 1.0
    ax.set_ylim(0, ymax * 1.15 + 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=25, ha="right")
    ax.set_ylabel("Error rate, %")
    ax.set_title(
        f"WER vs CER per model "
        f"(split: {rows[0].split or 'n/a'}, "
        f"N≈{rows[0].n_samples})"
    )
    ax.legend(loc="upper right")

    # NaN values produce no bar height; skip the label rather than print
    # "nan" over an invisible bar.
    for bar, value in zip(bars_w, wer_pct):
        if math.isnan(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + ymax * 0.015,
            f"{value:.1f}",
            ha="center", va="bottom", fontsize=8,
        )
    for bar, value in zip(bars_c, cer_pct):
        if math.isnan(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + ymax * 0.015,
            f"{value:.1f}",
            ha="center", va="bottom", fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Chart 2 — error breakdown (subs / ins / dels) per run
# ---------------------------------------------------------------------------


def plot_error_breakdown(run_dir: Path, out_path: Path, top_k: int = 10) -> None:
    data = _read_json(run_dir / "error_analysis.json")
    if data is None:
        return

    subs = (data.get("top_substitutions") or [])[:top_k]
    ins  = (data.get("top_insertions")   or [])[:top_k]
    dels = (data.get("top_deletions")    or [])[:top_k]

    if not (subs or ins or dels):
        logger.info("No error-analysis content for %s", run_dir.name)
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    def _bar(ax: plt.Axes, items: List[Dict[str, Any]], title: str,
             label_fn, color: str) -> None:
        if not items:
            ax.text(0.5, 0.5, "(no events)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="#888888")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_visible(False)
            return
        labels = [label_fn(e) for e in items][::-1]
        counts = [int(e["count"]) for e in items][::-1]
        ax.barh(labels, counts, color=color, edgecolor="white", linewidth=0.6)
        ax.set_title(title)
        ax.set_xlabel("count")
        for i, c in enumerate(counts):
            ax.text(c, i, f" {c}", va="center", fontsize=8)

    _bar(
        axes[0], subs,
        f"Top-{top_k} substitutions",
        lambda e: f"{e.get('ref','?')!s}→{e.get('hyp','?')!s}",
        COLOR_SUB,
    )
    _bar(
        axes[1], ins,
        f"Top-{top_k} insertions",
        lambda e: f"+{e.get('hyp','?')!s}",
        COLOR_INS,
    )
    _bar(
        axes[2], dels,
        f"Top-{top_k} deletions",
        lambda e: f"-{e.get('ref','?')!s}",
        COLOR_DEL,
    )

    fig.suptitle(f"Character-level error breakdown — {run_dir.name}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Chart 3 — dialect-pair confusions per run
# ---------------------------------------------------------------------------


def plot_dialect_pairs(run_dir: Path, out_path: Path) -> None:
    data = _read_json(run_dir / "error_analysis.json")
    if data is None:
        return
    pairs = data.get("dialect_pairs") or {}
    if not pairs:
        return

    items = sorted(pairs.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    counts = [int(v) for _, v in items]

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    bars = ax.bar(
        labels, counts,
        color=COLOR_DIALECT, edgecolor="white", linewidth=0.6,
    )
    ax.set_title(f"Hutsul ↔ standard substitution pairs — {run_dir.name}")
    ax.set_ylabel("count")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    ymax = max(counts) if counts else 1
    ax.set_ylim(0, ymax * 1.15 + 1)
    for bar, c in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.02,
            f"{c}",
            ha="center", va="bottom", fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Chart 4 — training curves from TensorBoard
# ---------------------------------------------------------------------------


_TB_TAG_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "train_loss": ("train/loss", "loss", "train_loss"),
    "eval_loss":  ("eval/loss", "eval_loss"),
    "eval_wer":   ("eval/wer", "eval_wer"),
    "eval_cer":   ("eval/cer", "eval_cer"),
}


def _load_event_accumulator(tb_dir: Path):
    """Return an :class:`EventAccumulator` over every event file inside
    ``tb_dir`` (recursively), or ``None`` when tensorboard isn't installed
    or no events are found."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        logger.warning("tensorboard not installed — skipping training curves")
        return None

    # Some trainers write directly into <variant>/, others into
    # <variant>/<run>/.  EventAccumulator picks the first subdir with events;
    # we therefore point it at any directory that contains events.
    candidates: List[Path] = []
    if any(tb_dir.glob("events.out.tfevents.*")):
        candidates.append(tb_dir)
    for sub in sorted(tb_dir.rglob("events.out.tfevents.*")):
        candidates.append(sub.parent)
    candidates = list(dict.fromkeys(candidates))  # de-dupe preserving order
    if not candidates:
        return None

    acc = EventAccumulator(str(candidates[0]), size_guidance={"scalars": 0})
    acc.Reload()
    return acc


def _series_from_acc(acc, candidate_tags: Iterable[str]
                     ) -> Tuple[Optional[List[int]], Optional[List[float]]]:
    available = set(acc.Tags().get("scalars", []))
    for tag in candidate_tags:
        if tag in available:
            events = acc.Scalars(tag)
            return [e.step for e in events], [e.value for e in events]
    return None, None


def plot_training_curves(tb_dir: Path, out_path: Path) -> None:
    acc = _load_event_accumulator(tb_dir)
    if acc is None:
        return

    series: Dict[str, Tuple[Optional[List[int]], Optional[List[float]]]] = {}
    for key, tags in _TB_TAG_CANDIDATES.items():
        series[key] = _series_from_acc(acc, tags)

    if all(steps is None for steps, _ in series.values()):
        logger.info("No scalar series found in %s — skipping", tb_dir)
        return

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.5))
    spec = [
        ("train_loss", "Train loss",  "step", "loss"),
        ("eval_loss",  "Eval loss",   "step", "loss"),
        ("eval_wer",   "Eval WER",    "step", "WER"),
        ("eval_cer",   "Eval CER",    "step", "CER"),
    ]
    for ax, (key, title, xlabel, ylabel) in zip(axes.flatten(), spec):
        steps, values = series[key]
        if not steps:
            ax.text(0.5, 0.5, f"{title}: not logged",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="#888888")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            continue
        ax.plot(steps, values, color=COLOR_CURVE, linewidth=1.4)
        ax.scatter(steps, values, s=10, color=COLOR_CURVE, zorder=3)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    fig.suptitle(f"Training curves — {tb_dir.name}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


_SAFE = re.compile(r"[^a-zA-Z0-9._-]+")


def _slug(name: str) -> str:
    return _SAFE.sub("_", name).strip("_") or "x"


def render_all(layout: StorageLayout, out_dir: Optional[Path] = None) -> None:
    _apply_global_style()

    json_root = layout.evaluations_json
    out_dir = out_dir or (layout.evaluations / "summary" / "charts")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Storage layout : %s", layout.root)
    logger.info("Charts → %s", out_dir)

    # 1) WER/CER bar charts — one per split, so test and validation are
    #    separable in the report.
    all_rows = _collect_results(json_root)
    if not all_rows:
        logger.warning(
            "No test_results.json files under %s — run evaluate.py first.",
            json_root,
        )
    for split, rows in _group_by_split(all_rows).items():
        plot_wer_cer_bar(
            rows, out_dir / f"wer_cer_bar__{_slug(split)}.png"
        )

    # 2 & 3) Per-run error breakdown + dialect-pair charts.
    if json_root.exists():
        for run_dir in sorted(p for p in json_root.iterdir() if p.is_dir()):
            if _is_deprecated(run_dir.name):
                continue
            plot_error_breakdown(
                run_dir,
                out_dir / f"error_breakdown__{_slug(run_dir.name)}.png",
            )
            plot_dialect_pairs(
                run_dir,
                out_dir / f"dialect_pairs__{_slug(run_dir.name)}.png",
            )

    # 4) Training curves — one per variant directory under tensorboard/.
    tb_root = layout.tensorboard
    if tb_root.exists():
        for tb_dir in sorted(p for p in tb_root.iterdir() if p.is_dir()):
            if _is_deprecated(tb_dir.name):
                continue
            plot_training_curves(
                tb_dir,
                out_dir / f"training_curves__{_slug(tb_dir.name)}.png",
            )
    else:
        logger.info("No tensorboard root at %s — skipping curve plots", tb_root)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render publication-quality evaluation charts.",
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
    render_all(
        layout,
        out_dir=Path(args.output_dir) if args.output_dir else None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
