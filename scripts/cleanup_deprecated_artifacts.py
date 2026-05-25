"""Delete stale on-Drive artefacts for retired model variants.

Use this once after Parakeet (or any future) variant has been removed
from the codebase, so that ``checkpoints/``, ``final_models/``,
``tensorboard/``, ``preprocessed/`` and ``evaluations/*/`` no longer
contain orphan subdirectories from runs that the project no longer
supports.

By default the script runs in DRY-RUN mode and only prints what it
would delete.  Pass ``--apply`` to actually remove the directories.

Usage
-----

    export HUTSUL_ASR_ROOT=/content/drive/MyDrive/hutsul_asr

    # 1. Dry run — see what would be removed.
    python scripts/cleanup_deprecated_artifacts.py

    # 2. Apply.
    python scripts/cleanup_deprecated_artifacts.py --apply

By default the prefixes matched are read from
``aggregate_results.DEPRECATED_VARIANT_PREFIXES``.  Override with
``--prefix`` to target a specific retired variant.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    StorageLayout,
    configure_logging,
    resolve_storage_layout,
)

# Reuse the same canonical list of retired variants used by the
# downstream tools so behaviour stays consistent.
from scripts.aggregate_results import (  # noqa: E402
    DEPRECATED_VARIANT_PREFIXES as DEFAULT_PREFIXES,
)

logger = logging.getLogger(__name__)


def _candidates(layout: StorageLayout,
                prefixes: Iterable[str]) -> List[Path]:
    """Return every Drive directory that matches a deprecated prefix.

    Searches:
        * checkpoints/<variant>/
        * final_models/<variant>/
        * tensorboard/<variant>/
        * preprocessed/<variant>/
        * evaluations/{csv,json,predictions}/<variant>/
    """
    out: List[Path] = []
    parents = [
        layout.checkpoints,
        layout.final_models,
        layout.tensorboard,
        layout.preprocessed,
        layout.evaluations_csv,
        layout.evaluations_json,
        layout.evaluations_predictions,
    ]
    for parent in parents:
        if not parent.exists():
            continue
        for child in parent.iterdir():
            if not child.is_dir():
                continue
            if any(child.name.startswith(p) for p in prefixes):
                out.append(child)
    return sorted(out)


def _humansize(p: Path) -> str:
    """Approximate directory size for logging — never raises."""
    try:
        total = 0
        for f in p.rglob("*"):
            try:
                total += f.stat().st_size
            except OSError:
                continue
        if total >= 1 << 30:
            return f"{total / (1 << 30):.2f} GB"
        if total >= 1 << 20:
            return f"{total / (1 << 20):.2f} MB"
        return f"{total / 1024:.2f} KB"
    except OSError:
        return "?"


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Remove orphan Drive directories for retired model variants. "
            "Defaults to a dry run; pass --apply to actually delete."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--storage_root",
        default=os.environ.get("HUTSUL_ASR_ROOT"),
    )
    p.add_argument(
        "--prefix",
        action="append",
        default=None,
        help=(
            "Variant-name prefix(es) to delete.  Repeat to add more. "
            f"Defaults to: {list(DEFAULT_PREFIXES)}"
        ),
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete (default is dry-run, prints only).",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: List[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    configure_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    layout = resolve_storage_layout(
        Path(args.storage_root) if args.storage_root else None,
        refresh=bool(args.storage_root),
    )
    layout.ensure()

    prefixes = tuple(args.prefix) if args.prefix else tuple(DEFAULT_PREFIXES)
    logger.info("Storage root  : %s", layout.root)
    logger.info("Mode          : %s", "APPLY" if args.apply else "DRY-RUN")
    logger.info("Match prefixes: %s", list(prefixes))

    candidates = _candidates(layout, prefixes)
    if not candidates:
        logger.info("No matching directories found — nothing to do.")
        return 0

    print()
    for path in candidates:
        size = _humansize(path)
        marker = "DELETE" if args.apply else "WOULD DELETE"
        print(f"  [{marker}] {path}  ({size})")

    if not args.apply:
        print(
            "\nDry run complete.  Re-run with --apply to actually delete."
        )
        return 0

    print()
    for path in candidates:
        try:
            shutil.rmtree(path)
            logger.info("Removed %s", path)
        except OSError as exc:
            logger.error("Could not remove %s: %s", path, exc)
            return 1
    logger.info("Done — removed %d director%s.",
                len(candidates), "y" if len(candidates) == 1 else "ies")
    return 0


if __name__ == "__main__":
    sys.exit(main())
