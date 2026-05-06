"""
train.py
========

Unified CLI training entry point for every ASR model family in this
project.

Usage examples
--------------

::

    # Whisper-small with LoRA, default hyper-parameters from the YAML
    python train.py --model_type whisper --variant whisper-small

    # Whisper-large-v3, override the output dir and turn on augmentation
    python train.py --model_type whisper --variant whisper-large-v3 \\
                    --output_dir outputs/whisper-large-v3 \\
                    --use_augmentation

    # Wav2Vec2-XLSR
    python train.py --model_type wav2vec2 --variant xlsr-300m-uk

    # Wav2Vec2-BERT-UK with adapter tuning (default)
    python train.py --model_type wav2vec2_bert --variant w2v-bert-uk-v2.1

    # OmniASR 1B with tri-stage scheduler
    python train.py --model_type omniasr --variant omniasr-1b

    # Resume the most recent checkpoint inside an output directory
    python train.py --model_type whisper --variant whisper-small \\
                    --output_dir outputs/whisper-small \\
                    --resume_from_checkpoint LATEST

Resuming
--------

``--resume_from_checkpoint LATEST`` triggers automatic detection of
the most recent ``checkpoint-*`` subdirectory under ``--output_dir``
via :func:`transformers.trainer_utils.get_last_checkpoint`.  The HF
:class:`Trainer` then transparently restores the optimizer state,
the LR scheduler state, the FP16 grad-scaler, the global step, the
``TrainerState`` (best metric / counters) and the RNG state, so
training continues exactly where it was interrupted.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make the project importable when ``train.py`` is launched from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local imports must come AFTER the sys.path tweak above.
from config import (  # noqa: E402
    DEFAULT_DATASET_NAME,
    DEFAULT_SEED,
    configure_logging,
    resolve_storage_layout,
)
from models import TRAINER_MODULE_MAP  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_LATEST_SENTINEL = "LATEST"


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified ASR training entry point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Mandatory dispatch keys ----------------------------------------
    parser.add_argument(
        "--model_type",
        required=True,
        choices=sorted(TRAINER_MODULE_MAP.keys()),
        help="Which model family to fine-tune.",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help=(
            "Named variant from the YAML config (e.g. whisper-small, "
            "xlsr-300m-uk). Defaults to the first variant in the file."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Optional path to a YAML config.  When omitted, the "
            "default ``configs/<model_type>.yaml`` is used."
        ),
    )

    # ---- Common overrides -----------------------------------------------
    parser.add_argument(
        "--model_name",
        dest="model_name_or_path",
        default=None,
        help="Override the YAML's ``model_name_or_path``.",
    )
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--audio_column", default=None)
    parser.add_argument("--text_column", default=None)
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Output directory.  When omitted, defaults to "
            "``outputs/<model_type>/<variant>``."
        ),
    )
    parser.add_argument("--run_name", default=None)

    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument(
        "--batch_size",
        dest="per_device_train_batch_size",
        type=int,
        default=None,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--eval_batch_size",
        dest="per_device_eval_batch_size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--grad_accum",
        dest="gradient_accumulation_steps",
        type=int,
        default=None,
    )
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    # ---- Switches -------------------------------------------------------
    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=None,
        help="Enable LoRA fine-tuning (Whisper only).",
    )
    parser.add_argument(
        "--no_lora",
        dest="use_lora",
        action="store_false",
        default=None,
        help="Disable LoRA (Whisper only).",
    )
    parser.add_argument(
        "--use_augmentation",
        action="store_true",
        default=None,
        help="Enable on-the-fly augmentation pipeline.",
    )
    parser.add_argument(
        "--no_augmentation",
        dest="use_augmentation",
        action="store_false",
        default=None,
        help="Disable augmentation regardless of YAML default.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=None,
        help="Enable deterministic CuDNN mode (slower, fully reproducible).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=None,
    )

    # ---- Resume ---------------------------------------------------------
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help=(
            "Either an explicit ``checkpoint-*`` directory, or the "
            f"sentinel ``{_LATEST_SENTINEL}`` to auto-detect the latest "
            "checkpoint inside ``--output_dir``."
        ),
    )

    # ---- Storage --------------------------------------------------------
    parser.add_argument(
        "--storage_root",
        default=os.environ.get("HUTSUL_ASR_ROOT"),
        help=(
            "Root directory for all training artefacts.  When omitted: "
            "uses Drive (``/content/drive/MyDrive/hutsul_asr``) on Colab "
            "if mounted, otherwise ``./outputs/`` locally.  Also "
            "configurable via the ``HUTSUL_ASR_ROOT`` env var."
        ),
    )

    # ---- HF / external -------------------------------------------------
    parser.add_argument(
        "--hf_token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token (overrides $HF_TOKEN).",
    )
    parser.add_argument("--trust_remote_code", action="store_true", default=None)

    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_config_path(args: argparse.Namespace) -> Path:
    """Pick the YAML config path to use."""
    if args.config:
        path = Path(args.config)
    else:
        path = PROJECT_ROOT / "configs" / f"{args.model_type}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"YAML config not found: {path}")
    return path


def _resolve_output_dir(
    args: argparse.Namespace, variant: str
) -> Path:
    """Pick the output directory.

    Resolution order:

    1. ``--output_dir`` (explicit CLI override).
    2. ``<storage_root>/checkpoints/<variant>/`` from the resolved
       :class:`StorageLayout` — Drive on Colab, project-local
       fallback otherwise.
    """
    if args.output_dir:
        return Path(args.output_dir)
    layout = resolve_storage_layout(
        Path(args.storage_root) if args.storage_root else None
    )
    return layout.checkpoint_dir(variant)


def _resolve_resume_path(
    requested: Optional[str], output_dir: Path
) -> Optional[str]:
    """Map the user-supplied resume hint to a real directory path."""
    if requested is None:
        return None

    if requested == _LATEST_SENTINEL:
        # Lazy import keeps `train.py --help` cheap.
        from transformers.trainer_utils import get_last_checkpoint

        if not output_dir.exists():
            logger.warning(
                "%s does not exist yet — cannot resume; starting fresh.",
                output_dir,
            )
            return None

        last = get_last_checkpoint(str(output_dir))
        if last is None:
            logger.warning(
                "No checkpoint-* directory found under %s — starting fresh.",
                output_dir,
            )
            return None
        logger.info("Auto-resuming from latest checkpoint: %s", last)
        return last

    path = Path(requested)
    if not path.exists():
        raise FileNotFoundError(
            f"--resume_from_checkpoint {requested} does not exist"
        )
    return str(path)


# ---------------------------------------------------------------------------
# Override builder
# ---------------------------------------------------------------------------


_OVERRIDE_KEYS = (
    "model_name_or_path",
    "dataset_name",
    "dataset_config",
    "audio_column",
    "text_column",
    "output_dir",
    "run_name",
    "learning_rate",
    "max_steps",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "warmup_steps",
    "seed",
    "use_lora",
    "use_augmentation",
    "deterministic",
    "fp16",
    "bf16",
    "resume_from_checkpoint",
    "hf_token",
    "trust_remote_code",
)


def _collect_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Translate CLI overrides to a flat dict for ``args_from_yaml``."""
    overrides: Dict[str, Any] = {}
    namespace = vars(args)
    for key in _OVERRIDE_KEYS:
        if key not in namespace:
            continue
        value = namespace[key]
        if value is None:
            continue
        overrides[key] = value
    return overrides


# ---------------------------------------------------------------------------
# Defaults coming from CLI
# ---------------------------------------------------------------------------


def _ensure_defaults(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Inject project-level defaults when the YAML omits them."""
    overrides.setdefault("dataset_name", DEFAULT_DATASET_NAME)
    overrides.setdefault("seed", DEFAULT_SEED)
    return overrides


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def _dispatch(args: argparse.Namespace) -> Dict[str, float]:
    """Resolve the trainer module and run training."""
    import importlib

    # Lock the storage layout up-front so all downstream callers see
    # the same root (Colab/Drive vs. local fallback vs. CLI override).
    layout = resolve_storage_layout(
        Path(args.storage_root) if args.storage_root else None,
        refresh=bool(args.storage_root),
    )
    layout.ensure()
    logger.info("Storage layout resolved:\n%s", layout.summary())

    yaml_path = _resolve_config_path(args)
    module_path = TRAINER_MODULE_MAP[args.model_type]
    trainer_module = importlib.import_module(module_path)

    if not hasattr(trainer_module, "args_from_yaml"):
        raise AttributeError(
            f"{module_path} does not expose ``args_from_yaml``"
        )

    train_fn_name = {
        "whisper": "train_whisper",
        "wav2vec2": "train_wav2vec2",
        "wav2vec2_bert": "train_wav2vec2_bert",
        "omniasr": "train_omniasr",
    }[args.model_type]

    if not hasattr(trainer_module, train_fn_name):
        raise AttributeError(
            f"{module_path} does not expose ``{train_fn_name}``"
        )

    overrides = _collect_overrides(args)
    overrides = _ensure_defaults(overrides)

    # Build args object from YAML + overrides.
    train_args = trainer_module.args_from_yaml(
        yaml_path,
        variant=args.variant,
        overrides=overrides,
    )

    # Resolve output dir AFTER ``args_from_yaml`` so the variant is final.
    output_dir = _resolve_output_dir(
        args, getattr(train_args, "variant", "default")
    )
    train_args.output_dir = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve LATEST sentinel using the now-finalised output dir.
    if args.resume_from_checkpoint is not None:
        train_args.resume_from_checkpoint = _resolve_resume_path(
            args.resume_from_checkpoint, output_dir
        )

    logger.info("Resolved training args: %s", train_args)

    train_fn = getattr(trainer_module, train_fn_name)
    return train_fn(train_args)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    configure_logging()
    parser = _build_argparser()
    args = parser.parse_args(argv)

    try:
        metrics = _dispatch(args)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        return 130
    except Exception:
        logger.exception("Training failed with an unhandled exception")
        return 1

    logger.info("Training finished. Final metrics:")
    for k, v in sorted(metrics.items()):
        logger.info("  %s = %s", k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
