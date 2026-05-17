

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DEFAULT_DATASET_NAME,
    DEFAULT_SEED,
    configure_logging,
    resolve_storage_layout,
)
from models import TRAINER_MODULE_MAP

logger = logging.getLogger(__name__)


_LATEST_SENTINEL = "LATEST"


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified ASR training entry point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model_type",
        required=True,
        choices=sorted(TRAINER_MODULE_MAP.keys()),
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="Named variant from the YAML; defaults to the first one.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML override; defaults to configs/<model_type>.yaml.",
    )

    parser.add_argument(
        "--model_name",
        dest="model_name_or_path",
        default=None,
    )
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--audio_column", default=None)
    parser.add_argument("--text_column", default=None)
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Defaults to <storage_root>/checkpoints/<variant>/.",
    )
    parser.add_argument("--run_name", default=None)

    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument(
        "--batch_size",
        dest="per_device_train_batch_size",
        type=int,
        default=None,
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

    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=None,
        help="Whisper only.",
    )
    parser.add_argument(
        "--no_lora",
        dest="use_lora",
        action="store_false",
        default=None,
    )
    parser.add_argument(
        "--use_augmentation",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no_augmentation",
        dest="use_augmentation",
        action="store_false",
        default=None,
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=None,
        help="Deterministic CuDNN mode (slower, reproducible).",
    )
    parser.add_argument("--fp16", action="store_true", default=None)
    parser.add_argument("--bf16", action="store_true", default=None)

    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help=(
            "Path to a checkpoint-* dir, or "
            f"``{_LATEST_SENTINEL}`` to auto-detect the newest one."
        ),
    )

    parser.add_argument(
        "--storage_root",
        default=os.environ.get("HUTSUL_ASR_ROOT"),
        help="Override the canonical storage root.",
    )

    parser.add_argument(
        "--hf_token",
        default=os.environ.get("HF_TOKEN"),
    )
    parser.add_argument("--trust_remote_code", action="store_true", default=None)

    return parser


def _resolve_config_path(args: argparse.Namespace) -> Path:
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
    if args.output_dir:
        return Path(args.output_dir)
    layout = resolve_storage_layout(
        Path(args.storage_root) if args.storage_root else None
    )
    return layout.checkpoint_dir(variant)


def _resolve_resume_path(
    requested: Optional[str], output_dir: Path
) -> Optional[str]:
    if requested is None:
        return None

    if requested == _LATEST_SENTINEL:
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


def _ensure_defaults(overrides: Dict[str, Any]) -> Dict[str, Any]:
    overrides.setdefault("dataset_name", DEFAULT_DATASET_NAME)
    overrides.setdefault("seed", DEFAULT_SEED)
    return overrides


def _dispatch(args: argparse.Namespace) -> Dict[str, float]:
    import importlib


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
        "parakeet": "train_parakeet",
    }[args.model_type]

    if not hasattr(trainer_module, train_fn_name):
        raise AttributeError(
            f"{module_path} does not expose ``{train_fn_name}``"
        )

    overrides = _collect_overrides(args)
    overrides = _ensure_defaults(overrides)

    train_args = trainer_module.args_from_yaml(
        yaml_path,
        variant=args.variant,
        overrides=overrides,
    )


    output_dir = _resolve_output_dir(
        args, getattr(train_args, "variant", "default")
    )
    train_args.output_dir = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume_from_checkpoint is not None:
        train_args.resume_from_checkpoint = _resolve_resume_path(
            args.resume_from_checkpoint, output_dir
        )

    logger.info("Resolved training args: %s", train_args)

    train_fn = getattr(trainer_module, train_fn_name)
    return train_fn(train_args)


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
