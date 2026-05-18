"""Real Meta OmniASR (fairseq2) launcher.

Replaces the previous fake HF ``AutoModelForCTC``-based trainer.  This
module is a thin orchestrator: it converts the project's HF dataset
into the fairseq2 manifest layout, renders the Meta recipe YAML with
the right dataset paths injected, and invokes the official entrypoint::

    python -m workflows.recipes.wav2vec2.asr <output_dir> \\
        --config-file <rendered_yaml>

fairseq2 and the ``omnilingual-asr`` repository are runtime-only
dependencies — not pinned in ``requirements.txt`` (same isolation
discipline used for the Parakeet/NeMo pipeline).  Install them in a
dedicated Colab runtime; see README "OmniASR / fairseq2" section.

Asset resolution
----------------

The Meta-schema YAML at ``configs/omniasr/ctc-finetune.yaml`` declares::

    model:     { name: omniASR_CTC_300M }
    tokenizer: { name: omniASR_tokenizer_v1 }

Both names resolve through the asset card at
``omnilingual-asr/src/omnilingual_asr/cards/models/rc_models_v1.yaml``
which points at:

    checkpoint: https://dl.fbaipublicfiles.com/mms/omniASR-CTC-300M.pt
    tokenizer:  https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer.model

fairseq2 downloads both into its asset cache on first use.

No HF Transformers code path runs anywhere in this module.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from config import (
    DEFAULT_DATASET_NAME,
    PROJECT_ROOT,
    ProjectConfig,
    configure_logging,
    resolve_storage_layout,
    set_global_seed,
)

logger = logging.getLogger(__name__)


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "omniasr.yaml"
META_YAML_DIR = PROJECT_ROOT / "configs" / "omniasr"
DEFAULT_META_TEMPLATE = META_YAML_DIR / "ctc-finetune.yaml"
DEFAULT_REPO_LOCAL = Path("/content/omnilingual-asr")
RECIPE_MODULE = "workflows.recipes.wav2vec2.asr"


# ---------------------------------------------------------------------------
# Legacy compatibility stub
# ---------------------------------------------------------------------------


class OmniASRProcessor:
    """Backward-compat stub for ``evaluate.py:_load_omniasr``.

    The new fairseq2 pipeline never instantiates this class.  It is
    kept only so that the legacy HF-format auto-detection branch in
    ``evaluate.py`` can still import the symbol when evaluating any
    older HF-format checkpoint that happens to live on disk.  Real
    Meta OmniASR fine-tunes save fairseq2 ``.pt`` archives and are
    evaluated through the official recipe, not through
    ``evaluate.py``.
    """

    def __init__(self, feature_extractor: Any, tokenizer: Any) -> None:
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    @property
    def model_input_names(self) -> List[str]:
        names = getattr(self.feature_extractor, "model_input_names", None)
        return list(names) if names else ["input_features"]

    def save_pretrained(self, directory: Union[str, Path]) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.feature_extractor.save_pretrained(str(directory))
        self.tokenizer.save_pretrained(str(directory))


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


@dataclass
class OmniASRTrainArgs:
    """CLI / YAML arguments for the Meta OmniASR launcher."""

    variant: str
    dataset_name: str = DEFAULT_DATASET_NAME
    dataset_config: Optional[str] = None
    audio_column: Optional[str] = None
    text_column: Optional[str] = None

    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    run_name: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None

    seed: int = 42
    deterministic: bool = False
    sample_rate: int = 16000

    # Paths to the Meta runtime
    meta_repo_path: Optional[str] = None
    meta_yaml_template: Optional[str] = None

    # Smoke-test toggle and step count
    smoke: bool = False
    smoke_train_size: int = 10
    smoke_dev_size: int = 5
    smoke_steps: int = 50

    # Paper-scale recipe overrides (mapped 1:1 into the rendered YAML)
    max_num_steps: int = 48000
    grad_accumulation_num_batches: int = 4
    batch_size: int = 8
    learning_rate: float = 5.0e-5
    warmup_ratio: float = 0.10
    hold_ratio: float = 0.40
    precision: str = "float16"
    checkpoint_every_n_steps: int = 1000
    validate_every_n_steps: int = 1000
    min_train_audio_duration_sec: float = 1.0

    # Carried for logging only; the real load is driven by the asset
    # name inside the rendered Meta YAML.
    model_name_or_path: str = "omniASR_CTC_300M"

    hf_token: Optional[str] = None
    trust_remote_code: bool = False


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


def load_yaml_config(
    path: Union[str, Path] = DEFAULT_CONFIG_PATH,
    *,
    variant: Optional[str] = None,
) -> Dict[str, Any]:
    """Same defaults+variants schema used by the other trainers."""
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
) -> OmniASRTrainArgs:
    cfg = load_yaml_config(yaml_path, variant=variant)
    for k, v in dict(overrides or {}).items():
        if v is None:
            continue
        cfg[k] = v
    valid_keys = set(OmniASRTrainArgs.__dataclass_fields__.keys())
    filtered = {k: v for k, v in cfg.items() if k in valid_keys}
    if "variant" not in filtered:
        raise ValueError("Missing required OmniASR field: variant")
    return OmniASRTrainArgs(**filtered)


# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------


def _check_fairseq2_available() -> None:
    try:
        import fairseq2  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "OmniASR training requires fairseq2 and the official "
            "omnilingual-asr repo.  Install in a dedicated Colab "
            "runtime, separate from the transformers stack:\n\n"
            "    pip install -q fairseq2\n"
            "    git clone https://github.com/facebookresearch/omnilingual-asr.git \\\n"
            "        /content/omnilingual-asr\n"
            "    pip install -e /content/omnilingual-asr\n\n"
            f"Original ImportError: {exc}"
        ) from exc


def _resolve_meta_repo(args: OmniASRTrainArgs) -> Path:
    candidate = (
        Path(args.meta_repo_path) if args.meta_repo_path else DEFAULT_REPO_LOCAL
    )
    if not candidate.exists():
        raise FileNotFoundError(
            f"omnilingual-asr repo not found at {candidate}.\n\n"
            "Clone it:\n"
            "    git clone https://github.com/facebookresearch/omnilingual-asr.git "
            f"{candidate}\n"
            "    pip install -e " + str(candidate)
        )
    if not (candidate / "workflows" / "recipes" / "wav2vec2" / "asr").exists():
        raise FileNotFoundError(
            f"{candidate} exists but does not contain the expected "
            "workflows/recipes/wav2vec2/asr/ directory.  Re-clone the "
            "official repo."
        )
    return candidate


# ---------------------------------------------------------------------------
# Manifest conversion + smoke truncation
# ---------------------------------------------------------------------------


def _convert_dataset_to_manifest(
    args: OmniASRTrainArgs, manifest_dir: Path
) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    expected = [manifest_dir / f for f in
                ("train.tsv", "train.wrd", "dev.tsv", "dev.wrd")]
    if all(p.exists() for p in expected):
        logger.info("Manifest cache present at %s; skipping conversion.",
                    manifest_dir)
        return
    script = PROJECT_ROOT / "scripts" / "convert_to_omniasr_manifest.py"
    if not script.exists():
        raise FileNotFoundError(f"Conversion script not found: {script}")
    cmd = [
        sys.executable, str(script),
        "--out_dir", str(manifest_dir),
        "--sample_rate", str(args.sample_rate),
    ]
    if args.hf_token:
        cmd += ["--hf_token", args.hf_token]
    logger.info("Running manifest conversion: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(
            f"Manifest conversion failed (exit code {result.returncode})."
        )


def _truncate_for_smoke(
    manifest_dir: Path, n_train: int, n_dev: int
) -> Path:
    """Build a subdirectory ``<manifest_dir>/smoke/`` whose own
    ``train.tsv`` / ``dev.tsv`` / ``train.wrd`` / ``dev.wrd`` contain
    the first N rows of the corresponding parent file.

    fairseq2's ``ManifestStorage.discover_splits`` scans the data
    directory for files named ``<split>.tsv``, so the smoke variant
    needs a directory of its own — the previous ``*.smoke.tsv``
    filename scheme is invisible to that discovery.

    Returns the path to the smoke directory (this is what the
    fairseq2 asset card points at for the smoke variant).
    """
    smoke_dir = manifest_dir / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", n_train), ("dev", n_dev)):
        src_tsv = manifest_dir / f"{split}.tsv"
        src_wrd = manifest_dir / f"{split}.wrd"
        if not src_tsv.exists() or not src_wrd.exists():
            raise FileNotFoundError(
                f"Cannot build smoke manifest: missing {src_tsv} or {src_wrd}"
            )
        tsv_lines = src_tsv.read_text(encoding="utf-8").splitlines()
        wrd_lines = src_wrd.read_text(encoding="utf-8").splitlines()
        # Line 0 of the parent TSV is an absolute audio-root path;
        # copy it as-is so the relative wav names in the truncated
        # rows still resolve.  No audio relocation needed.
        (smoke_dir / f"{split}.tsv").write_text(
            "\n".join([tsv_lines[0]] + tsv_lines[1:1 + n]) + "\n",
            encoding="utf-8",
        )
        (smoke_dir / f"{split}.wrd").write_text(
            "\n".join(wrd_lines[:n]) + "\n", encoding="utf-8"
        )
        logger.info(
            "Smoke %s manifest: %d entries -> %s",
            split, n, smoke_dir / f"{split}.tsv",
        )
    return smoke_dir


# ---------------------------------------------------------------------------
# fairseq2 asset-card registration
# ---------------------------------------------------------------------------

# fairseq2's user-asset directory.  The Meta recipe resolves
# ``dataset.name`` via the AssetStore, which scans this directory
# (plus paths in ``FAIRSEQ2_USER_ASSET_DIR``) for YAML files declaring
# ``name: ...`` + ``family: manifest_asr_dataset`` + ``data: <path>``.
USER_ASSET_DIR = Path.home() / ".config" / "fairseq2" / "assets"


def _register_manifest_asset_card(
    asset_name: str, data_dir: Path
) -> Path:
    """Write a fairseq2 asset card so the recipe resolves
    ``dataset.name == asset_name`` to ``data_dir`` at runtime.

    Schema (verified against
    ``src/omnilingual_asr/datasets/impl/manifest_asr_dataset.py``)::

        @dataclass
        class ManifestAsrDatasetConfig:
            data: Path

    ``dataset_config`` therefore accepts EXACTLY one field: ``data``.
    A previous version of this writer also emitted ``tokenizer_ref``
    under the nested block; fairseq2's ``AssetCard.parse_as`` rejected
    that with ``extra keys tokenizer_ref``.  The tokenizer is already
    declared in the recipe YAML via ``tokenizer.name``; the dataset
    asset card does not cross-reference it.

    Asset-card-level layout (also verified by the same load path):

        name: <asset_name>
        dataset_family: manifest_asr_dataset
        dataset_config:
          data: <absolute path to manifest dir>
    """
    USER_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    card_path = USER_ASSET_DIR / f"{asset_name}.yaml"
    payload = (
        f"name: {asset_name}\n"
        f"dataset_family: manifest_asr_dataset\n"
        f"dataset_config:\n"
        f"  data: {data_dir.resolve()}\n"
    )
    card_path.write_text(payload, encoding="utf-8")
    logger.info(
        "Wrote fairseq2 asset card -> %s  (data=%s)", card_path, data_dir
    )
    return card_path


# ---------------------------------------------------------------------------
# Meta-schema YAML rendering
# ---------------------------------------------------------------------------


def _render_meta_yaml(
    args: OmniASRTrainArgs,
    template_path: Path,
    asset_name: str,
    out_path: Path,
) -> Path:
    """Load the Meta-schema template, inject only fields that exist in
    ``Wav2Vec2AsrRecipeConfig``, save the resolved YAML next to the
    run output dir.

    The dataset manifest path is NOT a field of ``Wav2Vec2AsrDatasetSection``
    — it lives in a fairseq2 asset card the launcher writes separately
    (see :func:`_register_manifest_asset_card`).  This function only
    overrides ``dataset.name`` so the recipe resolves the right card.

    Verified valid OmegaConf paths (every other key that appeared in
    the old launcher has been removed; the recipe's dataclass schema
    rejects extras with the exact ``Recipe configuration cannot be
    structured`` error we hit before):

        dataset.name
        dataset.asr_task_config.{min_audio_len,max_audio_len,
                                 max_num_elements,batch_size,
                                 normalize_audio}
        optimizer.config.lr
        lr_scheduler.config.stage_ratio
        trainer.mixed_precision.dtype
        trainer.grad_accumulation.num_batches
        regime.{num_steps,checkpoint_every_n_steps,validate_every_n_steps,
                validate_after_n_steps,publish_metrics_every_n_steps}
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(template_path))

    # 1) Asset-card key.
    OmegaConf.update(cfg, "dataset.name", asset_name)

    # 2) Audio-length / batching (samples, not seconds).
    sr = int(args.sample_rate)
    min_samples = int(round(args.min_train_audio_duration_sec * sr))
    max_samples = int(round(30.0 * sr))   # 30 s cap, paper convention
    OmegaConf.update(cfg, "dataset.asr_task_config.min_audio_len", min_samples)
    OmegaConf.update(cfg, "dataset.asr_task_config.max_audio_len", max_samples)
    OmegaConf.update(cfg, "dataset.asr_task_config.max_num_elements", max_samples)
    OmegaConf.update(cfg, "dataset.asr_task_config.normalize_audio", True)

    # 3) Per-mode trainer / regime overrides.
    if args.smoke:
        OmegaConf.update(cfg, "dataset.asr_task_config.batch_size", 2)
        OmegaConf.update(cfg, "trainer.grad_accumulation.num_batches", 1)
        OmegaConf.update(cfg, "regime.num_steps", args.smoke_steps)
        # fairseq2 requires checkpoint_every_n_steps and
        # validate_every_n_steps to be multiples of
        # publish_metrics_every_n_steps.  The template's default
        # publish cadence is 200, which exceeds any reasonable smoke
        # step count — override all three so the constraint holds and
        # the smoke run actually emits metrics + checkpoints.
        publish_every = max(1, args.smoke_steps // 10)
        check_every = max(publish_every, args.smoke_steps // 2)
        # Round up to the next multiple of publish_every.
        check_every = (
            (check_every + publish_every - 1) // publish_every
        ) * publish_every
        OmegaConf.update(cfg, "regime.publish_metrics_every_n_steps",
                         publish_every)
        OmegaConf.update(cfg, "regime.checkpoint_every_n_steps", check_every)
        OmegaConf.update(cfg, "regime.validate_every_n_steps", check_every)
        OmegaConf.update(cfg, "regime.validate_after_n_steps", 0)
    else:
        OmegaConf.update(cfg, "dataset.asr_task_config.batch_size",
                         args.batch_size)
        OmegaConf.update(cfg, "trainer.grad_accumulation.num_batches",
                         args.grad_accumulation_num_batches)
        OmegaConf.update(cfg, "regime.num_steps", args.max_num_steps)
        OmegaConf.update(cfg, "regime.checkpoint_every_n_steps",
                         args.checkpoint_every_n_steps)
        OmegaConf.update(cfg, "regime.validate_every_n_steps",
                         args.validate_every_n_steps)

    # 4) Optimizer.
    OmegaConf.update(cfg, "optimizer.config.lr", float(args.learning_rate))

    # 5) Mixed-precision dtype.  fairseq2 accepts a torch-prefixed
    #    string; map our YAML's ``precision: float16|bfloat16|float32``
    #    to the corresponding ``torch.<dtype>`` string.
    _DTYPE_MAP = {
        "float16":  "torch.float16",
        "bfloat16": "torch.bfloat16",
        "float32":  "torch.float32",
    }
    dtype_str = _DTYPE_MAP.get(args.precision, "torch.float16")
    OmegaConf.update(cfg, "trainer.mixed_precision.dtype", dtype_str)

    # 6) Tri-stage LR scheduler.  TriStageLRConfig uses ``stage_ratio``
    #    (a 3-tuple summing to 1.0), NOT separate warmup/hold/decay
    #    fields.  Our project-level YAML exposes warmup_ratio +
    #    hold_ratio so we derive the third stage.
    warmup = float(args.warmup_ratio)
    hold   = float(args.hold_ratio)
    decay  = max(0.0, 1.0 - warmup - hold)
    OmegaConf.update(
        cfg, "lr_scheduler.config.stage_ratio", [warmup, hold, decay]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, str(out_path))
    logger.info("Rendered Meta-schema YAML -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------


def _detect_existing_checkpoint(output_dir: Path) -> Optional[Path]:
    if not output_dir.exists():
        return None
    candidates = [
        output_dir / "checkpoints" / "last.pt",
        output_dir / "checkpoints" / "latest.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    pts = sorted(
        output_dir.rglob("checkpoint_*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return pts[0] if pts else None


def _find_best_checkpoint(output_dir: Path) -> Optional[Path]:
    """Locate the best/latest fairseq2 checkpoint after a run completes."""
    candidates = [
        output_dir / "checkpoints" / "best.pt",
        output_dir / "best.pt",
        output_dir / "checkpoints" / "last.pt",
        output_dir / "checkpoints" / "latest.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    pts = sorted(output_dir.rglob("*.pt"),
                 key=lambda p: p.stat().st_mtime, reverse=True)
    return pts[0] if pts else None


# ---------------------------------------------------------------------------
# Recipe launch
# ---------------------------------------------------------------------------


def _launch_meta_recipe(
    repo_path: Path,
    output_dir: Path,
    resolved_yaml: Path,
) -> int:
    cmd = [
        sys.executable, "-m", RECIPE_MODULE,
        str(output_dir),
        "--config-file", str(resolved_yaml),
    ]
    logger.info("Launching Meta recipe:")
    logger.info("  cwd = %s", repo_path)
    logger.info("  cmd = %s", " ".join(cmd))
    env = os.environ.copy()
    # Ensure the Meta repo's package is importable when run via -m.
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(repo_path), env.get("PYTHONPATH", "")])
    )
    proc = subprocess.run(cmd, cwd=str(repo_path), env=env)
    return proc.returncode


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def train_omniasr(args: OmniASRTrainArgs) -> Dict[str, float]:
    """Launch the official Meta fairseq2 OmniASR fine-tuning recipe."""
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
    manifest_dir = layout.preprocessed_dir("omniasr") / "manifest"
    final_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("OmniASR (fairseq2) fine-tuning — variant=%s", args.variant)
    logger.info("Asset name      : %s", args.model_name_or_path)
    logger.info("Checkpoints     : %s", args.output_dir)
    logger.info("Final model     : %s", final_dir)
    logger.info("TensorBoard     : %s", tensorboard_dir)
    logger.info("Manifest dir    : %s", manifest_dir)
    logger.info("Smoke           : %s", args.smoke)
    logger.info("Storage root    : %s", layout.root)
    logger.info("=" * 70)

    # 1) Verify Meta runtime + repo.
    _check_fairseq2_available()
    repo_path = _resolve_meta_repo(args)

    # 2) HF dataset -> fairseq2 manifest (cached).
    _convert_dataset_to_manifest(args, manifest_dir)
    data_dir_for_recipe = manifest_dir
    if args.smoke:
        data_dir_for_recipe = _truncate_for_smoke(
            manifest_dir, args.smoke_train_size, args.smoke_dev_size,
        )

    # 3) Register a fairseq2 asset card so the recipe can resolve
    #    ``dataset.name`` to the manifest directory.  The asset name
    #    is per-variant so smoke and paper-scale runs do not collide.
    asset_name = f"hutsul_omniasr_{args.variant.replace('-', '_')}"
    _register_manifest_asset_card(asset_name, data_dir_for_recipe)

    # 4) Resolve Meta YAML template + render.
    template_path = (
        Path(args.meta_yaml_template) if args.meta_yaml_template
        else DEFAULT_META_TEMPLATE
    )
    if not template_path.exists():
        raise FileNotFoundError(
            f"Meta recipe template not found: {template_path}"
        )
    resolved_yaml = args.output_dir / "ctc-finetune.resolved.yaml"
    _render_meta_yaml(args, template_path, asset_name, resolved_yaml)

    # 4) Resume detection (informational — fairseq2 auto-resumes from
    #    the output_dir when checkpoints are present).
    existing = _detect_existing_checkpoint(args.output_dir)
    if args.resume_from_checkpoint:
        if existing is not None:
            logger.info(
                "Resume hint '%s' acknowledged.  fairseq2 will auto-"
                "resume from %s.",
                args.resume_from_checkpoint, existing,
            )
        else:
            logger.info(
                "Resume hint '%s' acknowledged but no checkpoint found "
                "under %s — starting fresh.",
                args.resume_from_checkpoint, args.output_dir,
            )
    elif existing is not None:
        logger.info("Found existing checkpoint %s — fairseq2 will resume.",
                    existing)

    # 5) Launch official recipe.
    rc = _launch_meta_recipe(repo_path, args.output_dir, resolved_yaml)
    if rc != 0:
        raise RuntimeError(
            f"Meta OmniASR recipe exited with code {rc}.  Inspect "
            f"{args.output_dir} for the training log."
        )

    # 6) Copy best/last checkpoint to <final_models>/<variant>.pt
    best = _find_best_checkpoint(args.output_dir)
    if best is not None:
        target = final_dir / f"{args.variant}.pt"
        shutil.copy2(str(best), str(target))
        logger.info("Copied final checkpoint to %s", target)
    else:
        logger.warning(
            "No *.pt produced under %s; final_models directory left empty.",
            args.output_dir,
        )

    logger.info("OmniASR fine-tuning complete.")
    return {"recipe_exit_code": 0.0}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "OmniASRProcessor",
    "OmniASRTrainArgs",
    "args_from_yaml",
    "load_yaml_config",
    "train_omniasr",
]
