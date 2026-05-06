"""
config.py
=========

Central project configuration for the ASR Hutsul reproduction study.

This module centralises every constant that more than one script needs:

* default dataset name and split ratios,
* the canonical Ukrainian alphabet (Hutsul-compatible),
* directory layout for outputs, caches and checkpoints,
* default reproducibility settings.

It is intentionally framework-agnostic so it can be imported from
preprocessing, training and evaluation scripts without pulling in heavy
dependencies (no transformers / datasets imports here).
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotent logger configuration used by every entry-point script."""
    root = logging.getLogger()
    if root.handlers:
        # Already configured (e.g. when imported by Jupyter).
        for handler in root.handlers:
            handler.setLevel(level)
        root.setLevel(level)
        return
    logging.basicConfig(level=level, format=LOG_FORMAT)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# The repository root is the directory that contains this file.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Storage layout
# ---------------------------------------------------------------------------
#
# We support three storage roots, in priority order:
#
# 1. Explicit override via ``HUTSUL_ASR_ROOT`` environment variable
#    (or via ``--storage_root`` CLI / ``StorageLayout(root=...)`` API).
# 2. Google Drive root when running in Colab AND Drive is mounted at
#    ``/content/drive/MyDrive``.  Path: ``/content/drive/MyDrive/hutsul_asr/``.
# 3. Project-local fallback: ``<repo>/outputs/`` for outputs and
#    ``<repo>/.cache/`` for caches.
#
# The layout exposes per-purpose subdirectories matching the
# documented project structure:
#
#   <root>/
#     checkpoints/<variant>/        ← Trainer ``output_dir`` (every checkpoint-N)
#     final_models/<variant>/       ← post-training save_model + processor
#     preprocessed/<model_type>/    ← cached normalized DatasetDict
#     evaluations/csv/              ← evaluate.py predictions.csv
#     evaluations/json/             ← evaluate.py test_results / error_analysis
#     evaluations/predictions/      ← raw decoded outputs
#     tensorboard/<variant>/        ← TensorBoard logging_dir
#     cache/                        ← HF_HOME / TRANSFORMERS_CACHE
#     datasets/                     ← HF_DATASETS_CACHE
#
# The previous ``DEFAULT_*`` constants are kept for backward
# compatibility with any external callers — they now resolve through
# the same layout.

DRIVE_ROOT: Final[Path] = Path("/content/drive/MyDrive/hutsul_asr")
DRIVE_MOUNT_PROBE: Final[Path] = Path("/content/drive/MyDrive")
LOCAL_FALLBACK_ROOT: Final[Path] = PROJECT_ROOT / "outputs"


def is_colab() -> bool:
    """Return ``True`` when we are running inside Google Colab."""
    if os.environ.get("COLAB_RELEASE_TAG"):
        return True
    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def is_drive_mounted() -> bool:
    """Return ``True`` when ``/content/drive/MyDrive`` is reachable."""
    try:
        return DRIVE_MOUNT_PROBE.exists() and DRIVE_MOUNT_PROBE.is_dir()
    except OSError:
        return False


def _select_storage_root(override: Optional[Path] = None) -> Path:
    """Resolve the canonical storage root for the current environment."""
    if override is not None:
        return Path(override)

    env = os.environ.get("HUTSUL_ASR_ROOT")
    if env:
        return Path(env)

    if is_colab() and is_drive_mounted():
        return DRIVE_ROOT

    return LOCAL_FALLBACK_ROOT


@dataclass(frozen=True)
class StorageLayout:
    """Filesystem layout for every artefact the project produces.

    The layout is rooted at :attr:`root` and provides per-purpose
    subdirectories.  All accessors return :class:`pathlib.Path`
    instances (never strings) so callers can compose paths
    ergonomically.

    Use :func:`resolve_storage_layout` rather than constructing this
    class directly — it picks the right root for Colab vs. local
    runs.
    """

    root: Path

    # ------------------------------------------------------------------
    # Top-level directories
    # ------------------------------------------------------------------
    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def final_models(self) -> Path:
        return self.root / "final_models"

    @property
    def preprocessed(self) -> Path:
        return self.root / "preprocessed"

    @property
    def evaluations(self) -> Path:
        return self.root / "evaluations"

    @property
    def tensorboard(self) -> Path:
        return self.root / "tensorboard"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def datasets_cache(self) -> Path:
        return self.root / "datasets"

    # ------------------------------------------------------------------
    # Per-purpose helpers
    # ------------------------------------------------------------------
    def checkpoint_dir(self, variant: str) -> Path:
        return self.checkpoints / variant

    def final_model_dir(self, variant: str) -> Path:
        return self.final_models / variant

    def preprocessed_dir(self, model_type: str) -> Path:
        return self.preprocessed / model_type

    @property
    def evaluations_csv(self) -> Path:
        return self.evaluations / "csv"

    @property
    def evaluations_json(self) -> Path:
        return self.evaluations / "json"

    @property
    def evaluations_predictions(self) -> Path:
        return self.evaluations / "predictions"

    def tensorboard_dir(self, variant: str) -> Path:
        return self.tensorboard / variant

    def evaluation_run_dir(self, name: str) -> Path:
        """Per-run evaluation directory for a single ``evaluate.py`` invocation.

        Returns a *flat* path inside ``evaluations/`` whose name
        identifies the evaluated checkpoint (e.g. ``whisper-small``
        or ``whisper-small__step1500``).  The CSV / JSON / predictions
        files for that run all live next to one another in the
        per-purpose subdirectories under :meth:`evaluations`.
        """
        return self.evaluations / name

    # ------------------------------------------------------------------
    # Bulk-create
    # ------------------------------------------------------------------
    def ensure(self) -> None:
        """Create every top-level directory if it does not yet exist."""
        for d in (
            self.root,
            self.checkpoints,
            self.final_models,
            self.preprocessed,
            self.evaluations,
            self.evaluations_csv,
            self.evaluations_json,
            self.evaluations_predictions,
            self.tensorboard,
            self.cache,
            self.datasets_cache,
        ):
            try:
                d.mkdir(parents=True, exist_ok=True)
            except OSError as exc:  # pragma: no cover — Drive read-only / quota
                logging.getLogger(__name__).warning(
                    "Could not create %s: %s", d, exc
                )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def summary(self) -> str:
        return (
            f"StorageLayout(root={self.root})\n"
            f"  checkpoints      = {self.checkpoints}\n"
            f"  final_models     = {self.final_models}\n"
            f"  preprocessed     = {self.preprocessed}\n"
            f"  evaluations      = {self.evaluations}\n"
            f"  tensorboard      = {self.tensorboard}\n"
            f"  cache            = {self.cache}\n"
            f"  datasets_cache   = {self.datasets_cache}\n"
        )


_RESOLVED_LAYOUT: Optional[StorageLayout] = None


def resolve_storage_layout(
    override_root: Optional[Path] = None,
    *,
    refresh: bool = False,
) -> StorageLayout:
    """Return the canonical :class:`StorageLayout` for this run.

    The result is cached so callers can call this freely; pass
    ``refresh=True`` to recompute (e.g. after Drive is mounted in a
    Colab cell).
    """
    global _RESOLVED_LAYOUT
    if refresh or _RESOLVED_LAYOUT is None or override_root is not None:
        _RESOLVED_LAYOUT = StorageLayout(root=_select_storage_root(override_root))
    return _RESOLVED_LAYOUT


def configure_hf_caches(layout: Optional[StorageLayout] = None) -> None:
    """Point the Hugging Face caches at the storage layout.

    Sets ``HF_HOME``, ``TRANSFORMERS_CACHE`` and ``HF_DATASETS_CACHE``
    via ``os.environ.setdefault`` (so user-supplied values win).

    This MUST be called before any ``transformers`` / ``datasets``
    imports take effect — those libraries cache the env vars on first
    import.  ``config.py`` invokes this automatically at import time
    when running inside a Colab session with Drive mounted.
    """
    layout = layout or resolve_storage_layout()
    layout.ensure()
    os.environ.setdefault("HF_HOME", str(layout.cache / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(layout.cache / "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(layout.datasets_cache))
    # Avoid HF telemetry during long-running Colab sessions.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


# Auto-apply HF cache redirection in Colab so subsequent
# ``from transformers import ...`` calls in the same Python process
# pick up the Drive-backed cache directory.  This is a no-op outside
# Colab.
if is_colab() and is_drive_mounted():
    try:
        configure_hf_caches()
    except Exception:  # pragma: no cover
        # Never let path setup crash module import.
        pass


# ---------------------------------------------------------------------------
# Backward-compatible default paths (resolved through the layout)
# ---------------------------------------------------------------------------

_DEFAULT_LAYOUT = resolve_storage_layout()

DEFAULT_OUTPUT_DIR: Final[Path] = _DEFAULT_LAYOUT.root
DEFAULT_CACHE_DIR: Final[Path] = _DEFAULT_LAYOUT.cache
DEFAULT_DATASET_CACHE_DIR: Final[Path] = _DEFAULT_LAYOUT.datasets_cache
DEFAULT_PREPROCESSED_DIR: Final[Path] = _DEFAULT_LAYOUT.preprocessed
DEFAULT_LOG_DIR: Final[Path] = _DEFAULT_LAYOUT.tensorboard


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

DEFAULT_DATASET_NAME: Final[str] = (
    "KSE-RESEARCH-Group/Dido-Yvanchyk-Audio-Dataset-v2"
)

# Paper-reported split: 80% train / 10% validation / 10% test.
DEFAULT_TRAIN_RATIO: Final[float] = 0.80
DEFAULT_VAL_RATIO: Final[float] = 0.10
DEFAULT_TEST_RATIO: Final[float] = 0.10

# Target audio sample rate.  All four model families are 16-kHz models.
TARGET_SAMPLE_RATE: Final[int] = 16_000

# Heuristic column-name candidates used when auto-detecting the audio /
# transcription columns of an arbitrary HF dataset.
AUDIO_COLUMN_CANDIDATES: Final[Tuple[str, ...]] = (
    "audio",
    "wav",
    "speech",
    "path",
    "file",
    "filepath",
    "filename",
)

TEXT_COLUMN_CANDIDATES: Final[Tuple[str, ...]] = (
    "sentence",
    "text",
    "transcription",
    "transcript",
    "normalized_text",
    "raw_text",
    "labels",
)

# ---------------------------------------------------------------------------
# Ukrainian / Hutsul alphabet
# ---------------------------------------------------------------------------
#
# The dialect uses the modern Ukrainian alphabet plus apostrophes.  We
# expose three helpful constants:
#
# * ``UKRAINIAN_ALPHABET`` — the 33-letter alphabet in canonical order;
# * ``ALLOWED_CHARS``      — alphabet ∪ {space, apostrophe};
# * ``CTC_VOCAB``          — base vocabulary for CTC tokenizers
#                            (alphabet + apostrophe + word delimiter).
#
# All apostrophe variants ('ʼ' U+02BC, '’' U+2019, "'" U+0027) are
# normalised to ``"'"`` (U+0027) by ``utils.text_normalization``.

UKRAINIAN_ALPHABET: Final[Tuple[str, ...]] = tuple(
    "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя"
)

APOSTROPHE: Final[str] = "'"

ALLOWED_CHARS: Final[frozenset] = frozenset(
    list(UKRAINIAN_ALPHABET) + [APOSTROPHE, " "]
)

# Word delimiter used by CTC tokenizers.
CTC_WORD_DELIMITER: Final[str] = "|"
CTC_PAD_TOKEN: Final[str] = "[PAD]"
CTC_UNK_TOKEN: Final[str] = "[UNK]"

CTC_VOCAB: Final[Tuple[str, ...]] = tuple(
    list(UKRAINIAN_ALPHABET)
    + [APOSTROPHE, CTC_WORD_DELIMITER, CTC_UNK_TOKEN, CTC_PAD_TOKEN]
)

# Dialect substitution pairs we explicitly track in error analysis.
# (reference, hypothesis) pairs that frequently appear in Hutsul data.
DIALECT_SUBSTITUTION_PAIRS: Final[Tuple[Tuple[str, str], ...]] = (
    ("и", "і"),
    ("і", "и"),
    ("е", "є"),
    ("є", "е"),
    ("о", "у"),  # frequent dialect rounding
    ("ї", "і"),
    ("щ", "ш"),
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

DEFAULT_SEED: Final[int] = 42


def set_global_seed(seed: int = DEFAULT_SEED, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch RNGs.

    The PyTorch import is deferred so that lightweight scripts (e.g. the
    text-normalization unit tests) can import ``config.py`` without
    pulling in torch.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # local import on purpose
    except ImportError:  # pragma: no cover — torch is required at training time
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # CuDNN-deterministic mode noticeably reduces throughput but is
        # required if you want bit-for-bit reproducibility across runs.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # ``use_deterministic_algorithms`` raises if a non-deterministic
        # op is hit; we set ``warn_only=True`` so that training does not
        # crash when a kernel has no deterministic implementation.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover
            torch.use_deterministic_algorithms(True)
        # Required for deterministic reductions on some CUDA kernels.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


# ---------------------------------------------------------------------------
# Project configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class ProjectConfig:
    """Runtime configuration shared across preprocessing / training / eval.

    A dataclass is used (instead of a global module) so that the values
    can be customised from CLI arguments or a YAML file without
    monkey-patching module-level constants.
    """

    # ---- Dataset ------------------------------------------------------
    dataset_name: str = DEFAULT_DATASET_NAME
    dataset_config: Optional[str] = None
    dataset_split: Optional[str] = None  # if None, auto-detect / build
    train_ratio: float = DEFAULT_TRAIN_RATIO
    val_ratio: float = DEFAULT_VAL_RATIO
    test_ratio: float = DEFAULT_TEST_RATIO

    audio_column: Optional[str] = None  # auto-detect when None
    text_column: Optional[str] = None
    sample_rate: int = TARGET_SAMPLE_RATE

    # ---- Filesystem ---------------------------------------------------
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    cache_dir: Path = field(default_factory=lambda: DEFAULT_CACHE_DIR)
    dataset_cache_dir: Path = field(
        default_factory=lambda: DEFAULT_DATASET_CACHE_DIR
    )
    preprocessed_dir: Path = field(
        default_factory=lambda: DEFAULT_PREPROCESSED_DIR
    )
    log_dir: Path = field(default_factory=lambda: DEFAULT_LOG_DIR)

    # ---- Reproducibility ----------------------------------------------
    seed: int = DEFAULT_SEED
    deterministic: bool = False

    # ---- Misc ---------------------------------------------------------
    num_workers: int = 2
    use_augmentation: bool = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        # Coerce strings to ``Path`` for ergonomic CLI integration.
        for name in (
            "output_dir",
            "cache_dir",
            "dataset_cache_dir",
            "preprocessed_dir",
            "log_dir",
        ):
            value = getattr(self, name)
            if not isinstance(value, Path):
                setattr(self, name, Path(value))

        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-6:
            raise ValueError(
                "train/val/test ratios must sum to 1.0 (got "
                f"{self.train_ratio} + {self.val_ratio} + {self.test_ratio})"
            )

    def ensure_dirs(self) -> None:
        """Create every directory referenced by this config."""
        for d in (
            self.output_dir,
            self.cache_dir,
            self.dataset_cache_dir,
            self.preprocessed_dir,
            self.log_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def child_output_dir(self, name: str) -> Path:
        """Return ``output_dir / <name>``, creating the directory."""
        path = self.output_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

__all__: List[str] = [
    "ALLOWED_CHARS",
    "APOSTROPHE",
    "AUDIO_COLUMN_CANDIDATES",
    "CTC_PAD_TOKEN",
    "CTC_UNK_TOKEN",
    "CTC_VOCAB",
    "CTC_WORD_DELIMITER",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_DATASET_CACHE_DIR",
    "DEFAULT_DATASET_NAME",
    "DEFAULT_LOG_DIR",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_PREPROCESSED_DIR",
    "DEFAULT_SEED",
    "DEFAULT_TEST_RATIO",
    "DEFAULT_TRAIN_RATIO",
    "DEFAULT_VAL_RATIO",
    "DIALECT_SUBSTITUTION_PAIRS",
    "DRIVE_ROOT",
    "LOCAL_FALLBACK_ROOT",
    "PROJECT_ROOT",
    "ProjectConfig",
    "StorageLayout",
    "TARGET_SAMPLE_RATE",
    "TEXT_COLUMN_CANDIDATES",
    "UKRAINIAN_ALPHABET",
    "configure_hf_caches",
    "configure_logging",
    "is_colab",
    "is_drive_mounted",
    "resolve_storage_layout",
    "set_global_seed",
]
