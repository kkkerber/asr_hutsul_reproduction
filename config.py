from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, List, Optional, Tuple

import numpy as np

LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)


def configure_logging(level: int = logging.INFO) -> None:

    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            handler.setLevel(level)
        root.setLevel(level)
        return
    logging.basicConfig(level=level, format=LOG_FORMAT)


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent

DRIVE_ROOT: Final[Path] = Path("/content/drive/MyDrive/hutsul_asr")
DRIVE_MOUNT_PROBE: Final[Path] = Path("/content/drive/MyDrive")
LOCAL_FALLBACK_ROOT: Final[Path] = PROJECT_ROOT / "outputs"


def is_colab() -> bool:
    if os.environ.get("COLAB_RELEASE_TAG"):
        return True
    try:
        import google.colab

        return True
    except Exception:
        return False


def is_drive_mounted() -> bool:
    try:
        return DRIVE_MOUNT_PROBE.exists() and DRIVE_MOUNT_PROBE.is_dir()
    except OSError:
        return False


def _select_storage_root(override: Optional[Path] = None) -> Path:
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

    root: Path

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
        return self.evaluations / name

    def ensure(self) -> None:
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
            except OSError as exc:
                logging.getLogger(__name__).warning(
                    "Could not create %s: %s", d, exc
                )

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
    global _RESOLVED_LAYOUT
    if refresh or _RESOLVED_LAYOUT is None or override_root is not None:
        _RESOLVED_LAYOUT = StorageLayout(root=_select_storage_root(override_root))
    return _RESOLVED_LAYOUT


def configure_hf_caches(layout: Optional[StorageLayout] = None) -> None:

    layout = layout or resolve_storage_layout()
    layout.ensure()
    os.environ.setdefault("HF_HOME", str(layout.cache / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(layout.cache / "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(layout.datasets_cache))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


if is_colab() and is_drive_mounted():
    try:
        configure_hf_caches()
    except Exception:
        pass


_DEFAULT_LAYOUT = resolve_storage_layout()

DEFAULT_OUTPUT_DIR: Final[Path] = _DEFAULT_LAYOUT.root
DEFAULT_CACHE_DIR: Final[Path] = _DEFAULT_LAYOUT.cache
DEFAULT_DATASET_CACHE_DIR: Final[Path] = _DEFAULT_LAYOUT.datasets_cache
DEFAULT_PREPROCESSED_DIR: Final[Path] = _DEFAULT_LAYOUT.preprocessed
DEFAULT_LOG_DIR: Final[Path] = _DEFAULT_LAYOUT.tensorboard


DEFAULT_DATASET_NAME: Final[str] = (
    "KSE-RESEARCH-Group/Dido-Yvanchyk-Audio-Dataset-v2"
)

DEFAULT_TRAIN_RATIO: Final[float] = 0.80
DEFAULT_VAL_RATIO: Final[float] = 0.10
DEFAULT_TEST_RATIO: Final[float] = 0.10

TARGET_SAMPLE_RATE: Final[int] = 16_000

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


UKRAINIAN_ALPHABET: Final[Tuple[str, ...]] = tuple(
    "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя"
)

APOSTROPHE: Final[str] = "'"

ALLOWED_CHARS: Final[frozenset] = frozenset(
    list(UKRAINIAN_ALPHABET) + [APOSTROPHE, " "]
)

CTC_WORD_DELIMITER: Final[str] = "|"
CTC_PAD_TOKEN: Final[str] = "[PAD]"
CTC_UNK_TOKEN: Final[str] = "[UNK]"

CTC_VOCAB: Final[Tuple[str, ...]] = tuple(
    list(UKRAINIAN_ALPHABET)
    + [APOSTROPHE, CTC_WORD_DELIMITER, CTC_UNK_TOKEN, CTC_PAD_TOKEN]
)

DIALECT_SUBSTITUTION_PAIRS: Final[Tuple[Tuple[str, str], ...]] = (
    ("и", "і"),
    ("і", "и"),
    ("е", "є"),
    ("є", "е"),
    ("о", "у"),
    ("ї", "і"),
)


DEFAULT_SEED: Final[int] = 42

def set_global_seed(seed: int = DEFAULT_SEED, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            torch.use_deterministic_algorithms(True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@dataclass
class ProjectConfig:

    dataset_name: str = DEFAULT_DATASET_NAME
    dataset_config: Optional[str] = None
    dataset_split: Optional[str] = None
    train_ratio: float = DEFAULT_TRAIN_RATIO
    val_ratio: float = DEFAULT_VAL_RATIO
    test_ratio: float = DEFAULT_TEST_RATIO

    audio_column: Optional[str] = None
    text_column: Optional[str] = None
    sample_rate: int = TARGET_SAMPLE_RATE

    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    cache_dir: Path = field(default_factory=lambda: DEFAULT_CACHE_DIR)
    dataset_cache_dir: Path = field(
        default_factory=lambda: DEFAULT_DATASET_CACHE_DIR
    )
    preprocessed_dir: Path = field(
        default_factory=lambda: DEFAULT_PREPROCESSED_DIR
    )
    log_dir: Path = field(default_factory=lambda: DEFAULT_LOG_DIR)

    seed: int = DEFAULT_SEED
    deterministic: bool = False

    num_workers: int = 2
    use_augmentation: bool = False

    min_train_audio_duration_sec: Optional[float] = None
    min_collator_input_samples: Optional[int] = None

    def __post_init__(self) -> None:
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
        for d in (
            self.output_dir,
            self.cache_dir,
            self.dataset_cache_dir,
            self.preprocessed_dir,
            self.log_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def child_output_dir(self, name: str) -> Path:
        path = self.output_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path


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
