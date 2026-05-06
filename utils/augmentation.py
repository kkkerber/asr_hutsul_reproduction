"""
utils/augmentation.py
=====================

Optional on-the-fly data augmentation for ASR training.

Two complementary augmentation stages are implemented:

1. **Waveform augmentation** (:class:`WaveformAugmenter`) — applied
   *before* feature extraction, on raw 16-kHz mono audio.  Powered by
   the ``audiomentations`` library:

   * Gaussian noise injection,
   * pitch shifting (semitones),
   * speed / time stretching (0.8×–1.2×),
   * gain modulation (dB).

2. **Feature-level augmentation** (:class:`SpecAugment`) — applied
   *after* feature extraction, directly on log-mel / fbank tensors.
   This is a faithful reimplementation of Park et al. (2019)
   SpecAugment with independent time and frequency masking.

The :func:`build_augmentation_pipeline` factory returns the appropriate
augmenter for a given model family and respects the project
``ProjectConfig.use_augmentation`` flag (or the equivalent
``--use_augmentation`` CLI flag).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: audiomentations
# ---------------------------------------------------------------------------
#
# ``audiomentations`` is listed in ``requirements.txt``, but we still
# guard the import — the module is importable on machines that have not
# installed the optional dep, and only fails when the user actually
# enables augmentation.

try:
    from audiomentations import (
        AddGaussianNoise,
        Compose,
        Gain,
        PitchShift,
        TimeStretch,
    )

    _AUDIOMENTATIONS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _AUDIOMENTATIONS_AVAILABLE = False
    AddGaussianNoise = Compose = Gain = PitchShift = TimeStretch = None  # type: ignore


# ---------------------------------------------------------------------------
# Waveform augmentation
# ---------------------------------------------------------------------------


@dataclass
class WaveformAugmentConfig:
    """Hyper-parameters for :class:`WaveformAugmenter`.

    Defaults are conservative — strong enough to regularise but mild
    enough that they do not destroy phonetic content for Hutsul vowel
    pairs (which are sensitive to pitch and formant shifts).
    """

    sample_rate: int = 16_000

    # Gaussian noise
    p_noise: float = 0.5
    min_amplitude: float = 0.0005
    max_amplitude: float = 0.015

    # Pitch shift (semitones)
    p_pitch: float = 0.4
    min_semitones: float = -2.0
    max_semitones: float = 2.0

    # Time stretch (speed perturbation, 0.8x–1.2x)
    p_time: float = 0.4
    min_rate: float = 0.8
    max_rate: float = 1.2

    # Gain modulation (dB)
    p_gain: float = 0.4
    min_gain_db: float = -6.0
    max_gain_db: float = 6.0


class WaveformAugmenter:
    """Composable waveform augmenter built on ``audiomentations``.

    The object is callable: ``augmenter(samples, sample_rate)`` returns
    an augmented ``np.ndarray`` of the same dtype.  Both arguments
    mirror the audiomentations call convention so that the augmenter
    can be slotted into any preprocessing pipeline that already uses
    that library.
    """

    def __init__(self, config: Optional[WaveformAugmentConfig] = None) -> None:
        if not _AUDIOMENTATIONS_AVAILABLE:
            raise ImportError(
                "WaveformAugmenter requires the optional package "
                "'audiomentations'.  Install it via: pip install audiomentations"
            )

        self.config = config or WaveformAugmentConfig()

        # ``Compose`` runs each transform with the configured per-call
        # probability.  audiomentations >= 0.34 dropped the
        # ``leave_length_unchanged`` parameter on ``TimeStretch`` —
        # the audio length always changes now, which is what we want
        # for speed perturbation anyway.  We only pass kwargs that
        # exist across all supported versions (>= 0.36 per
        # requirements.txt).
        self._pipeline = Compose(
            [
                AddGaussianNoise(
                    min_amplitude=self.config.min_amplitude,
                    max_amplitude=self.config.max_amplitude,
                    p=self.config.p_noise,
                ),
                PitchShift(
                    min_semitones=self.config.min_semitones,
                    max_semitones=self.config.max_semitones,
                    p=self.config.p_pitch,
                ),
                TimeStretch(
                    min_rate=self.config.min_rate,
                    max_rate=self.config.max_rate,
                    p=self.config.p_time,
                ),
                Gain(
                    min_gain_db=self.config.min_gain_db,
                    max_gain_db=self.config.max_gain_db,
                    p=self.config.p_gain,
                ),
            ]
        )

    # ------------------------------------------------------------------
    def __call__(
        self,
        samples: np.ndarray,
        sample_rate: Optional[int] = None,
    ) -> np.ndarray:
        sr = sample_rate if sample_rate is not None else self.config.sample_rate
        if not isinstance(samples, np.ndarray):
            samples = np.asarray(samples, dtype=np.float32)
        elif samples.dtype != np.float32:
            samples = samples.astype(np.float32, copy=False)
        try:
            return self._pipeline(samples=samples, sample_rate=sr)
        except Exception as exc:  # pragma: no cover — augmentation must never crash training
            logger.warning(
                "Waveform augmentation failed (%s); returning the original "
                "audio.",
                exc,
            )
            return samples


# ---------------------------------------------------------------------------
# SpecAugment (feature-level)
# ---------------------------------------------------------------------------


@dataclass
class SpecAugmentConfig:
    """Hyper-parameters for :class:`SpecAugment`.

    Defaults follow the "LibriSpeech basic" policy from Park et al.,
    scaled down slightly for the much smaller Hutsul corpus.
    """

    # Time masking
    time_mask_param: int = 40
    n_time_masks: int = 2
    time_mask_p: float = 1.0  # cap on fraction of total length

    # Frequency masking
    freq_mask_param: int = 27
    n_freq_masks: int = 2

    # Probabilities (per call)
    apply_prob: float = 0.8

    # Mask fill value.  ``None`` -> mean of the input tensor.
    mask_value: Optional[float] = None


class SpecAugment:
    """Feature-domain SpecAugment.

    The transform operates on a tensor of shape ``(..., n_mels, n_frames)``
    (the layout produced by all four model families' feature extractors
    once converted to ``torch.Tensor``).  Numpy arrays are accepted as
    well — the type is preserved on output.
    """

    def __init__(self, config: Optional[SpecAugmentConfig] = None) -> None:
        self.config = config or SpecAugmentConfig()

    # ------------------------------------------------------------------
    def __call__(self, features: Any) -> Any:
        if random.random() > self.config.apply_prob:
            return features

        # Lazy import torch so that this module can be imported in
        # environments where torch is missing (e.g. CI lint).
        try:
            import torch  # noqa: WPS433

            if isinstance(features, torch.Tensor):
                return self._apply_torch(features)
        except ImportError:  # pragma: no cover
            pass

        return self._apply_numpy(np.asarray(features))

    # ------------------------------------------------------------------
    def _apply_numpy(self, x: np.ndarray) -> np.ndarray:
        if x.ndim < 2:
            raise ValueError(
                f"SpecAugment expects at least 2-D input, got {x.shape}"
            )
        out = x.copy()
        n_mels, n_frames = out.shape[-2], out.shape[-1]

        fill = (
            float(out.mean())
            if self.config.mask_value is None
            else float(self.config.mask_value)
        )

        # Time masks
        max_time = int(min(self.config.time_mask_param,
                           max(1, int(n_frames * self.config.time_mask_p))))
        for _ in range(self.config.n_time_masks):
            t = random.randint(0, max_time)
            if t == 0 or t >= n_frames:
                continue
            t0 = random.randint(0, n_frames - t)
            out[..., :, t0 : t0 + t] = fill

        # Frequency masks
        for _ in range(self.config.n_freq_masks):
            f = random.randint(0, min(self.config.freq_mask_param, n_mels))
            if f == 0 or f >= n_mels:
                continue
            f0 = random.randint(0, n_mels - f)
            out[..., f0 : f0 + f, :] = fill

        return out

    # ------------------------------------------------------------------
    def _apply_torch(self, x: "Any") -> "Any":  # noqa: F821 — runtime-only torch
        import torch  # local import

        if x.dim() < 2:
            raise ValueError(
                f"SpecAugment expects at least 2-D input, got {tuple(x.shape)}"
            )
        out = x.clone()
        n_mels = out.size(-2)
        n_frames = out.size(-1)

        fill = (
            out.float().mean().item()
            if self.config.mask_value is None
            else float(self.config.mask_value)
        )

        # Time masks
        max_time = int(min(self.config.time_mask_param,
                           max(1, int(n_frames * self.config.time_mask_p))))
        for _ in range(self.config.n_time_masks):
            t = torch.randint(0, max_time + 1, (1,)).item()
            if t == 0 or t >= n_frames:
                continue
            t0 = torch.randint(0, n_frames - t + 1, (1,)).item()
            out[..., :, t0 : t0 + t] = fill

        # Frequency masks
        for _ in range(self.config.n_freq_masks):
            f = torch.randint(
                0, min(self.config.freq_mask_param, n_mels) + 1, (1,)
            ).item()
            if f == 0 or f >= n_mels:
                continue
            f0 = torch.randint(0, n_mels - f + 1, (1,)).item()
            out[..., f0 : f0 + f, :] = fill

        return out


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------


@dataclass
class AugmentationPipeline:
    """Bundles a waveform augmenter and a SpecAugment instance.

    Either component is optional — set the corresponding field to
    ``None`` to disable it.  This is the structure consumed by every
    Trainer in ``models/``.
    """

    waveform: Optional[WaveformAugmenter] = None
    specaugment: Optional[SpecAugment] = None

    # ------------------------------------------------------------------
    def apply_waveform(
        self, samples: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        if self.waveform is None:
            return samples
        return self.waveform(samples, sample_rate)

    # ------------------------------------------------------------------
    def apply_features(self, features: Any) -> Any:
        if self.specaugment is None:
            return features
        return self.specaugment(features)

    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.waveform is not None or self.specaugment is not None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_augmentation_pipeline(
    *,
    use_augmentation: bool,
    sample_rate: int = 16_000,
    waveform_config: Optional[WaveformAugmentConfig] = None,
    spec_config: Optional[SpecAugmentConfig] = None,
    enable_waveform: bool = True,
    enable_specaugment: bool = True,
    strong: bool = False,
) -> AugmentationPipeline:
    """Build a project-wide :class:`AugmentationPipeline`.

    Parameters
    ----------
    use_augmentation
        Master switch — if ``False`` returns an empty (no-op) pipeline.
    sample_rate
        Target audio sample rate; defaults to 16 kHz.
    waveform_config / spec_config
        Optional explicit overrides.  When omitted the defaults are
        derived from ``strong``.
    enable_waveform / enable_specaugment
        Allow callers to turn off individual stages even when
        ``use_augmentation=True``.
    strong
        Use the "strong" augmentation policy described in the paper for
        Wav2Vec2-BERT (wider masks, higher noise, larger pitch range).
    """

    if not use_augmentation:
        return AugmentationPipeline(waveform=None, specaugment=None)

    if strong:
        waveform_config = waveform_config or WaveformAugmentConfig(
            sample_rate=sample_rate,
            p_noise=0.6,
            min_amplitude=0.001,
            max_amplitude=0.025,
            p_pitch=0.5,
            min_semitones=-3.0,
            max_semitones=3.0,
            p_time=0.5,
            min_rate=0.85,
            max_rate=1.15,
            p_gain=0.5,
            min_gain_db=-8.0,
            max_gain_db=8.0,
        )
        spec_config = spec_config or SpecAugmentConfig(
            time_mask_param=60,
            n_time_masks=2,
            freq_mask_param=40,
            n_freq_masks=2,
            apply_prob=0.9,
        )
    else:
        waveform_config = waveform_config or WaveformAugmentConfig(
            sample_rate=sample_rate
        )
        spec_config = spec_config or SpecAugmentConfig()

    waveform_aug: Optional[WaveformAugmenter] = None
    if enable_waveform:
        if not _AUDIOMENTATIONS_AVAILABLE:
            logger.warning(
                "audiomentations is not installed — waveform augmentation "
                "disabled."
            )
        else:
            waveform_aug = WaveformAugmenter(waveform_config)

    spec_aug: Optional[SpecAugment] = None
    if enable_specaugment:
        spec_aug = SpecAugment(spec_config)

    return AugmentationPipeline(waveform=waveform_aug, specaugment=spec_aug)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "AugmentationPipeline",
    "SpecAugment",
    "SpecAugmentConfig",
    "WaveformAugmentConfig",
    "WaveformAugmenter",
    "build_augmentation_pipeline",
]
