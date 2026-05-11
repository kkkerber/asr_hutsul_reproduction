"""On-the-fly augmentation.

WaveformAugmenter: audiomentations (noise / pitch / time-stretch / gain),
applied before feature extraction.
SpecAugment: time + frequency masking on the padded log-mel tensor.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# audiomentations is optional — only required when augmentation is enabled.
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


@dataclass
class WaveformAugmentConfig:
    """Conservative defaults — mild enough not to break Hutsul vowels."""

    sample_rate: int = 16_000

    p_noise: float = 0.5
    min_amplitude: float = 0.0005
    max_amplitude: float = 0.015

    p_pitch: float = 0.4
    min_semitones: float = -2.0
    max_semitones: float = 2.0

    # 0.8x–1.2x speed perturbation
    p_time: float = 0.4
    min_rate: float = 0.8
    max_rate: float = 1.2

    p_gain: float = 0.4
    min_gain_db: float = -6.0
    max_gain_db: float = 6.0


class WaveformAugmenter:
    """audiomentations Compose: noise / pitch / time-stretch / gain."""

    def __init__(self, config: Optional[WaveformAugmentConfig] = None) -> None:
        if not _AUDIOMENTATIONS_AVAILABLE:
            raise ImportError(
                "WaveformAugmenter requires audiomentations. "
                "pip install audiomentations"
            )

        self.config = config or WaveformAugmentConfig()

        # audiomentations >= 0.34 removed TimeStretch.leave_length_unchanged.
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


@dataclass
class SpecAugmentConfig:
    """Park et al. (2019) "LibriSpeech basic", scaled for Hutsul."""

    time_mask_param: int = 40
    n_time_masks: int = 2
    time_mask_p: float = 1.0

    freq_mask_param: int = 27
    n_freq_masks: int = 2

    apply_prob: float = 0.8

    # None -> mean of the input tensor
    mask_value: Optional[float] = None


class SpecAugment:
    """Time + frequency masking on shape ``(..., n_mels, n_frames)``."""

    def __init__(self, config: Optional[SpecAugmentConfig] = None) -> None:
        self.config = config or SpecAugmentConfig()

    def __call__(self, features: Any) -> Any:
        if random.random() > self.config.apply_prob:
            return features

        try:
            import torch  # noqa: WPS433

            if isinstance(features, torch.Tensor):
                return self._apply_torch(features)
        except ImportError:  # pragma: no cover
            pass

        return self._apply_numpy(np.asarray(features))

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

        max_time = int(min(self.config.time_mask_param,
                           max(1, int(n_frames * self.config.time_mask_p))))
        for _ in range(self.config.n_time_masks):
            t = random.randint(0, max_time)
            if t == 0 or t >= n_frames:
                continue
            t0 = random.randint(0, n_frames - t)
            out[..., :, t0 : t0 + t] = fill

        for _ in range(self.config.n_freq_masks):
            f = random.randint(0, min(self.config.freq_mask_param, n_mels))
            if f == 0 or f >= n_mels:
                continue
            f0 = random.randint(0, n_mels - f)
            out[..., f0 : f0 + f, :] = fill

        return out

    def _apply_torch(self, x: "Any") -> "Any":  # noqa: F821
        import torch

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

        max_time = int(min(self.config.time_mask_param,
                           max(1, int(n_frames * self.config.time_mask_p))))
        for _ in range(self.config.n_time_masks):
            t = torch.randint(0, max_time + 1, (1,)).item()
            if t == 0 or t >= n_frames:
                continue
            t0 = torch.randint(0, n_frames - t + 1, (1,)).item()
            out[..., :, t0 : t0 + t] = fill

        for _ in range(self.config.n_freq_masks):
            f = torch.randint(
                0, min(self.config.freq_mask_param, n_mels) + 1, (1,)
            ).item()
            if f == 0 or f >= n_mels:
                continue
            f0 = torch.randint(0, n_mels - f + 1, (1,)).item()
            out[..., f0 : f0 + f, :] = fill

        return out


@dataclass
class AugmentationPipeline:
    """Waveform + SpecAugment bundle. Either field can be ``None``."""

    waveform: Optional[WaveformAugmenter] = None
    specaugment: Optional[SpecAugment] = None

    def apply_waveform(
        self, samples: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        if self.waveform is None:
            return samples
        return self.waveform(samples, sample_rate)

    def apply_features(self, features: Any) -> Any:
        if self.specaugment is None:
            return features
        return self.specaugment(features)

    @property
    def enabled(self) -> bool:
        return self.waveform is not None or self.specaugment is not None


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
    """``strong=True`` uses the W2V-BERT preset (wider masks, more noise)."""

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


__all__ = [
    "AugmentationPipeline",
    "SpecAugment",
    "SpecAugmentConfig",
    "WaveformAugmentConfig",
    "WaveformAugmenter",
    "build_augmentation_pipeline",
]
