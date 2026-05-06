"""Utility subpackage for the ASR Hutsul reproduction project.

This package groups small, dependency-light helpers that are reused
across preprocessing, training and evaluation:

* :mod:`utils.text_normalization` — Ukrainian / Hutsul transcription
  cleanup utilities;
* :mod:`utils.augmentation`       — on-the-fly waveform augmentation
  (Gaussian noise, pitch shift, speed perturbation, gain) plus
  SpecAugment;
* :mod:`utils.collators`          — Trainer-compatible data collators
  for Whisper (seq2seq) and CTC models;
* :mod:`utils.callbacks`          — Trainer callbacks (best-CER
  selection, TensorBoard wiring, early stopping factory).
"""

from .text_normalization import (
    TextNormalizer,
    build_default_normalizer,
    normalize_text,
)

__all__ = [
    "TextNormalizer",
    "build_default_normalizer",
    "normalize_text",
]
