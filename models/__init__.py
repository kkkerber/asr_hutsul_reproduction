"""Model trainer subpackage.

Each module exposes a ``train(config)`` entry point and a small set of
helpers for building the corresponding :class:`Trainer`.

* :mod:`models.whisper_trainer`        — Whisper (Seq2Seq + LoRA)
* :mod:`models.wav2vec2_trainer`       — Wav2Vec2-XLSR (CTC)
* :mod:`models.wav2vec2_bert_trainer`  — Wav2Vec2-BERT-UK (CTC + adapters)
* :mod:`models.omniasr_trainer`        — OmniASR (CTC + tri-stage LR)

The dispatcher in :mod:`train` imports a trainer by ``--model_type`` so
modules are *not* eagerly imported here — that keeps ``import models``
cheap on machines that only need one model family installed.
"""

from __future__ import annotations

from typing import Callable, Dict

# Mapping from --model_type CLI value to the dotted import path of its
# trainer module.  ``train.py`` resolves this lazily via ``importlib``.
TRAINER_MODULE_MAP: Dict[str, str] = {
    "whisper": "models.whisper_trainer",
    "wav2vec2": "models.wav2vec2_trainer",
    "wav2vec2_bert": "models.wav2vec2_bert_trainer",
    "omniasr": "models.omniasr_trainer",
}


def get_trainer_module(model_type: str) -> str:
    """Return the dotted module path for a given ``--model_type``."""
    if model_type not in TRAINER_MODULE_MAP:
        raise KeyError(
            f"Unknown model_type {model_type!r}. "
            f"Choose one of: {sorted(TRAINER_MODULE_MAP)}"
        )
    return TRAINER_MODULE_MAP[model_type]


__all__ = ["TRAINER_MODULE_MAP", "get_trainer_module"]
