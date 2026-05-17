

from __future__ import annotations

from typing import Callable, Dict

TRAINER_MODULE_MAP: Dict[str, str] = {
    "whisper": "models.whisper_trainer",
    "wav2vec2": "models.wav2vec2_trainer",
    "wav2vec2_bert": "models.wav2vec2_bert_trainer",
    "omniasr": "models.omniasr_trainer",
    "parakeet": "models.parakeet_trainer",
}


def get_trainer_module(model_type: str) -> str:
    if model_type not in TRAINER_MODULE_MAP:
        raise KeyError(
            f"Unknown model_type {model_type!r}. "
            f"Choose one of: {sorted(TRAINER_MODULE_MAP)}"
        )
    return TRAINER_MODULE_MAP[model_type]


__all__ = ["TRAINER_MODULE_MAP", "get_trainer_module"]
