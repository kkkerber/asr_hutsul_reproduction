

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# We import lazily inside functions where reasonable, but for the
# callback class itself we import at module top — transformers is a
from transformers import (
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


@dataclass
class BestCERTrackerCallback(TrainerCallback):
    """Logs best eval_cer + writes ``best_metric.json``."""

    metric_name: str = "eval_cer"
    secondary_metric_name: str = "eval_wer"
    output_filename: str = "best_metric.json"

    best_value: float = float("inf")
    best_step: int = -1
    best_secondary: Optional[float] = None

    def on_evaluate(  # type: ignore[override]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> TrainerControl:
        if metrics is None:
            return control

        value = metrics.get(self.metric_name)
        if value is None:
            value = metrics.get(self.metric_name.replace("eval_", ""))

        if value is None:
            return control

        if value < self.best_value:
            self.best_value = float(value)
            self.best_step = int(state.global_step)
            self.best_secondary = (
                float(metrics.get(self.secondary_metric_name, float("nan")))
                if self.secondary_metric_name
                else None
            )

            logger.info(
                "[BestCERTracker] new best %s = %.4f at step %d (wer=%.4f)",
                self.metric_name,
                self.best_value,
                self.best_step,
                self.best_secondary if self.best_secondary is not None else float("nan"),
            )

            try:
                out_path = Path(args.output_dir) / self.output_filename
                out_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "best_step": self.best_step,
                    "best_metric": self.metric_name,
                    "best_value": self.best_value,
                    "secondary_metric": self.secondary_metric_name,
                    "secondary_value": self.best_secondary,
                }
                out_path.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "Could not write %s: %s", self.output_filename, exc
                )

        return control


class MemoryMonitorCallback(TrainerCallback):
    """Log peak CUDA memory after each eval; reset for the next window."""

    def on_evaluate(  # type: ignore[override]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        try:
            import torch
        except ImportError:  # pragma: no cover
            return control

        if not torch.cuda.is_available():
            return control

        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        logger.info(
            "[MemoryMonitor] step=%d peak_cuda_memory=%.0f MiB",
            state.global_step,
            peak_mb,
        )
        torch.cuda.reset_peak_memory_stats()
        return control


def build_early_stopping_callback(
    *,
    patience: int = 5,
    threshold: float = 0.0,
) -> EarlyStoppingCallback:
    if patience < 1:
        raise ValueError(f"patience must be >= 1, got {patience}")
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")

    return EarlyStoppingCallback(
        early_stopping_patience=patience,
        early_stopping_threshold=threshold,
    )


def build_default_callbacks(
    *,
    enable_early_stopping: bool = True,
    early_stopping_patience: int = 5,
    early_stopping_threshold: float = 0.0,
    enable_best_cer_tracker: bool = True,
    enable_memory_monitor: bool = True,
) -> List[TrainerCallback]:
    callbacks: List[TrainerCallback] = []

    if enable_best_cer_tracker:
        callbacks.append(BestCERTrackerCallback())

    if enable_memory_monitor:
        callbacks.append(MemoryMonitorCallback())

    if enable_early_stopping:
        callbacks.append(
            build_early_stopping_callback(
                patience=early_stopping_patience,
                threshold=early_stopping_threshold,
            )
        )

    return callbacks




__all__ = [
    "BestCERTrackerCallback",
    "MemoryMonitorCallback",
    "build_default_callbacks",
    "build_early_stopping_callback",
]
