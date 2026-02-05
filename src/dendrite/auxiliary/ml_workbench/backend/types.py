"""Dataclasses for offline ML training and evaluation."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TrainResult:
    """Result of a training run."""

    decoder: Any  # The trained decoder object
    accuracy: float  # Final training accuracy
    val_accuracy: float  # Validation accuracy
    confusion_matrix: np.ndarray  # Confusion matrix
    train_history: dict[str, list[float]]  # Loss/accuracy curves
    cv_results: dict[str, Any] | None = None  # Cross-validation results


@dataclass
class EvalResult:
    """Result from epoch-based decoder evaluation."""

    accuracy: float
    confusion_matrix: np.ndarray
    per_class_accuracy: dict[int, float]
    avg_inference_time_ms: float

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            "accuracy": self.accuracy,
            "per_class_accuracy": self.per_class_accuracy,
            "avg_inference_time_ms": self.avg_inference_time_ms,
            "confusion_matrix": self.confusion_matrix.tolist(),
        }
