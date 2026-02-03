"""Evaluation result types for offline ML.

Shared dataclasses for epoch-based and async evaluation results.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class EvalResult:
    """Result from epoch-based decoder evaluation."""

    accuracy: float
    confusion_matrix: np.ndarray
    per_class_accuracy: dict[int, float]
    avg_inference_time_ms: float
    predictions: np.ndarray | None = None
    probabilities: np.ndarray | None = None

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            "accuracy": self.accuracy,
            "per_class_accuracy": self.per_class_accuracy,
            "avg_inference_time_ms": self.avg_inference_time_ms,
            "confusion_matrix": self.confusion_matrix.tolist(),
        }
