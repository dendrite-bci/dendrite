"""Configuration dataclasses for offline training."""

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
