"""Epoch-based evaluation for offline ML.

Standard train/test evaluation on labeled epochs.
"""

import time
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from .eval_types import EvalResult


class EpochEvaluator:
    """Evaluate decoders on labeled epoch data."""

    def evaluate(
        self,
        decoder: Any,
        X: np.ndarray,
        y: np.ndarray,
        modality: str = "eeg",
    ) -> EvalResult:
        """Evaluate decoder on test data.

        Args:
            decoder: Trained decoder with predict() method
            X: Test data, shape (n_samples, n_channels, n_times)
            y: True labels, shape (n_samples,)
            modality: Data modality key for decoder (eeg, emg)

        Returns:
            EvalResult with accuracy, confusion matrix, and timing
        """
        n_samples = X.shape[0]
        inference_times = []
        predictions = []

        # Run predictions sample by sample for timing
        for i in range(n_samples):
            sample = X[i : i + 1]  # Keep batch dimension

            # Decoder expects array directly
            start = time.perf_counter()
            pred = decoder.predict(sample)
            elapsed = time.perf_counter() - start

            inference_times.append(elapsed * 1000)  # Convert to ms
            predictions.append(pred[0] if hasattr(pred, "__len__") else pred)

        predictions = np.array(predictions)

        # Compute metrics
        acc = accuracy_score(y, predictions)
        cm = confusion_matrix(y, predictions)

        # Per-class accuracy
        per_class = {}
        classes = np.unique(y)
        for cls in classes:
            mask = y == cls
            if mask.sum() > 0:
                per_class[int(cls)] = float(accuracy_score(y[mask], predictions[mask]))

        avg_time = float(np.mean(inference_times))

        return EvalResult(
            accuracy=acc,
            confusion_matrix=cm,
            per_class_accuracy=per_class,
            avg_inference_time_ms=avg_time,
            predictions=predictions,
        )
