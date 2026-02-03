"""Synchronous (trial-based) metrics for BCI evaluation."""

from collections import Counter
from typing import Any

import numpy as np


class SynchronousMetrics:
    """
    Metrics for synchronous (trial-based) Dendrite classification.

    Computes prequential accuracy, confusion matrix, and Cohen's kappa
    for discrete trial paradigms.
    """

    def __init__(self, num_classes: int = 2):
        """Initialize metrics tracker."""
        self.num_classes = num_classes

        # Core tracking
        self.predictions = []
        self.true_labels = []
        self.prequential_accuracy = []

        # Confusion matrix
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    def add_prediction(
        self,
        prediction: int,
        true_label: int,
        forgetting_factor: float = 0.95,
    ):
        """Add a prediction and update all metrics."""
        self.predictions.append(prediction)
        self.true_labels.append(true_label)

        is_correct = prediction == true_label
        current_accuracy = 1.0 if is_correct else 0.0

        if self.prequential_accuracy:
            new_acc = (
                forgetting_factor * self.prequential_accuracy[-1]
                + (1 - forgetting_factor) * current_accuracy
            )
        else:
            new_acc = current_accuracy
        self.prequential_accuracy.append(new_acc)

        # Update confusion matrix
        if 0 <= true_label < self.num_classes and 0 <= prediction < self.num_classes:
            self.confusion_matrix[true_label, prediction] += 1

        return is_correct, current_accuracy

    def _class_distribution(self) -> dict[int, float]:
        """Calculate class distribution from true labels."""
        if not self.true_labels:
            return {}
        counts = Counter(self.true_labels)
        total = sum(counts.values())
        return {cls: count / total for cls, count in counts.items()}

    def calculate_overall_accuracy(self) -> float:
        """Calculate overall accuracy."""
        if not self.predictions:
            return 0.0
        correct = sum(
            1
            for pred, true in zip(self.predictions, self.true_labels, strict=False)
            if pred == true
        )
        return correct / len(self.predictions)

    def calculate_cohens_kappa(self) -> float:
        """Calculate Cohen's kappa coefficient."""
        if not self.predictions or len(set(self.true_labels)) <= 1:
            return 0.0

        try:
            p_o = self.calculate_overall_accuracy()
            class_dist = self._class_distribution()
            pred_counts = Counter(self.predictions)
            total = len(self.predictions)

            p_e = sum(
                class_dist.get(i, 0) * (pred_counts.get(i, 0) / total)
                for i in range(self.num_classes)
            )

            if p_e == 1.0:
                return 1.0
            return (p_o - p_e) / (1 - p_e)
        except (ZeroDivisionError, ValueError):
            return 0.0

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all synchronous metrics."""
        class_dist = self._class_distribution()
        chance_level = max(class_dist.values()) if class_dist else 1.0 / self.num_classes

        return {
            "prequential_accuracy": self.prequential_accuracy[-1]
            if self.prequential_accuracy
            else 0.0,
            "samples_processed": len(self.predictions),
            "chance_level": chance_level,
            "class_distribution": class_dist,
            "cohens_kappa": self.calculate_cohens_kappa(),
            "confusion_matrix": self.confusion_matrix.tolist(),
        }

    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.true_labels = []
        self.prequential_accuracy = []
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
