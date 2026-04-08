"""Online BCI metrics for processing modes.

SynchronousMetrics: prequential accuracy, confusion matrix, kappa (trial-based paradigms).
AsynchronousMetrics: trial detection via DecisionGate, FAR, TTD (continuous paradigms).

Async metrics delegate aggregation to ``compute_trial_metrics()`` in
``dendrite.ml.metrics_utils`` (shared with offline evaluate_sliding_window).
"""

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dendrite.ml.decision_gate import DecisionGate
from dendrite.ml.metrics_utils import (
    calculate_itr,
    class_distribution,
    compute_trial_metrics,
)

# ---------------------------------------------------------------------------
# Synchronous (trial-based) metrics
# ---------------------------------------------------------------------------


class SynchronousMetrics:
    """Prequential accuracy, confusion matrix, and Cohen's kappa for trial paradigms."""

    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.predictions: list[int] = []
        self.true_labels: list[int] = []
        self.prequential_accuracy: list[float] = []
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    def add_prediction(
        self, prediction: int, true_label: int, forgetting_factor: float = 0.95,
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

        if 0 <= true_label < self.num_classes and 0 <= prediction < self.num_classes:
            self.confusion_matrix[true_label, prediction] += 1

        return is_correct, current_accuracy

    def calculate_overall_accuracy(self) -> float:
        if not self.predictions:
            return 0.0
        correct = sum(1 for p, t in zip(self.predictions, self.true_labels) if p == t)
        return correct / len(self.predictions)

    def calculate_cohens_kappa(self) -> float:
        if not self.predictions or len(set(self.true_labels)) <= 1:
            return 0.0
        try:
            p_o = self.calculate_overall_accuracy()
            dist = class_distribution(self.true_labels)
            pred_counts = Counter(self.predictions)
            total = len(self.predictions)
            p_e = sum(
                dist.get(i, 0) * (pred_counts.get(i, 0) / total)
                for i in range(self.num_classes)
            )
            if p_e == 1.0:
                return 1.0
            return (p_o - p_e) / (1 - p_e)
        except (ZeroDivisionError, ValueError):
            return 0.0

    def get_all_metrics(self) -> dict[str, Any]:
        dist = class_distribution(self.true_labels)
        chance_level = max(dist.values()) if dist else 1.0 / self.num_classes
        return {
            "prequential_accuracy": (
                self.prequential_accuracy[-1] if self.prequential_accuracy else 0.0
            ),
            "samples_processed": len(self.predictions),
            "chance_level": chance_level,
            "class_distribution": dist,
            "cohens_kappa": self.calculate_cohens_kappa(),
            "confusion_matrix": self.confusion_matrix.tolist(),
        }

    def reset(self):
        self.predictions = []
        self.true_labels = []
        self.prequential_accuracy = []
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)


# ---------------------------------------------------------------------------
# Asynchronous (continuous) metrics
# ---------------------------------------------------------------------------


@dataclass
class Trial:
    """Track state for a single trial."""

    onset_sample: int
    label: int
    predictions: list[int] = field(default_factory=list)
    n_correct: int = 0
    detected: bool = False
    _streak: int = 0


class AsynchronousMetrics:
    """Online trial-level metrics for continuous BCI paradigms.

    Collects predictions into trial windows, tracks background predictions,
    then delegates metric computation to ``compute_trial_metrics()``.
    """

    def __init__(
        self,
        detection_window_samples: int,
        sample_rate: int = 250,
        step_size_ms: float = 100.0,
        label_mapping: dict[int, str] | None = None,
        gate: DecisionGate | None = None,
    ):
        self.gate = gate or DecisionGate()
        self.num_classes = len(label_mapping) if label_mapping else 2
        self.detection_window_samples = detection_window_samples
        self.sample_rate = sample_rate
        self.step_size_ms = step_size_ms
        self.label_mapping = label_mapping or {}

        self.trials: list[Trial] = []
        self._background_preds: deque[int] = deque(maxlen=6000)
        self.last_sample_idx: int = 0

    def register_event(self, sample_idx: int, label: int) -> None:
        self.trials.append(Trial(onset_sample=sample_idx, label=label))

    def add_prediction(
        self, prediction: int, current_sample_idx: int,
        confidence: float = 1.0,
    ) -> tuple[bool, bool]:
        """Add a prediction. Returns (in_trial, just_detected).

        ``just_detected`` is True on the exact step the dwell streak completes
        (fires once per trial).  Predictions below the gate's confidence
        threshold become ``-1`` (abstain) and break dwell streaks.
        """
        prediction = self.gate.filter_prediction(prediction, confidence)
        self.last_sample_idx = current_sample_idx
        trial = self._get_active_trial(current_sample_idx)

        if trial is not None:
            trial.predictions.append(prediction)
            if prediction == trial.label:
                trial.n_correct += 1
                trial._streak += 1
                if not trial.detected and trial._streak >= self.gate.dwell_n:
                    trial.detected = True
                    return True, True
            else:
                trial._streak = 0
            return True, False
        else:
            self._background_preds.append(prediction)
            return False, False

    def _get_active_trial(self, current_sample_idx: int) -> Trial | None:
        for trial in reversed(self.trials):
            # Causal offset: only count predictions after a full window of
            # post-event data.  Earlier predictions use pre-event data and
            # form wrong-class dwell streaks.
            window_start = trial.onset_sample + self.detection_window_samples
            window_end = window_start + self.detection_window_samples
            if window_start <= current_sample_idx < window_end:
                return trial
            if current_sample_idx >= window_end:
                break
        return None

    def get_all_metrics(self) -> dict[str, Any]:
        trial_dicts = [
            {"predictions": t.predictions, "label": t.label, "n_correct": t.n_correct}
            for t in self.trials
        ]
        return compute_trial_metrics(
            trial_dicts, list(self._background_preds), self.gate,
            num_classes=self.num_classes,
            step_duration_ms=self.step_size_ms,
            label_mapping=self.label_mapping,
        )

    def get_itr(self, mean_selection_time_sec: float) -> float:
        metrics = self.get_all_metrics()
        return calculate_itr(
            self.num_classes, metrics.get("balanced_accuracy", 0.0),
            mean_selection_time_sec,
        )

    def reset(self):
        self.trials = []
        self._background_preds = deque(maxlen=6000)
        self.last_sample_idx = 0
