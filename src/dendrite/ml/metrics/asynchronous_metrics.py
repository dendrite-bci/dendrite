"""Asynchronous (continuous) metrics for BCI evaluation.

Trial-level classification accuracy metrics for N-way classification.
Per-class accuracy measures correct classification rate (prediction matches trial label).
FAR counts any prediction made outside trial windows (per BCI literature conventions).
"""

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Trial:
    """Track state for a single trial."""

    onset_sample: int
    label: int  # Class index (0 to num_classes-1)
    predictions: list[int] = None  # All predictions in trial window
    first_correct_ms: float | None = None  # TTD: time to first correct prediction

    def __post_init__(self):
        if self.predictions is None:
            self.predictions = []


class AsynchronousMetrics:
    """
    Trial-level metrics for asynchronous Dendrite classification.

    Tracks whether each event (error/correct) was correctly classified within
    the detection window. A trial is considered "detected" when the model
    predicts the correct label for that trial type.
    """

    def __init__(
        self,
        detection_window_samples: int,
        sample_rate: int = 250,
        label_mapping: dict[int, str] | None = None,
        background_class: int | None = None,
    ):
        """Initialize async metrics tracker.

        Args:
            detection_window_samples: Window after event onset for valid detection
                                      (typically = decoder epoch length)
            sample_rate: Sampling rate in Hz
            label_mapping: Optional mapping from class index to name (e.g., {0: 'left', 1: 'right'})
            background_class: Class index for background/idle state. Predictions of this
                              class outside trial windows are not counted as false alarms.
                              If None, all predictions outside trial windows are false alarms.
        """
        self.detection_window_samples = detection_window_samples
        self.sample_rate = sample_rate
        self.label_mapping = label_mapping or {}
        self.background_class = background_class

        # Trial tracking
        self.trials: list[Trial] = []

        # Background false positive tracking (for FAR)
        self.background_fp_samples: list[int] = []

        # Track last sample for FAR calculation
        self.last_sample_idx: int = 0

    def register_event(self, sample_idx: int, label: int) -> None:
        """Register a new event/trial onset.

        Called when an event marker is received (error or correct action).

        Args:
            sample_idx: Sample index of event onset
            label: Event label (0=correct, 1=error)
        """
        self.trials.append(Trial(onset_sample=sample_idx, label=label))

    def add_prediction(
        self, prediction: int, current_sample_idx: int
    ) -> bool | None:
        """Add a prediction and update trial tracking.

        Args:
            prediction: Predicted class (0 or 1)
            current_sample_idx: Current sample index

        Returns:
            True if valid detection in trial, False if FP, None if background
        """
        self.last_sample_idx = current_sample_idx

        # Find which trial (if any) this prediction belongs to
        trial = self._get_active_trial(current_sample_idx)

        if trial is not None:
            # Collect all predictions in trial window
            trial.predictions.append(prediction)
            # Track time to first correct prediction (for TTD)
            if prediction == trial.label and trial.first_correct_ms is None:
                ttd_samples = current_sample_idx - trial.onset_sample
                trial.first_correct_ms = (ttd_samples / self.sample_rate) * 1000
            return True  # Prediction within trial
        else:
            # Inter-trial interval - only count non-background predictions as false alarms
            if self.background_class is None or prediction != self.background_class:
                self.background_fp_samples.append(current_sample_idx)
                return False
            return None  # Correct non-detection (background prediction during background period)

    def _get_active_trial(self, current_sample_idx: int) -> Trial | None:
        """Get the trial whose detection window contains current sample."""
        for trial in self.trials:
            window_end = trial.onset_sample + self.detection_window_samples
            if trial.onset_sample <= current_sample_idx < window_end:
                return trial
        return None

    def _get_majority_class(self, predictions: list[int]) -> int | None:
        """Get majority class from predictions.

        Returns None if empty or tied (model couldn't decide).
        """
        if not predictions:
            return None
        counts = Counter(predictions)
        ((winner, top_count),) = counts.most_common(1)
        # Check for ties: any other class with same count means undecided
        if sum(1 for cnt in counts.values() if cnt == top_count) > 1:
            return None
        return winner

    def get_metrics(self) -> dict[str, Any]:
        """Get trial-level evaluation metrics.

        Returns:
            per_class_accuracy: Accuracy per class (dict of class_index -> accuracy)
            per_class_accuracy_named: Accuracy per class with names (if label_mapping provided)
            balanced_accuracy: Mean of per-class accuracies
            far_per_min: False alarms per minute in background (any prediction outside trial window)
            mean_ttd_ms: Mean time-to-detection for all correctly classified trials
            n_trials: Total number of trials
            per_class_trials: Trial count per class
        """
        if not self.trials:
            return {
                "per_class_accuracy": {},
                "per_class_accuracy_named": {},
                "balanced_accuracy": 0.0,
                "far_per_min": 0.0,
                "mean_ttd_ms": float("nan"),
                "n_trials": 0,
                "per_class_trials": {},
                "accuracy": 0.0,
            }

        # Group trials by class
        trials_by_class: dict[int, list[Trial]] = {}
        for t in self.trials:
            trials_by_class.setdefault(t.label, []).append(t)

        # Compute per-class accuracy (% of trials where majority vote = correct label)
        per_class_accuracy: dict[int, float] = {}
        for class_idx, class_trials in trials_by_class.items():
            n_correct = sum(
                1 for t in class_trials if self._get_majority_class(t.predictions) == t.label
            )
            per_class_accuracy[class_idx] = n_correct / len(class_trials)

        # Create named version if label_mapping provided
        per_class_accuracy_named: dict[str, float] = {}
        for class_idx, acc in per_class_accuracy.items():
            name = self.label_mapping.get(class_idx, str(class_idx))
            per_class_accuracy_named[name] = acc

        # Balanced accuracy = mean of per-class accuracies
        balanced_accuracy = (
            sum(per_class_accuracy.values()) / len(per_class_accuracy)
            if per_class_accuracy
            else 0.0
        )

        # TTD: mean time to first correct prediction (for all correctly classified trials)
        correctly_classified_trials = [
            t
            for t in self.trials
            if self._get_majority_class(t.predictions) == t.label and t.first_correct_ms is not None
        ]
        ttds = [t.first_correct_ms for t in correctly_classified_trials]
        mean_ttd_ms = float(sum(ttds) / len(ttds)) if ttds else float("nan")

        # FAR: false positives per minute in background
        far_per_min = self._calculate_far()

        # Per-class trial counts
        per_class_trials = {k: len(v) for k, v in trials_by_class.items()}

        return {
            "per_class_accuracy": per_class_accuracy,
            "per_class_accuracy_named": per_class_accuracy_named,
            "balanced_accuracy": float(balanced_accuracy),
            "far_per_min": float(far_per_min),
            "mean_ttd_ms": mean_ttd_ms,
            "n_trials": len(self.trials),
            "per_class_trials": per_class_trials,
        }

    def _calculate_far(self) -> float:
        """Calculate false alarm rate per minute in background."""
        if not self.background_fp_samples:
            return 0.0

        # Deduplicate FPs that are close together (within 1 second)
        dedup_threshold = self.sample_rate  # 1 second
        unique_fps = []
        sorted_fps = sorted(self.background_fp_samples)
        for fp in sorted_fps:
            if not unique_fps or (fp - unique_fps[-1]) > dedup_threshold:
                unique_fps.append(fp)

        # Calculate background duration (total time minus trial windows)
        total_samples = self.last_sample_idx
        trial_samples = len(self.trials) * self.detection_window_samples
        background_samples = max(1, total_samples - trial_samples)
        background_minutes = (background_samples / self.sample_rate) / 60.0

        return len(unique_fps) / background_minutes if background_minutes > 0 else 0.0

    def calculate_itr(self, num_classes: int, mean_selection_time_sec: float) -> float:
        """Calculate ITR in bits/min using Wolpaw formula.

        Args:
            num_classes: Number of classification targets
            mean_selection_time_sec: Average time per selection in seconds

        Returns:
            ITR in bits per minute, or 0.0 if inputs invalid
        """
        metrics = self.get_metrics()
        accuracy = metrics.get("balanced_accuracy", 0.0)

        if mean_selection_time_sec <= 0 or accuracy <= 0 or accuracy >= 1.0 or num_classes < 2:
            return 0.0

        P, N = accuracy, num_classes
        bits_per_selection = np.log2(N) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (N - 1))
        selections_per_min = 60.0 / mean_selection_time_sec

        return max(0.0, float(bits_per_selection * selections_per_min))

    def reset(self):
        """Reset all metrics."""
        self.trials = []
        self.background_fp_samples = []
        self.last_sample_idx = 0
