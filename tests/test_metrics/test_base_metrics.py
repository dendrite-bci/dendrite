"""Tests for shared metrics utility functions."""

from dendrite.ml.decision_gate import DecisionGate
from dendrite.ml.decision_gate import _dwell_decide as dwell_decide
from dendrite.ml.decision_gate import _majority_vote as majority_vote
from dendrite.ml.metrics_utils import class_distribution, compute_trial_metrics


class TestClassDistribution:
    def test_balanced_two_class(self):
        result = class_distribution([0, 1, 0, 1])
        assert result == {0: 0.5, 1: 0.5}

    def test_imbalanced(self):
        result = class_distribution([0, 0, 0, 1])
        assert result == {0: 0.75, 1: 0.25}

    def test_single_class(self):
        result = class_distribution([0, 0, 0])
        assert result == {0: 1.0}

    def test_empty(self):
        result = class_distribution([])
        assert result == {}

    def test_three_class(self):
        result = class_distribution([0, 1, 2, 0, 1, 2])
        for v in result.values():
            assert abs(v - 1 / 3) < 1e-9


class TestMajorityVote:
    def test_clear_winner(self):
        assert majority_vote([0, 0, 1]) == 0

    def test_tie_returns_none(self):
        assert majority_vote([0, 1]) is None

    def test_empty_returns_none(self):
        assert majority_vote([]) is None

    def test_single(self):
        assert majority_vote([2]) == 2


class TestDwellDecide:
    def test_first_class_at_start(self):
        cls, step = dwell_decide([1, 1, 1, 0, 0], dwell_n=3)
        assert cls == 1
        assert step == 2

    def test_first_class_mid_sequence(self):
        cls, step = dwell_decide([0, 1, 1, 1, 0], dwell_n=3)
        assert cls == 1
        assert step == 3

    def test_no_class_reaches_dwell(self):
        cls, step = dwell_decide([0, 1, 1, 0, 1], dwell_n=3)
        assert cls is None
        assert step is None

    def test_dwell_1_picks_first_prediction(self):
        cls, step = dwell_decide([0, 1, 0], dwell_n=1)
        assert cls == 0
        assert step == 0

    def test_wrong_class_reaches_dwell_first(self):
        # True label would be 1, but class 0 reaches dwell first
        cls, step = dwell_decide([0, 0, 0, 1, 1, 1], dwell_n=3)
        assert cls == 0
        assert step == 2

    def test_abstained_resets_streak(self):
        cls, step = dwell_decide([1, -1, 1, 1, 1], dwell_n=3)
        assert cls == 1
        assert step == 4

    def test_empty(self):
        cls, step = dwell_decide([], dwell_n=3)
        assert cls is None
        assert step is None


class TestComputeTrialMetrics:
    def test_all_abstained_counts_as_wrong(self):
        """When all predictions are abstained, trial must NOT use ground truth."""
        gate = DecisionGate(confidence_threshold=0.9)
        trials = [
            {"predictions": [-1, -1, -1], "confidences": [0.3, 0.2, 0.4],
             "label": 0, "n_correct": 0},
            {"predictions": [-1, -1, -1], "confidences": [0.3, 0.2, 0.4],
             "label": 1, "n_correct": 0},
        ]
        result = compute_trial_metrics(trials, [], gate, num_classes=2)
        assert result["balanced_accuracy"] == 0.0
        assert result["accuracy"] == 0.0

    def test_majority_fallback_when_dwell_fails(self):
        """Dwell can't fire → fallback to majority vote (not ground truth)."""
        gate = DecisionGate(strategy="dwell", dwell_n=5)
        trials = [
            {"predictions": [0, 0, 0, 1, 1], "label": 0, "n_correct": 3},
        ]
        result = compute_trial_metrics(trials, [], gate, num_classes=2)
        # Majority is 0, label is 0 → correct
        assert result["accuracy"] == 1.0

    def test_majority_fallback_wrong(self):
        """Dwell can't fire, majority disagrees with label → wrong."""
        gate = DecisionGate(strategy="dwell", dwell_n=5)
        trials = [
            {"predictions": [1, 1, 1, 0, 0], "label": 0, "n_correct": 2},
        ]
        result = compute_trial_metrics(trials, [], gate, num_classes=2)
        # Majority is 1, label is 0 → wrong
        assert result["accuracy"] == 0.0
