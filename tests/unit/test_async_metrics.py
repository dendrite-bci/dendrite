"""Unit tests for trial-level async metrics."""

import pytest
import numpy as np
from dendrite.ml.metrics.asynchronous_metrics import AsynchronousMetrics, Trial
from dendrite.ml.metrics.metrics_manager import MetricsManager


class TestAsynchronousMetrics:
    """Tests for AsynchronousMetrics trial-level tracking."""

    def test_basic_initialization(self):
        """Test basic initialization with detection window."""
        metrics = AsynchronousMetrics(
            detection_window_samples=250,
            sample_rate=250,
        )
        assert metrics.detection_window_samples == 250
        assert metrics.sample_rate == 250
        assert len(metrics.trials) == 0

    def test_register_event(self):
        """Test event registration creates trials."""
        metrics = AsynchronousMetrics(detection_window_samples=250, sample_rate=250)

        metrics.register_event(sample_idx=0, label=1)
        metrics.register_event(sample_idx=500, label=0)

        assert len(metrics.trials) == 2
        assert metrics.trials[0].onset_sample == 0
        assert metrics.trials[0].label == 1
        assert metrics.trials[1].onset_sample == 500
        assert metrics.trials[1].label == 0

    def test_detection_within_window(self):
        """Test that prediction within detection window is recorded."""
        metrics = AsynchronousMetrics(detection_window_samples=250, sample_rate=250)

        # Register error event at sample 0
        metrics.register_event(sample_idx=0, label=1)

        # Make prediction within window (at sample 100)
        result = metrics.add_prediction(prediction=1, current_sample_idx=100)

        assert result is True
        assert metrics.trials[0].predictions == [1]
        assert metrics.trials[0].first_correct_ms == pytest.approx(400.0)  # 100 samples / 250 Hz * 1000

    def test_detection_outside_window(self):
        """Test that prediction outside detection window is background FP."""
        metrics = AsynchronousMetrics(detection_window_samples=250, sample_rate=250)

        # Register error event at sample 0
        metrics.register_event(sample_idx=0, label=1)

        # Make prediction outside window (at sample 500)
        result = metrics.add_prediction(prediction=1, current_sample_idx=500)

        assert result is False  # False positive
        assert metrics.trials[0].predictions == []  # No predictions in trial window
        assert len(metrics.background_fp_samples) == 1

    def test_class_1_accuracy_calculation(self):
        """Test class 1 accuracy calculation - majority vote matches error label."""
        metrics = AsynchronousMetrics(detection_window_samples=250, sample_rate=250)

        # Two error trials
        metrics.register_event(sample_idx=0, label=1)
        metrics.register_event(sample_idx=500, label=1)

        # First trial: majority=1 (correct), second trial: no predictions (majority=None)
        metrics.add_prediction(prediction=1, current_sample_idx=100)

        result = metrics.get_metrics()
        assert result['per_class_accuracy'][1] == pytest.approx(0.5)  # 1/2 errors correctly classified

    def test_class_0_accuracy_calculation(self):
        """Test class 0 accuracy calculation - majority vote matches correct label."""
        metrics = AsynchronousMetrics(detection_window_samples=250, sample_rate=250)

        # Two correct trials
        metrics.register_event(sample_idx=0, label=0)
        metrics.register_event(sample_idx=500, label=0)

        # First trial: majority=0 (correct), second trial: no predictions (majority=None)
        metrics.add_prediction(prediction=0, current_sample_idx=100)

        result = metrics.get_metrics()
        assert result['per_class_accuracy'][0] == pytest.approx(0.5)  # 1/2 corrects correctly classified

    def test_perfect_classification(self):
        """Test 100% per-class accuracy scenario - all trials correctly classified."""
        metrics = AsynchronousMetrics(detection_window_samples=250, sample_rate=250)

        # One error, one correct
        metrics.register_event(sample_idx=0, label=1)
        metrics.register_event(sample_idx=500, label=0)

        # Correctly classify both: prediction matches label for each
        metrics.add_prediction(prediction=1, current_sample_idx=100)
        metrics.add_prediction(prediction=0, current_sample_idx=600)

        result = metrics.get_metrics()
        assert result['per_class_accuracy'][1] == pytest.approx(1.0)
        assert result['per_class_accuracy'][0] == pytest.approx(1.0)

    def test_first_correct_time(self):
        """Test that first correct prediction time is recorded correctly."""
        metrics = AsynchronousMetrics(detection_window_samples=500, sample_rate=500)

        metrics.register_event(sample_idx=0, label=1)

        # Multiple predictions - should only record first correct
        metrics.add_prediction(prediction=1, current_sample_idx=100)
        metrics.add_prediction(prediction=1, current_sample_idx=200)

        assert metrics.trials[0].predictions == [1, 1]
        assert metrics.trials[0].first_correct_ms == pytest.approx(200.0)  # 100 samples / 500 Hz * 1000

    def test_majority_vote_decides_classification(self):
        """Test that majority class determines trial classification."""
        metrics = AsynchronousMetrics(detection_window_samples=250, sample_rate=250)

        # Error trial (label=1)
        metrics.register_event(sample_idx=0, label=1)
        # Correct trial (label=0)
        metrics.register_event(sample_idx=500, label=0)

        # First trial: 3 error predictions, 2 correct -> majority=1 (correct for error)
        metrics.add_prediction(prediction=1, current_sample_idx=50)
        metrics.add_prediction(prediction=0, current_sample_idx=100)
        metrics.add_prediction(prediction=1, current_sample_idx=150)
        metrics.add_prediction(prediction=0, current_sample_idx=175)
        metrics.add_prediction(prediction=1, current_sample_idx=200)

        # Second trial: 2 error predictions, 3 correct -> majority=0 (correct for correct)
        metrics.add_prediction(prediction=1, current_sample_idx=550)
        metrics.add_prediction(prediction=0, current_sample_idx=600)
        metrics.add_prediction(prediction=1, current_sample_idx=650)
        metrics.add_prediction(prediction=0, current_sample_idx=700)
        metrics.add_prediction(prediction=0, current_sample_idx=720)

        result = metrics.get_metrics()
        # Both trials correctly classified by majority vote
        assert result['per_class_accuracy'][1] == pytest.approx(1.0)
        assert result['per_class_accuracy'][0] == pytest.approx(1.0)

    def test_majority_vote_wrong_classification(self):
        """Test that wrong majority class results in incorrect classification."""
        metrics = AsynchronousMetrics(detection_window_samples=250, sample_rate=250)

        # Error trial (label=1)
        metrics.register_event(sample_idx=0, label=1)

        # 3 correct predictions, 2 error -> majority=0 (WRONG for error trial)
        metrics.add_prediction(prediction=0, current_sample_idx=50)
        metrics.add_prediction(prediction=1, current_sample_idx=100)
        metrics.add_prediction(prediction=0, current_sample_idx=150)
        metrics.add_prediction(prediction=1, current_sample_idx=175)
        metrics.add_prediction(prediction=0, current_sample_idx=200)

        result = metrics.get_metrics()
        # Error trial was incorrectly classified (majority=0, should be 1)
        assert result['per_class_accuracy'][1] == pytest.approx(0.0)

    def test_far_calculation(self):
        """Test false alarm rate calculation in background."""
        metrics = AsynchronousMetrics(detection_window_samples=250, sample_rate=250)

        # One event, large gap
        metrics.register_event(sample_idx=0, label=1)

        # Multiple FPs in background (after event window)
        metrics.add_prediction(prediction=1, current_sample_idx=500)
        metrics.add_prediction(prediction=1, current_sample_idx=1000)
        metrics.last_sample_idx = 15000  # 1 minute of data at 250 Hz

        result = metrics.get_metrics()
        assert result['far_per_min'] > 0

    def test_majority_vote_tie_returns_none(self):
        """Test that tied predictions return None (unclassified)."""
        metrics = AsynchronousMetrics(detection_window_samples=250, sample_rate=250)

        # Equal 0s and 1s -> tie
        assert metrics._get_majority_class([0, 1, 0, 1]) is None
        assert metrics._get_majority_class([1, 0, 1, 0, 1, 0]) is None

        # Clear majorities still work
        assert metrics._get_majority_class([0, 0, 1]) == 0
        assert metrics._get_majority_class([1, 1, 0]) == 1

    def test_reset(self):
        """Test reset clears all state."""
        metrics = AsynchronousMetrics(detection_window_samples=250, sample_rate=250)

        metrics.register_event(sample_idx=0, label=1)
        metrics.add_prediction(prediction=1, current_sample_idx=100)

        metrics.reset()

        assert len(metrics.trials) == 0
        assert len(metrics.background_fp_samples) == 0
        assert metrics.last_sample_idx == 0

    def test_derived_accuracy_metrics(self):
        """Test that balanced_accuracy and accuracy are derived from TPR/TNR."""
        metrics = AsynchronousMetrics(detection_window_samples=250, sample_rate=250)

        # Perfect classification scenario
        metrics.register_event(sample_idx=0, label=1)
        metrics.register_event(sample_idx=500, label=0)
        # Correctly classify both trials
        metrics.add_prediction(prediction=1, current_sample_idx=100)
        metrics.add_prediction(prediction=0, current_sample_idx=600)

        result = metrics.get_metrics()
        # TPR = 1.0, TNR = 1.0, balanced_accuracy = (1.0 + 1.0) / 2 = 1.0
        assert result['balanced_accuracy'] == pytest.approx(1.0)

    def test_derived_accuracy_partial(self):
        """Test derived accuracy with partial performance."""
        metrics = AsynchronousMetrics(detection_window_samples=250, sample_rate=250)

        # Two errors, one correct
        metrics.register_event(sample_idx=0, label=1)
        metrics.register_event(sample_idx=500, label=1)
        metrics.register_event(sample_idx=1000, label=0)

        # Correctly classify first error and the correct trial, miss second error
        metrics.add_prediction(prediction=1, current_sample_idx=100)
        metrics.add_prediction(prediction=0, current_sample_idx=1100)

        result = metrics.get_metrics()
        # class 1 accuracy = 0.5 (1/2 errors), class 0 accuracy = 1.0 (1/1 correct), balanced_accuracy = 0.75
        assert result['per_class_accuracy'][1] == pytest.approx(0.5)
        assert result['per_class_accuracy'][0] == pytest.approx(1.0)
        assert result['balanced_accuracy'] == pytest.approx(0.75)

    def test_calculate_itr(self):
        """Test ITR calculation using Wolpaw formula."""
        metrics = AsynchronousMetrics(detection_window_samples=250, sample_rate=250)

        # Set up for 75% balanced accuracy (1/2 errors + 1/1 corrects correctly classified)
        metrics.register_event(sample_idx=0, label=1)
        metrics.register_event(sample_idx=500, label=1)
        metrics.register_event(sample_idx=1000, label=0)
        metrics.add_prediction(prediction=1, current_sample_idx=100)
        metrics.add_prediction(prediction=0, current_sample_idx=1100)

        # 100ms selection time = 0.1s
        itr = metrics.calculate_itr(num_classes=2, mean_selection_time_sec=0.1)

        # ITR should be positive for >50% accuracy
        assert itr > 0
        # 75% accuracy with 2 classes and 0.1s selection time should give ~100-200 bits/min
        assert itr > 50

    def test_calculate_itr_edge_cases(self):
        """Test ITR returns 0.0 for invalid inputs."""
        metrics = AsynchronousMetrics(detection_window_samples=250, sample_rate=250)

        # No trials - 0 accuracy
        assert metrics.calculate_itr(num_classes=2, mean_selection_time_sec=0.1) == 0.0

        # Invalid inputs
        metrics.register_event(sample_idx=0, label=1)
        metrics.add_prediction(prediction=1, current_sample_idx=100)

        assert metrics.calculate_itr(num_classes=1, mean_selection_time_sec=0.1) == 0.0  # < 2 classes
        assert metrics.calculate_itr(num_classes=2, mean_selection_time_sec=0.0) == 0.0  # 0 time
        assert metrics.calculate_itr(num_classes=2, mean_selection_time_sec=-1) == 0.0  # negative time


class TestMetricsManagerAsync:
    """Tests for MetricsManager in async mode."""

    def test_async_mode_initialization(self):
        """Test MetricsManager initializes async metrics correctly."""
        manager = MetricsManager(
            mode_type='asynchronous',
            sample_rate=250,
            num_classes=2,
            detection_window_samples=250
        )

        assert manager.mode_type == 'asynchronous'
        assert manager.async_metrics is not None
        assert manager.sync_metrics is None

    def test_register_event_passthrough(self):
        """Test that register_event passes through to async metrics."""
        manager = MetricsManager(
            mode_type='asynchronous',
            sample_rate=250,
            detection_window_samples=250
        )

        manager.register_event(sample_idx=0, label=1)

        assert len(manager.async_metrics.trials) == 1

    def test_update_metrics_async(self):
        """Test update_metrics works with new async interface."""
        manager = MetricsManager(
            mode_type='asynchronous',
            sample_rate=250,
            detection_window_samples=250
        )

        manager.register_event(sample_idx=0, label=1)
        manager.update_metrics(
            prediction=1,
            true_label=-1,
            current_sample_idx=100
        )

        metrics = manager.get_current_metrics()
        assert metrics['per_class_accuracy'][1] == pytest.approx(1.0)

    def test_sync_mode_unchanged(self):
        """Test that sync mode still works with existing interface."""
        manager = MetricsManager(
            mode_type='synchronous',
            sample_rate=250,
            num_classes=2
        )

        assert manager.mode_type == 'synchronous'
        assert manager.sync_metrics is not None
        assert manager.async_metrics is None


class TestTrialDataclass:
    """Tests for Trial dataclass."""

    def test_trial_defaults(self):
        """Test Trial has correct defaults."""
        trial = Trial(onset_sample=100, label=1)

        assert trial.onset_sample == 100
        assert trial.label == 1
        assert trial.predictions == []
        assert trial.first_correct_ms is None

    def test_trial_modification(self):
        """Test Trial fields can be modified."""
        trial = Trial(onset_sample=0, label=1)

        trial.predictions.append(1)
        trial.first_correct_ms = 150.0

        assert trial.predictions == [1]
        assert trial.first_correct_ms == 150.0


class TestMulticlassMetrics:
    """Tests for N-way classification support."""

    def test_multiclass_per_class_accuracy(self):
        """Test per-class accuracy with 3+ classes."""
        metrics = AsynchronousMetrics(
            detection_window_samples=250,
            sample_rate=250,
            label_mapping={0: 'rest', 1: 'left', 2: 'right'}
        )

        # Register trials for each class
        metrics.register_event(sample_idx=0, label=0)     # rest
        metrics.register_event(sample_idx=500, label=1)   # left
        metrics.register_event(sample_idx=1000, label=2)  # right

        # Correctly classify all trials
        metrics.add_prediction(prediction=0, current_sample_idx=100)
        metrics.add_prediction(prediction=1, current_sample_idx=600)
        metrics.add_prediction(prediction=2, current_sample_idx=1100)

        result = metrics.get_metrics()

        # Check per-class accuracy
        assert result['per_class_accuracy'] == {0: 1.0, 1: 1.0, 2: 1.0}
        assert result['per_class_accuracy_named'] == {'rest': 1.0, 'left': 1.0, 'right': 1.0}
        assert result['balanced_accuracy'] == pytest.approx(1.0)
        assert result['per_class_trials'] == {0: 1, 1: 1, 2: 1}

    def test_multiclass_partial_accuracy(self):
        """Test multiclass with partial accuracy."""
        metrics = AsynchronousMetrics(
            detection_window_samples=250,
            sample_rate=250,
            label_mapping={0: 'rest', 1: 'left', 2: 'right'}
        )

        # 2 trials per class
        metrics.register_event(sample_idx=0, label=0)
        metrics.register_event(sample_idx=500, label=0)
        metrics.register_event(sample_idx=1000, label=1)
        metrics.register_event(sample_idx=1500, label=1)
        metrics.register_event(sample_idx=2000, label=2)
        metrics.register_event(sample_idx=2500, label=2)

        # Correctly classify 1 of 2 for each class
        metrics.add_prediction(prediction=0, current_sample_idx=100)
        metrics.add_prediction(prediction=1, current_sample_idx=600)  # wrong for class 0
        metrics.add_prediction(prediction=1, current_sample_idx=1100)
        metrics.add_prediction(prediction=2, current_sample_idx=1600)  # wrong for class 1
        metrics.add_prediction(prediction=2, current_sample_idx=2100)
        metrics.add_prediction(prediction=0, current_sample_idx=2600)  # wrong for class 2

        result = metrics.get_metrics()

        # Each class has 50% accuracy
        assert result['per_class_accuracy'][0] == pytest.approx(0.5)
        assert result['per_class_accuracy'][1] == pytest.approx(0.5)
        assert result['per_class_accuracy'][2] == pytest.approx(0.5)
        assert result['balanced_accuracy'] == pytest.approx(0.5)

    def test_label_mapping_in_output(self):
        """Test that named classes appear in output."""
        metrics = AsynchronousMetrics(
            detection_window_samples=250,
            sample_rate=250,
            label_mapping={0: 'correct', 1: 'error'}
        )

        metrics.register_event(sample_idx=0, label=0)
        metrics.register_event(sample_idx=500, label=1)
        metrics.add_prediction(prediction=0, current_sample_idx=100)
        metrics.add_prediction(prediction=1, current_sample_idx=600)

        result = metrics.get_metrics()

        assert 'correct' in result['per_class_accuracy_named']
        assert 'error' in result['per_class_accuracy_named']
        assert result['per_class_accuracy_named']['correct'] == 1.0
        assert result['per_class_accuracy_named']['error'] == 1.0

    def test_far_any_prediction_outside_trial(self):
        """Test that any prediction outside trial window is a false alarm when background_class=None."""
        metrics = AsynchronousMetrics(
            detection_window_samples=250,
            sample_rate=250,
            background_class=None  # Explicit: no background class
        )

        # Register trial for class 0
        metrics.register_event(sample_idx=0, label=0)

        # Any prediction outside trial window is a false alarm
        result1 = metrics.add_prediction(prediction=1, current_sample_idx=500)
        assert result1 is False
        assert len(metrics.background_fp_samples) == 1

        result2 = metrics.add_prediction(prediction=2, current_sample_idx=600)
        assert result2 is False
        assert len(metrics.background_fp_samples) == 2

        # Even class 0 prediction outside trial is a false alarm
        result3 = metrics.add_prediction(prediction=0, current_sample_idx=700)
        assert result3 is False
        assert len(metrics.background_fp_samples) == 3

    def test_ttd_for_all_classes(self):
        """Test TTD is computed for all correctly classified trials."""
        metrics = AsynchronousMetrics(
            detection_window_samples=250,
            sample_rate=250,
        )

        # Trial for each class - all correctly classified
        metrics.register_event(sample_idx=0, label=0)
        metrics.add_prediction(prediction=0, current_sample_idx=50)  # 50 samples = 200ms

        metrics.register_event(sample_idx=500, label=1)
        metrics.add_prediction(prediction=1, current_sample_idx=600)  # 100 samples = 400ms

        metrics.register_event(sample_idx=1000, label=2)
        metrics.add_prediction(prediction=2, current_sample_idx=1150)  # 150 samples = 600ms

        result = metrics.get_metrics()

        # TTD should be mean of 200ms, 400ms and 600ms = 400ms
        assert result['mean_ttd_ms'] == pytest.approx(400.0)

    def test_binary_classification_uses_per_class_accuracy(self):
        """Test that binary classification uses per_class_accuracy."""
        metrics = AsynchronousMetrics(
            detection_window_samples=250,
            sample_rate=250,
        )

        # Same setup as older tests
        metrics.register_event(sample_idx=0, label=1)  # error
        metrics.register_event(sample_idx=500, label=0)  # correct
        metrics.add_prediction(prediction=1, current_sample_idx=100)
        metrics.add_prediction(prediction=0, current_sample_idx=600)

        result = metrics.get_metrics()

        # Per-class accuracy keys
        assert result['per_class_accuracy'][0] == 1.0
        assert result['per_class_accuracy'][1] == 1.0
        assert result['per_class_trials'] == {0: 1, 1: 1}
        assert result['balanced_accuracy'] == pytest.approx(1.0)

        # Verify backward compat keys are removed
        assert 'tpr' not in result
        assert 'tnr' not in result
        assert 'n_errors' not in result
        assert 'n_corrects' not in result

    def test_empty_label_mapping_uses_indices(self):
        """Test that missing label_mapping uses class indices as names."""
        metrics = AsynchronousMetrics(
            detection_window_samples=250,
            sample_rate=250,
        )

        metrics.register_event(sample_idx=0, label=0)
        metrics.register_event(sample_idx=500, label=1)
        metrics.register_event(sample_idx=1000, label=2)
        metrics.add_prediction(prediction=0, current_sample_idx=100)
        metrics.add_prediction(prediction=1, current_sample_idx=600)
        metrics.add_prediction(prediction=2, current_sample_idx=1100)

        result = metrics.get_metrics()

        # Named dict should use string indices
        assert result['per_class_accuracy_named'] == {'0': 1.0, '1': 1.0, '2': 1.0}

    def test_background_class_excluded_from_far(self):
        """Test that background_class predictions between trials don't count as FAR."""
        metrics = AsynchronousMetrics(
            detection_window_samples=250,
            sample_rate=250,
            background_class=0  # Class 0 is background/idle
        )
        metrics.register_event(sample_idx=0, label=1)

        # Background prediction outside trial - NOT a false alarm
        result = metrics.add_prediction(prediction=0, current_sample_idx=500)
        assert result is None  # Correct non-detection
        assert len(metrics.background_fp_samples) == 0

        # Active prediction outside trial - IS a false alarm
        result = metrics.add_prediction(prediction=1, current_sample_idx=600)
        assert result is False
        assert len(metrics.background_fp_samples) == 1

        # Another active class outside trial - also a false alarm
        result = metrics.add_prediction(prediction=2, current_sample_idx=700)
        assert result is False
        assert len(metrics.background_fp_samples) == 2

    def test_no_background_class_all_predictions_far(self):
        """Test that without background_class, all predictions outside trial are FAR."""
        metrics = AsynchronousMetrics(
            detection_window_samples=250,
            sample_rate=250,
            background_class=None  # No background class (default)
        )
        metrics.register_event(sample_idx=0, label=1)

        # Any prediction outside trial = false alarm
        metrics.add_prediction(prediction=0, current_sample_idx=500)
        metrics.add_prediction(prediction=1, current_sample_idx=600)
        metrics.add_prediction(prediction=2, current_sample_idx=700)
        assert len(metrics.background_fp_samples) == 3


class TestMetricsManagerMulticlass:
    """Tests for MetricsManager with multiclass support."""

    def test_async_mode_with_label_mapping(self):
        """Test MetricsManager passes label_mapping to async metrics."""
        manager = MetricsManager(
            mode_type='asynchronous',
            sample_rate=250,
            num_classes=3,
            detection_window_samples=250,
            label_mapping={0: 'rest', 1: 'left', 2: 'right'}
        )

        assert manager.async_metrics.label_mapping == {0: 'rest', 1: 'left', 2: 'right'}

    def test_sync_mode_ignores_async_params(self):
        """Test that sync mode doesn't break with async-only params."""
        # Should not raise
        manager = MetricsManager(
            mode_type='synchronous',
            sample_rate=250,
            num_classes=3,
            label_mapping={0: 'a', 1: 'b', 2: 'c'}
        )

        assert manager.sync_metrics is not None
        assert manager.async_metrics is None

    def test_async_mode_with_background_class(self):
        """Test MetricsManager passes background_class to async metrics."""
        manager = MetricsManager(
            mode_type='asynchronous',
            sample_rate=250,
            num_classes=3,
            detection_window_samples=250,
            background_class=0
        )

        assert manager.async_metrics.background_class == 0

        # Verify behavior: background prediction outside trial is not a false alarm
        manager.register_event(sample_idx=0, label=1)
        manager.update_metrics(
            prediction=0,  # background class
            true_label=-1,
            current_sample_idx=500
        )
        assert len(manager.async_metrics.background_fp_samples) == 0

        # Non-background prediction outside trial is a false alarm
        manager.update_metrics(
            prediction=1,
            true_label=-1,
            current_sample_idx=600
        )
        assert len(manager.async_metrics.background_fp_samples) == 1
