"""Tests for synchronous/asynchronous metrics and MetricsManager."""

import numpy as np
import pytest

from dendrite.processing.modes._metrics import AsynchronousMetrics, SynchronousMetrics

# ---------------------------------------------------------------------------
# Synchronous Metrics
# ---------------------------------------------------------------------------


class TestSynchronousMetrics:
    def test_add_prediction_correct(self):
        m = SynchronousMetrics(num_classes=2)
        is_correct, acc = m.add_prediction(prediction=1, true_label=1)
        assert is_correct is True
        assert acc == 1.0

    def test_add_prediction_wrong(self):
        m = SynchronousMetrics(num_classes=2)
        is_correct, acc = m.add_prediction(prediction=0, true_label=1)
        assert is_correct is False
        assert acc == 0.0

    def test_prequential_accuracy_smoothing(self):
        m = SynchronousMetrics(num_classes=2)
        # Feed many correct predictions
        for _ in range(50):
            m.add_prediction(prediction=1, true_label=1)
        high_acc = m.prequential_accuracy[-1]

        # Now feed wrong predictions — EMA should decrease
        for _ in range(20):
            m.add_prediction(prediction=0, true_label=1)
        low_acc = m.prequential_accuracy[-1]
        assert low_acc < high_acc

    def test_confusion_matrix_shape(self):
        m = SynchronousMetrics(num_classes=3)
        m.add_prediction(0, 0)
        m.add_prediction(1, 1)
        m.add_prediction(2, 2)
        assert m.confusion_matrix.shape == (3, 3)

    def test_confusion_matrix_values(self):
        m = SynchronousMetrics(num_classes=2)
        m.add_prediction(0, 0)  # correct class 0
        m.add_prediction(1, 1)  # correct class 1
        m.add_prediction(0, 1)  # wrong: predicted 0, true 1
        assert m.confusion_matrix[0, 0] == 1
        assert m.confusion_matrix[1, 1] == 1
        assert m.confusion_matrix[1, 0] == 1  # row=true, col=predicted

    def test_cohens_kappa_perfect(self):
        m = SynchronousMetrics(num_classes=2)
        for _ in range(20):
            m.add_prediction(0, 0)
            m.add_prediction(1, 1)
        kappa = m.calculate_cohens_kappa()
        assert kappa == pytest.approx(1.0, abs=0.01)

    def test_cohens_kappa_random(self):
        m = SynchronousMetrics(num_classes=2)
        rng = np.random.RandomState(42)
        for _ in range(200):
            pred = rng.randint(0, 2)
            true = rng.randint(0, 2)
            m.add_prediction(pred, true)
        kappa = m.calculate_cohens_kappa()
        # Random predictions should give kappa near 0
        assert abs(kappa) < 0.3

    def test_get_all_metrics_keys(self):
        m = SynchronousMetrics(num_classes=2)
        m.add_prediction(0, 0)
        metrics = m.get_all_metrics()
        expected_keys = {
            "prequential_accuracy",
            "samples_processed",
            "chance_level",
            "class_distribution",
            "cohens_kappa",
            "confusion_matrix",
        }
        assert expected_keys == set(metrics.keys())

    def test_reset_clears_state(self):
        m = SynchronousMetrics(num_classes=2)
        for _ in range(10):
            m.add_prediction(0, 0)
        m.reset()
        assert len(m.predictions) == 0
        assert len(m.true_labels) == 0
        assert len(m.prequential_accuracy) == 0
        assert m.confusion_matrix.sum() == 0

    def test_class_distribution(self):
        m = SynchronousMetrics(num_classes=2)
        for _ in range(6):
            m.add_prediction(0, 0)
        for _ in range(4):
            m.add_prediction(1, 1)
        metrics = m.get_all_metrics()
        dist = metrics["class_distribution"]
        assert dist[0] == pytest.approx(0.6)
        assert dist[1] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Asynchronous Metrics
# ---------------------------------------------------------------------------


class TestAsynchronousMetrics:
    def _make_async(self, **kwargs):
        from dendrite.ml.decision_gate import DecisionGate

        gate = kwargs.pop("gate", None)
        dwell_n = kwargs.pop("dwell_n", 3)
        if gate is None:
            gate = DecisionGate(dwell_n=dwell_n)
        return AsynchronousMetrics(
            detection_window_samples=kwargs.pop("detection_window_samples", 250),
            sample_rate=kwargs.pop("sample_rate", 250),
            step_size_ms=kwargs.pop("step_size_ms", 100.0),
            label_mapping=kwargs.pop("label_mapping", {0: "left", 1: "right"}),
            gate=gate,
        )

    def test_register_event_creates_trial(self):
        m = self._make_async()
        m.register_event(sample_idx=1000, label=0)
        assert len(m.trials) == 1
        assert m.trials[0].onset_sample == 1000
        assert m.trials[0].label == 0

    def test_prediction_in_trial_window(self):
        m = self._make_async()
        m.register_event(sample_idx=1000, label=0)
        # Causal window: [1000+250, 1000+500) = [1250, 1500)
        in_trial, detected = m.add_prediction(prediction=0, current_sample_idx=1300)
        assert in_trial is True

    def test_prediction_near_onset_is_background(self):
        m = self._make_async()
        m.register_event(sample_idx=1000, label=0)
        # Prediction at 1100: decoder window still contains mostly pre-event
        # data (causal offset = 250 samples).  Goes to background.
        in_trial, detected = m.add_prediction(prediction=0, current_sample_idx=1100)
        assert in_trial is False

    def test_prediction_outside_window_is_false_alarm(self):
        m = self._make_async()
        m.register_event(sample_idx=1000, label=0)
        # Well outside trial window
        in_trial, detected = m.add_prediction(prediction=1, current_sample_idx=2000)
        assert in_trial is False

    def test_per_class_accuracy_dwell(self):
        m = self._make_async(detection_window_samples=500, dwell_n=3)
        m.register_event(sample_idx=0, label=0)
        # Causal window: [500, 1000). Predictions inside it.
        m.add_prediction(0, 550)
        m.add_prediction(0, 600)
        m.add_prediction(0, 650)
        m.add_prediction(1, 700)

        metrics = m.get_all_metrics()
        assert metrics["per_class_accuracy"][0] == 1.0  # Dwell detected

    def test_balanced_accuracy(self):
        m = self._make_async(detection_window_samples=250, dwell_n=1)
        # Class 0: causal window [250, 500)
        m.register_event(0, label=0)
        m.add_prediction(0, 300)
        # Class 1: causal window [750, 1000)
        m.register_event(500, label=1)
        m.add_prediction(1, 800)

        metrics = m.get_all_metrics()
        assert metrics["balanced_accuracy"] == pytest.approx(1.0)

    def test_far_per_minute(self):
        m = self._make_async(detection_window_samples=250, sample_rate=250, dwell_n=2)
        m.register_event(0, label=0)
        # Causal window: [250, 500). Predictions at 600, 700 are outside → background
        m.add_prediction(1, 600)
        m.add_prediction(1, 700)

        metrics = m.get_all_metrics()
        assert metrics["far_per_min"] > 0

    def test_ttd_calculation(self):
        m = self._make_async(detection_window_samples=500, sample_rate=250, dwell_n=1)
        m.register_event(0, label=0)
        # Causal window: [500, 1000). First prediction at 550.
        m.add_prediction(0, 550)

        metrics = m.get_all_metrics()
        assert metrics["mean_ttd_ms"] == pytest.approx(0.0)

    def test_step_accuracy(self):
        m = self._make_async(detection_window_samples=500, dwell_n=1)
        m.register_event(0, label=0)
        # Causal window: [500, 1000)
        m.add_prediction(0, 550)   # correct
        m.add_prediction(0, 600)   # correct
        m.add_prediction(1, 650)   # wrong
        m.add_prediction(0, 700)   # correct

        metrics = m.get_all_metrics()
        assert metrics["mean_step_accuracy"] == pytest.approx(0.75, abs=0.01)

    def test_far_dwell_based(self):
        m = self._make_async(detection_window_samples=250, sample_rate=250, dwell_n=3)
        m.register_event(0, label=0)
        # Causal window: [250, 500). Predictions at 600+ are background.
        m.add_prediction(1, 600)
        m.add_prediction(1, 700)
        m.add_prediction(1, 800)

        metrics = m.get_all_metrics()
        assert metrics["far_per_min"] > 0

    def test_far_no_streak_no_detection(self):
        m = self._make_async(detection_window_samples=250, sample_rate=250, dwell_n=3)
        m.register_event(0, label=0)
        # Alternating predictions in background → no dwell streak → FAR = 0
        m.add_prediction(0, 600)
        m.add_prediction(1, 700)

        metrics = m.get_all_metrics()
        assert metrics["far_per_min"] == 0.0

    def test_reset_clears_state(self):
        m = self._make_async()
        m.register_event(0, label=0)
        # Causal window: [250, 500)
        m.add_prediction(0, 300)
        m.reset()
        assert len(m.trials) == 0
        assert len(m._background_preds) == 0
        assert m.last_sample_idx == 0

    def test_trials_survive_detection_window_update(self):
        """Simulates decoder reload: updating detection_window_samples preserves trial history."""
        m = self._make_async(detection_window_samples=250, dwell_n=3)

        # Trial 0: causal window [250, 500)
        m.register_event(sample_idx=0, label=0)
        for step in range(10):
            m.add_prediction(0, 250 + 25 * step)
        # Trial 1: causal window [750, 1000)
        m.register_event(sample_idx=500, label=1)
        for step in range(10):
            m.add_prediction(1, 750 + 25 * step)

        metrics_before = m.get_all_metrics()
        assert metrics_before["n_trials"] == 2
        assert metrics_before["balanced_accuracy"] == pytest.approx(1.0)

        # Simulate decoder reload: update window size (don't recreate)
        m.detection_window_samples = 500

        # Old trials still present
        metrics_after = m.get_all_metrics()
        assert metrics_after["n_trials"] == 2
        assert metrics_after["balanced_accuracy"] == pytest.approx(1.0)

        # New trial: causal window [2500, 3000)
        m.register_event(sample_idx=2000, label=0)
        m.add_prediction(1, 2600)  # wrong
        m.add_prediction(1, 2700)  # wrong
        m.add_prediction(1, 2800)  # wrong

        metrics_final = m.get_all_metrics()
        assert metrics_final["n_trials"] == 3
        # Class 0: 1 detected + 1 not detected = 0.5, Class 1: 1 detected = 1.0
        assert metrics_final["balanced_accuracy"] == pytest.approx(0.75)

    def test_step_size_ms_used_for_ttd(self):
        """step_size_ms passed directly — no derived formula."""
        m = self._make_async(detection_window_samples=500, dwell_n=1, step_size_ms=200.0)
        m.register_event(0, label=0)
        # First prediction at step 0 → dwell fires immediately
        m.add_prediction(0, 500)
        metrics = m.get_all_metrics()
        # TTD = detect_step (0) * step_duration_ms (200) = 0.0
        assert metrics["mean_ttd_ms"] == pytest.approx(0.0)

        # Second trial: dwell fires on step 2 → TTD = 2 * 200 = 400ms
        m2 = self._make_async(detection_window_samples=500, dwell_n=3, step_size_ms=200.0)
        m2.register_event(0, label=0)
        m2.add_prediction(0, 500)
        m2.add_prediction(0, 600)
        m2.add_prediction(0, 700)
        metrics2 = m2.get_all_metrics()
        assert metrics2["mean_ttd_ms"] == pytest.approx(400.0)

    def test_itr_calculation(self):
        m = self._make_async(detection_window_samples=250, dwell_n=1)
        # Class 0: causal window [250, 500)
        m.register_event(0, label=0)
        m.add_prediction(0, 300)
        # Class 1: causal window [750, 1000)
        m.register_event(500, label=1)
        m.add_prediction(1, 800)

        itr = m.get_itr(mean_selection_time_sec=1.0)
        assert isinstance(itr, float)

        # Sub-perfect accuracy should give positive ITR
        m.reset()
        # Class 0: causal window [250, 500)
        m.register_event(0, label=0)
        m.add_prediction(0, 300)
        # Class 1 trial 1: causal window [750, 1000)
        m.register_event(500, label=1)
        m.add_prediction(0, 800)  # Wrong
        # Class 1 trial 2: causal window [1250, 1500)
        m.register_event(1000, label=1)
        m.add_prediction(1, 1300)  # Correct

        itr2 = m.get_itr(mean_selection_time_sec=1.0)
        assert itr2 >= 0.0

    def test_dwell_detection_fires_once(self):
        """detected=True fires on the exact step dwell_n is reached, then never again."""
        m = self._make_async(detection_window_samples=500, dwell_n=3)
        m.register_event(sample_idx=0, label=0)
        # Causal window: [500, 1000)
        _, d1 = m.add_prediction(0, 550)  # streak=1
        _, d2 = m.add_prediction(0, 600)  # streak=2
        _, d3 = m.add_prediction(0, 650)  # streak=3 → detected!
        _, d4 = m.add_prediction(0, 700)  # streak=4, already detected

        assert (d1, d2, d3, d4) == (False, False, True, False)

    def test_dwell_detection_resets_on_wrong(self):
        """Streak resets when a wrong prediction interrupts."""
        m = self._make_async(detection_window_samples=500, dwell_n=3)
        m.register_event(sample_idx=0, label=0)
        # Causal window: [500, 1000)
        _, d1 = m.add_prediction(0, 550)  # streak=1
        _, d2 = m.add_prediction(0, 600)  # streak=2
        _, d3 = m.add_prediction(1, 650)  # wrong → streak=0
        _, d4 = m.add_prediction(0, 700)  # streak=1
        _, d5 = m.add_prediction(0, 750)  # streak=2
        _, d6 = m.add_prediction(0, 800)  # streak=3 → detected!

        assert (d1, d2, d3, d4, d5, d6) == (False, False, False, False, False, True)

    def test_detection_independent_across_trials(self):
        """Each trial tracks its own streak independently."""
        m = self._make_async(detection_window_samples=250, dwell_n=2)
        # Trial 0: causal window [250, 500)
        m.register_event(0, label=0)
        _, d1 = m.add_prediction(0, 300)
        _, d2 = m.add_prediction(0, 350)  # detected
        assert (d1, d2) == (False, True)

        # Trial 1: causal window [750, 1000)
        m.register_event(500, label=1)
        _, d3 = m.add_prediction(1, 800)
        _, d4 = m.add_prediction(1, 850)  # detected
        assert (d3, d4) == (False, True)

