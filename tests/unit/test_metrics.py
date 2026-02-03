"""
Unit tests for BMI metrics module.

Tests cover:
- SynchronousMetrics: trial-based evaluation
- AsynchronousMetrics: window-based confusion matrix metrics
- MetricsManager: mode routing
"""

import pytest
import numpy as np
from dendrite.ml.metrics import SynchronousMetrics, AsynchronousMetrics, MetricsManager


class TestSynchronousMetrics:
    """Tests for trial-based synchronous metrics."""

    def test_basic_accuracy_calculation(self):
        """Test that accuracy is calculated correctly."""
        metrics = SynchronousMetrics(num_classes=2)

        for i in range(7):
            metrics.add_prediction(prediction=1, true_label=1)
        for i in range(3):
            metrics.add_prediction(prediction=0, true_label=1)

        result = metrics.get_all_metrics()
        assert 0.5 <= result['prequential_accuracy'] <= 0.9
        assert result['samples_processed'] == 10

    def test_cohens_kappa_perfect_agreement(self):
        """Test Cohen's kappa with perfect agreement."""
        metrics = SynchronousMetrics(num_classes=2)

        for _ in range(5):
            metrics.add_prediction(prediction=0, true_label=0)
        for _ in range(5):
            metrics.add_prediction(prediction=1, true_label=1)

        result = metrics.get_all_metrics()
        assert result['cohens_kappa'] == pytest.approx(1.0, rel=0.01)

    def test_cohens_kappa_chance_agreement(self):
        """Test Cohen's kappa with chance-level agreement."""
        metrics = SynchronousMetrics(num_classes=2)

        np.random.seed(42)
        for _ in range(100):
            pred = np.random.randint(0, 2)
            label = np.random.randint(0, 2)
            metrics.add_prediction(prediction=pred, true_label=label)

        result = metrics.get_all_metrics()
        assert -0.3 <= result['cohens_kappa'] <= 0.3

    def test_reset_clears_all_metrics(self):
        """Test that reset clears all accumulated metrics."""
        metrics = SynchronousMetrics(num_classes=2)

        for _ in range(10):
            metrics.add_prediction(prediction=1, true_label=1)

        metrics.reset()

        result = metrics.get_all_metrics()
        assert result['samples_processed'] == 0
        assert result['prequential_accuracy'] == 0.0


class TestAsynchronousMetrics:
    """Tests for trial-level asynchronous metrics (TPR, TNR, FAR, TTD)."""

    def test_initialization(self):
        """Test AsynchronousMetrics initialization."""
        metrics = AsynchronousMetrics(
            detection_window_samples=250,
            sample_rate=250,
        )
        assert metrics.detection_window_samples == 250
        assert metrics.sample_rate == 250

    def test_register_event(self):
        """Test event registration."""
        metrics = AsynchronousMetrics(detection_window_samples=100, sample_rate=250)
        metrics.register_event(sample_idx=1000, label=1)
        metrics.register_event(sample_idx=2000, label=0)

        assert len(metrics.trials) == 2
        assert metrics.trials[0].onset_sample == 1000
        assert metrics.trials[0].label == 1
        assert metrics.trials[1].label == 0

    def test_add_prediction_detection_in_window(self):
        """Test prediction within trial detection window."""
        metrics = AsynchronousMetrics(detection_window_samples=100, sample_rate=250)
        metrics.register_event(sample_idx=1000, label=1)

        # Prediction within window should be a detection
        result = metrics.add_prediction(prediction=1, current_sample_idx=1050)
        assert result is True
        assert metrics.trials[0].first_correct_ms is not None

    def test_add_prediction_false_positive(self):
        """Test false positive in background."""
        metrics = AsynchronousMetrics(detection_window_samples=100, sample_rate=250)
        # No events registered - all predictions are in background
        result = metrics.add_prediction(prediction=1, current_sample_idx=500)
        assert result is False
        assert len(metrics.background_fp_samples) == 1

    def test_class_1_accuracy_calculation(self):
        """Test class 1 accuracy calculation."""
        metrics = AsynchronousMetrics(detection_window_samples=100, sample_rate=250)

        # Register 2 error trials
        metrics.register_event(sample_idx=1000, label=1)
        metrics.register_event(sample_idx=2000, label=1)

        # Detect first error
        metrics.add_prediction(prediction=1, current_sample_idx=1050)
        # Miss second error (no detection in window)

        result = metrics.get_metrics()
        assert result['per_class_accuracy'][1] == pytest.approx(0.5, rel=0.01)  # 1/2 detected

    def test_reset_clears_all(self):
        """Test that reset clears all tracking."""
        metrics = AsynchronousMetrics(detection_window_samples=100, sample_rate=250)
        metrics.register_event(sample_idx=1000, label=1)
        metrics.add_prediction(prediction=1, current_sample_idx=1050)

        metrics.reset()

        assert len(metrics.trials) == 0
        assert len(metrics.background_fp_samples) == 0
        result = metrics.get_metrics()
        assert result['n_trials'] == 0


class TestMetricsManager:
    """Tests for MetricsManager routing."""

    def test_synchronous_mode_initialization(self):
        """Test MetricsManager initializes correctly for sync mode."""
        manager = MetricsManager(
            mode_type='synchronous',
            sample_rate=250,
            num_classes=2
        )

        assert manager.mode_type == 'synchronous'
        assert manager.sync_metrics is not None
        assert manager.async_metrics is None

    def test_asynchronous_mode_initialization(self):
        """Test MetricsManager initializes correctly for async mode."""
        manager = MetricsManager(
            mode_type='asynchronous',
            sample_rate=250,
            num_classes=2,
            detection_window_samples=250
        )

        assert manager.mode_type == 'asynchronous'
        assert manager.async_metrics is not None
        assert manager.sync_metrics is None

    def test_sync_mode_update_returns_metrics(self):
        """Test sync mode update returns metrics dict."""
        manager = MetricsManager(mode_type='synchronous', num_classes=2)

        result = manager.update_metrics(
            prediction=1,
            true_label=1
        )

        assert isinstance(result, dict)
        assert 'prequential_accuracy' in result

    def test_async_mode_update(self):
        """Test async mode update works."""
        manager = MetricsManager(
            mode_type='asynchronous',
            sample_rate=250,
            num_classes=2,
            detection_window_samples=100
        )

        result = manager.update_metrics(
            prediction=0,
            true_label=0,
            current_sample_idx=15
        )

        assert isinstance(result, dict)

    def test_reset_delegates_to_backend(self):
        """Test reset delegates to appropriate backend."""
        manager = MetricsManager(mode_type='synchronous', num_classes=2)

        manager.update_metrics(prediction=1, true_label=1)
        manager.reset()

        metrics = manager.get_current_metrics()
        assert metrics['samples_processed'] == 0

    def test_invalid_mode_raises_error(self):
        """Test invalid mode type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid mode_type"):
            MetricsManager(mode_type='invalid', num_classes=2)


from dendrite.utils.state_keys import (
    stream_latency_key,
    stream_timestamp_key,
    mode_metric_key,
    streamer_metric_key,
    STATIC_METRICS,
    MetricType,
)


class TestStreamMetricKeys:
    """Test suite for stream metric key generation."""

    def test_stream_latency_key_lowercase(self):
        """Test stream latency key generation with uppercase input."""
        assert stream_latency_key('EEG') == 'eeg_latency_p50'
        assert stream_latency_key('Events') == 'events_latency_p50'
        assert stream_latency_key('EMG') == 'emg_latency_p50'

    def test_stream_timestamp_key(self):
        """Test stream timestamp key generation."""
        assert stream_timestamp_key('EEG') == 'eeg_latency_ts'
        assert stream_timestamp_key('Events') == 'events_latency_ts'


class TestModeMetricKeys:
    """Test suite for mode metric key generation."""

    def test_mode_metric_key_internal(self):
        """Test mode internal latency key."""
        assert mode_metric_key('sync_mode_1', 'internal_ms') == 'sync_mode_1_internal_ms'

    def test_mode_metric_key_inference(self):
        """Test mode inference time key."""
        assert mode_metric_key('async_mode_1', 'inference_ms') == 'async_mode_1_inference_ms'


class TestStreamerMetricKeys:
    """Test suite for streamer metric key generation."""

    def test_streamer_metric_key_lowercase(self):
        """Test streamer metric key with uppercase input."""
        assert streamer_metric_key('LSL', 'bandwidth_kbps') == 'lsl_bandwidth_kbps'

    def test_streamer_metric_key_with_spaces(self):
        """Test streamer metric key with spaces in name."""
        assert streamer_metric_key('BMI Visualization', 'bandwidth_kbps') == 'bmi_visualization_bandwidth_kbps'


class TestStaticMetrics:
    """Test suite for static metrics registry."""

    def test_static_metrics_defined(self):
        """Test that expected static metrics exist."""
        assert 'e2e_latency_ms' in STATIC_METRICS
        assert 'viz_stream_bandwidth_kbps' in STATIC_METRICS
        assert 'channel_info' in STATIC_METRICS

    def test_static_metrics_types(self):
        """Test that static metrics have correct types."""
        assert STATIC_METRICS['e2e_latency_ms'].metric_type == MetricType.FLOAT
        assert STATIC_METRICS['channel_info'].metric_type == MetricType.DICT
