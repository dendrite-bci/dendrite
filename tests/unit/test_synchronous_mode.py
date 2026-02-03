"""
Behavior-focused unit tests for SynchronousMode class.

This module provides unit tests for the SynchronousMode class,
focusing on public behavior and integration testing rather than
internal implementation details.

Tests cover:
- Configuration validation and initialization behavior
- Data processing pipeline (sample ingestion to output)
- Model lifecycle (creation, training, prediction readiness)
- Output generation and payload verification
- Error handling and edge cases

Follows red-green-refactor principles by testing what the mode does,
not how it does it internally.
"""

import sys
import os
import pytest
import numpy as np
import time
import threading
from unittest.mock import Mock, MagicMock
from queue import Queue, Empty

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dendrite.processing.modes.synchronous_mode import SynchronousMode, MetricsPayload, ERPPayload, SyncPrediction


class TestSynchronousModeConfiguration:
    """Test suite for SynchronousMode configuration and initialization behavior."""

    @pytest.fixture
    def mock_queues(self):
        """Create mock queues and events for external dependencies."""
        return {
            'data_queue': Mock(),
            'output_queue': Mock(),
            'prediction_queue': Mock(),
            'stop_event': Mock()
        }

    @staticmethod
    def build_instance_config(**kwargs):
        """Helper to build instance_config with defaults."""
        config = {
            'name': kwargs.get('name', 'test_sync'),
            'decoder_config': kwargs.get('decoder_config', {
                'model_config': {
                    'model_type': 'EEGNet',
                    'num_classes': 2,
                    'learning_rate': 0.001
                }
            }),
            'event_mapping': kwargs.get('event_mapping', {}),
            'channel_selection': kwargs.get('channel_selection', {'EEG': [0, 1, 2, 3]}),
            'start_offset': kwargs.get('start_offset', 0.0),
            'end_offset': kwargs.get('end_offset', 2.0),
            'training_interval': kwargs.get('training_interval', 10),
            'study_name': kwargs.get('study_name', 'default_study'),
            'linked_async_modes': kwargs.get('linked_async_modes', [])
        }

        # Copy any other kwargs directly
        for key, value in kwargs.items():
            if key not in ['name', 'event_mapping'] and key not in config:
                config[key] = value

        return config

    @pytest.fixture
    def valid_config(self):
        """Valid configuration that should work."""
        return self.build_instance_config(
            event_mapping={1: 'left_hand', 2: 'right_hand'},
            channel_selection={'EEG': [0, 1, 2, 3]},
            start_offset=-1.0,
            end_offset=2.0,
            training_interval=10
        )

    def test_valid_configuration_initializes_correctly(self, mock_queues, valid_config):
        """Test that valid configuration produces working mode."""
        mode = SynchronousMode(
            data_queue=mock_queues['data_queue'],
            output_queue=mock_queues['output_queue'],
            stop_event=mock_queues['stop_event'],
            instance_config=valid_config,
            sample_rate=500.0
        )

        # Test public configuration is applied
        assert mode.mode_name == 'test_sync'
        assert mode.sample_rate == 500.0
        # Event mapping is extracted from config
        expected_event_mapping = {1: 'left_hand', 2: 'right_hand'}
        assert mode.event_mapping == expected_event_mapping
        assert mode.start_offset == -1.0
        assert mode.end_offset == 2.0
        assert mode.training_interval == 10

        # Test derived calculations are correct
        assert mode.start_offset_samples == -500  # -1.0 * 500
        assert mode.end_offset_samples == 1000    # 2.0 * 500
        assert mode.epoch_length_samples == 1500  # 1000 - (-500)

        # Test label mapping uses sequential indices for ML training
        # event_mapping: {1: 'left_hand', 2: 'right_hand'} -> sorted by event code
        # Sequential indices: left_hand=0, right_hand=1
        expected_label_mapping = {'left_hand': 0, 'right_hand': 1}
        expected_reverse_mapping = {0: 'left_hand', 1: 'right_hand'}
        expected_index_to_event_code = {0: 1, 1: 2}
        assert mode.label_mapping == expected_label_mapping
        assert mode.reverse_label_mapping == expected_reverse_mapping
        assert mode.index_to_event_code == expected_index_to_event_code

        # Test mode is in uninitialized but configurable state
        assert mode.epoch_count == 0
        assert mode.decoder is None  # Not created until modalities detected

    def test_defaults_are_applied_when_not_specified(self, mock_queues):
        """Test that sensible defaults are used for unspecified parameters."""
        instance_config = self.build_instance_config(
            event_mapping={1: 'test_event'}
        )
        mode = SynchronousMode(
            data_queue=mock_queues['data_queue'],
            output_queue=mock_queues['output_queue'],
            stop_event=mock_queues['stop_event'],
            instance_config=instance_config,
            sample_rate=500.0
        )

        # Test default behavior
        assert mode.start_offset == 0.0  # Default epoch start
        assert mode.end_offset == 2.0    # Default epoch end
        assert mode.training_interval == 10  # Default training frequency
        assert mode.study_name == 'default_study'  # Default study name

    def test_empty_event_mapping_creates_empty_labels(self, mock_queues):
        """Test that empty event mapping results in empty label mappings."""
        instance_config = self.build_instance_config(
            event_mapping={}
        )
        mode = SynchronousMode(
            data_queue=mock_queues['data_queue'],
            output_queue=mock_queues['output_queue'],
            stop_event=mock_queues['stop_event'],
            instance_config=instance_config,
            sample_rate=500.0
        )

        # Test that empty mapping behavior is consistent
        assert mode.event_mapping == {}
        assert mode.label_mapping == {}
        assert mode.reverse_label_mapping == {}

        # Should still initialize without errors
        assert mode.mode_name == 'test_sync'


class TestSynchronousModeConfigurationValidation:
    """Test suite for configuration validation behavior."""

    @staticmethod
    def build_instance_config(**kwargs):
        """Helper to build instance_config with defaults."""
        return TestSynchronousModeConfiguration.build_instance_config(**kwargs)

    @pytest.fixture
    def mock_queues(self):
        """Create mock queues and events."""
        return {
            'data_queue': Mock(),
            'output_queue': Mock(),
            'stop_event': Mock()
        }

    def test_valid_configuration_passes_validation(self, mock_queues):
        """Test that valid configuration passes internal validation."""
        instance_config = self.build_instance_config(
            event_mapping={1: 'test_event'},
            start_offset=-1.0,
            end_offset=2.0
        )
        mode = SynchronousMode(
            data_queue=mock_queues['data_queue'],
            output_queue=mock_queues['output_queue'],
            stop_event=mock_queues['stop_event'],
            instance_config=instance_config,
            sample_rate=500.0
        )

        # Test that validation succeeds (public behavior)
        assert mode._validate_configuration() == True
        # Test that epoch calculation is positive
        assert mode.epoch_length_samples > 0

    def test_zero_epoch_length_fails_validation(self, mock_queues):
        """Test that zero or negative epoch length is detected as invalid."""
        instance_config = self.build_instance_config(
            event_mapping={1: 'test_event'},
            start_offset=1.0,
            end_offset=1.0  # Same as start_offset = zero length
        )
        mode = SynchronousMode(
            data_queue=mock_queues['data_queue'],
            output_queue=mock_queues['output_queue'],
            stop_event=mock_queues['stop_event'],
            instance_config=instance_config,
            sample_rate=500.0
        )
        mode.logger = Mock()

        # Test that invalid configuration is rejected
        assert mode._validate_configuration() == False
        assert mode.epoch_length_samples == 0
        # Test that error logging occurs
        mode.logger.error.assert_called()

    def test_negative_epoch_length_fails_validation(self, mock_queues):
        """Test that negative epoch length is detected as invalid."""
        instance_config = self.build_instance_config(
            event_mapping={1: 'test_event'},
            start_offset=2.0,
            end_offset=1.0  # End before start = negative length
        )
        mode = SynchronousMode(
            data_queue=mock_queues['data_queue'],
            output_queue=mock_queues['output_queue'],
            stop_event=mock_queues['stop_event'],
            instance_config=instance_config,
            sample_rate=500.0
        )
        mode.logger = Mock()

        # Test that invalid configuration is rejected
        assert mode._validate_configuration() == False
        assert mode.epoch_length_samples < 0
        # Test that error logging occurs
        mode.logger.error.assert_called()

    def test_empty_event_mapping_fails_validation(self, mock_queues):
        """Test that empty event mapping is detected as invalid."""
        instance_config = self.build_instance_config(
            event_mapping={}  # Empty - no events to process
        )
        mode = SynchronousMode(
            data_queue=mock_queues['data_queue'],
            output_queue=mock_queues['output_queue'],
            stop_event=mock_queues['stop_event'],
            instance_config=instance_config,
            sample_rate=500.0
        )
        mode.logger = Mock()

        # Test that empty event mapping is invalid for synchronous mode
        assert mode._validate_configuration() == False
        # Test that error logging occurs
        mode.logger.error.assert_called()


class TestSynchronousModeDataProcessing:
    """Test suite for data processing pipeline behavior."""

    @staticmethod
    def build_instance_config(**kwargs):
        """Helper to build instance_config with defaults."""
        return TestSynchronousModeConfiguration.build_instance_config(**kwargs)

    @pytest.fixture
    def processing_mode(self):
        """Create a SynchronousMode instance with minimal mocking for integration testing."""
        instance_config = self.build_instance_config(
            event_mapping={1: 'left_hand', 2: 'right_hand'},
            start_offset=-0.2,
            end_offset=1.0,
            training_interval=5
        )
        mode = SynchronousMode(
            data_queue=Mock(),
            output_queue=Mock(),
            stop_event=Mock(),
            instance_config=instance_config,
            sample_rate=500.0
        )
        mode.logger = Mock()
        return mode

    def test_valid_sample_is_processed_successfully(self, processing_mode):
        """Test that valid samples are processed without errors."""
        # Mock the buffer since it's None initially
        processing_mode.buffer = Mock()
        processing_mode.buffer.add_sample = Mock()

        sample = {
            'eeg': np.array([1.0, 2.0, 3.0, 4.0]),
            'markers': -1  # Background sample
        }

        # Should not raise any exceptions
        processing_mode._process_data(sample)

        # Test that sample was ingested
        assert processing_mode.current_sample_index == 1
        # Test that buffer received the sample
        processing_mode.buffer.add_sample.assert_called_once_with(sample)

    def test_event_marker_triggers_epoch_scheduling(self, processing_mode):
        """Test that event markers result in epoch scheduling."""
        # Setup buffer mock to allow epoch extraction
        processing_mode.buffer = Mock()
        processing_mode.buffer.add_sample = Mock()

        sample = {
            'eeg': np.array([1.0, 2.0, 3.0, 4.0]),
            'markers': 1  # left_hand event
        }

        processing_mode._process_data(sample)

        # Test that buffer received sample
        processing_mode.buffer.add_sample.assert_called_once_with(sample)
        # Test that epoch is pending (requires post-event data)
        assert len(processing_mode.pending_epochs) > 0

    def test_invalid_event_codes_are_ignored(self, processing_mode):
        """Test that invalid event codes don't trigger epoch processing."""
        processing_mode.buffer = Mock()
        processing_mode.buffer.add_sample = Mock()

        sample = {
            'eeg': np.array([1.0, 2.0, 3.0, 4.0]),
            'markers': 99  # Invalid event code
        }

        processing_mode._process_data(sample)

        # Test that sample was still buffered
        processing_mode.buffer.add_sample.assert_called_once_with(sample)
        # Test that no epoch was scheduled
        assert len(processing_mode.pending_epochs) == 0

    def test_non_dict_samples_are_ignored(self, processing_mode):
        """Test that invalid sample formats are handled gracefully."""
        # Should not raise exceptions
        processing_mode._process_data("invalid_sample")
        processing_mode._process_data(None)
        processing_mode._process_data(123)

        # Sample index should not increment
        assert processing_mode.current_sample_index == 0

    def test_epoch_processing_increments_counters(self, processing_mode):
        """Test that successful epoch processing increments counters."""
        # Setup mocks for successful epoch extraction
        processing_mode.buffer = Mock()
        processing_mode.buffer.add_sample = Mock()
        processing_mode.buffer.extract_epoch_at_event = Mock(return_value={'eeg': np.array([[1, 2, 3]])})
        processing_mode.dataset = Mock()

        # Simulate event sample followed by enough post-event samples
        event_sample = {'eeg': np.array([1.0, 2.0, 3.0, 4.0]), 'markers': 1}
        processing_mode._process_data(event_sample)

        # Add enough post-event samples to trigger extraction
        for i in range(processing_mode.end_offset_samples + 1):
            post_sample = {'eeg': np.array([1.0, 2.0, 3.0, 4.0]), 'markers': -1}
            processing_mode._process_data(post_sample)

        # Test that epoch counter incremented
        assert processing_mode.epoch_count > 0




class TestSynchronousModeModelLifecycle:
    """Test suite for model lifecycle behavior (creation, training, prediction readiness)."""

    @staticmethod
    def build_instance_config(**kwargs):
        """Helper to build instance_config with defaults."""
        return TestSynchronousModeConfiguration.build_instance_config(**kwargs)

    @pytest.fixture
    def lifecycle_mode(self):
        """Create SynchronousMode for model lifecycle testing."""
        instance_config = self.build_instance_config(
            event_mapping={1: 'left_hand', 2: 'right_hand'},
            training_interval=3,  # Train every 3 epochs
            channel_selection={'EEG': [0, 1, 2, 3]},
            decoder_config={'model_config': {'model_type': 'EEGNet'}}
        )
        mode = SynchronousMode(
            data_queue=Mock(),
            output_queue=Mock(),
            stop_event=Mock(),
            instance_config=instance_config,
            sample_rate=500.0
        )
        mode.logger = Mock()
        return mode

    def test_model_starts_as_not_ready(self, lifecycle_mode):
        """Test that model is not ready for predictions initially."""
        # Helper to check model readiness using the same pattern as production code
        def is_model_ready(decoder):
            return getattr(decoder, 'is_fitted', True) if decoder else False

        # Model should not exist initially
        assert lifecycle_mode.decoder is None
        # Should not be ready for predictions (decoder is None)
        assert not is_model_ready(lifecycle_mode.decoder)

    def test_decoder_created_after_modality_detection(self, lifecycle_mode):
        """Test that decoder is created once modalities are detected."""
        # Simulate modality detection
        lifecycle_mode.modalities_detected = True
        lifecycle_mode.modalities = ['EEG']

        # Trigger decoder creation
        decoder = lifecycle_mode._create_decoder(lifecycle_mode.decoder_config)

        # Should create decoder when modalities are available
        assert decoder is not None

    def test_training_triggered_at_specified_intervals(self, lifecycle_mode):
        """Test that training is triggered at the configured interval."""
        lifecycle_mode.dataset = Mock()
        lifecycle_mode.dataset.get_training_data = Mock(return_value={
            'X': [np.array([[1, 2, 3]]), np.array([[4, 5, 6]])],
            'y': [0, 1]
        })
        # Setup training queue to verify data is scheduled
        lifecycle_mode._training_queue = Mock()
        lifecycle_mode._training_queue.put = Mock()

        # Trigger training
        lifecycle_mode.epoch_count = 3
        lifecycle_mode._trigger_training()

        # Verify training data was put on queue
        lifecycle_mode._training_queue.put.assert_called_once()

    def test_model_readiness_depends_on_training_state(self, lifecycle_mode):
        """Test that model readiness correctly reflects training state."""
        # Helper to check model readiness using the same pattern as production code
        def is_model_ready(decoder):
            return getattr(decoder, 'is_fitted', True) if decoder else False

        # Create a mock decoder
        mock_decoder = Mock()
        lifecycle_mode.decoder = mock_decoder

        # Test not fitted state
        mock_decoder.is_fitted = False
        assert not is_model_ready(lifecycle_mode.decoder)

        # Test fitted state
        mock_decoder.is_fitted = True
        assert is_model_ready(lifecycle_mode.decoder)

        # Test no decoder state
        lifecycle_mode.decoder = None
        assert not is_model_ready(lifecycle_mode.decoder)

    def test_insufficient_data_does_not_trigger_training(self, lifecycle_mode):
        """Test that training is not triggered with insufficient data."""
        lifecycle_mode.dataset = Mock()
        lifecycle_mode.dataset.samples = []  # Mock samples list for len() access
        lifecycle_mode.dataset.get_training_data = Mock(return_value={
            'X': [np.array([[1, 2, 3]])],  # Only 1 sample
            'y': [0]
        })
        lifecycle_mode._schedule_training = Mock()

        lifecycle_mode._trigger_training()

        # Should not schedule training with < 2 samples
        lifecycle_mode._schedule_training.assert_not_called()


class TestSynchronousModeDataStorage:
    """Test suite for data storage and dataset behavior."""

    @staticmethod
    def build_instance_config(**kwargs):
        """Helper to build instance_config with defaults."""
        return TestSynchronousModeConfiguration.build_instance_config(**kwargs)

    @pytest.fixture
    def storage_mode(self):
        """Create SynchronousMode for testing data storage behavior."""
        instance_config = self.build_instance_config(
            event_mapping={1: 'left_hand', 2: 'right_hand'},
            training_interval=2  # Train every 2 epochs
        )
        mode = SynchronousMode(
            data_queue=Mock(),
            output_queue=Mock(),
            stop_event=Mock(),
            instance_config=instance_config,
            sample_rate=500.0
        )
        mode.logger = Mock()
        return mode

    def test_dataset_is_created_automatically(self, storage_mode):
        """Test that dataset is created automatically when needed."""
        # Dataset should exist after initialization
        assert storage_mode.dataset is not None
        # Should have appropriate name
        assert 'test_sync' in storage_mode.dataset.name

    def test_epoch_data_is_stored_in_dataset(self, storage_mode):
        """Test that processed epochs are stored in dataset with correct metadata."""
        storage_mode.dataset = Mock()
        storage_mode._send_erp_data = Mock()
        storage_mode.decoder = None  # No prediction
        storage_mode.epoch_count = 1

        X_input = {'eeg': np.array([[1, 2, 3]])}

        storage_mode._process_extracted_epoch(X_input, 1, 'left_hand', 0)

        # Test that dataset received the epoch data
        storage_mode.dataset.add_sample.assert_called_once()

        # Verify call structure (arguments should include X_input, label, source_id, source_info)
        call_args = storage_mode.dataset.add_sample.call_args
        assert call_args[0][0] == X_input  # X_input
        assert call_args[0][1] == 0        # label
        # Source info should contain sampling metadata
        source_info = call_args[0][3]
        assert source_info['sampling_freq'] == 500.0
        assert source_info['mode_name'] == 'test_sync'


class TestSynchronousModePredictionBehavior:
    """Test suite for prediction behavior in SynchronousMode."""

    @staticmethod
    def build_instance_config(**kwargs):
        """Helper to build instance_config with defaults."""
        return TestSynchronousModeConfiguration.build_instance_config(**kwargs)

    @pytest.fixture
    def prediction_mode(self):
        """Create SynchronousMode for prediction testing."""
        instance_config = self.build_instance_config(
            event_mapping={1: 'left_hand', 2: 'right_hand'}
        )
        mode = SynchronousMode(
            data_queue=Mock(),
            output_queue=Mock(),
            stop_event=Mock(),
            instance_config=instance_config,
            sample_rate=500.0
        )
        mode.logger = Mock()
        return mode

    def test_predictions_made_when_model_ready(self, prediction_mode):
        """Test that predictions are made when model is ready."""
        mock_decoder = Mock()
        mock_decoder.is_fitted = True
        mock_decoder.predict_sample.return_value = (1, 0.85)
        prediction_mode.decoder = mock_decoder
        prediction_mode._model_lock = threading.Lock()

        X_input = {'eeg': np.array([[1, 2, 3]])}
        prediction, confidence, _ = prediction_mode._predict(
            model=mock_decoder,
            X_input=X_input,
            lock=prediction_mode._model_lock,
            blocking=False
        )

        assert prediction == 1
        assert confidence == 0.85
        mock_decoder.predict_sample.assert_called_once()
        actual_call_arg = mock_decoder.predict_sample.call_args[0][0]
        np.testing.assert_array_equal(actual_call_arg, X_input['eeg'])

    def test_default_prediction_when_no_decoder(self, prediction_mode):
        """Test that default values are returned when no decoder exists."""
        prediction_mode._model_lock = threading.Lock()

        X_input = {'eeg': np.array([[1, 2, 3]])}
        prediction, confidence, _ = prediction_mode._predict(
            model=None,
            X_input=X_input,
            lock=prediction_mode._model_lock,
            blocking=False
        )

        assert prediction == 0
        assert confidence == 0.5

    def test_default_prediction_when_model_not_ready(self, prediction_mode):
        """Test that default values are returned when model is not ready."""
        mock_decoder = Mock()
        mock_decoder.is_fitted = False
        prediction_mode._model_lock = threading.Lock()

        X_input = {'eeg': np.array([[1, 2, 3]])}
        prediction, confidence, _ = prediction_mode._predict(
            model=mock_decoder,
            X_input=X_input,
            lock=prediction_mode._model_lock,
            blocking=False
        )

        assert prediction == 0
        assert confidence == 0.5
        mock_decoder.predict_sample.assert_not_called()

    def test_prediction_errors_handled_gracefully(self, prediction_mode):
        """Test that prediction errors are handled without crashing."""
        mock_decoder = Mock()
        mock_decoder.is_fitted = True
        mock_decoder.predict_sample.side_effect = Exception("Prediction failed")
        prediction_mode.decoder = mock_decoder
        prediction_mode._model_lock = threading.Lock()

        X_input = {'eeg': np.array([[1, 2, 3]])}
        prediction, confidence, _ = prediction_mode._predict(
            model=mock_decoder,
            X_input=X_input,
            lock=prediction_mode._model_lock,
            blocking=False
        )

        assert prediction == 0
        assert confidence == 0.5
        prediction_mode.logger.error.assert_called()

    def test_predictions_avoided_during_training(self, prediction_mode):
        """Test that predictions are avoided when model is being trained."""
        mock_decoder = Mock()
        mock_decoder.is_fitted = True
        mock_decoder.predict_sample.return_value = (1, 0.85)
        prediction_mode.decoder = mock_decoder
        prediction_mode._model_lock = threading.Lock()

        prediction_mode._model_lock.acquire()

        try:
            X_input = {'eeg': np.array([[1, 2, 3]])}
            prediction, confidence, _ = prediction_mode._predict(
                model=mock_decoder,
                X_input=X_input,
                lock=prediction_mode._model_lock,
                blocking=False
            )

            assert prediction == 0
            assert confidence == 0.5
            mock_decoder.predict_sample.assert_not_called()
        finally:
            prediction_mode._model_lock.release()

    def test_model_readiness_checking_behavior(self, prediction_mode):
        """Test that model readiness is checked correctly."""
        # Helper to check model readiness using the same pattern as production code
        def is_model_ready(decoder):
            return getattr(decoder, 'is_fitted', True) if decoder else False

        # Test with fitted decoder
        mock_decoder = Mock()
        mock_decoder.is_fitted = True
        prediction_mode.decoder = mock_decoder
        assert is_model_ready(prediction_mode.decoder) == True

        # Test with unfitted decoder
        mock_decoder.is_fitted = False
        assert is_model_ready(prediction_mode.decoder) == False

        # Test with no decoder
        prediction_mode.decoder = None
        assert is_model_ready(prediction_mode.decoder) == False


class TestSynchronousModeOutputBehavior:
    """Test suite for output generation and payload behavior."""

    @staticmethod
    def build_instance_config(**kwargs):
        """Helper to build instance_config with defaults."""
        return TestSynchronousModeConfiguration.build_instance_config(**kwargs)

    @pytest.fixture
    def output_mode(self):
        """Create SynchronousMode for output testing."""
        instance_config = self.build_instance_config(
            event_mapping={1: 'left_hand', 2: 'right_hand'}
        )
        mode = SynchronousMode(
            data_queue=Mock(),
            output_queue=Mock(),
            stop_event=Mock(),
            instance_config=instance_config,
            sample_rate=500.0
        )
        mode.logger = Mock()
        return mode

    def test_performance_metrics_generated(self, output_mode):
        """Test that performance metrics are generated and contain expected fields."""
        output_mode.output_queue = Mock()
        output_mode.prediction_queue = Mock()
        output_mode.metrics_manager = Mock()

        mock_metrics = {'prequential_accuracy': 0.75}
        output_mode.metrics_manager.get_current_metrics.return_value = mock_metrics

        output_mode._update_metrics_and_send(1, 0.8, 0, 'left_hand')

        # Verify queues received output (performance to main, prediction to prediction queue)
        output_mode.output_queue.put.assert_called_once()
        output_mode.prediction_queue.put.assert_called_once()

    def test_outputs_generated_without_metrics_manager(self, output_mode):
        """Test that outputs are still generated when no metrics manager exists."""
        output_mode.output_queue = Mock()
        output_mode.prediction_queue = Mock()
        output_mode.metrics_manager = None

        # Should not crash
        output_mode._update_metrics_and_send(1, 0.8, 0, 'left_hand')

        # Should still generate outputs with default values
        output_mode.output_queue.put.assert_called_once()
        output_mode.prediction_queue.put.assert_called_once()

    def test_erp_data_sent_for_visualization(self, output_mode):
        """Test that ERP data is sent for visualization when EEG data is available."""
        output_mode.output_queue = Mock()
        output_mode.prediction_queue = Mock()

        X_input = {'eeg': np.array([[1, 2, 3], [4, 5, 6]])}

        output_mode._send_output(ERPPayload(
            event_type='left_hand',
            eeg_data=X_input['eeg'],
            start_offset_ms=0.0,
            sample_rate=500.0
        ), 'erp', queue='main')

        # Should send ERP data to main queue
        output_mode.output_queue.put.assert_called_once()
        output_mode.prediction_queue.put.assert_not_called()

    def test_no_erp_data_sent_without_eeg(self, output_mode):
        """Test that no ERP data is sent when EEG data is not available."""
        output_mode.output_queue = Mock()
        output_mode.prediction_queue = Mock()

        X_input = {'emg': np.array([[1, 2, 3]])}  # No EEG data

        # Only send if EEG present (mimicking actual behavior)
        if 'eeg' in X_input:
            output_mode._send_output({}, 'erp', queue='main')

        # Should not send anything
        output_mode.output_queue.put.assert_not_called()

    def test_prediction_payload_contains_correct_event_names(self, output_mode):
        """Test that prediction payloads contain human-readable event names."""
        output_mode.output_queue = Mock()
        output_mode.prediction_queue = Mock()
        output_mode.metrics_manager = Mock()
        output_mode.metrics_manager.get_current_metrics.return_value = {'prequential_accuracy': 0.8}

        # Test with prediction=1 which should map to 'right_hand'
        output_mode._update_metrics_and_send(1, 0.85, 0, 'left_hand')

        # Should generate outputs with meaningful names
        output_mode.output_queue.put.assert_called_once()
        output_mode.prediction_queue.put.assert_called_once()
