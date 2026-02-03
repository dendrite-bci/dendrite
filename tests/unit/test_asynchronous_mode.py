"""
Comprehensive unit tests for AsynchronousMode class.

This module provides unit tests for the AsynchronousMode class,
focusing on testing individual components and methods in isolation.

Tests cover:
- AsynchronousMode initialization and configuration
- Continuous data processing and sliding window management
- Model loading and prediction functionality
- Model sharing from sync modes
- Metrics handling and temporal evaluation
- Error handling and edge cases
"""

import sys
import os
import pytest
import numpy as np
import time
import threading
from unittest.mock import Mock, patch, MagicMock, call
from queue import Queue, Empty
import json

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dendrite.processing.modes.asynchronous_mode import AsynchronousMode, AsyncPrediction


class TestAsynchronousModeInitialization:
    """Test suite for AsynchronousMode initialization."""
    
    @pytest.fixture
    def mock_queues(self):
        """Create mock queues and events."""
        return {
            'data_queue': Mock(),
            'output_queue': Mock(),
            'prediction_queue': Mock(),
            'stop_event': Mock()
        }
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for AsynchronousMode."""
        return {
            'name': 'test_async',
            'mode': 'asynchronous',
            'decoder_config': {
                'model_config': {
                    'model_type': 'EEGNet',
                    'num_classes': 2,
                    'learning_rate': 0.001
                }
            },
            'event_mapping': {1: 'left_hand', 2: 'right_hand'},
            'channel_selection': {'EEG': [0, 1, 2, 3]},
            'step_size_ms': 100,
            'window_length_sec': 1.0  # 500 samples / 500 Hz = 1.0 sec
        }
    
    def test_basic_initialization(self, mock_queues, basic_config):
        """Test basic AsynchronousMode initialization."""
        mode = AsynchronousMode(
            data_queue=mock_queues['data_queue'],
            output_queue=mock_queues['output_queue'],
            stop_event=mock_queues['stop_event'],
            instance_config=basic_config,
            sample_rate=500.0
        )

        assert mode.mode_name == 'test_async'
        assert mode.sample_rate == 500.0
        assert mode.decoder_config == basic_config['decoder_config']
        assert mode.event_mapping == {1: 'left_hand', 2: 'right_hand'}
        assert mode.step_size_ms == 100
        assert mode.epoch_length_samples == 500  # Derived from window_length_sec * sample_rate

        # Check derived attributes
        assert mode.samples_per_prediction_step == 50  # 500 * (100/1000)

        # Check label mapping uses sequential indices for ML training
        # event_mapping: {1: 'left_hand', 2: 'right_hand'} -> sorted by event code
        # Sequential indices: left_hand=0, right_hand=1
        expected_label_mapping = {'left_hand': 0, 'right_hand': 1}
        expected_reverse_mapping = {0: 'left_hand', 1: 'right_hand'}
        expected_index_to_event_code = {0: 1, 1: 2}
        assert mode.label_mapping == expected_label_mapping
        assert mode.reverse_label_mapping == expected_reverse_mapping
        assert mode.index_to_event_code == expected_index_to_event_code
    
    def test_initialization_with_defaults(self, mock_queues):
        """Test initialization with default parameters."""
        config = {
            'name': 'test_async',
            'mode': 'asynchronous',
            'decoder_config': {'model_config': {'model_type': 'EEGNet', 'num_classes': 2}},
            'event_mapping': {1: 'test_event'}
        }
        mode = AsynchronousMode(
            data_queue=mock_queues['data_queue'],
            output_queue=mock_queues['output_queue'],
            stop_event=mock_queues['stop_event'],
            instance_config=config,
            sample_rate=500.0
        )

        assert mode.step_size_ms == 100
        assert mode.epoch_length_samples == 500  # Default to 1 second at 500 Hz
        assert mode.decoder_source == 'pretrained'
        assert mode.source_sync_mode == ''
        assert mode.study_name == 'default_study'
    
    def test_initialization_with_sync_mode_source(self, mock_queues, basic_config):
        """Test initialization with sync mode decoder source."""
        config = basic_config.copy()
        config.update({
            'decoder_source': 'sync_mode',
            'source_sync_mode': 'test_sync',
            'study_name': 'test_study',
            'file_identifier': 'session_1'
        })
        mode = AsynchronousMode(
            data_queue=mock_queues['data_queue'],
            output_queue=mock_queues['output_queue'],
            stop_event=mock_queues['stop_event'],
            instance_config=config,
            sample_rate=500.0
        )

        assert mode.decoder_source == 'sync_mode'
        assert mode.source_sync_mode == 'test_sync'
        assert mode.study_name == 'test_study'
        assert mode.file_identifier == 'session_1'
    
    def test_initialization_without_event_mapping(self, mock_queues):
        """Test initialization with empty event mapping."""
        config = {
            'name': 'test_async',
            'mode': 'asynchronous',
            'decoder_config': {'model_config': {'model_type': 'EEGNet', 'num_classes': 2}},
            'event_mapping': {}
        }
        mode = AsynchronousMode(
            data_queue=mock_queues['data_queue'],
            output_queue=mock_queues['output_queue'],
            stop_event=mock_queues['stop_event'],
            instance_config=config,
            sample_rate=500.0
        )

        assert mode.event_mapping == {}
        assert mode.label_mapping == {}
        assert mode.reverse_label_mapping == {}
    
    def test_label_mapping_generation_from_base_class(self, mock_queues):
        """Test that label mapping generation uses BaseMode helper method."""
        config = {
            'name': 'test_async',
            'mode': 'asynchronous',
            'decoder_config': {'model_config': {'model_type': 'EEGNet', 'num_classes': 2}},
            'event_mapping': {1: 'target', 2: 'non_target', 3: 'distractor'}
        }
        mode = AsynchronousMode(
            data_queue=mock_queues['data_queue'],
            output_queue=mock_queues['output_queue'],
            stop_event=mock_queues['stop_event'],
            instance_config=config,
            sample_rate=500.0
        )

        # Test that label mapping uses sequential indices for ML training
        # event_mapping: {1: 'target', 2: 'non_target', 3: 'distractor'} -> sorted by event code
        # Sequential indices: target=0, non_target=1, distractor=2
        expected_label_mapping = {'target': 0, 'non_target': 1, 'distractor': 2}
        expected_reverse_mapping = {0: 'target', 1: 'non_target', 2: 'distractor'}
        expected_index_to_event_code = {0: 1, 1: 2, 2: 3}
        assert mode.label_mapping == expected_label_mapping
        assert mode.reverse_label_mapping == expected_reverse_mapping
        assert mode.index_to_event_code == expected_index_to_event_code


class TestAsynchronousModeValidation:
    """Test suite for AsynchronousMode configuration validation."""
    
    @pytest.fixture
    def mock_queues(self):
        """Create mock queues and events."""
        return {
            'data_queue': Mock(),
            'output_queue': Mock(),
            'stop_event': Mock()
        }
    
    def test_validate_configuration_valid(self, mock_queues):
        """Test configuration validation with valid parameters."""
        config = {
            'name': 'test_async',
            'mode': 'asynchronous',
            'decoder_config': {'model_config': {'model_type': 'EEGNet', 'num_classes': 2}},
            'event_mapping': {1: 'test_event'},
            'channel_selection': {'EEG': [0, 1]},
            'window_length_sec': 2.0,  # 1000 samples / 500 Hz = 2.0 sec
            'step_size_ms': 100
        }
        mode = AsynchronousMode(
            data_queue=mock_queues['data_queue'],
            output_queue=mock_queues['output_queue'],
            stop_event=mock_queues['stop_event'],
            instance_config=config,
            sample_rate=500.0
        )

        assert mode._validate_configuration() == True
    
    def test_validate_configuration_zero_epoch_length(self, mock_queues):
        """Test configuration validation with zero epoch length."""
        config = {
            'name': 'test_async',
            'mode': 'asynchronous',
            'decoder_config': {'model_config': {'model_type': 'EEGNet', 'num_classes': 2}},
            'event_mapping': {1: 'test_event'},
            'window_length_sec': 0.0  # Will result in 0 epoch_length_samples
        }
        mode = AsynchronousMode(
            data_queue=mock_queues['data_queue'],
            output_queue=mock_queues['output_queue'],
            stop_event=mock_queues['stop_event'],
            instance_config=config,
            sample_rate=500.0
        )
        mode.logger = Mock()

        assert mode._validate_configuration() == False
        mode.logger.error.assert_called()
    
    def test_validate_configuration_zero_prediction_step(self, mock_queues):
        """Test configuration validation with zero prediction step."""
        config = {
            'name': 'test_async',
            'mode': 'asynchronous',
            'decoder_config': {'model_config': {'model_type': 'EEGNet', 'num_classes': 2}},
            'event_mapping': {1: 'test_event'},
            'step_size_ms': 0  # Will result in 0 samples_per_prediction_step
        }
        mode = AsynchronousMode(
            data_queue=mock_queues['data_queue'],
            output_queue=mock_queues['output_queue'],
            stop_event=mock_queues['stop_event'],
            instance_config=config,
            sample_rate=500.0
        )
        mode.logger = Mock()

        assert mode._validate_configuration() == False
        mode.logger.error.assert_called()
    
    def test_validate_configuration_no_decoder_config(self, mock_queues):
        """Test mode handles missing decoder_config gracefully."""
        config = {
            'name': 'test_async',
            'mode': 'asynchronous',
            # No decoder_config key at all
            'event_mapping': {1: 'test_event'}
        }
        # Mode handles missing decoder_config with empty dict default
        mode = AsynchronousMode(
            data_queue=mock_queues['data_queue'],
            output_queue=mock_queues['output_queue'],
            stop_event=mock_queues['stop_event'],
            instance_config=config,
            sample_rate=500.0
        )
        assert mode.decoder_config == {}
    
    def test_validate_configuration_no_channel_selection(self, mock_queues):
        """Test configuration validation without modality config."""
        config = {
            'name': 'test_async',
            'mode': 'asynchronous',
            'decoder_config': {'model_config': {'model_type': 'EEGNet', 'num_classes': 2}},
            'event_mapping': {1: 'test_event'},
            'channel_selection': None
        }
        mode = AsynchronousMode(
            data_queue=mock_queues['data_queue'],
            output_queue=mock_queues['output_queue'],
            stop_event=mock_queues['stop_event'],
            instance_config=config,
            sample_rate=500.0
        )
        mode.logger = Mock()

        assert mode._validate_configuration() == False
        mode.logger.error.assert_called()


class TestAsynchronousModeDataProcessing:
    """Test suite for continuous data processing in AsynchronousMode."""
    
    @pytest.fixture
    def async_mode(self):
        """Create an AsynchronousMode instance for testing."""
        config = {
            'name': 'test_async',
            'mode': 'asynchronous',
            'decoder_config': {'model_config': {'model_type': 'EEGNet', 'num_classes': 2}},
            'event_mapping': {1: 'left_hand', 2: 'right_hand'},
            'channel_selection': {'EEG': [0, 1, 2, 3]},
            'window_length_sec': 1.0,  # 500 samples / 500 Hz = 1.0 sec
            'step_size_ms': 100
        }
        mode = AsynchronousMode(
            data_queue=Mock(),
            output_queue=Mock(),
            stop_event=Mock(),
            instance_config=config,
            sample_rate=500.0
        )
        mode.logger = Mock()
        mode.buffer = Mock()
        mode._trigger_prediction = Mock()
        mode._check_for_model_updates = Mock()
        mode.stop_event.is_set = Mock(return_value=False)  # Ensure stop event is not set
        mode.model = Mock()  # Add model to avoid sync mode check

        # Initialize attributes that are normally set in _initialize_mode
        mode.continuous_labels = []
        mode.max_continuous_labels = 10000
        mode._current_label = -1
        mode._active_label = -1
        mode._labeling_samples_remaining = 0

        return mode

    def test_process_data_valid(self, async_mode):
        """Test processing of valid continuous data."""
        sample_chunk = {
            'eeg': np.array([1, 2, 3, 4]),
            'markers': 1
        }

        async_mode._process_data(sample_chunk)

        # Check buffer add_sample was called
        async_mode.buffer.add_sample.assert_called_once_with(sample_chunk)

        # Check sample index incremented
        assert async_mode.current_sample_index == 1

        # Check current label is set (event code 1 maps to class 0)
        assert async_mode._current_label == 0

    def test_process_data_no_markers(self, async_mode):
        """Test processing data without markers."""
        sample_chunk = {
            'eeg': np.array([1, 2, 3, 4])
            # No markers
        }

        async_mode._process_data(sample_chunk)

        # Buffer should still be called
        async_mode.buffer.add_sample.assert_called_once_with(sample_chunk)
        # No event, so label remains -1
        assert async_mode._current_label == -1

    def test_process_data_invalid_marker(self, async_mode):
        """Test processing data with invalid marker."""
        sample_chunk = {
            'eeg': np.array([1, 2, 3, 4]),
            'markers': 'invalid'
        }

        async_mode._process_data(sample_chunk)

        # Buffer should still be called
        async_mode.buffer.add_sample.assert_called_once_with(sample_chunk)
        # Invalid marker treated as -1, so label remains -1
        assert async_mode._current_label == -1

    def test_process_data_invalid_input(self, async_mode):
        """Test processing invalid input."""
        # Non-dict input
        async_mode._process_data("invalid")

        # Should not process anything
        async_mode.buffer.add_sample.assert_not_called()

    def test_process_data_multiple_samples(self, async_mode):
        """Test processing multiple samples updates index correctly."""
        # Process multiple samples
        for i in range(5):
            sample_chunk = {'eeg': np.array([i]), 'markers': 1}
            async_mode._process_data(sample_chunk)

        # Check sample index incremented for each sample
        assert async_mode.current_sample_index == 5
        # Check buffer add_sample was called for each
        assert async_mode.buffer.add_sample.call_count == 5


class TestAsynchronousModeEventLabeling:
    """Test event labeling state machine in async mode."""

    @pytest.fixture
    def async_mode_labeling(self):
        """Create async mode for labeling tests (no buffer mock)."""
        config = {
            'name': 'test_async',
            'mode': 'asynchronous',
            'decoder_config': {'model_config': {'model_type': 'EEGNet', 'num_classes': 2}},
            'event_mapping': {1: 'left_hand', 2: 'right_hand'},
            'channel_selection': {'eeg': [0, 1, 2, 3]},
            'window_length_sec': 1.0,
            'step_size_ms': 100
        }
        mode = AsynchronousMode(
            data_queue=Mock(),
            output_queue=Mock(),
            stop_event=Mock(),
            instance_config=config,
            sample_rate=500.0
        )
        mode.logger = Mock()
        mode.stop_event.is_set = Mock(return_value=False)
        mode._check_for_model_updates = Mock()

        # Initialize buffer properly (not mocked)
        mode._setup_buffer = Mock()
        mode.buffer = Mock()
        mode.buffer.add_sample = Mock(return_value=True)

        # Initialize labeling state
        mode._current_label = -1
        mode._active_label = -1
        mode._labeling_samples_remaining = 0
        mode.metrics_manager = None

        return mode

    def test_event_sets_label_for_duration(self, async_mode_labeling):
        """Test that event sets label for epoch_length_samples."""
        mode = async_mode_labeling
        # epoch_length_samples = 500 (1.0 sec * 500 Hz)
        sample_with_event = {'eeg': np.array([1]), 'markers': 1}
        sample_no_event = {'eeg': np.array([1]), 'markers': -1}

        # Process event sample - sets label
        mode._process_data(sample_with_event)
        assert mode._current_label == 0  # class 0 for event code 1
        assert mode._labeling_samples_remaining == 499  # decremented once

        # Process 498 more samples (total 499 processed)
        for _ in range(498):
            mode._process_data(sample_no_event)
        assert mode._current_label == 0  # Still labeled
        assert mode._labeling_samples_remaining == 1  # One left

        # Process one more - should still be labeled (last sample)
        mode._process_data(sample_no_event)
        assert mode._current_label == 0  # Last labeled sample
        assert mode._labeling_samples_remaining == 0

        # Process next sample - label expires
        mode._process_data(sample_no_event)
        assert mode._current_label == -1  # Expired

    def test_new_event_resets_duration(self, async_mode_labeling):
        """Test that new event resets the labeling counter."""
        mode = async_mode_labeling
        sample_event_1 = {'eeg': np.array([1]), 'markers': 1}
        sample_event_2 = {'eeg': np.array([1]), 'markers': 2}
        sample_no_event = {'eeg': np.array([1]), 'markers': -1}

        # Process first event
        mode._process_data(sample_event_1)
        assert mode._current_label == 0  # class 0 for event code 1

        # Process some samples
        for _ in range(100):
            mode._process_data(sample_no_event)
        assert mode._current_label == 0
        assert mode._labeling_samples_remaining == 399  # 500 - 1 - 100

        # New event overwrites and resets counter
        mode._process_data(sample_event_2)
        assert mode._current_label == 1  # class 1 for event code 2
        assert mode._labeling_samples_remaining == 499  # Reset

    def test_unknown_event_code_ignored(self, async_mode_labeling):
        """Test that unknown event codes don't set labels."""
        mode = async_mode_labeling
        sample_unknown = {'eeg': np.array([1]), 'markers': 99}  # Not in event_mapping

        mode._process_data(sample_unknown)
        assert mode._current_label == -1  # No label set
        assert mode._labeling_samples_remaining == 0


class TestAsynchronousModePrediction:
    """Test suite for prediction functionality in AsynchronousMode."""
    
    @pytest.fixture
    def async_mode_with_model(self):
        """Create AsynchronousMode with model setup."""
        config = {
            'name': 'test_async',
            'mode': 'asynchronous',
            'decoder_config': {'model_config': {'model_type': 'EEGNet', 'num_classes': 2}},
            'event_mapping': {1: 'left_hand', 2: 'right_hand'},
            'channel_selection': {'EEG': [0, 1]},
            'window_length_sec': 1.0,  # 500 samples / 500 Hz = 1.0 sec
            'step_size_ms': 100
        }
        mode = AsynchronousMode(
            data_queue=Mock(),
            output_queue=Mock(),
            stop_event=Mock(),
            instance_config=config,
            sample_rate=500.0
        )
        mode.logger = Mock()
        mode.decoder = Mock()
        mode.decoder.is_fitted = True  # Model is ready by default
        mode.decoder.predict_sample = Mock(return_value=(0, 0.8))
        mode._model_lock = threading.Lock()
        mode.buffer = Mock()
        mode._get_true_label_from_events = Mock(return_value=(0, None))
        mode._update_metrics_and_send = Mock()

        # Initialize attributes that are normally set in _initialize_mode
        mode.continuous_labels = []
        mode.max_continuous_labels = 10000
        mode._current_label = -1
        mode._active_label = -1
        mode._labeling_samples_remaining = 0

        return mode

    def test_trigger_prediction_success(self, async_mode_with_model):
        """Test successful prediction triggering."""
        # Mock buffer extraction
        X_input = {'eeg': np.array([[1, 2, 3], [4, 5, 6]])}
        async_mode_with_model.buffer.extract_window.return_value = X_input

        # Mock prediction
        async_mode_with_model.decoder.predict_sample.return_value = (1, 0.8)

        async_mode_with_model._trigger_prediction()

        # Check prediction was made (base_mode._predict extracts array from dict)
        async_mode_with_model.decoder.predict_sample.assert_called_once()
        actual_call_arg = async_mode_with_model.decoder.predict_sample.call_args[0][0]
        np.testing.assert_array_equal(actual_call_arg, X_input['eeg'])

        # Check metrics and outputs were updated
        async_mode_with_model._update_metrics_and_send.assert_called_once_with(
            1, 0.8  # prediction, confidence
        )

        # Check prediction count incremented
        assert async_mode_with_model.prediction_count == 1
    
    def test_trigger_prediction_no_model(self, async_mode_with_model):
        """Test prediction triggering without model."""
        async_mode_with_model.decoder = None

        async_mode_with_model._trigger_prediction()
        
        # Should not crash, should not process anything
        async_mode_with_model.buffer.extract_window.assert_not_called()
        async_mode_with_model._update_metrics_and_send.assert_not_called()
    
    def test_trigger_prediction_model_not_ready(self, async_mode_with_model):
        """Test prediction triggering when model not ready."""
        # Mock model as not fitted (readiness check is in base class now)
        async_mode_with_model.decoder.is_fitted = False

        async_mode_with_model._trigger_prediction()

        # Should not make prediction (early return before buffer extraction)
        async_mode_with_model.buffer.extract_window.assert_not_called()
        async_mode_with_model._update_metrics_and_send.assert_not_called()
    
    def test_trigger_prediction_no_input_data(self, async_mode_with_model):
        """Test prediction triggering with no input data."""
        async_mode_with_model.buffer.extract_window.return_value = None

        async_mode_with_model._trigger_prediction()

        # Should not make prediction
        async_mode_with_model.decoder.predict_sample.assert_not_called()
        async_mode_with_model._update_metrics_and_send.assert_not_called()
    
    def test_predict_success(self, async_mode_with_model):
        """Test successful prediction using base class _predict."""
        async_mode_with_model.decoder.predict_sample.return_value = (1, 0.75)

        X_input = {'eeg': np.array([[1, 2, 3]])}
        prediction, confidence, _ = async_mode_with_model._predict(
            model=async_mode_with_model.decoder,
            X_input=X_input,
            lock=async_mode_with_model._model_lock,
            blocking=True
        )

        assert prediction == 1
        assert confidence == 0.75
        async_mode_with_model.decoder.predict_sample.assert_called_once()
        actual_call_arg = async_mode_with_model.decoder.predict_sample.call_args[0][0]
        np.testing.assert_array_equal(actual_call_arg, X_input['eeg'])

    def test_predict_no_model(self, async_mode_with_model):
        """Test prediction when no model exists."""
        X_input = {'eeg': np.array([[1, 2, 3]])}
        prediction, confidence, _ = async_mode_with_model._predict(
            model=None,
            X_input=X_input,
            lock=async_mode_with_model._model_lock,
            blocking=True
        )

        assert prediction == 0
        assert confidence == 0.5

    def test_predict_no_predict_method(self, async_mode_with_model):
        """Test prediction when model lacks predict_sample method."""
        delattr(async_mode_with_model.decoder, 'predict_sample')

        X_input = {'eeg': np.array([[1, 2, 3]])}
        prediction, confidence, _ = async_mode_with_model._predict(
            model=async_mode_with_model.decoder,
            X_input=X_input,
            lock=async_mode_with_model._model_lock,
            blocking=True
        )

        assert prediction == 0
        assert confidence == 0.5

    def test_predict_model_not_ready(self, async_mode_with_model):
        """Test prediction when model not ready."""
        async_mode_with_model.decoder.is_fitted = False

        X_input = {'eeg': np.array([[1, 2, 3]])}
        prediction, confidence, _ = async_mode_with_model._predict(
            model=async_mode_with_model.decoder,
            X_input=X_input,
            lock=async_mode_with_model._model_lock,
            blocking=True
        )

        assert prediction == 0
        assert confidence == 0.5
        async_mode_with_model.decoder.predict_sample.assert_not_called()
    
    def test_is_model_ready_various_attributes(self, async_mode_with_model):
        """Test model readiness checking with various model attributes."""
        # Model readiness is now checked inline using: getattr(decoder, 'is_fitted', True)
        def is_model_ready(decoder):
            """Helper to check model readiness using the same pattern as production code."""
            return getattr(decoder, 'is_fitted', True) if decoder else False

        # Test with is_fitted attribute
        async_mode_with_model.decoder.is_fitted = True
        assert is_model_ready(async_mode_with_model.decoder) == True

        async_mode_with_model.decoder.is_fitted = False
        assert is_model_ready(async_mode_with_model.decoder) == False

        # Test fallback behavior when is_fitted doesn't exist (should default to True)
        delattr(async_mode_with_model.decoder, 'is_fitted')
        assert is_model_ready(async_mode_with_model.decoder) == True


class TestAsynchronousModeModelUpdates:
    """Test suite for model updating from sync modes."""
    
    @pytest.fixture
    def async_mode_sync_source(self):
        """Create AsynchronousMode with sync mode source."""
        config = {
            'name': 'test_async',
            'mode': 'asynchronous',
            'decoder_config': {'model_config': {'model_type': 'EEGNet', 'num_classes': 2}},
            'event_mapping': {1: 'test_event'},
            'channel_selection': {'EEG': [0, 1]},
            'decoder_source': 'sync_mode',
            'source_sync_mode': 'test_sync',
            'study_name': 'test_study',
            'file_identifier': 'session_1'
        }
        mode = AsynchronousMode(
            data_queue=Mock(),
            output_queue=Mock(),
            stop_event=Mock(),
            instance_config=config,
            sample_rate=500.0
        )
        mode.logger = Mock()
        mode.decoder = Mock()
        mode.decoder.input_shapes = {'EEG': (32, 500)}  # Mock input shapes for epoch length update
        mode._model_lock = threading.Lock()
        mode.last_model_check_time = 0
        mode.model_check_interval = 1  # Short interval for testing
        mode._setup_buffer = Mock()  # Mock buffer setup method
        return mode
    
    @patch('dendrite.processing.modes.asynchronous_mode.os.path.exists')
    @patch('dendrite.processing.modes.asynchronous_mode.os.path.getmtime')
    @patch('dendrite.processing.modes.asynchronous_mode.time.time')
    def test_check_for_model_updates_new_model(self, mock_time, mock_getmtime, mock_exists, async_mode_sync_source):
        """Test checking for model updates when new model available."""
        from dendrite import DATA_DIR

        mock_time.return_value = 10.0  # Current time
        mock_exists.return_value = True
        mock_getmtime.return_value = 9.0  # Model timestamp
        async_mode_sync_source.last_model_modification_time = 5.0  # Older timestamp

        async_mode_sync_source._load_updated_model = Mock()

        async_mode_sync_source._check_for_model_updates()

        # Should detect and load new model (checks .json file with full path)
        # New structure: studies/{study}/decoders/shared/{mode}_{file}_latest
        relative_path = 'shared/test_sync_session_1_latest'
        full_path = DATA_DIR / "studies" / "test_study" / "decoders" / relative_path
        expected_json_path = f'{full_path}.json'
        mock_exists.assert_called_once_with(expected_json_path)
        mock_getmtime.assert_called_once_with(expected_json_path)
        async_mode_sync_source._load_updated_model.assert_called_once_with(str(full_path))
        assert async_mode_sync_source.last_model_modification_time == 9.0
    
    @patch('dendrite.processing.modes.asynchronous_mode.os.path.exists')
    @patch('dendrite.processing.modes.asynchronous_mode.time.time')
    def test_check_for_model_updates_no_model_file(self, mock_time, mock_exists, async_mode_sync_source):
        """Test checking for model updates when no model file exists."""
        mock_time.return_value = 10.0
        mock_exists.return_value = False
        
        async_mode_sync_source._load_updated_model = Mock()
        
        async_mode_sync_source._check_for_model_updates()
        
        # Should not try to load anything
        async_mode_sync_source._load_updated_model.assert_not_called()
        assert async_mode_sync_source.last_model_check_time == 10.0
    
    @patch('dendrite.processing.modes.asynchronous_mode.time.time')
    def test_check_for_model_updates_too_soon(self, mock_time, async_mode_sync_source):
        """Test checking for model updates when called too soon."""
        mock_time.return_value = 0.5  # Less than model_check_interval
        async_mode_sync_source.last_model_check_time = 0.0
        
        async_mode_sync_source._check_for_model_updates()
        
        # Should return early without checking
        assert async_mode_sync_source.last_model_check_time == 0.0  # Unchanged
    
    def test_check_for_model_updates_pretrained_mode(self, async_mode_sync_source):
        """Test model updates check in pretrained mode."""
        async_mode_sync_source.decoder_source = 'pretrained'
        
        async_mode_sync_source._check_for_model_updates()
        
        # Should not check for updates in pretrained mode
        # (No specific assertions needed, just ensure no crash)
    
    def test_load_updated_model_success(self, async_mode_sync_source):
        """Test successful updated model loading."""
        model_path = '/path/to/model.json'
        mock_decoder = MagicMock()
        mock_decoder.is_fitted = True

        with patch('dendrite.ml.decoders.load_decoder', return_value=mock_decoder) as mock_load:
            with patch('os.path.exists', return_value=True):
                async_mode_sync_source._load_updated_model(model_path)

            mock_load.assert_called_once_with(model_path)
            # Check that success logging occurred
            async_mode_sync_source.logger.info.assert_any_call(
                'Successfully loaded updated'
            )

    def test_load_updated_model_failure(self, async_mode_sync_source):
        """Test failed updated model loading."""
        model_path = '/path/to/model.json'

        with patch('dendrite.ml.decoders.load_decoder', side_effect=RuntimeError("Load failed")):
            with patch('os.path.exists', return_value=True):
                async_mode_sync_source._load_updated_model(model_path)

            async_mode_sync_source.logger.error.assert_called()

    def test_load_updated_model_file_not_found(self, async_mode_sync_source):
        """Test loading when model file doesn't exist."""
        model_path = '/path/to/model.json'

        with patch('dendrite.ml.decoders.load_decoder', side_effect=FileNotFoundError("Not found")):
            with patch('os.path.exists', return_value=True):
                async_mode_sync_source._load_updated_model(model_path)

            async_mode_sync_source.logger.error.assert_called()


class TestAsynchronousModeMetrics:
    """Test suite for metrics handling in AsynchronousMode."""
    
    @pytest.fixture
    def async_mode_with_metrics(self):
        """Create AsynchronousMode with metrics setup."""
        # Use 0-indexed keys to match ML model class indices
        config = {
            'name': 'test_async',
            'mode': 'asynchronous',
            'decoder_config': {'model_config': {'model_type': 'EEGNet', 'num_classes': 2}},
            'event_mapping': {0: 'left_hand', 1: 'right_hand'},
            'channel_selection': {'EEG': [0, 1]}
        }
        mode = AsynchronousMode(
            data_queue=Mock(),
            output_queue=Mock(),
            stop_event=Mock(),
            instance_config=config,
            sample_rate=500.0
        )
        mode.logger = Mock()
        mode.metrics_manager = Mock()
        mode.output_queue = Mock()
        mode.prediction_queue = Mock()
        mode._get_true_label_from_events = Mock(return_value=(1, None))
        mode.continuous_labels = [0, 1, 0, 1, 0]
        mode.current_sample_index = 100
        mode._current_label = -1
        mode.num_classes = 2
        return mode
    
    def test_update_metrics_and_send_with_temporal(self, async_mode_with_metrics):
        """Test metrics update with temporal context."""
        # Setup mock metrics
        mock_metrics = {
            'prequential_accuracy': 0.75,
            'balanced_accuracy': 0.80,
        }
        async_mode_with_metrics.metrics_manager.get_current_metrics.return_value = mock_metrics
        
        async_mode_with_metrics._update_metrics_and_send(0, 0.7)  # prediction, confidence

        # Check metrics manager was called with correct args
        call_args = async_mode_with_metrics.metrics_manager.update_metrics.call_args
        assert call_args[1]['prediction'] == 0
        assert call_args[1]['true_label'] == -1  # _current_label from fixture setup
        assert call_args[1]['current_sample_idx'] == 100  # From fixture
    
    def test_update_metrics_and_send_prediction_payload(self, async_mode_with_metrics):
        """Test prediction payload creation and sending."""
        mock_metrics = {'prequential_accuracy': 0.75}
        async_mode_with_metrics.metrics_manager.get_current_metrics.return_value = mock_metrics

        async_mode_with_metrics._update_metrics_and_send(0, 0.8)

        # Check queues received data (prediction and visualization)
        async_mode_with_metrics.prediction_queue.put.assert_called_once()
        async_mode_with_metrics.output_queue.put.assert_called_once()

        # Check prediction queue payload
        pred_call_args = async_mode_with_metrics.prediction_queue.put.call_args[0][0]
        assert pred_call_args["type"] == 'prediction'
        assert pred_call_args["data"]['event_name'] == 'left_hand'
        assert pred_call_args["data"]['confidence'] == 0.8
        assert pred_call_args["data"]['prediction'] == 0
    
    def test_update_metrics_and_send_visualization_payload(self, async_mode_with_metrics):
        """Test visualization payload creation and sending."""
        mock_metrics = {
            'balanced_accuracy': 0.75,
            'overall_temporal_accuracy': 0.80,
            'stable_window_accuracy': 0.85
        }
        async_mode_with_metrics.metrics_manager.get_current_metrics.return_value = mock_metrics

        async_mode_with_metrics._update_metrics_and_send(0, 0.8)

        # Check queues received data (prediction and visualization)
        async_mode_with_metrics.prediction_queue.put.assert_called_once()
        async_mode_with_metrics.output_queue.put.assert_called_once()

        # Check visualization payload (output_queue)
        viz_call_args = async_mode_with_metrics.output_queue.put.call_args[0][0]
        assert viz_call_args["type"] == 'performance'
        assert viz_call_args["data"]['prediction'] == 0
        assert viz_call_args["data"]['confidence'] == 0.8
        assert viz_call_args["data"]['event_name'] == 'left_hand'
        assert viz_call_args["data"]['balanced_accuracy'] == 0.75
    
    def test_update_metrics_and_send_no_metrics_manager(self, async_mode_with_metrics):
        """Test metrics update when no metrics manager exists."""
        async_mode_with_metrics.metrics_manager = None

        async_mode_with_metrics._update_metrics_and_send(0, 0.8)

        # Should still send payloads with default values
        async_mode_with_metrics.prediction_queue.put.assert_called_once()
        async_mode_with_metrics.output_queue.put.assert_called_once()
    
    def test_get_true_label_from_events(self, async_mode_with_metrics):
        """Test true label extraction from events."""
        async_mode_with_metrics._current_label = 1
        
        true_label, event_type = async_mode_with_metrics._get_true_label_from_events()
        
        assert true_label == 1
        assert event_type is None


class TestAsynchronousModeMainLoop:
    """Test suite for main loop functionality in AsynchronousMode."""
    
    @pytest.fixture
    def async_mode_for_loop(self):
        """Create AsynchronousMode for main loop testing."""
        config = {
            'name': 'test_async',
            'mode': 'asynchronous',
            'decoder_config': {'model_config': {'model_type': 'EEGNet', 'num_classes': 2}},
            'event_mapping': {1: 'test_event'},
            'channel_selection': {'EEG': [0, 1]}
        }
        mode = AsynchronousMode(
            data_queue=Queue(),
            output_queue=Mock(),
            stop_event=threading.Event(),
            instance_config=config,
            sample_rate=500.0
        )
        mode.logger = Mock()
        mode.buffer = Mock()
        mode.buffer.is_ready_for_step.return_value = False  # Default: not ready
        mode._process_data = Mock()
        mode._trigger_prediction = Mock()
        mode._check_for_model_updates = Mock()
        return mode
    
    def test_run_main_loop_data_processing(self, async_mode_for_loop):
        """Test main loop data processing."""
        # Add sample to queue
        test_sample = {'EEG': np.array([1, 2, 3])}
        async_mode_for_loop.data_queue.put(test_sample)
        
        # Stop after processing one sample
        def stop_after_first_sample(*args):
            async_mode_for_loop.stop_event.set()
        
        async_mode_for_loop._process_data.side_effect = stop_after_first_sample
        
        async_mode_for_loop._run_main_loop()
        
        # Should process the sample
        async_mode_for_loop._process_data.assert_called_once_with(test_sample)
        async_mode_for_loop._check_for_model_updates.assert_called()
    
    def test_run_main_loop_prediction_trigger(self, async_mode_for_loop):
        """Test main loop prediction triggering."""
        # Setup buffer to be ready for prediction
        async_mode_for_loop.buffer.is_ready_for_step.return_value = True
        
        # Add sample and stop after processing
        test_sample = {'EEG': np.array([1, 2, 3])}
        async_mode_for_loop.data_queue.put(test_sample)
        
        def stop_after_trigger(*args):
            async_mode_for_loop.stop_event.set()
        
        async_mode_for_loop._trigger_prediction.side_effect = stop_after_trigger
        
        async_mode_for_loop._run_main_loop()
        
        # Should trigger prediction
        async_mode_for_loop._trigger_prediction.assert_called_once()
    
    def test_run_main_loop_empty_queue(self, async_mode_for_loop):
        """Test main loop handling of empty queue."""
        # Immediately stop to avoid infinite loop
        async_mode_for_loop.stop_event.set()
        
        async_mode_for_loop._run_main_loop()
        
        # Should not process any samples
        async_mode_for_loop._process_data.assert_not_called()


class TestAsynchronousModeCleanup:
    """Test suite for AsynchronousMode cleanup functionality."""
    
    def test_cleanup_with_metrics(self):
        """Test cleanup with metrics manager."""
        config = {
            'name': 'test_async',
            'mode': 'asynchronous',
            'decoder_config': {'model_config': {'model_type': 'EEGNet', 'num_classes': 2}},
            'event_mapping': {1: 'test_event'}
        }
        mode = AsynchronousMode(
            data_queue=Mock(),
            output_queue=Mock(),
            stop_event=Mock(),
            instance_config=config,
            sample_rate=500.0
        )

        # Mock metrics and components
        mode.metrics_manager = Mock()
        mode.prediction_count = 500
        mode.logger = Mock()

        mode._cleanup()

        # Check logging calls
        assert mode.logger.info.call_count >= 2  # Cleanup message + prediction count

        # Check prediction count was logged
        log_calls = [call[0][0] for call in mode.logger.info.call_args_list]
        prediction_log = next((log for log in log_calls if 'Total predictions made: 500' in log), None)
        assert prediction_log is not None
    
    def test_cleanup_no_metrics(self):
        """Test cleanup without metrics manager."""
        config = {
            'name': 'test_async',
            'mode': 'asynchronous',
            'decoder_config': {'model_config': {'model_type': 'EEGNet', 'num_classes': 2}},
            'event_mapping': {1: 'test_event'}
        }
        mode = AsynchronousMode(
            data_queue=Mock(),
            output_queue=Mock(),
            stop_event=Mock(),
            instance_config=config,
            sample_rate=500.0
        )

        mode.metrics_manager = None
        mode.prediction_count = 100
        mode.logger = Mock()

        # Should not crash
        mode._cleanup()

        # Should still log basic cleanup info
        mode.logger.info.assert_called()


class TestAsynchronousModeCreateDecoder:
    """Test suite for decoder creation in AsynchronousMode."""
    
    @pytest.fixture
    def async_mode_for_decoder(self):
        """Create AsynchronousMode for decoder testing."""
        config = {
            'name': 'test_async',
            'mode': 'asynchronous',
            'decoder_config': {
                'model_config': {
                    'model_type': 'EEGNet',
                    'num_classes': 3,
                    'learning_rate': 0.002
                }
            },
            'event_mapping': {1: 'test_event'}
        }
        mode = AsynchronousMode(
            data_queue=Mock(),
            output_queue=Mock(),
            stop_event=Mock(),
            instance_config=config,
            sample_rate=500.0
        )
        mode.logger = Mock()
        return mode
    
    @patch('dendrite.ml.decoders.create_decoder')  # Patch at source, not where it's imported
    def test_create_decoder_success(self, mock_create_decoder, async_mode_for_decoder):
        """Test successful decoder creation."""
        mock_decoder = Mock()
        mock_create_decoder.return_value = mock_decoder

        decoder = async_mode_for_decoder._create_decoder(async_mode_for_decoder.decoder_config)

        assert decoder == mock_decoder

        # Check create_decoder was called with correct parameters
        mock_create_decoder.assert_called_once_with(
            model_type='EEGNet',
            num_classes=3,
            learning_rate=0.002
        )
    
    @patch('dendrite.ml.decoders.create_decoder')  # Patch at source
    def test_create_decoder_with_defaults(self, mock_create_decoder, async_mode_for_decoder):
        """Test decoder creation with default configuration."""
        # Minimal config
        minimal_config = {'model_config': {'model_type': 'EEGNet', 'num_classes': 2}}

        mock_decoder = Mock()
        mock_create_decoder.return_value = mock_decoder

        decoder = async_mode_for_decoder._create_decoder(minimal_config)

        # Check defaults were used (only num_classes is set as default)
        call_kwargs = mock_create_decoder.call_args[1]
        # Model source is 'pretrained' by default, so uses default of 2
        assert call_kwargs['num_classes'] == 2
        mock_create_decoder.assert_called_once()
    
    @patch('dendrite.ml.decoders.create_decoder')  # Patch at source (imported locally in base_mode)
    def test_create_decoder_exception(self, mock_create_decoder, async_mode_for_decoder):
        """Test decoder creation with exception."""
        mock_create_decoder.side_effect = Exception("Creation failed")

        decoder = async_mode_for_decoder._create_decoder(async_mode_for_decoder.decoder_config)

        assert decoder is None
        async_mode_for_decoder.logger.error.assert_called()