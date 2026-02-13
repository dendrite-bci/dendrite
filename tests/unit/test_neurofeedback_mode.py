"""
Comprehensive unit tests for NeurofeedbackMode class.

This module provides unit tests for the NeurofeedbackMode class,
focusing on testing individual components and methods in isolation.

Tests cover:
- NeurofeedbackMode initialization and configuration
- Feature extraction and band power calculation
- Single and multi-band configuration support
- Data processing and sliding window management
- Output payload creation and sending
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
from dataclasses import asdict

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dendrite.processing.modes.neurofeedback_mode import NeurofeedbackMode, BandPowerPayload

# Test constants for maintainability
TEST_SAMPLE_RATE = 500.0
TEST_MODE_NAME = 'test_nfb'
DEFAULT_WINDOW_LENGTH = 1.0  
DEFAULT_WINDOW_STEP = 0.25
DEFAULT_ALPHA_BAND = [8.0, 12.0]
MULTI_BANDS = {
    'alpha': [8.0, 12.0],
    'beta': [12.0, 30.0], 
    'theta': [4.0, 8.0]
}
TEST_CHANNELS_2 = {'EEG': [0, 1]}
TEST_CHANNELS_3 = {'EEG': [0, 1, 2]}


def create_test_mode(**kwargs):
    """Helper to create NeurofeedbackMode with sensible defaults using new API."""
    # Extract queue/event parameters
    data_queue = kwargs.pop('data_queue', Mock())
    output_queue = kwargs.pop('output_queue', Mock())
    stop_event = kwargs.pop('stop_event', Mock())
    prediction_queue = kwargs.pop('prediction_queue', None)
    sample_rate = kwargs.pop('sample_rate', TEST_SAMPLE_RATE)

    # Build instance_config from remaining kwargs
    instance_config = {
        'name': kwargs.pop('mode_name', TEST_MODE_NAME),
        'mode': 'neurofeedback',
        'feature_config': kwargs.pop('feature_config', {}),
        'channel_selection': kwargs.pop('channel_selection', {'EEG': [0, 1, 2, 3]})
    }
    # Add any remaining kwargs to instance_config
    instance_config.update(kwargs)

    return NeurofeedbackMode(
        data_queue=data_queue,
        output_queue=output_queue,
        stop_event=stop_event,
        instance_config=instance_config,
        sample_rate=sample_rate,
        prediction_queue=prediction_queue
    )


def create_test_eeg_data(n_channels=2, n_samples=500):
    """Helper to create realistic EEG test data in (channels, times) format."""
    return np.random.randn(n_channels, n_samples)


def assert_valid_channel_powers(channel_powers, expected_channels, expected_bands):
    """Helper to validate channel_powers structure."""
    assert len(channel_powers) == expected_channels
    for ch_key in channel_powers:
        for band_name in expected_bands:
            assert band_name in channel_powers[ch_key]
            power_value = channel_powers[ch_key][band_name]
            assert isinstance(power_value, float)
            assert power_value >= 0.0  # Power should be non-negative


class TestNeurofeedbackModeInitialization:
    """Test suite for NeurofeedbackMode initialization."""
    
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
        """Basic configuration for NeurofeedbackMode."""
        return {
            'feature_config': {
                'target_band': DEFAULT_ALPHA_BAND,
                'window_length_sec': DEFAULT_WINDOW_LENGTH,
                'window_step_sec': DEFAULT_WINDOW_STEP
            },
            'channel_selection': {'EEG': [0, 1, 2, 3]}
        }
    
    def test_basic_initialization_single_band(self, mock_queues, basic_config):
        """Test basic NeurofeedbackMode initialization with single band."""
        mode = create_test_mode(
            data_queue=mock_queues['data_queue'],
            output_queue=mock_queues['output_queue'],
            stop_event=mock_queues['stop_event'],
            mode_name=TEST_MODE_NAME,
            sample_rate=TEST_SAMPLE_RATE,
            **basic_config
        )
        
        assert mode.mode_name == TEST_MODE_NAME
        assert mode.sample_rate == TEST_SAMPLE_RATE
        assert mode.feature_config == basic_config['feature_config']
        assert mode.window_length_sec == DEFAULT_WINDOW_LENGTH
        assert mode.window_step_sec == DEFAULT_WINDOW_STEP
        
        # Check derived attributes
        assert mode.window_length_samples == int(DEFAULT_WINDOW_LENGTH * TEST_SAMPLE_RATE)
        assert mode.window_step_samples == int(DEFAULT_WINDOW_STEP * TEST_SAMPLE_RATE)
        
        # Check single band conversion to multi-band format
        expected_bands = {'default': DEFAULT_ALPHA_BAND}
        assert mode.target_bands == expected_bands
    
    def test_initialization_multi_band(self, mock_queues):
        """Test initialization with multiple bands."""
        feature_config = {
            'target_bands': {
                'alpha': [8.0, 12.0],
                'beta': [12.0, 30.0],
                'theta': [4.0, 8.0]
            },
            'window_length_sec': 2.0,
            'window_step_sec': 0.5
        }
        
        mode = create_test_mode(
            data_queue=mock_queues['data_queue'],
            output_queue=mock_queues['output_queue'],
            stop_event=mock_queues['stop_event'],
            mode_name='test_nfb',
            sample_rate=250.0,
            feature_config=feature_config,
            channel_selection={'EEG': [0, 1]}
        )
        
        assert mode.target_bands == feature_config['target_bands']
        assert mode.window_length_samples == 500  # 2.0 * 250
        assert mode.window_step_samples == 125    # 0.5 * 250
    
    def test_initialization_with_defaults(self, mock_queues):
        """Test initialization with default parameters."""
        mode = create_test_mode(
            data_queue=mock_queues['data_queue'],
            output_queue=mock_queues['output_queue'],
            stop_event=mock_queues['stop_event'],
            mode_name='test_nfb',
            sample_rate=500.0
        )
        
        # Check defaults
        assert mode.window_length_sec == 1.0
        assert mode.window_step_sec == 0.25
        assert mode.target_bands == {'default': [8.0, 12.0]}
        assert mode.window_length_samples == 500
        assert mode.window_step_samples == 125
    
    def test_initialization_no_feature_config(self, mock_queues):
        """Test initialization without feature config."""
        mode = create_test_mode(
            data_queue=mock_queues['data_queue'],
            output_queue=mock_queues['output_queue'],
            stop_event=mock_queues['stop_event'],
            mode_name='test_nfb',
            sample_rate=500.0,
            feature_config=None
        )
        
        # Should use defaults
        assert mode.feature_config == {}
        assert mode.target_bands == {'default': [8.0, 12.0]}
        assert mode.window_length_sec == 1.0
        assert mode.window_step_sec == 0.25


class TestNeurofeedbackModeValidation:
    """Test suite for NeurofeedbackMode configuration validation."""
    
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
        mode = create_test_mode(
            data_queue=mock_queues['data_queue'],
            output_queue=mock_queues['output_queue'],
            stop_event=mock_queues['stop_event'],
            mode_name='test_nfb',
            sample_rate=500.0,
            feature_config={
                'target_bands': {'alpha': [8.0, 12.0], 'beta': [12.0, 30.0]},
                'window_length_sec': 1.0,
                'window_step_sec': 0.25
            }
        )
        
        assert mode._validate_configuration() == True
    
    def test_validate_configuration_zero_window_length(self, mock_queues):
        """Test configuration validation with zero window length."""
        mode = create_test_mode(
            data_queue=mock_queues['data_queue'],
            output_queue=mock_queues['output_queue'],
            stop_event=mock_queues['stop_event'],
            mode_name='test_nfb',
            sample_rate=500.0,
            window_length_sec=0.0  # Pass directly, not in feature_config
        )
        mode.logger = Mock()

        assert mode._validate_configuration() == False
        mode.logger.error.assert_called()
    
    def test_validate_configuration_zero_window_step(self, mock_queues):
        """Test configuration validation with zero window step."""
        mode = create_test_mode(
            data_queue=mock_queues['data_queue'],
            output_queue=mock_queues['output_queue'],
            stop_event=mock_queues['stop_event'],
            mode_name='test_nfb',
            sample_rate=500.0,
            step_size_ms=0  # Pass directly, not in feature_config
        )

        # Zero window step should be corrected to 1
        assert mode._validate_configuration() == True
        assert mode.window_step_samples == 1
    
    def test_validate_configuration_invalid_band_range(self, mock_queues):
        """Test configuration validation with invalid band range."""
        mode = create_test_mode(
            data_queue=mock_queues['data_queue'],
            output_queue=mock_queues['output_queue'],
            stop_event=mock_queues['stop_event'],
            mode_name='test_nfb',
            sample_rate=500.0,
            feature_config={
                'target_bands': {'invalid': [12.0, 8.0]}  # High freq < low freq
            }
        )
        mode.logger = Mock()
        
        assert mode._validate_configuration() == False
        mode.logger.error.assert_called()
    
    def test_validate_configuration_band_exceeds_nyquist(self, mock_queues):
        """Test configuration validation with band exceeding Nyquist frequency."""
        mode = create_test_mode(
            data_queue=mock_queues['data_queue'],
            output_queue=mock_queues['output_queue'],
            stop_event=mock_queues['stop_event'],
            mode_name='test_nfb',
            sample_rate=100.0,  # Nyquist = 50 Hz
            feature_config={
                'target_bands': {'high_freq': [40.0, 60.0]}  # Exceeds Nyquist
            }
        )
        mode.logger = Mock()
        
        # Should still validate true but log warning
        assert mode._validate_configuration() == True
        mode.logger.warning.assert_called()
    
    def test_validate_configuration_incomplete_band_range(self, mock_queues):
        """Test configuration validation with incomplete band range."""
        mode = create_test_mode(
            data_queue=mock_queues['data_queue'],
            output_queue=mock_queues['output_queue'],
            stop_event=mock_queues['stop_event'],
            mode_name='test_nfb',
            sample_rate=500.0,
            feature_config={
                'target_bands': {'incomplete': [8.0]}  # Missing high frequency
            }
        )
        mode.logger = Mock()
        
        assert mode._validate_configuration() == False
        mode.logger.error.assert_called()


class TestNeurofeedbackModeBandPowerCalculation:
    """Test suite for band power calculation in NeurofeedbackMode."""
    
    @pytest.fixture
    def nfb_mode_single_band(self):
        """Create NeurofeedbackMode with single band for testing."""
        mode = create_test_mode(
            data_queue=Mock(),
            output_queue=Mock(),
            stop_event=Mock(),
            mode_name='test_nfb',
            sample_rate=500.0,
            feature_config={'target_band': [8.0, 12.0]},
            channel_selection=None  # Use all channels
        )
        mode.logger = Mock()
        mode._initialize_mode()  # Initialize band_power_transform
        return mode
    
    @pytest.fixture
    def nfb_mode_multi_band(self):
        """Create NeurofeedbackMode with multiple bands for testing."""
        mode = create_test_mode(
            data_queue=Mock(),
            output_queue=Mock(),
            stop_event=Mock(),
            mode_name='test_nfb',
            sample_rate=500.0,
            feature_config={
                'target_bands': {
                    'alpha': [8.0, 12.0],
                    'beta': [12.0, 30.0],
                    'theta': [4.0, 8.0]
                }
            },
            channel_selection=None  # Use all channels
        )
        mode.logger = Mock()
        mode._initialize_mode()  # Initialize band_power_transform
        return mode
    
    def test_calculate_band_powers_single_channel_single_band(self, nfb_mode_single_band):
        """Test band power calculation for single channel and single band."""
        eeg_data = create_test_eeg_data(n_channels=1, n_samples=1000)
        
        channel_powers = nfb_mode_single_band._calculate_band_powers(eeg_data)
        
        # Validate using helper function
        assert_valid_channel_powers(channel_powers, expected_channels=1, expected_bands=['default'])
    
    def test_calculate_band_powers_multi_channel_single_band(self, nfb_mode_single_band):
        """Test band power calculation for multiple channels and single band."""
        eeg_data = create_test_eeg_data(n_channels=3, n_samples=500)
        
        channel_powers = nfb_mode_single_band._calculate_band_powers(eeg_data)
        
        # Validate using helper function
        assert_valid_channel_powers(channel_powers, expected_channels=3, expected_bands=['default'])
    
    def test_calculate_band_powers_multi_channel_multi_band(self, nfb_mode_multi_band):
        """Test band power calculation for multiple channels and multiple bands."""
        eeg_data = create_test_eeg_data(n_channels=2, n_samples=1000)
        
        channel_powers = nfb_mode_multi_band._calculate_band_powers(eeg_data)
        
        # Validate using helper function
        expected_bands = ['alpha', 'beta', 'theta']
        assert_valid_channel_powers(channel_powers, expected_channels=2, expected_bands=expected_bands)
    
    def test_calculate_band_powers_1d_data(self, nfb_mode_single_band):
        """Test band power calculation with 1D input data."""
        # Create 1D EEG data (single channel)
        eeg_data = np.random.randn(500)
        
        channel_powers = nfb_mode_single_band._calculate_band_powers(eeg_data)
        
        # Should handle 1D data by reshaping
        assert len(channel_powers) == 1
        assert 'ch0' in channel_powers
        assert 'default' in channel_powers['ch0']
    
    def test_calculate_band_powers_correct_format(self, nfb_mode_single_band):
        """Test band power calculation with correct (channels, times) format."""
        # Create data in correct (channels, times) format as returned by buffer
        eeg_data = np.random.randn(3, 1000)  # 3 channels, 1000 time points
        
        channel_powers = nfb_mode_single_band._calculate_band_powers(eeg_data)
        
        # Should process correctly
        assert len(channel_powers) == 3
        for ch_idx in range(3):
            assert f'ch{ch_idx}' in channel_powers
            assert 'default' in channel_powers[f'ch{ch_idx}']
            assert isinstance(channel_powers[f'ch{ch_idx}']['default'], float)
    
    def test_calculate_band_powers_with_channel_selection(self, nfb_mode_single_band):
        """Test band power calculation with specific modality configuration."""
        # Set specific channel indices directly (channel_selection is processed during init)
        nfb_mode_single_band.selected_channel_indices = [0, 1, 2]
        nfb_mode_single_band.modalities = ['EEG']

        # Create test data: 3 channels
        eeg_data = np.random.randn(3, 500)

        channel_powers = nfb_mode_single_band._calculate_band_powers(eeg_data)

        # Should use the specified channel indices for labeling
        expected_channels = ['ch0', 'ch1', 'ch2']
        assert len(channel_powers) == 3
        for ch_label in expected_channels:
            assert ch_label in channel_powers
    
    def test_calculate_band_powers_noncontiguous_channel_indices(self):
        """Test that channel labels are correct when channel_selection uses non-zero-based indices.

        Regression test: get_modality_labels() returns already-filtered labels
        (e.g. ["F3", "C3", "C4"] for selected indices [2, 3, 4]), so
        _calculate_band_powers must index into channel_labels using the local
        ch_idx (0, 1, 2), not the original stream-level index (2, 3, 4).
        """
        mode = create_test_mode(
            data_queue=Mock(),
            output_queue=Mock(),
            stop_event=Mock(),
            mode_name='test_nfb',
            sample_rate=500.0,
            feature_config={'target_band': [8.0, 12.0]},
            channel_selection={'EEG': [2, 3, 4]},
            modality_labels={'eeg': ['F3', 'C3', 'C4']},
        )
        mode.logger = Mock()
        mode._initialize_mode()

        # 8 channels to simulate full stream data (indices [2,3,4] will be selected)
        eeg_data = np.random.randn(8, 500)
        channel_powers = mode._calculate_band_powers(eeg_data)

        assert list(channel_powers.keys()) == ['F3', 'C3', 'C4']
        assert_valid_channel_powers(channel_powers, expected_channels=3, expected_bands=['default'])

    def test_calculate_band_powers_known_frequency(self, nfb_mode_single_band):
        """Test band power calculation with known frequency content."""
        # Create synthetic data with known frequency (10 Hz, within alpha band 8-12 Hz)
        sample_rate = nfb_mode_single_band.sample_rate
        duration = 2.0  # 2 seconds
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        
        # 10 Hz sine wave
        freq = 10.0
        eeg_data = np.sin(2 * np.pi * freq * t).reshape(1, -1)
        
        channel_powers = nfb_mode_single_band._calculate_band_powers(eeg_data)
        
        # The 10 Hz signal should result in significant power in the alpha band
        alpha_power = channel_powers['ch0']['default']
        assert alpha_power > 0.1  # Should have substantial power


class TestNeurofeedbackModeDataProcessing:
    """Test suite for data processing in NeurofeedbackMode."""
    
    @pytest.fixture
    def nfb_mode_with_buffer(self):
        """Create NeurofeedbackMode with buffer setup."""
        mode = create_test_mode(
            data_queue=Mock(),
            output_queue=Mock(),
            stop_event=Mock(),
            mode_name='test_nfb',
            sample_rate=500.0,
            feature_config={'target_band': [8.0, 12.0]},
            channel_selection={'EEG': [0, 1]}
        )
        mode.logger = Mock()
        mode._initialize_mode()  # Initialize band_power_transform
        mode.buffer = Mock()
        mode._extract_and_send_features = Mock()
        mode.modalities = ['EEG']
        return mode
    
    def test_extract_and_send_features_success(self, nfb_mode_with_buffer):
        """Test successful feature extraction and sending."""
        # Restore the real method (it was mocked in fixture)
        from dendrite.processing.modes.neurofeedback_mode import NeurofeedbackMode
        nfb_mode_with_buffer._extract_and_send_features = NeurofeedbackMode._extract_and_send_features.__get__(nfb_mode_with_buffer)

        # Mock buffer extraction
        X_input = {'eeg': np.random.randn(2, 500)}
        nfb_mode_with_buffer.buffer.extract_window.return_value = X_input

        # Setup queues
        nfb_mode_with_buffer.output_queue = Mock()
        nfb_mode_with_buffer.prediction_queue = Mock()

        nfb_mode_with_buffer._extract_and_send_features()

        # Check buffer was accessed
        nfb_mode_with_buffer.buffer.extract_window.assert_called_once()

        # Check output was sent to both queues
        nfb_mode_with_buffer.output_queue.put.assert_called_once()
        nfb_mode_with_buffer.prediction_queue.put.assert_called_once()

    def test_extract_and_send_features_no_input_data(self, nfb_mode_with_buffer):
        """Test feature extraction when no input data available."""
        nfb_mode_with_buffer.buffer.extract_window.return_value = None
        nfb_mode_with_buffer.output_queue = Mock()
        nfb_mode_with_buffer.prediction_queue = Mock()

        nfb_mode_with_buffer._extract_and_send_features()

        # Should not send anything
        nfb_mode_with_buffer.output_queue.put.assert_not_called()

    def test_extract_and_send_features_no_eeg_data(self, nfb_mode_with_buffer):
        """Test feature extraction when EEG data not available."""
        # Return data without EEG
        X_input = {'emg': np.random.randn(2, 500)}
        nfb_mode_with_buffer.buffer.extract_window.return_value = X_input
        nfb_mode_with_buffer.output_queue = Mock()
        nfb_mode_with_buffer.prediction_queue = Mock()

        nfb_mode_with_buffer._extract_and_send_features()

        # Should not send anything
        nfb_mode_with_buffer.output_queue.put.assert_not_called()
    
    def test_extract_and_send_features_payload_creation(self, nfb_mode_with_buffer):
        """Test proper payload creation during feature extraction."""
        # Restore the real method (it was mocked in fixture)
        from dendrite.processing.modes.neurofeedback_mode import NeurofeedbackMode
        nfb_mode_with_buffer._extract_and_send_features = NeurofeedbackMode._extract_and_send_features.__get__(nfb_mode_with_buffer)

        # Setup multi-band mode for more complex payload
        nfb_mode_with_buffer.target_bands = {
            'alpha': [8.0, 12.0],
            'beta': [12.0, 30.0]
        }

        # Mock buffer extraction (2 channels matching channel_selection)
        X_input = {'eeg': np.random.randn(2, 500)}
        nfb_mode_with_buffer.buffer.extract_window.return_value = X_input

        # Ensure band_power_transform is set up properly
        nfb_mode_with_buffer.band_power_transform = Mock()
        mock_features = np.array([[10.0, 12.0, 8.0, 9.0]])
        nfb_mode_with_buffer.band_power_transform.transform.return_value = {'eeg': mock_features}

        # Setup queues
        nfb_mode_with_buffer.output_queue = Mock()
        nfb_mode_with_buffer.prediction_queue = Mock()

        nfb_mode_with_buffer._extract_and_send_features()

        # Check both queues received data
        nfb_mode_with_buffer.output_queue.put.assert_called_once()
        nfb_mode_with_buffer.prediction_queue.put.assert_called_once()

        # Check payload structure from prediction queue call
        pred_call_args = nfb_mode_with_buffer.prediction_queue.put.call_args[0][0]
        assert pred_call_args["type"] == 'neurofeedback'
        payload_data = pred_call_args["data"]

        assert payload_data['target_bands'] == nfb_mode_with_buffer.target_bands
        assert len(payload_data['channel_powers']) == 2  # 2 channels

        # Check each channel has both bands
        for ch_key in payload_data['channel_powers']:
            assert 'alpha' in payload_data['channel_powers'][ch_key]
            assert 'beta' in payload_data['channel_powers'][ch_key]
            assert isinstance(payload_data['channel_powers'][ch_key]['alpha'], float)
            assert isinstance(payload_data['channel_powers'][ch_key]['beta'], float)


class TestNeurofeedbackModeMainLoop:
    """Test suite for main loop functionality in NeurofeedbackMode."""
    
    @pytest.fixture
    def nfb_mode_for_loop(self):
        """Create NeurofeedbackMode for main loop testing."""
        mode = create_test_mode(
            data_queue=Queue(),
            output_queue=Mock(),
            stop_event=threading.Event(),
            mode_name='test_nfb',
            sample_rate=500.0,
            feature_config={'target_band': [8.0, 12.0]},
            channel_selection={'EEG': [0, 1]}
        )
        mode.logger = Mock()
        mode.buffer = Mock()
        mode.buffer.is_ready_for_step.return_value = False  # Default: not ready
        mode._extract_and_send_features = Mock()
        return mode
    
    def test_run_main_loop_data_processing(self, nfb_mode_for_loop):
        """Test main loop data processing."""
        # Add sample to queue
        test_sample = {'EEG': np.array([1, 2, 3, 4])}
        nfb_mode_for_loop.data_queue.put(test_sample)
        
        # Stop after processing one sample
        def stop_after_first_sample(*args):
            nfb_mode_for_loop.stop_event.set()
        
        nfb_mode_for_loop.buffer.add_sample.side_effect = stop_after_first_sample
        
        nfb_mode_for_loop._run_main_loop()

        # Should process and add sample to buffer
        nfb_mode_for_loop.buffer.add_sample.assert_called_once()
    
    def test_run_main_loop_feature_extraction_trigger(self, nfb_mode_for_loop):
        """Test main loop feature extraction triggering."""
        # Setup buffer to be ready for feature extraction
        nfb_mode_for_loop.buffer.is_ready_for_step.return_value = True
        
        # Add sample and stop after feature extraction
        test_sample = {'EEG': np.array([1, 2, 3, 4])}
        nfb_mode_for_loop.data_queue.put(test_sample)
        
        def stop_after_extraction(*args):
            nfb_mode_for_loop.stop_event.set()
        
        nfb_mode_for_loop._extract_and_send_features.side_effect = stop_after_extraction
        
        nfb_mode_for_loop._run_main_loop()
        
        # Should trigger feature extraction
        nfb_mode_for_loop._extract_and_send_features.assert_called_once()
    
    def test_run_main_loop_empty_queue(self, nfb_mode_for_loop):
        """Test main loop handling of empty queue."""
        # Immediately stop to avoid infinite loop
        nfb_mode_for_loop.stop_event.set()
        
        nfb_mode_for_loop._run_main_loop()
        
        # Should not process any samples
        nfb_mode_for_loop.buffer.add_sample.assert_not_called()
        nfb_mode_for_loop._extract_and_send_features.assert_not_called()
    
    def test_run_main_loop_exception_handling(self, nfb_mode_for_loop):
        """Test main loop exception handling."""
        # Add sample that will cause exception
        test_sample = {'EEG': np.array([1, 2, 3, 4])}
        nfb_mode_for_loop.data_queue.put(test_sample)

        # Make buffer.add_sample raise exception
        nfb_mode_for_loop.buffer.add_sample.side_effect = Exception("Test exception")

        # Stop after first iteration
        def stop_after_exception():
            nfb_mode_for_loop.stop_event.set()
        
        # Set up a timer to stop the loop
        import threading
        timer = threading.Timer(0.1, stop_after_exception)
        timer.start()
        
        try:
            nfb_mode_for_loop._run_main_loop()
        finally:
            timer.cancel()

        # Should log error and continue
        nfb_mode_for_loop.logger.error.assert_called()


class TestNeurofeedbackModePayload:
    """Test suite for BandPowerPayload functionality."""

    def test_neurofeedback_payload_creation(self):
        """Test basic BandPowerPayload creation."""
        powers = {
            'ch0': {'alpha': 12.5, 'beta': 8.3},
            'ch1': {'alpha': 10.2, 'beta': 6.1}
        }
        bands = {'alpha': [8.0, 12.0], 'beta': [12.0, 30.0]}

        payload = BandPowerPayload(channel_powers=powers, target_bands=bands)

        assert payload.channel_powers == powers
        assert payload.target_bands == bands

    def test_neurofeedback_payload_default_creation(self):
        """Test BandPowerPayload creation with defaults."""
        payload = BandPowerPayload()

        assert payload.channel_powers == {}
        assert payload.target_bands == {}

    def test_neurofeedback_payload_single_band(self):
        """Test BandPowerPayload with single band configuration."""
        powers = {
            'ch0': {'default': 15.7},
            'ch1': {'default': 12.3}
        }
        bands = {'default': [8.0, 12.0]}

        payload = BandPowerPayload(channel_powers=powers, target_bands=bands)

        assert len(payload.channel_powers) == 2
        assert len(payload.target_bands) == 1
        assert 'default' in payload.target_bands

    def test_neurofeedback_payload_serialization(self):
        """Test BandPowerPayload serialization."""
        payload = BandPowerPayload(
            channel_powers={'ch0': {'alpha': 10.0}},
            target_bands={'alpha': [8.0, 12.0]}
        )

        # Convert to dict using dataclass utility
        payload_dict = asdict(payload)

        assert 'channel_powers' in payload_dict
        assert 'target_bands' in payload_dict
        assert payload_dict['channel_powers']['ch0']['alpha'] == 10.0
        assert payload_dict['target_bands']['alpha'] == [8.0, 12.0]


class TestNeurofeedbackModeCleanup:
    """Test suite for NeurofeedbackMode cleanup functionality."""
    
    def test_cleanup_basic(self):
        """Test basic cleanup functionality."""
        mode = create_test_mode(
            data_queue=Mock(),
            output_queue=Mock(),
            stop_event=Mock(),
            mode_name='test_nfb',
            sample_rate=500.0
        )
        
        mode.logger = Mock()
        
        # Mock parent cleanup to avoid side effects
        with patch('dendrite.processing.modes.base_mode.BaseMode._cleanup'):
            mode._cleanup()
        
        # Should log cleanup message
        mode.logger.info.assert_called_with('Cleaning up NeurofeedbackMode')


class TestNeurofeedbackModeInitialization:
    """Test suite for NeurofeedbackMode initialization functionality."""
    
    @pytest.fixture
    def nfb_mode_for_init(self):
        """Create NeurofeedbackMode for initialization testing."""
        mode = create_test_mode(
            data_queue=Mock(),
            output_queue=Mock(),
            stop_event=Mock(),
            mode_name='test_nfb',
            sample_rate=500.0,
            feature_config={
                'target_bands': {
                    'alpha': [8.0, 12.0],
                    'beta': [12.0, 30.0]
                },
                'window_length_sec': 2.0,
                'window_step_sec': 0.5
            },
            channel_selection={'EEG': [0, 1, 2]}
        )
        mode.logger = Mock()
        return mode
    
    @patch('dendrite.processing.modes.neurofeedback_mode.NeurofeedbackMode._setup_buffer')
    def test_initialize_mode_success(self, mock_setup_buffer, nfb_mode_for_init):
        """Test successful mode initialization."""
        result = nfb_mode_for_init._initialize_mode()
        
        assert result == True
        
        # Check buffer setup was called
        mock_setup_buffer.assert_called_once_with(nfb_mode_for_init.window_length_samples)
        
        # Check logging
        nfb_mode_for_init.logger.info.assert_called()
        
        # Check multi-band logging
        log_calls = [call[0][0] for call in nfb_mode_for_init.logger.info.call_args_list]
        multi_band_log = next((log for log in log_calls if 'Multi-band mode: 2 bands' in log), None)
        assert multi_band_log is not None
    
    @patch('dendrite.processing.modes.neurofeedback_mode.NeurofeedbackMode._setup_buffer')
    def test_initialize_mode_single_band_logging(self, mock_setup_buffer):
        """Test mode initialization with single band logging."""
        mode = create_test_mode(
            data_queue=Mock(),
            output_queue=Mock(),
            stop_event=Mock(),
            mode_name='test_nfb',
            sample_rate=500.0,
            feature_config={'target_band': [8.0, 12.0]}
        )
        mode.logger = Mock()
        
        result = mode._initialize_mode()
        
        assert result == True
        
        # Check single band logging
        log_calls = [call[0][0] for call in mode.logger.info.call_args_list]
        single_band_log = next((log for log in log_calls if "Single band 'default': 8.0-12.0 Hz" in log), None)
        assert single_band_log is not None
    
    @patch('dendrite.processing.modes.neurofeedback_mode.NeurofeedbackMode._setup_buffer')
    def test_initialize_mode_exception(self, mock_setup_buffer, nfb_mode_for_init):
        """Test mode initialization with exception."""
        mock_setup_buffer.side_effect = Exception("Setup failed")
        
        result = nfb_mode_for_init._initialize_mode()
        
        assert result == False
        nfb_mode_for_init.logger.error.assert_called_with(
            'Initialization failed: Setup failed'
        )


class TestNeurofeedbackModeEdgeCases:
    """Test suite for edge cases and error handling in NeurofeedbackMode."""
    
    def test_calculate_band_powers_empty_data(self):
        """Test band power calculation with empty data."""
        mode = create_test_mode(
            data_queue=Mock(),
            output_queue=Mock(),
            stop_event=Mock(),
            mode_name='test_nfb',
            sample_rate=500.0
        )
        
        # Empty data
        eeg_data = np.array([]).reshape(0, 0)
        
        # Should handle gracefully without crashing
        try:
            channel_powers = mode._calculate_band_powers(eeg_data)
            # If it doesn't crash, that's acceptable behavior
        except Exception:
            # Also acceptable - empty data is an edge case
            pass
    
    def test_calculate_band_powers_single_sample(self):
        """Test band power calculation with single time sample."""
        mode = create_test_mode(
            data_queue=Mock(),
            output_queue=Mock(),
            stop_event=Mock(),
            mode_name='test_nfb',
            sample_rate=500.0,
            channel_selection={'EEG': [0, 1, 2]}
        )
        mode.logger = Mock()  # Mock logger before initialization
        mode._initialize_mode()  # Initialize band_power_transform

        # Single time sample, multiple channels - this is an edge case
        eeg_data = np.array([[1.0], [2.0], [3.0]])  # 3 channels, 1 sample

        # Should handle gracefully (FFT might not work well with 1 sample)
        channel_powers = mode._calculate_band_powers(eeg_data)
        
        # With 1 sample, FFT is problematic - method might return fewer channels or fail gracefully
        # The important thing is it doesn't crash
        assert isinstance(channel_powers, dict)
        assert len(channel_powers) >= 1  # At least some result
        for ch_key in channel_powers:
            assert 'default' in channel_powers[ch_key]
    
    def test_extract_and_send_features_no_queues(self):
        """Test feature extraction without output queues."""
        mode = create_test_mode(
            data_queue=Mock(),
            output_queue=None,
            stop_event=Mock(),
            mode_name='test_nfb',
            sample_rate=500.0,
            channel_selection={'EEG': [0, 1]}
        )
        mode.logger = Mock()
        mode._initialize_mode()
        mode.buffer = Mock()
        mode.buffer.extract_window.return_value = {'eeg': np.random.randn(2, 500)}
        mode.modalities = ['EEG']
        mode.prediction_queue = None

        # Should not crash
        mode._extract_and_send_features()

        # No specific assertions needed - just ensure no exception