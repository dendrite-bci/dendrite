"""
Unit tests for BaseMode class.

Tests cover:
- BaseMode initialization and configuration
- Status reporting  
- Error handling and edge cases

Note: Buffer tests are in test_mode_utils.py
"""

import sys
import os
import pytest
import numpy as np
import time
import multiprocessing
from unittest.mock import Mock, patch, MagicMock
from collections import deque
import logging

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dendrite.processing.modes.base_mode import BaseMode
from dendrite.processing.modes.mode_utils import Buffer
from tests.unit.test_output_schema import assert_valid_mode_output


class TestBaseModeInitialization:
    """Test suite for BaseMode initialization."""
    
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
    def base_mode_impl(self, mock_queues):
        """Create a concrete implementation of BaseMode for testing."""
        class TestMode(BaseMode):
            def _validate_configuration(self):
                return True

            def _initialize_mode(self):
                return True

            def _run_main_loop(self):
                pass

        instance_config = {
            'name': 'test_mode',
            'channel_selection': None
        }

        return TestMode(
            data_queue=mock_queues['data_queue'],
            output_queue=mock_queues['output_queue'],
            stop_event=mock_queues['stop_event'],
            instance_config=instance_config,
            sample_rate=500.0,
            prediction_queue=mock_queues['prediction_queue']
        )
    
    def test_base_mode_initialization(self, base_mode_impl, mock_queues):
        """Test BaseMode basic initialization."""
        mode = base_mode_impl
        
        assert mode.data_queue == mock_queues['data_queue']
        assert mode.output_queue == mock_queues['output_queue']
        assert mode.stop_event == mock_queues['stop_event']
        assert mode.prediction_queue == mock_queues['prediction_queue']
        assert mode.mode_name == 'test_mode'
        assert mode.sample_rate == 500.0
        
        # Check process inheritance
        assert isinstance(mode, multiprocessing.Process)
    
    def test_base_mode_with_channel_selection(self, mock_queues):
        """Test BaseMode initialization with modality configuration."""
        class TestMode(BaseMode):
            def _validate_configuration(self):
                return True
            def _initialize_mode(self):
                return True
            def _run_main_loop(self):
                pass

        channel_selection = {'eeg': [0, 1, 2, 3], 'emg': [0, 1]}
        instance_config = {
            'name': 'test_mode',
            'channel_selection': channel_selection
        }

        mode = TestMode(
            data_queue=mock_queues['data_queue'],
            output_queue=mock_queues['output_queue'],
            stop_event=mock_queues['stop_event'],
            instance_config=instance_config,
            sample_rate=500.0
        )

        assert mode.channel_selection == channel_selection
        assert mode.modalities == ['eeg', 'emg']
        assert mode.modality_num_channels == {'eeg': 4, 'emg': 2}
        assert mode.modalities_detected == True
    
    def test_base_mode_without_channel_selection(self, mock_queues):
        """Test BaseMode initialization without modality configuration."""
        class TestMode(BaseMode):
            def _validate_configuration(self):
                return True
            def _initialize_mode(self):
                return True
            def _run_main_loop(self):
                pass

        instance_config = {
            'name': 'test_mode',
            'channel_selection': None
        }

        mode = TestMode(
            data_queue=mock_queues['data_queue'],
            output_queue=mock_queues['output_queue'],
            stop_event=mock_queues['stop_event'],
            instance_config=instance_config,
            sample_rate=500.0
        )

        assert mode.channel_selection == {}
        assert mode.modalities == []
        assert mode.modality_num_channels == {}
        assert mode.modalities_detected == False



class TestBaseModeStatus:
    """Test suite for BaseMode status reporting."""

    @pytest.fixture
    def base_mode_impl(self):
        """Create a concrete implementation of BaseMode for testing."""
        class TestMode(BaseMode):
            def _validate_configuration(self):
                return True
            def _initialize_mode(self):
                return True
            def _run_main_loop(self):
                pass

        instance_config = {
            'name': 'test_mode',
            'channel_selection': {'eeg': [0, 1, 2, 3]}
        }

        return TestMode(
            data_queue=Mock(),
            output_queue=Mock(),
            stop_event=Mock(),
            instance_config=instance_config,
            sample_rate=500.0
        )
    
    def test_get_status_basic(self, base_mode_impl):
        """Test basic status reporting."""
        mode = base_mode_impl
        mode._start_time = time.time() - 10.0  # 10 seconds ago
        mode._is_running = True
        
        status = mode.get_status()
        
        assert status['mode_name'] == 'test_mode'
        assert status['mode_type'] == 'TestMode'
        assert status['is_running'] == True
        assert status['modalities'] == ['eeg']
        assert status['sample_rate'] == 500.0
        assert 'uptime_seconds' in status
        assert 8 <= status['uptime_seconds'] <= 12  # Approximate check
    
    def test_get_status_with_metrics(self, base_mode_impl):
        """Test status reporting with metrics manager."""
        mode = base_mode_impl
        mode.metrics_manager = Mock()
        mode.metrics_manager.get_current_metrics.return_value = {
            'accuracy': 0.85,
            'samples_processed': 100
        }
        
        status = mode.get_status()
        
        assert 'metrics' in status
        assert status['metrics']['accuracy'] == 0.85
        assert status['metrics']['samples_processed'] == 100
    
    def test_get_status_with_buffer(self, base_mode_impl):
        """Test status reporting with buffer."""
        mode = base_mode_impl
        mode.buffer = Mock()
        mode.buffer.get_status.return_value = {
            'buffer_size': 1000,
            'current_size': 500,
            'samples_since_last_step': 10
        }
        
        status = mode.get_status()
        
        assert 'buffer' in status
        assert status['buffer']['buffer_size'] == 1000
        assert status['buffer']['current_size'] == 500



class TestBaseModeSetup:
    """Test suite for BaseMode setup methods."""

    @pytest.fixture
    def base_mode_impl(self):
        """Create a concrete implementation of BaseMode for testing."""
        class TestMode(BaseMode):
            def _validate_configuration(self):
                return True
            def _initialize_mode(self):
                return True
            def _run_main_loop(self):
                pass

        instance_config = {
            'name': 'test_mode',
            'channel_selection': None
        }

        return TestMode(
            data_queue=Mock(),
            output_queue=Mock(),
            stop_event=Mock(),
            instance_config=instance_config,
            sample_rate=500.0
        )
    
    @patch('dendrite.processing.modes.base_mode.setup_logger')
    def test_setup_logger(self, mock_setup_logger, base_mode_impl):
        """Test logger setup."""
        mock_logger = Mock()
        mock_setup_logger.return_value = mock_logger
        
        mode = base_mode_impl
        mode._setup_logger()
        
        mock_setup_logger.assert_called_once_with('TestMode-test_mode', level=logging.INFO)
        assert mode.logger == mock_logger
    
    def test_send_output_to_main_queue(self, base_mode_impl):
        """Test _send_output sends to main queue only."""
        mode = base_mode_impl
        mode.logger = Mock()
        mode.output_queue = Mock()
        mode.prediction_queue = Mock()

        test_payload = {'accuracy': 0.85}
        mode._send_output(test_payload, 'performance', queue='main')

        mode.output_queue.put.assert_called_once()
        mode.prediction_queue.put.assert_not_called()

        call_args = mode.output_queue.put.call_args[0][0]
        assert_valid_mode_output(call_args)  # Schema validation
        assert call_args["type"] == 'performance'
        assert call_args["mode_name"] == 'test_mode'
        assert call_args["data"] == test_payload

    def test_send_output_to_prediction_queue(self, base_mode_impl):
        """Test _send_output sends to prediction queue only."""
        mode = base_mode_impl
        mode.logger = Mock()
        mode.output_queue = Mock()
        mode.prediction_queue = Mock()

        test_payload = {'prediction': 1}
        mode._send_output(test_payload, 'prediction', queue='prediction')

        mode.prediction_queue.put.assert_called_once()
        mode.output_queue.put.assert_not_called()

        call_args = mode.prediction_queue.put.call_args[0][0]
        assert_valid_mode_output(call_args)  # Schema validation

    def test_send_output_to_both_queues(self, base_mode_impl):
        """Test _send_output sends to both queues."""
        mode = base_mode_impl
        mode.logger = Mock()
        mode.output_queue = Mock()
        mode.prediction_queue = Mock()

        test_payload = {'data': 'test'}
        mode._send_output(test_payload, 'test_type', queue='both')

        mode.output_queue.put.assert_called_once()
        mode.prediction_queue.put.assert_called_once()

        # Validate both queue outputs
        main_output = mode.output_queue.put.call_args[0][0]
        pred_output = mode.prediction_queue.put.call_args[0][0]
        assert_valid_mode_output(main_output)
        assert_valid_mode_output(pred_output)

    def test_send_output_handles_none_queues(self, base_mode_impl):
        """Test _send_output handles None queues gracefully."""
        mode = base_mode_impl
        mode.logger = Mock()
        mode.output_queue = None
        mode.prediction_queue = None

        # Should not crash
        mode._send_output({'data': 'test'}, 'test_type', queue='both')

    @patch('dendrite.processing.modes.base_mode.MetricsManager')
    def test_setup_metrics_manager(self, mock_metrics_manager, base_mode_impl):
        """Test metrics manager setup."""
        mock_manager = Mock()
        mock_metrics_manager.return_value = mock_manager

        mode = base_mode_impl
        mode.logger = Mock()  # Setup logger first
        mode._setup_metrics_manager(num_classes=3, mode_type='test')

        mock_metrics_manager.assert_called_once_with(
            mode_type='test',
            sample_rate=500,
            num_classes=3,
            detection_window_samples=None,
            label_mapping=None,
            background_class=None
        )
        assert mode.metrics_manager == mock_manager
    
    def test_setup_buffer(self, base_mode_impl):
        """Test buffer setup."""
        mode = base_mode_impl
        mode.modalities = ['eeg', 'emg']
        mode.logger = Mock()
        
        mode._setup_buffer(1000)
        
        assert mode.buffer is not None
        assert isinstance(mode.buffer, Buffer)
        assert mode.buffer.modalities == ['eeg', 'emg']
        assert mode.buffer.buffer_size == 1000



class TestBaseModeErrorHandling:
    """Test suite for BaseMode error handling."""
    
    def test_abstract_methods_not_implemented(self):
        """Test that abstract methods raise NotImplementedError."""
        instance_config = {
            'name': 'test',
            'channel_selection': None
        }

        with pytest.raises(TypeError):
            # Cannot instantiate abstract class
            BaseMode(
                data_queue=Mock(),
                output_queue=Mock(),
                stop_event=Mock(),
                instance_config=instance_config,
                sample_rate=500.0
            )
    
    def test_cleanup_runs_without_error(self):
        """Test cleanup runs without errors."""
        class TestMode(BaseMode):
            def _validate_configuration(self):
                return True
            def _initialize_mode(self):
                return True
            def _run_main_loop(self):
                pass

        instance_config = {
            'name': 'test_mode',
            'channel_selection': None
        }

        mode = TestMode(
            data_queue=Mock(),
            output_queue=Mock(),
            stop_event=Mock(),
            instance_config=instance_config,
            sample_rate=500.0
        )

        mode.logger = Mock()

        # Should not raise exception
        mode._cleanup()