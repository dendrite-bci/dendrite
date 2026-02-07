"""
Unit tests for DataProcessor class.

This module provides comprehensive unit tests for the synchronized data processor,
focusing on testing the core functionality in isolation with mocked dependencies.

Tests cover:
- DataProcessor initialization and configuration
- Channel information setup and validation
- Sample collection and buffering
- Chunk processing and data reshaping
- Fan-out distribution to mode queues
- Preprocessing integration
- Error handling and edge cases
- Performance characteristics
"""

import sys
import os
import pytest
import numpy as np
import time
import queue
import multiprocessing
import threading
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, Any, List

# Add the project root to the path so BMI can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dendrite.data.stream_schemas import StreamConfig
from dendrite.processing.processor import DataProcessor


class TestDataProcessorInitialization:
    """Test suite for DataProcessor initialization."""
    
    @pytest.fixture
    def mock_queues(self):
        """Create mock queues for testing."""
        return {
            'data_queue': Mock(),
            'plot_queue': Mock(),
            'shared_state': Mock(),
            'stop_event': Mock()
        }
    
    @pytest.fixture
    def sample_preprocessing_config(self):
        """Sample preprocessing configuration."""
        return {
            'EEG': {
                'sample_rate': 500,
                'lowcut': 0.5,
                'highcut': 40.0,
                'num_channels': 32
            },
            'EMG': {
                'sample_rate': 500,
                'lowcut': 10.0,
                'highcut': 200.0,
                'num_channels': 8
            }
        }
    
    def test_basic_initialization(self, mock_queues, sample_preprocessing_config):
        """Test basic DataProcessor initialization."""
        mode_configs = {'sync_mode': {}, 'async_mode': {}}

        processor = DataProcessor(
            data_queue=mock_queues['data_queue'],
            plot_queue=mock_queues['plot_queue'],
            stop_event=mock_queues['stop_event'],
            shared_state=mock_queues['shared_state'],
            mode_configs=mode_configs,
            preprocessing_config={},
        )

        assert processor.data_queue is mock_queues['data_queue']
        assert processor.plot_queue is mock_queues['plot_queue']
        assert processor.stop_event is mock_queues['stop_event']
        assert processor.shared_state is mock_queues['shared_state']
        assert processor.mode_names == ['sync_mode', 'async_mode']
        assert processor.preprocessing_config == {}

        # Check mode queues are created
        assert len(processor.mode_queues) == 2
        assert 'sync_mode' in processor.mode_queues
        assert 'async_mode' in processor.mode_queues

        # Check initialization state
        assert processor.dropped_samples == {'sync_mode': 0, 'async_mode': 0}
        assert processor.preprocessors == {}  # Empty dict, not None
    
    def test_initialization_with_preprocessing(self, mock_queues, sample_preprocessing_config):
        """Test initialization with preprocessing enabled."""
        processor = DataProcessor(
            data_queue=mock_queues['data_queue'],
            plot_queue=mock_queues['plot_queue'],
            stop_event=mock_queues['stop_event'],
            shared_state=mock_queues['shared_state'],
            mode_configs={'test_mode': {}},
            preprocessing_config=sample_preprocessing_config,
            mode_queue_size=500
        )

        assert processor.preprocessing_config == sample_preprocessing_config
        # Queues should have custom size
        test_queue = processor.get_queue_for_mode('test_mode')
        assert test_queue is not None
    
    def test_get_queue_for_mode(self, mock_queues, sample_preprocessing_config):
        """Test getting mode-specific queues."""
        processor = DataProcessor(
            data_queue=mock_queues['data_queue'],
            plot_queue=mock_queues['plot_queue'],
            stop_event=mock_queues['stop_event'],
            shared_state=mock_queues['shared_state'],
            mode_configs={'mode1': {}, 'mode2': {}},
            preprocessing_config=sample_preprocessing_config
        )
        
        queue1 = processor.get_queue_for_mode('mode1')
        queue2 = processor.get_queue_for_mode('mode2')
        nonexistent = processor.get_queue_for_mode('nonexistent')
        
        assert queue1 is not None
        assert queue2 is not None
        assert nonexistent is None
        assert queue1 is not queue2


class TestWaitForFirstSample:
    """Test suite for waiting for first data sample."""

    @pytest.fixture
    def processor_with_mocks(self):
        """Create processor with mocked dependencies."""
        mock_queues = {
            'data_queue': Mock(),
            'plot_queue': Mock(),
            'shared_state': Mock(),
            'stop_event': Mock()
        }
        mock_queues['stop_event'].is_set.return_value = False

        processor = DataProcessor(
            data_queue=mock_queues['data_queue'],
            plot_queue=mock_queues['plot_queue'],
            stop_event=mock_queues['stop_event'],
            shared_state=mock_queues['shared_state'],
            mode_configs={'test_mode': {}},
            preprocessing_config={}
        )
        processor.logger = Mock()

        return processor, mock_queues

    @pytest.fixture
    def sample_data_dict(self):
        """Sample data dictionary from DAQ."""
        return {
            'eeg': np.zeros((8, 1)),      # 8 EEG channels
            'emg': np.zeros((4, 1)),      # 4 EMG channels
            'markers': np.array([[0]]),   # 1 marker channel
        }

    def test_wait_for_first_sample_returns_data(self, processor_with_mocks, sample_data_dict):
        """Test waiting for first data sample."""
        processor, mock_queues = processor_with_mocks
        mock_queues['data_queue'].get.return_value = {'data': sample_data_dict}

        result = processor._wait_for_first_sample()

        assert result is not None
        assert 'data' in result

    def test_wait_for_first_sample_skips_events(self, processor_with_mocks, sample_data_dict):
        """Test that event payloads are skipped while waiting."""
        processor, mock_queues = processor_with_mocks
        # First call returns event, second returns data
        mock_queues['data_queue'].get.side_effect = [
            {'event': {'event_id': 1}},
            {'data': sample_data_dict}
        ]

        result = processor._wait_for_first_sample()

        assert result is not None
        assert 'data' in result
        assert mock_queues['data_queue'].get.call_count == 2


class TestSampleProcessing:
    """Test suite for per-sample processing (replaced buffer-based collection)."""

    @pytest.fixture
    def processor_with_initialized_channels(self):
        """Create processor with initialized channel information."""
        processor = DataProcessor(
            data_queue=Mock(),
            plot_queue=Mock(),
            stop_event=Mock(),
            shared_state=Mock(),
            mode_configs={'test_mode': {}},
            preprocessing_config={}
        )

        # Mock mode queue
        processor.mode_queues['test_mode'] = Mock()
        processor.mode_queues['test_mode'].full.return_value = False

        return processor

    def test_distribute_sample(self, processor_with_initialized_channels):
        """Test distributing a single sample to mode queues."""
        processor = processor_with_initialized_channels

        # Latencies are now added directly to filtered_dict with _ prefix
        # (done by _handle_data_sample before calling _distribute_sample)
        filtered_dict = {
            'EEG': np.random.randn(4, 1),
            'EMG': np.random.randn(2, 1),
            '_eeg_latency_ms': 5.2,
        }
        lsl_timestamp = 12345.67
        daq_receive_ns = 123456789

        processor._distribute_sample(filtered_dict, lsl_timestamp, daq_receive_ns)

        # Mode queue should receive the sample
        mode_queue = processor.mode_queues['test_mode']
        mode_queue.put_nowait.assert_called_once()

        # Verify sample structure - mode only receives its required modalities (default: eeg)
        call_args = mode_queue.put_nowait.call_args[0][0]
        assert 'EEG' in call_args  # eeg is default required modality
        # EMG is NOT in call_args since mode didn't request it (default required_modalities=['eeg'])
        assert call_args['lsl_timestamp'] == lsl_timestamp
        assert call_args['_daq_receive_ns'] == daq_receive_ns
        assert '_eeg_latency_ms' in call_args

    def test_process_next_sample_success(self, processor_with_initialized_channels):
        """Test successful single sample processing."""
        processor = processor_with_initialized_channels

        # Mock data queue payload - DAQ now sends dict format directly
        sample_payload = {
            'data': {
                'eeg': np.random.randn(4, 1),
                'emg': np.random.randn(2, 1),
            },
            'lsl_timestamp': 12345.67,
            '_daq_receive_ns': 123456789,
            'eeg_latency_ms': 5.2
        }
        processor.data_queue.get.return_value = sample_payload

        processor._process_next_sample()

        # Mode queue should receive sample
        processor.mode_queues['test_mode'].put_nowait.assert_called_once()

    def test_process_next_sample_empty_queue(self, processor_with_initialized_channels):
        """Test sample processing with empty queue raises Empty."""
        processor = processor_with_initialized_channels
        processor.data_queue.get.side_effect = queue.Empty()

        # Should raise queue.Empty (caught in main loop)
        with pytest.raises(queue.Empty):
            processor._process_next_sample()


class TestSampleDistribution:
    """Test suite for sample distribution to mode queues."""
    
    @pytest.fixture
    def processor_with_modes(self):
        """Create processor with multiple modes with different required modalities."""
        mode_configs = {
            'mode1': {'required_modalities': ['eeg']},
            'mode2': {'required_modalities': ['eeg', 'emg']},  # Multi-modal
            'mode3': {'required_modalities': ['emg']},
        }
        processor = DataProcessor(
            data_queue=Mock(),
            plot_queue=Mock(),
            stop_event=Mock(),
            shared_state=Mock(),
            mode_configs=mode_configs,
            preprocessing_config={}
        )

        # Replace real queues with mocks for testing
        for mode_name in processor.mode_names:
            processor.mode_queues[mode_name] = Mock()

        return processor

    def test_fan_out_sample_success(self, processor_with_modes):
        """Test sample fan-out filters by required modalities."""
        processor = processor_with_modes

        sample_dict = {
            'EEG': np.random.randn(4, 1),
            'EMG': np.random.randn(2, 1),
        }

        processor._fan_out_sample(sample_dict)

        # Each mode receives only its required modalities
        # mode1: eeg only
        call_args_1 = processor.mode_queues['mode1'].put_nowait.call_args[0][0]
        assert 'EEG' in call_args_1
        assert 'EMG' not in call_args_1

        # mode2: eeg + emg
        call_args_2 = processor.mode_queues['mode2'].put_nowait.call_args[0][0]
        assert 'EEG' in call_args_2
        assert 'EMG' in call_args_2

        # mode3: emg only
        call_args_3 = processor.mode_queues['mode3'].put_nowait.call_args[0][0]
        assert 'EEG' not in call_args_3
        assert 'EMG' in call_args_3

    def test_fan_out_sample_full_queue(self, processor_with_modes):
        """Test fan-out with full queue handling."""
        processor = processor_with_modes

        # Make one queue raise Full exception (EAFP pattern)
        processor.mode_queues['mode2'].put_nowait.side_effect = queue.Full()

        sample_dict = {
            'EEG': np.random.randn(4, 1),
            'EMG': np.random.randn(2, 1),
        }

        with patch.object(processor, '_handle_full_queue') as mock_handle:
            processor._fan_out_sample(sample_dict)

            # Handle full queue should be called for mode2
            mock_handle.assert_called_once_with('mode2')

            # Other queues should receive filtered samples
            processor.mode_queues['mode1'].put_nowait.assert_called_once()
            processor.mode_queues['mode3'].put_nowait.assert_called_once()
    
    def test_handle_full_queue(self, processor_with_modes):
        """Test handling of full queue by dropping the sample (real-time priority)."""
        processor = processor_with_modes

        processor._handle_full_queue('mode1')

        # Should increment dropped counter
        assert processor.dropped_samples['mode1'] == 1

    def test_send_to_plot_queue(self, processor_with_modes):
        """Test sending samples to plot queue with decimation."""
        processor = processor_with_modes

        sample_dict = {'test': 'data'}

        # Decimation is 5 by default - first 4 calls are skipped
        for _ in range(4):
            processor._send_to_plot_queue(sample_dict)
        processor.plot_queue.put_nowait.assert_not_called()  # Should be skipped

        # 5th call should send (EAFP pattern - no full() check)
        processor._send_to_plot_queue(sample_dict)
        processor.plot_queue.put_nowait.assert_called_once()

        # Test with full queue (next sample after decimation)
        processor.plot_queue.reset_mock()
        processor.plot_queue.put_nowait.side_effect = queue.Full()

        for _ in range(5):  # Need 5 more calls to hit next send
            processor._send_to_plot_queue(sample_dict)

        # put_nowait is called but raises Full (silently dropped)
        processor.plot_queue.put_nowait.assert_called_once()

    def test_send_to_plot_queue_event_bypass_decimation(self, processor_with_modes):
        """Test samples with markers != 0 bypass decimation."""
        processor = processor_with_modes
        processor.plot_queue.full.return_value = False

        # Sample without event (markers = 0) - should be decimated
        sample_no_event = {'eeg': np.array([[1.0]]), 'markers': np.array([[0.0]])}
        processor._viz_state = dict(sample_no_event)  # Simulate accumulation

        # First 4 calls should be skipped (decimation = 5)
        for _ in range(4):
            processor._send_to_plot_queue(sample_no_event)
        processor.plot_queue.put_nowait.assert_not_called()

        # Sample WITH event (markers != 0) - should bypass decimation
        processor.plot_queue.reset_mock()
        processor._plot_counter = 1  # Reset to non-5th position
        sample_with_event = {'eeg': np.array([[1.0]]), 'markers': np.array([[42.0]])}
        processor._viz_state = dict(sample_with_event)  # Simulate accumulation

        processor._send_to_plot_queue(sample_with_event)
        # Should be sent despite decimation counter not being at 5
        processor.plot_queue.put_nowait.assert_called_once()

    def test_send_to_plot_queue_event_with_zero_markers(self, processor_with_modes):
        """Test that markers = 0 does NOT bypass decimation."""
        processor = processor_with_modes
        processor.plot_queue.full.return_value = False
        processor._plot_counter = 1  # Not at decimation point

        sample_zero_marker = {'eeg': np.array([[1.0]]), 'markers': np.array([[0.0]])}
        processor._viz_state = dict(sample_zero_marker)  # Simulate accumulation
        processor._send_to_plot_queue(sample_zero_marker)

        # Should NOT be sent (decimation applies)
        processor.plot_queue.put_nowait.assert_not_called()


class TestEventHandling:
    """Test suite for event handling and marker distribution."""

    @pytest.fixture
    def processor_with_pending_marker(self):
        """Create processor for event handling tests."""
        processor = DataProcessor(
            data_queue=Mock(),
            plot_queue=Mock(),
            stop_event=Mock(),
            shared_state=Mock(),
            mode_configs={'test_mode': {}},
            preprocessing_config={}
        )

        # Replace real queue with mock
        processor.mode_queues['test_mode'] = Mock()

        return processor

    def test_handle_event_stores_pending_marker(self, processor_with_pending_marker):
        """Test that _handle_event queues event as pending marker."""
        processor = processor_with_pending_marker
        # Configure stream so event handling works
        processor.stream_configs = [StreamConfig(name='EEG_Stream', type='EEG', channel_count=1, sample_rate=500)]

        event_payload = {
            'event': {
                'event_id': 42.0,
                'event_json': {'event_id': 42, 'event_type': 'test'},
            },
            'lsl_timestamp': 12345.67,
            '_daq_receive_ns': 123456789,
        }

        processor._handle_event(event_payload)

        # Pending markers should have one event
        assert len(processor.pending_markers) == 1
        event_id, timestamp, streams_pending = processor.pending_markers[0]
        assert event_id == 42.0
        assert timestamp == 12345.67
        assert streams_pending == {'EEG_Stream'}

    def test_handle_data_sample_attaches_marker(self, processor_with_pending_marker):
        """Test that pending marker is attached to EEG sample."""
        processor = processor_with_pending_marker
        # Configure stream for marker distribution
        processor.stream_configs = [StreamConfig(name='EEG_Stream', type='EEG', channel_count=1, sample_rate=500)]

        # Queue a pending marker
        processor.pending_markers.append((42.0, 12345.67, {'EEG_Stream'}))

        # Create EEG sample payload with stream_name
        data_payload = {
            'data': {
                'eeg': np.random.randn(4, 1),
                'markers': np.array([[0.0]]),
            },
            'stream_name': 'EEG_Stream',
            'lsl_timestamp': 12346.0,
            '_daq_receive_ns': 123456790,
        }

        processor._handle_data_sample(data_payload)

        # Pending markers should be empty (consumed)
        assert len(processor.pending_markers) == 0

        # Mode queue should receive sample with marker
        call_args = processor.mode_queues['test_mode'].put_nowait.call_args[0][0]
        assert 'markers' in call_args
        np.testing.assert_array_equal(call_args['markers'], np.array([[42.0]]))

    def test_handle_data_sample_no_pending_marker(self, processor_with_pending_marker):
        """Test that samples without pending markers have original marker value."""
        processor = processor_with_pending_marker
        processor.stream_configs = [StreamConfig(name='EEG_Stream', type='EEG', channel_count=1, sample_rate=500)]

        # No pending markers (empty deque)
        assert len(processor.pending_markers) == 0

        data_payload = {
            'data': {
                'eeg': np.random.randn(4, 1),
                'markers': np.array([[0.0]]),
            },
            'stream_name': 'EEG_Stream',
            'lsl_timestamp': 12346.0,
            '_daq_receive_ns': 123456790,
        }

        processor._handle_data_sample(data_payload)

        # Mode queue should receive sample with original marker
        call_args = processor.mode_queues['test_mode'].put_nowait.call_args[0][0]
        assert 'markers' in call_args
        np.testing.assert_array_equal(call_args['markers'], np.array([[0.0]]))

    def test_process_next_sample_event_payload(self, processor_with_pending_marker):
        """Test that event payloads are routed to _handle_event."""
        processor = processor_with_pending_marker
        processor.stream_configs = [StreamConfig(name='EEG_Stream', type='EEG', channel_count=1, sample_rate=500)]

        event_payload = {
            'event': {
                'event_id': 99.0,
                'event_json': {'event_id': 99, 'event_type': 'test'},
            },
            'lsl_timestamp': 12345.67,
        }
        processor.data_queue.get.return_value = event_payload

        processor._process_next_sample()

        # Event should be queued (with streams_pending set)
        assert len(processor.pending_markers) == 1
        event_id, timestamp, streams_pending = processor.pending_markers[0]
        assert event_id == 99.0
        assert timestamp == 12345.67
        assert streams_pending == {'EEG_Stream'}

    def test_process_next_sample_data_payload(self, processor_with_pending_marker):
        """Test that data payloads are routed to _handle_data_sample."""
        processor = processor_with_pending_marker
        processor.stream_configs = [StreamConfig(name='EEG_Stream', type='EEG', channel_count=1, sample_rate=500)]

        data_payload = {
            'data': {
                'eeg': np.random.randn(4, 1),
                'markers': np.array([[0.0]]),
            },
            'stream_name': 'EEG_Stream',
            'lsl_timestamp': 12346.0,
        }
        processor.data_queue.get.return_value = data_payload

        processor._process_next_sample()

        # Mode queue should have received the sample
        processor.mode_queues['test_mode'].put_nowait.assert_called_once()

    def test_marker_distributed_to_all_streams(self, processor_with_pending_marker):
        """Test that markers are distributed to ALL configured streams."""
        processor = processor_with_pending_marker

        # Configure two data streams
        processor.stream_configs = [
            StreamConfig(name='EEG_Stream', type='EEG', channel_count=4, sample_rate=500),
            StreamConfig(name='EMG_Stream', type='EMG', channel_count=2, sample_rate=500),
        ]

        # Queue an event (simulates _handle_event with stream tracking)
        processor.pending_markers.append((42.0, 12345.67, {'EEG_Stream', 'EMG_Stream'}))

        # First stream sample (EEG)
        eeg_payload = {
            'data': {'eeg': np.random.randn(4, 1)},
            'stream_name': 'EEG_Stream',
            'lsl_timestamp': 12346.0,
        }
        processor._handle_data_sample(eeg_payload)

        # Marker should still be pending (waiting for EMG)
        assert len(processor.pending_markers) == 1

        # Second stream sample (EMG)
        emg_payload = {
            'data': {'emg': np.random.randn(2, 1)},
            'stream_name': 'EMG_Stream',
            'lsl_timestamp': 12346.5,
        }
        processor._handle_data_sample(emg_payload)

        # Now marker should be fully consumed
        assert len(processor.pending_markers) == 0

    def test_single_stream_marker_behavior_equivalent(self, processor_with_pending_marker):
        """Verify single-stream markers work identically to pre-refactor behavior.

        Pre-refactor: marker attached to first EEG sample, consumed immediately.
        Post-refactor: same behavior when only one data stream configured.
        """
        processor = processor_with_pending_marker

        # Configure single EEG stream (new architecture)
        processor.stream_configs = [StreamConfig(name='EEG_Stream', type='EEG', channel_count=1, sample_rate=500)]

        # Event arrives
        event_payload = {
            'event': {'event_id': 42.0},
            'lsl_timestamp': 100.0,
        }
        processor._handle_event(event_payload)

        # Verify event queued with single stream pending
        assert len(processor.pending_markers) == 1
        _, _, streams_pending = processor.pending_markers[0]
        assert streams_pending == {'EEG_Stream'}

        # First EEG sample arrives
        eeg_payload = {
            'data': {'eeg': np.random.randn(4, 1)},
            'stream_name': 'EEG_Stream',
            'lsl_timestamp': 101.0,
        }
        processor._handle_data_sample(eeg_payload)

        # Marker consumed on first sample (same as pre-refactor)
        assert len(processor.pending_markers) == 0

        # Marker attached to sample
        call_args = processor.mode_queues['test_mode'].put_nowait.call_args[0][0]
        assert 'markers' in call_args
        assert call_args['markers'][0, 0] == 42.0

    def test_marker_single_stream_consumed_immediately(self, processor_with_pending_marker):
        """Test marker distribution with single stream is consumed immediately."""
        processor = processor_with_pending_marker
        processor.stream_configs = [StreamConfig(name='EEG_Stream', type='EEG', channel_count=1, sample_rate=500)]

        # Queue marker
        processor.pending_markers.append((42.0, 12345.67, {'EEG_Stream'}))

        data_payload = {
            'data': {'eeg': np.random.randn(4, 1)},
            'stream_name': 'EEG_Stream',
            'lsl_timestamp': 12346.0,
        }
        processor._handle_data_sample(data_payload)

        # Should be consumed immediately (only one stream)
        assert len(processor.pending_markers) == 0

    def test_marker_preserved_during_downsampling_empty_eeg(self, processor_with_pending_marker):
        """Test that markers are NOT consumed when samples are skipped due to empty EEG.

        During downsampling, the preprocessor accumulates samples and only outputs
        data when enough samples have been collected. Empty samples (shape n_channels, 0)
        are skipped in _distribute_sample. This test verifies that pending markers
        survive these skipped samples and are attached to the next non-empty sample.
        """
        processor = processor_with_pending_marker
        processor.stream_configs = [StreamConfig(name='EEG_Stream', type='EEG', channel_count=1, sample_rate=500)]

        # Queue a pending marker
        processor.pending_markers.append((42.0, 12345.67, {'EEG_Stream'}))

        # First sample: empty EEG (simulates downsampling accumulation)
        empty_sample = {
            'eeg': np.zeros((4, 0)),  # Empty - will be skipped
        }
        processor._distribute_sample(empty_sample, 12346.0, None, 'EEG_Stream')

        # Marker should still be pending (not consumed)
        assert len(processor.pending_markers) == 1

        # Second sample: has data
        data_sample = {
            'eeg': np.random.randn(4, 1),
        }
        processor._distribute_sample(data_sample, 12347.0, None, 'EEG_Stream')

        # Now marker should be consumed
        assert len(processor.pending_markers) == 0

        # Mode queue should have received sample with marker
        call_args = processor.mode_queues['test_mode'].put_nowait.call_args[0][0]
        assert 'markers' in call_args
        np.testing.assert_array_equal(call_args['markers'], np.array([[42.0]]))

    def test_marker_preserved_during_downsampling_empty_emg(self):
        """Test that markers are NOT consumed when EMG-only samples are skipped.

        This verifies the generalized empty check works for EMG-only modes,
        not just EEG.
        """
        # Create processor with EMG-only mode
        processor = DataProcessor(
            data_queue=Mock(),
            plot_queue=Mock(),
            stop_event=Mock(),
            shared_state=Mock(),
            mode_configs={'emg_mode': {'required_modalities': ['emg']}},
            stream_configs=[StreamConfig(name='EMG_Stream', type='EMG', channel_count=8, sample_rate=500)],
            preprocessing_config={}
        )
        processor.mode_queues['emg_mode'] = Mock()

        # Queue a pending marker
        processor.pending_markers.append((42.0, 12345.67, {'EMG_Stream'}))

        # First sample: empty EMG (simulates downsampling accumulation)
        empty_sample = {
            'emg': np.zeros((8, 0)),  # Empty - will be skipped
        }
        processor._distribute_sample(empty_sample, 12346.0, None, 'EMG_Stream')

        # Marker should still be pending (not consumed)
        assert len(processor.pending_markers) == 1

        # Second sample: has EMG data
        data_sample = {
            'emg': np.random.randn(8, 1),
        }
        processor._distribute_sample(data_sample, 12347.0, None, 'EMG_Stream')

        # Now marker should be consumed
        assert len(processor.pending_markers) == 0

        # Mode queue should have received sample with marker
        call_args = processor.mode_queues['emg_mode'].put_nowait.call_args[0][0]
        assert 'markers' in call_args
        np.testing.assert_array_equal(call_args['markers'], np.array([[42.0]]))

    def test_marker_distributed_when_partial_modalities_have_data(self, processor_with_pending_marker):
        """Test that samples with mixed empty/non-empty modalities are distributed.

        When some modalities have data and others are empty (e.g., EEG has data but
        EMG is still accumulating), the sample should still be distributed because
        at least one modality has data.
        """
        processor = processor_with_pending_marker
        processor.stream_configs = [StreamConfig(name='EEG_Stream', type='EEG', channel_count=1, sample_rate=500)]

        # Queue a pending marker
        processor.pending_markers.append((42.0, 12345.67, {'EEG_Stream'}))

        # Sample with EEG data but empty EMG
        mixed_sample = {
            'eeg': np.random.randn(4, 1),  # Has data
            'emg': np.zeros((8, 0)),        # Empty (still accumulating)
        }
        processor._distribute_sample(mixed_sample, 12346.0, None, 'EEG_Stream')

        # Marker should be consumed (sample was distributed)
        assert len(processor.pending_markers) == 0

        # Mode queue should have received sample with marker
        call_args = processor.mode_queues['test_mode'].put_nowait.call_args[0][0]
        assert 'markers' in call_args
        np.testing.assert_array_equal(call_args['markers'], np.array([[42.0]]))

    def test_multiple_events_before_sample_all_preserved(self, processor_with_pending_marker):
        """Test that multiple events are queued, not overwritten."""
        processor = processor_with_pending_marker
        processor.stream_configs = [StreamConfig(name='EEG_Stream', type='EEG', channel_count=1, sample_rate=500)]

        # Send two events before any EEG sample
        event1 = {'event': {'event_id': 1.0}, 'lsl_timestamp': 100.0}
        event2 = {'event': {'event_id': 2.0}, 'lsl_timestamp': 101.0}

        processor._handle_event(event1)
        processor._handle_event(event2)

        # First EEG sample should get event 1
        sample1 = {
            'data': {'eeg': np.random.randn(4, 1), 'markers': np.array([[0.0]])},
            'stream_name': 'EEG_Stream',
            'lsl_timestamp': 102.0
        }
        processor._handle_data_sample(sample1)

        call1 = processor.mode_queues['test_mode'].put_nowait.call_args_list[0][0][0]
        assert call1['markers'][0, 0] == 1.0  # First event

        # Second EEG sample should get event 2
        processor.mode_queues['test_mode'].reset_mock()
        sample2 = {
            'data': {'eeg': np.random.randn(4, 1), 'markers': np.array([[0.0]])},
            'stream_name': 'EEG_Stream',
            'lsl_timestamp': 103.0
        }
        processor._handle_data_sample(sample2)

        call2 = processor.mode_queues['test_mode'].put_nowait.call_args_list[0][0][0]
        assert call2['markers'][0, 0] == 2.0  # Second event

    def test_event_queue_fifo_order(self, processor_with_pending_marker):
        """Test that events are consumed in FIFO order."""
        processor = processor_with_pending_marker
        processor.stream_configs = [StreamConfig(name='EEG_Stream', type='EEG', channel_count=1, sample_rate=500)]

        # Queue 3 events
        for i in range(1, 4):
            processor._handle_event({
                'event': {'event_id': float(i)},
                'lsl_timestamp': float(100 + i)
            })

        # Process 3 EEG samples, verify FIFO order
        for expected_id in [1.0, 2.0, 3.0]:
            processor.mode_queues['test_mode'].reset_mock()
            sample = {
                'data': {'eeg': np.random.randn(4, 1), 'markers': np.array([[0.0]])},
                'stream_name': 'EEG_Stream',
                'lsl_timestamp': 200.0
            }
            processor._handle_data_sample(sample)

            call = processor.mode_queues['test_mode'].put_nowait.call_args[0][0]
            assert call['markers'][0, 0] == expected_id


class TestPreprocessingIntegration:
    """Test suite for preprocessing integration."""

    @pytest.fixture
    def processor_with_eog(self):
        """Create processor with EOG configuration."""
        preprocessing_config = {
            'modality_preprocessing': {
                'eeg': {
                    'sample_rate': 500,
                    'lowcut': 0.5,
                    'highcut': 40.0,
                    'num_channels': 8
                }
            }
        }

        processor = DataProcessor(
            data_queue=Mock(),
            plot_queue=Mock(),
            stop_event=Mock(),
            shared_state=Mock(),
            mode_configs={'test_mode': {}},
            preprocessing_config=preprocessing_config
        )

        return processor

    def test_create_preprocessors_no_streams(self, processor_with_eog):
        """Test preprocessor creation with no stream_configs creates no preprocessors."""
        processor = processor_with_eog
        processor.stream_configs = []  # No streams

        with patch('dendrite.processing.processor.OnlinePreprocessor') as mock_preprocessor:
            processor._create_preprocessors()

            # No preprocessors created when no stream configs
            assert len(processor.preprocessors) == 0
            mock_preprocessor.assert_not_called()

    def test_create_preprocessors_with_stream_configs(self, processor_with_eog):
        """Test preprocessor creation with stream_configs (new mode)."""
        processor = processor_with_eog

        # Add stream_configs
        processor.stream_configs = [
            StreamConfig(name='BioSemi', type='EEG', sample_rate=512, channel_count=8, channel_types=[]),
            StreamConfig(name='Trigno', type='EMG', sample_rate=2000, channel_count=4, channel_types=[]),
        ]

        with patch('dendrite.processing.processor.OnlinePreprocessor') as mock_preprocessor:
            mock_instance = Mock()
            mock_preprocessor.return_value = mock_instance

            processor._create_preprocessors()

            # Should create one preprocessor per stream
            assert 'BioSemi' in processor.preprocessors
            assert 'Trigno' in processor.preprocessors
            assert mock_preprocessor.call_count == 2

    def test_create_preprocessors_error(self, processor_with_eog):
        """Test preprocessor creation error handling."""
        processor = processor_with_eog

        with patch('dendrite.processing.processor.OnlinePreprocessor', side_effect=Exception("Test error")):
            processor._create_preprocessors()

            # Should have no preprocessors on error
            assert len(processor.preprocessors) == 0


class TestErrorHandling:
    """Test suite for error handling and edge cases."""

    def test_queue_operation_errors(self):
        """Test handling of queue operation errors."""
        processor = DataProcessor(
            data_queue=Mock(),
            plot_queue=Mock(),
            stop_event=Mock(),
            shared_state=Mock(),
            mode_configs={'test_mode': {}},
            preprocessing_config={}
        )
        
        # Mock queue that raises exceptions
        error_queue = Mock()
        error_queue.full.return_value = False
        error_queue.put_nowait.side_effect = Exception("Queue error")
        processor.mode_queues['test_mode'] = error_queue
        
        sample_dict = {'test': 'data'}
        
        # Should not crash on queue errors
        processor._fan_out_sample(sample_dict)


class TestProcessLifecycle:
    """Test suite for process lifecycle management."""
    
    def test_start_returns_process(self):
        """Test that start method returns a Process object."""
        processor = DataProcessor(
            data_queue=Mock(),
            plot_queue=Mock(),
            stop_event=Mock(),
            shared_state=Mock(),
            mode_configs={'test_mode': {}},
            preprocessing_config={}
        )
        
        with patch('multiprocessing.Process') as mock_process:
            mock_instance = Mock()
            mock_process.return_value = mock_instance
            
            result = processor.start()
            
            assert result is mock_instance
            mock_process.assert_called_once()
            mock_instance.start.assert_called_once()
    


# Integration-style tests that test multiple components together
class TestDataProcessorIntegration:
    """Integration tests for DataProcessor functionality."""

    def test_end_to_end_sample_flow(self):
        """Test complete sample flow from queue to distribution (per-sample)."""
        # Create processor with real-ish setup
        data_queue = queue.Queue()
        plot_queue = queue.Queue()
        stop_event = threading.Event()

        processor = DataProcessor(
            data_queue=data_queue,
            plot_queue=plot_queue,
            stop_event=stop_event,
            shared_state=Mock(),
            mode_configs={'test_mode': {}},
            preprocessing_config={}
        )

        # Add test data - DAQ sends dict format
        for i in range(5):
            sample_payload = {
                'data': {
                    'eeg': np.array([[i], [i+1], [i+2]], dtype=float),  # (n_ch, 1)
                },
                'lsl_timestamp': 1000.0 + i * 0.002
            }
            data_queue.put(sample_payload)

        # Mock mode queue to avoid multiprocessing issues
        mock_mode_queue = Mock()
        mock_mode_queue.full.return_value = False
        processor.mode_queues['test_mode'] = mock_mode_queue

        # Process all samples (per-sample, no chunking)
        for _ in range(5):
            processor._process_next_sample()

        # Mode queue should have received all 5 samples
        assert mock_mode_queue.put_nowait.call_count == 5

        # Verify sample structure - modality keys are lowercase
        calls = mock_mode_queue.put_nowait.call_args_list
        for call in calls:
            sample_dict = call[0][0]
            assert 'eeg' in sample_dict  # lowercase from DAQ
            assert sample_dict['eeg'].shape == (3, 1)  # 3 channels, 1 sample


class TestChannelTypesFlow:
    """Test that user-changed channel_types flow through the entire pipeline."""

    def test_channel_types_preserved_through_processor(self):
        """Test that channel_types are preserved when passed to DataProcessor."""
        from dendrite.data.stream_schemas import StreamConfig

        # Create stream with mixed channel types (user changed some EEG to EOG)
        stream = StreamConfig(
            name='BioSemi',
            type='EEG',
            channel_count=5,
            sample_rate=512.0,
            channel_types=['eeg', 'eeg', 'eeg', 'eog', 'eog']
        )

        # DataProcessor now stores StreamConfig objects directly
        processor = DataProcessor(
            data_queue=Mock(),
            plot_queue=Mock(),
            stop_event=Mock(),
            shared_state=Mock(),
            mode_configs={'test_mode': {}},
            stream_configs=[stream],
        )

        # Check stream config preserved (no longer converts to dict)
        stored = processor.stream_configs[0]
        assert stored.channel_types == ['eeg', 'eeg', 'eeg', 'eog', 'eog']
        assert stored.channel_count == 5

    def test_modality_preprocessing_built_from_channel_types(self):
        """Test that processor builds correct modality configs from channel_types."""
        processor = DataProcessor(
            data_queue=Mock(),
            plot_queue=Mock(),
            stop_event=Mock(),
            shared_state=Mock(),
            mode_configs={'test_mode': {}},
            preprocessing_config={
                'modality_preprocessing': {
                    'eeg': {'sample_rate': 512, 'lowcut': 0.5, 'highcut': 40.0},
                    'eog': {'sample_rate': 512, 'lowcut': 0.1, 'highcut': 15.0}
                }
            }
        )

        # Stream with mixed channel types
        stream_config = StreamConfig(
            name='BioSemi', type='EEG', channel_count=5,
            sample_rate=512.0, channel_types=['eeg', 'eeg', 'eeg', 'eog', 'eog']
        )

        modality_preprocessing = processor._build_modality_preprocessing_from_stream(stream_config)

        # Should have configs for both EEG and EOG
        assert 'eeg' in modality_preprocessing
        assert 'eog' in modality_preprocessing

        # EEG should have 3 channels
        assert modality_preprocessing['eeg']['num_channels'] == 3
        assert modality_preprocessing['eeg']['sample_rate'] == 512.0

        # EOG should have 2 channels
        assert modality_preprocessing['eog']['num_channels'] == 2
        assert modality_preprocessing['eog']['sample_rate'] == 512.0

    def test_modality_preprocessing_with_markers_channel(self):
        """Test that markers channel is excluded from modality configs."""
        processor = DataProcessor(
            data_queue=Mock(),
            plot_queue=Mock(),
            stop_event=Mock(),
            shared_state=Mock(),
            mode_configs={'test_mode': {}},
            preprocessing_config={'modality_preprocessing': {}}
        )

        # Stream with markers channel
        stream_config = StreamConfig(
            name='EEG_with_markers', type='EEG', channel_count=4,
            sample_rate=512.0, channel_types=['eeg', 'eeg', 'eeg', 'markers']
        )

        modality_preprocessing = processor._build_modality_preprocessing_from_stream(stream_config)

        # Should have EEG but NOT markers
        assert 'eeg' in modality_preprocessing
        assert 'markers' not in modality_preprocessing
        assert modality_preprocessing['eeg']['num_channels'] == 3

    def test_case_insensitive_channel_types(self):
        """Test that channel types are normalized to lowercase."""
        processor = DataProcessor(
            data_queue=Mock(),
            plot_queue=Mock(),
            stop_event=Mock(),
            shared_state=Mock(),
            mode_configs={'test_mode': {}},
            preprocessing_config={'modality_preprocessing': {}}
        )

        # Mixed case channel types
        stream_config = StreamConfig(
            name='BioSemi', type='EEG', channel_count=4,
            sample_rate=512.0, channel_types=['EEG', 'Eeg', 'eeg', 'EOG']
        )

        modality_preprocessing = processor._build_modality_preprocessing_from_stream(stream_config)

        # All EEG variants should be grouped together
        assert 'eeg' in modality_preprocessing
        assert 'eog' in modality_preprocessing
        assert modality_preprocessing['eeg']['num_channels'] == 3
        assert modality_preprocessing['eog']['num_channels'] == 1

    def test_unknown_modality_gets_config(self):
        """Test that unknown modality types still get minimal config."""
        processor = DataProcessor(
            data_queue=Mock(),
            plot_queue=Mock(),
            stop_event=Mock(),
            shared_state=Mock(),
            mode_configs={'test_mode': {}},
            preprocessing_config={'modality_preprocessing': {}}
        )

        # Custom modality type (e.g., ECG)
        stream_config = StreamConfig(
            name='Custom', type='Other', channel_count=2,
            sample_rate=256.0, channel_types=['ecg', 'ecg']
        )

        modality_preprocessing = processor._build_modality_preprocessing_from_stream(stream_config)

        # Unknown modality should still get minimal config
        assert 'ecg' in modality_preprocessing
        assert modality_preprocessing['ecg']['num_channels'] == 2
        assert modality_preprocessing['ecg']['sample_rate'] == 256.0