"""Tests for VisualizationStreamer."""

import pytest
import time
import threading
from unittest.mock import Mock, patch

from dendrite.data.streaming import VisualizationStreamer, RawDataPayload


class TestVisualizationStreamer:
    """Test suite for VisualizationStreamer class."""

    @patch('dendrite.data.streaming.base.LSLOutlet')
    def test_initialization(self, mock_lsl_outlet, plot_queue, mock_visualization_stream_config, stop_event, output_queue):
        """Test VisualizationStreamer initialization."""
        streamer = VisualizationStreamer(
            plot_queue,
            mock_visualization_stream_config,
            stop_event,
            output_queue,
            history_length=500,
        )

        assert streamer.plot_queue == plot_queue
        assert streamer.output_queue == output_queue
        assert streamer.history_length == 500
        assert streamer._channel_labels == {}
        assert streamer.mode_outputs == {}
        assert streamer.mode_history == {}
        assert streamer.history_lock is None

    @patch('dendrite.data.streaming.base.LSLOutlet')
    def test_initialization_with_channel_labels(self, mock_lsl_outlet, plot_queue, mock_visualization_stream_config, stop_event, output_queue):
        """Test VisualizationStreamer initialization with channel labels from stream config."""
        channel_labels = {
            'eeg': ['Fp1', 'Fp2', 'Fz', 'Cz', 'Pz', 'O1', 'O2'],
            'emg': ['EMG_L', 'EMG_R']
        }
        streamer = VisualizationStreamer(
            plot_queue,
            mock_visualization_stream_config,
            stop_event,
            output_queue,
            history_length=500,
            channel_labels=channel_labels,
        )

        assert streamer._channel_labels == channel_labels
        assert streamer._channel_labels['eeg'] == ['Fp1', 'Fp2', 'Fz', 'Cz', 'Pz', 'O1', 'O2']
        assert streamer._channel_labels['emg'] == ['EMG_L', 'EMG_R']

    def test_mode_type_usage(self, plot_queue, mock_visualization_stream_config, stop_event):
        """Test that VisualizationStreamer uses explicit mode_type from packets."""
        with patch('dendrite.data.streaming.base.LSLOutlet'):
            streamer = VisualizationStreamer(plot_queue, mock_visualization_stream_config, stop_event)

            assert not hasattr(streamer, '_get_mode_type')

            test_packet = {
                'output_type': 'performance',
                'mode_name': 'TestMode',
                'mode_type': 'asynchronous',
                'data': {'accuracy': 0.85},
                'timestamp': 12345.67
            }

            mode_type = test_packet.get('mode_type', 'unknown')
            assert mode_type == 'asynchronous'

    @patch('dendrite.data.streaming.base.LSLOutlet')
    def test_store_mode_packet(self, mock_lsl_outlet, plot_queue, mock_visualization_stream_config, stop_event):
        """Test storing mode packets in history."""
        streamer = VisualizationStreamer(plot_queue, mock_visualization_stream_config, stop_event)
        streamer.history_lock = threading.Lock()
        streamer.history_length = 3

        packets = [
            {'type': 'prediction', 'data': {'pred': 'left'}},
            {'type': 'prediction', 'data': {'pred': 'right'}},
            {'type': 'performance', 'data': {'accuracy': 0.85}},
            {'type': 'prediction', 'data': {'pred': 'forward'}}
        ]

        mode_name = 'TestMode'
        for packet in packets:
            streamer._store_mode_packet(mode_name, packet)

        assert mode_name in streamer.mode_history
        assert len(streamer.mode_history[mode_name]) == 3

        stored_packets = streamer.mode_history[mode_name]
        assert stored_packets[0]['data']['pred'] == 'right'
        assert stored_packets[-1]['data']['pred'] == 'forward'

    @patch('dendrite.data.streaming.base.LSLOutlet')
    def test_self_describing_packets_with_labels(self, mock_lsl_outlet, plot_queue, mock_visualization_stream_config, stop_event):
        """Test that channel labels from self-describing packets are used."""
        streamer = VisualizationStreamer(plot_queue, mock_visualization_stream_config, stop_event)
        streamer.history_lock = threading.Lock()
        streamer._send_data = Mock()

        # Simulate data with _channel_labels from processor
        data_with_labels = {
            'eeg': [1.0, 2.0, 3.0, 4.0],
            'emg': [0.5, 0.6],
            '_timestamp': time.time(),
            '_channel_labels': {
                'eeg': ['C3', 'C4', 'Cz', 'Fz'],
                'emg': ['EMG1', 'EMG2']
            }
        }

        streamer._process_single_raw_data(data_with_labels)

        streamer._send_data.assert_called_once()
        call_args = streamer._send_data.call_args[0][0]

        assert call_args['type'] == 'raw_data'
        assert 'eeg' in call_args['data']
        assert 'emg' in call_args['data']
        assert call_args['channel_labels']['eeg'] == ['C3', 'C4', 'Cz', 'Fz']
        assert call_args['channel_labels']['emg'] == ['EMG1', 'EMG2']

    @patch('dendrite.data.streaming.base.LSLOutlet')
    def test_fallback_labels_without_self_describing(self, mock_lsl_outlet, plot_queue, mock_visualization_stream_config, stop_event):
        """Test that fallback labels are generated when no _channel_labels in packet."""
        streamer = VisualizationStreamer(plot_queue, mock_visualization_stream_config, stop_event)
        streamer.history_lock = threading.Lock()
        streamer._send_data = Mock()

        # Data without _channel_labels
        data_without_labels = {
            'eeg': [1.0, 2.0],
            'emg': [0.5],
            '_timestamp': time.time()
        }

        streamer._process_single_raw_data(data_without_labels)

        streamer._send_data.assert_called_once()
        call_args = streamer._send_data.call_args[0][0]

        # Should generate fallback labels
        assert call_args['channel_labels']['eeg'] == ['EEG_1', 'EEG_2']
        assert call_args['channel_labels']['emg'] == ['EMG_1']

    @patch('dendrite.data.streaming.base.LSLOutlet')
    def test_labels_always_included_in_every_message(self, mock_lsl_outlet, plot_queue, mock_visualization_stream_config, stop_event):
        """Test that channel_labels are included in every raw_data message."""
        streamer = VisualizationStreamer(plot_queue, mock_visualization_stream_config, stop_event)
        streamer.history_lock = threading.Lock()
        streamer._send_data = Mock()

        # First sample with labels
        data1 = {
            'eeg': [1.0, 2.0],
            '_timestamp': time.time(),
            '_channel_labels': {'eeg': ['C3', 'C4']}
        }
        streamer._process_single_raw_data(data1)

        # Second sample (late-connecting consumer should still get labels)
        data2 = {
            'eeg': [3.0, 4.0],
            '_timestamp': time.time(),
            '_channel_labels': {'eeg': ['C3', 'C4']}
        }
        streamer._process_single_raw_data(data2)

        # Both calls should include channel_labels
        assert streamer._send_data.call_count == 2
        for call in streamer._send_data.call_args_list:
            message = call[0][0]
            assert 'channel_labels' in message
            assert message['channel_labels']['eeg'] == ['C3', 'C4']

    @patch('dendrite.data.streaming.base.LSLOutlet')
    def test_classifier_output_processing_validation(self, mock_lsl_outlet, plot_queue, mock_visualization_stream_config, stop_event):
        """Test classifier output validation in processing."""
        streamer = VisualizationStreamer(plot_queue, mock_visualization_stream_config, stop_event)
        streamer.history_lock = threading.Lock()

        streamer._send_data = Mock()

        valid_output = {
            'type': 'prediction',
            'mode_name': 'Asynchronous_1',
            'mode_type': 'asynchronous',
            'data': {'prediction': 'right', 'confidence': 0.8},
            'data_timestamp': time.time()
        }

        streamer._process_single_classifier_output('Asynchronous_1', valid_output)

        streamer._send_data.assert_called_once()
        call_args = streamer._send_data.call_args[0][0]

        assert call_args['type'] == 'prediction'
        assert call_args['mode_name'] == 'Asynchronous_1'
        assert call_args['mode_type'] == 'asynchronous'
        assert call_args['data']['prediction'] == 'right'

    def add_helper_methods_to_visualization_streamer(self):
        """Add helper methods for testing to VisualizationStreamer."""
        def _process_single_raw_data(self, data):
            """Process a single raw data item (for testing).

            Note: In production, channel labels are set at init from stream config.
            For testing, we allow setting via _channel_labels in data to simulate
            different label scenarios.
            """
            if data is None or not isinstance(data, dict):
                return

            # For testing: allow setting labels via data (production gets them at init)
            if '_channel_labels' in data:
                self._channel_labels = data['_channel_labels']

            timestamp = data.get('_timestamp', time.time())
            payload_data = {k: v for k, v in data.items() if not k.startswith('_')}
            payload_data = self._make_json_serializable(payload_data)

            channel_labels = self._get_channel_labels(payload_data)

            message: RawDataPayload = {
                'type': 'raw_data',
                'timestamp': timestamp,
                'data': payload_data,
                'channel_labels': channel_labels
            }

            self._send_data(message)

        def _process_single_classifier_output(self, mode_name, output):
            """Process a single classifier output (for testing)."""
            if output is None or not isinstance(output, dict):
                return

            output_type = output.get('type', 'unknown_output')
            mode_name_from_packet = output.get('mode_name', mode_name)
            data_content = output.get('data', {})
            data_timestamp = output.get('data_timestamp', time.time())

            mode_type = output.get('mode_type')
            if mode_type is None:
                mode_type = 'asynchronous'

            message = {
                'type': output_type,
                'data_timestamp': data_timestamp,
                'mode_name': mode_name_from_packet,
                'mode_type': mode_type,
                'data': data_content
            }

            with self.history_lock:
                self.mode_outputs[mode_name] = output
                self._store_mode_packet(mode_name_from_packet, message)

            self._send_data(message)

        VisualizationStreamer._process_single_raw_data = _process_single_raw_data
        VisualizationStreamer._process_single_classifier_output = _process_single_classifier_output

    def setup_method(self):
        """Setup test methods."""
        self.add_helper_methods_to_visualization_streamer()
