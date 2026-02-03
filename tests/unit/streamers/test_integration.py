"""Tests for multiprocessing integration and performance."""

import pytest
import time
import numpy as np
import multiprocessing
from unittest.mock import Mock, patch

from dendrite.data.streaming.base import LSLBaseStreamer
from dendrite.data.streaming.lsl import LSLStreamer
from dendrite.data.lsl_helpers import StreamConfig
from .test_base import MockOutputStreamer

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from tests.conftest import cleanup_process


class TestMultiProcessingIntegration:
    """Test suite for multiprocessing integration."""

    def teardown_method(self):
        """Cleanup after each test method."""
        time.sleep(0.1)
        try:
            active_children = multiprocessing.active_children()
            if active_children:
                for child in active_children:
                    cleanup_process(child)
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def test_process_lifecycle(self, input_queue, stop_event):
        """Test complete process lifecycle."""
        streamer = MockOutputStreamer(input_queue, "LifecycleTest", stop_event)

        assert isinstance(streamer, multiprocessing.Process)
        assert not streamer.is_alive()

        assert hasattr(streamer, 'start')
        assert hasattr(streamer, 'join')
        assert hasattr(streamer, 'terminate')

    def test_queue_communication_setup(self, input_queue, stop_event):
        """Test queue communication setup."""
        streamer = MockOutputStreamer(input_queue, "QueueTest", stop_event)

        assert streamer.input_queue == input_queue

        assert hasattr(input_queue, 'get')
        assert hasattr(input_queue, 'put')
        assert hasattr(input_queue, 'empty')

    def test_stop_event_functionality(self, input_queue):
        """Test stop event functionality."""
        stop_event = multiprocessing.Event()
        streamer = MockOutputStreamer(input_queue, "StopTest", stop_event)

        assert not stop_event.is_set()

        stop_event.set()
        assert stop_event.is_set()

        assert streamer.stop_event == stop_event

    @patch('dendrite.data.streaming.base.LSLOutlet')
    def test_multiple_streamers_isolation(self, mock_lsl_outlet, stop_event):
        """Test that multiple streamers are properly isolated."""
        queue1 = multiprocessing.Queue()
        queue2 = multiprocessing.Queue()

        config1 = Mock(spec=StreamConfig)
        config1.name = "Stream1"
        config1.type = "Type1"
        config1.source_id = "source1"

        config2 = Mock(spec=StreamConfig)
        config2.name = "Stream2"
        config2.type = "Type2"
        config2.source_id = "source2"

        streamer1 = LSLStreamer(queue1, config1, stop_event)
        streamer2 = LSLStreamer(queue2, config2, stop_event)

        assert streamer1.stream_info.name != streamer2.stream_info.name
        assert streamer1.input_queue != streamer2.input_queue

        assert streamer1 != streamer2
        assert streamer1.name != streamer2.name

        try:
            while not queue1.empty():
                queue1.get_nowait()
            while not queue2.empty():
                queue2.get_nowait()
        except Exception:
            pass


class TestPerformanceAndReliability:
    """Test suite for performance and reliability aspects."""

    def test_bandwidth_calculation_accuracy(self, input_queue, mock_lsl_stream_config, stop_event):
        """Test bandwidth calculation accuracy."""
        with patch('dendrite.data.streaming.base.LSLOutlet'):
            streamer = LSLBaseStreamer(input_queue, mock_lsl_stream_config, stop_event)

            assert "500.00 B/s" == streamer._format_bandwidth(500)
            assert "2.00 KB/s" == streamer._format_bandwidth(2048)
            assert "1.00 MB/s" == streamer._format_bandwidth(1048576)

    def test_message_counting_accuracy(self, input_queue, stop_event):
        """Test message counting accuracy."""
        streamer = MockOutputStreamer(input_queue, "CountTest", stop_event)

        test_data = [{'msg': i} for i in range(5)]

        for data in test_data:
            streamer._send_data(data)

        assert len(streamer.sent_data) == 5
        assert all(data in streamer.sent_data for data in test_data)

    def test_large_data_payload_handling(self, input_queue, stop_event):
        """Test handling of large data payloads."""
        streamer = MockOutputStreamer(input_queue, "LargeDataTest", stop_event)

        large_array = np.random.randn(1000, 64)
        large_data = {
            'EEG': large_array.tolist(),
            'metadata': {'info': 'large dataset test'},
            'timestamp': time.time()
        }

        serializable_data = streamer._make_json_serializable(large_data)

        assert isinstance(serializable_data['EEG'], list)
        assert len(serializable_data['EEG']) == 1000
        assert len(serializable_data['EEG'][0]) == 64
        assert serializable_data['metadata']['info'] == 'large dataset test'

    def test_rapid_message_processing(self, input_queue, stop_event):
        """Test processing many messages rapidly."""
        streamer = MockOutputStreamer(input_queue, "RapidTest", stop_event)

        num_messages = 100
        start_time = time.time()

        for i in range(num_messages):
            streamer._send_data({'sequence': i, 'data': f'message_{i}'})

        end_time = time.time()

        assert len(streamer.sent_data) == num_messages

        for i, data in enumerate(streamer.sent_data):
            assert data['sequence'] == i

        processing_time = end_time - start_time
        assert processing_time < 1.0
