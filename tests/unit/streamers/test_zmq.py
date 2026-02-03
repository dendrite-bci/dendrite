"""Tests for ZMQStreamer."""

import pytest
import json
from unittest.mock import Mock, patch

from dendrite.data.streaming import ZMQStreamer


class TestZMQStreamer:
    """Test suite for ZMQStreamer class."""

    def test_initialization_with_defaults(self, input_queue, stop_event):
        """Test ZMQStreamer initialization with default config."""
        streamer = ZMQStreamer(input_queue, stop_event)

        assert streamer.ip == '127.0.0.1'
        assert streamer.port == 5556
        assert streamer.message_format == 'JSON'
        assert streamer.context is None
        assert streamer.publisher is None

    def test_initialization_with_zmq_config(self, input_queue, stop_event):
        """Test ZMQStreamer initialization with custom config."""
        zmq_config = {
            'ip': '192.168.1.50',
            'port': 7777,
            'message_format': 'Binary'
        }

        streamer = ZMQStreamer(input_queue, stop_event, zmq_config)

        assert streamer.ip == '192.168.1.50'
        assert streamer.port == 7777
        assert streamer.message_format == 'Binary'

    @patch('dendrite.data.streaming.zmq.HAS_ZMQ', True)
    def test_initialize_output_with_zmq_available(self, input_queue, stop_event):
        """Test ZMQ initialization when ZMQ is available."""
        with patch('builtins.__import__') as mock_import:
            mock_zmq = Mock()
            mock_context = Mock()
            mock_publisher = Mock()
            mock_zmq.Context.return_value = mock_context
            mock_context.socket.return_value = mock_publisher
            mock_zmq.PUB = "PUB_CONSTANT"

            def mock_import_func(name, *args, **kwargs):
                if name == 'zmq':
                    return mock_zmq
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = mock_import_func

            streamer = ZMQStreamer(input_queue, stop_event)
            streamer._initialize_output()

            mock_zmq.Context.assert_called_once()
            mock_context.socket.assert_called_once_with("PUB_CONSTANT")
            mock_publisher.bind.assert_called_once_with("tcp://127.0.0.1:5556")

            assert streamer.context == mock_context
            assert streamer.publisher == mock_publisher

    @patch('dendrite.data.streaming.zmq.HAS_ZMQ', False)
    def test_initialize_output_without_zmq(self, input_queue, stop_event):
        """Test ZMQ initialization when ZMQ is not available."""
        streamer = ZMQStreamer(input_queue, stop_event)

        with pytest.raises(RuntimeError, match="ZeroMQ not available"):
            streamer._initialize_output()

    @patch('dendrite.data.streaming.zmq.HAS_ZMQ', True)
    def test_send_data_json_format(self, input_queue, stop_event):
        """Test sending data in JSON format."""
        mock_publisher = Mock()

        streamer = ZMQStreamer(input_queue, stop_event)
        streamer.publisher = mock_publisher
        streamer.message_format = 'JSON'

        test_data = {'prediction': 'left', 'confidence': 0.75}

        streamer._send_data(test_data)

        mock_publisher.send_string.assert_called_once()
        sent_json = mock_publisher.send_string.call_args[0][0]

        parsed_data = json.loads(sent_json)
        assert parsed_data['prediction'] == 'left'
        assert parsed_data['confidence'] == 0.75

    @patch('dendrite.data.streaming.zmq.HAS_ZMQ', True)
    def test_send_data_binary_format(self, input_queue, stop_event):
        """Test sending data in binary format."""
        mock_publisher = Mock()

        streamer = ZMQStreamer(input_queue, stop_event)
        streamer.publisher = mock_publisher
        streamer.message_format = 'Binary'

        test_data = {'prediction': 'right'}

        streamer._send_data(test_data)

        mock_publisher.send.assert_called_once()
        sent_bytes = mock_publisher.send.call_args[0][0]

        assert isinstance(sent_bytes, bytes)
        parsed_data = json.loads(sent_bytes.decode('utf-8'))
        assert parsed_data['prediction'] == 'right'

    @patch('dendrite.data.streaming.zmq.HAS_ZMQ', True)
    def test_cleanup(self, input_queue, stop_event):
        """Test ZMQ resource cleanup."""
        mock_context = Mock()
        mock_publisher = Mock()

        streamer = ZMQStreamer(input_queue, stop_event)
        streamer.context = mock_context
        streamer.publisher = mock_publisher

        streamer._cleanup()

        mock_publisher.close.assert_called_once()
        mock_context.term.assert_called_once()

    @patch('dendrite.data.streaming.zmq.HAS_ZMQ', False)
    def test_cleanup_without_zmq(self, input_queue, stop_event):
        """Test cleanup when ZMQ is not available."""
        streamer = ZMQStreamer(input_queue, stop_event)
        streamer._cleanup()
