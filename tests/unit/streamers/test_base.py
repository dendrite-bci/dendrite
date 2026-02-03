"""Tests for BaseOutputStreamer and LSLBaseStreamer."""

import pytest
import time
import json
import multiprocessing
from unittest.mock import Mock, patch

from dendrite.data.streaming.base import BaseOutputStreamer, LSLBaseStreamer
from dendrite.data.streaming.socket import SocketStreamer


class MockOutputStreamer(BaseOutputStreamer):
    """Concrete implementation of BaseOutputStreamer for testing."""

    def __init__(self, input_queue, stream_name="TestStreamer", stop_event=None):
        super().__init__(input_queue, stream_name, stop_event)
        self.initialized = False
        self.sent_data = []
        self.cleaned_up = False

    def _initialize_output(self):
        """Mock initialization."""
        self.initialized = True

    def _send_data(self, data):
        """Mock data sending."""
        if data is None:
            return
        self.sent_data.append(data)
        if isinstance(data, str):
            self.bytes_sent += len(data.encode('utf-8'))
        else:
            self.bytes_sent += len(str(data).encode('utf-8'))

    def _cleanup(self):
        """Mock cleanup."""
        self.cleaned_up = True


class TestBaseOutputStreamer:
    """Test suite for BaseOutputStreamer abstract base class."""

    def test_initialization(self, input_queue, stop_event):
        """Test BaseOutputStreamer initialization."""
        streamer = MockOutputStreamer(input_queue, "TestStream", stop_event)

        assert streamer.input_queue == input_queue
        assert streamer.stream_name == "TestStream"
        assert streamer.stop_event == stop_event
        assert streamer.messages_sent == 0
        assert streamer.bytes_sent == 0
        assert streamer.start_time is None
        assert hasattr(streamer, 'logger')

    def test_process_name_setting(self, input_queue, stop_event):
        """Test that process name is set correctly."""
        streamer = MockOutputStreamer(input_queue, "CustomName", stop_event)
        assert streamer.name == "CustomNameStreamer"

    def test_cannot_instantiate_abstract_base(self, input_queue):
        """Test that BaseOutputStreamer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseOutputStreamer(input_queue, "Test")

    def test_make_json_serializable_with_numpy_arrays(self, sample_numpy_data):
        """Test JSON serialization of numpy arrays."""
        streamer = MockOutputStreamer(multiprocessing.Queue(), "Test")

        result = streamer._make_json_serializable(sample_numpy_data)

        assert isinstance(result['array_2d'], list)
        assert result['array_2d'] == [[1, 2], [3, 4]]
        assert isinstance(result['array_1d'], list)
        assert result['array_1d'] == [1.5, 2.5, 3.5]
        assert isinstance(result['scalar'], float)
        assert result['scalar'] == 42.0
        assert isinstance(result['nested']['inner_array'], list)
        assert result['nested']['inner_array'] == [7, 8, 9]
        assert isinstance(result['nested']['value'], int)
        assert result['nested']['value'] == 123

    def test_make_json_serializable_with_none(self):
        """Test JSON serialization with None input."""
        streamer = MockOutputStreamer(multiprocessing.Queue(), "Test")
        result = streamer._make_json_serializable(None)
        assert result == {}

    def test_make_json_serializable_with_complex_objects(self):
        """Test JSON serialization with complex objects falls back to str()."""
        streamer = MockOutputStreamer(multiprocessing.Queue(), "Test")

        class CustomObject:
            def __init__(self):
                self.value = "test"
                self.number = 42

        obj = CustomObject()
        result = streamer._make_json_serializable(obj)

        # Unknown objects are converted to string representation
        assert isinstance(result, str)
        assert "CustomObject" in result

    def test_make_json_serializable_with_unserializable_objects(self):
        """Test JSON serialization with unserializable objects falls back to str()."""
        streamer = MockOutputStreamer(multiprocessing.Queue(), "Test")

        def test_func():
            pass

        result = streamer._make_json_serializable(test_func)
        # Functions are converted to string representation
        assert isinstance(result, str)
        assert "function" in result

class TestLSLBaseStreamer:
    """Test suite for LSLBaseStreamer class."""

    def test_initialization(self, input_queue, mock_lsl_stream_config, stop_event):
        """Test LSLBaseStreamer initialization."""
        with patch('dendrite.data.streaming.base.LSLOutlet'):
            streamer = LSLBaseStreamer(
                input_queue,
                mock_lsl_stream_config,
                stop_event,
                bandwidth_reporting_interval=30.0
            )

            assert streamer.input_queue == input_queue
            assert streamer.stream_info == mock_lsl_stream_config
            assert streamer.stop_event == stop_event
            assert streamer.bandwidth_reporting_interval == 30.0
            assert streamer.streamer is None
            assert streamer.bandwidth_lock is None

    @patch('dendrite.data.streaming.base.LSLOutlet')
    @patch('threading.Thread')
    def test_initialize_output(self, mock_thread, mock_lsl_outlet, input_queue, mock_lsl_stream_config, stop_event):
        """Test LSL output initialization."""
        streamer = LSLBaseStreamer(input_queue, mock_lsl_stream_config, stop_event)

        mock_outlet = Mock()
        mock_lsl_outlet.return_value = mock_outlet

        mock_bandwidth_thread = Mock()
        mock_thread.return_value = mock_bandwidth_thread

        streamer._initialize_output()

        mock_lsl_outlet.assert_called_once_with(mock_lsl_stream_config)
        assert streamer.streamer == mock_outlet

        mock_thread.assert_called_once()
        mock_bandwidth_thread.start.assert_called_once()

    @patch('dendrite.data.streaming.base.LSLOutlet')
    def test_send_data_valid_json(self, mock_lsl_outlet, input_queue, mock_lsl_stream_config, stop_event):
        """Test sending valid JSON data via LSL."""
        streamer = LSLBaseStreamer(input_queue, mock_lsl_stream_config, stop_event)

        mock_outlet = Mock()
        mock_lsl_outlet.return_value = mock_outlet
        streamer._initialize_output()

        test_data = {'prediction': 'right', 'confidence': 0.85}

        with patch('threading.Lock'):
            streamer._send_data(test_data)

        mock_outlet.push_sample.assert_called_once()
        args = mock_outlet.push_sample.call_args[0]
        assert len(args[0]) == 1

        json_data = json.loads(args[0][0])
        assert json_data['prediction'] == 'right'
        assert json_data['confidence'] == 0.85

    @patch('dendrite.data.streaming.base.LSLOutlet')
    def test_send_data_with_none(self, mock_lsl_outlet, input_queue, mock_lsl_stream_config, stop_event):
        """Test sending None data doesn't cause errors."""
        streamer = LSLBaseStreamer(input_queue, mock_lsl_stream_config, stop_event)

        mock_outlet = Mock()
        mock_lsl_outlet.return_value = mock_outlet
        streamer._initialize_output()

        with patch('threading.Lock'):
            streamer._send_data(None)

        mock_outlet.push_sample.assert_not_called()

    @patch('dendrite.data.streaming.base.LSLOutlet')
    def test_cleanup(self, mock_lsl_outlet, input_queue, mock_lsl_stream_config, stop_event):
        """Test LSL resource cleanup."""
        streamer = LSLBaseStreamer(input_queue, mock_lsl_stream_config, stop_event)

        mock_outlet = Mock()
        mock_lsl_outlet.return_value = mock_outlet

        with patch('threading.Thread'):
            streamer._initialize_output()

        assert streamer.streamer == mock_outlet
        streamer._cleanup()

    def test_format_bandwidth(self, input_queue, mock_lsl_stream_config, stop_event):
        """Test bandwidth formatting utility."""
        with patch('dendrite.data.streaming.base.LSLOutlet'):
            streamer = LSLBaseStreamer(input_queue, mock_lsl_stream_config, stop_event)

            assert "B/s" in streamer._format_bandwidth(100)
            assert "KB/s" in streamer._format_bandwidth(2048)
            assert "MB/s" in streamer._format_bandwidth(2097152)

    @patch('dendrite.data.streaming.base.LSLOutlet')
    def test_report_bandwidth(self, mock_lsl_outlet, input_queue, mock_lsl_stream_config, stop_event):
        """Test bandwidth reporting."""
        streamer = LSLBaseStreamer(input_queue, mock_lsl_stream_config, stop_event)
        streamer._initialize_output()

        streamer.bytes_sent = 1024
        streamer.last_report_time = time.time() - 5

        with patch.object(streamer, 'logger') as mock_logger:
            streamer._report_bandwidth()

            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            assert "Bandwidth usage" in call_args
            assert streamer.stream_info.name in call_args

    @patch('dendrite.data.streaming.base.LSLOutlet')
    def test_send_data_error_logs(self, mock_lsl_outlet, input_queue, mock_lsl_stream_config, stop_event):
        """Test that LSL send errors are logged."""
        mock_outlet = Mock()
        mock_outlet.push_sample.side_effect = Exception("LSL error")
        mock_lsl_outlet.return_value = mock_outlet

        streamer = LSLBaseStreamer(input_queue, mock_lsl_stream_config, stop_event)
        streamer._initialize_output()

        with patch.object(streamer, 'logger') as mock_logger:
            streamer._send_data({'test': 'data'})
            mock_logger.error.assert_called_once()


class TestSocketStreamerErrors:
    """Test suite for SocketStreamer error handling."""

    @patch('socket.socket')
    def test_permission_error_on_bind(self, mock_socket_class, input_queue, stop_event):
        """Test socket permission error handling."""
        socket_config = {
            'protocol': 'TCP',
            'ip': '127.0.0.1',
            'port': 80
        }

        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket
        permission_error = OSError("Permission denied")
        permission_error.errno = 13
        mock_socket.bind.side_effect = permission_error

        streamer = SocketStreamer(input_queue, stop_event, socket_config)

        with pytest.raises(OSError):
            streamer._initialize_output()
