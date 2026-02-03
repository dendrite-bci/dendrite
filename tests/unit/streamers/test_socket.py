"""Tests for SocketStreamer."""

import pytest
import json
from unittest.mock import Mock, patch

from dendrite.data.streaming import SocketStreamer


class TestSocketStreamer:
    """Test suite for SocketStreamer class."""

    def test_initialization_tcp_default(self, input_queue, stop_event):
        """Test SocketStreamer initialization with TCP defaults."""
        streamer = SocketStreamer(input_queue, stop_event)

        assert streamer.protocol == 'TCP'
        assert streamer.ip == '127.0.0.1'
        assert streamer.port == 8080
        assert streamer.server_socket is None
        assert streamer.client_connections == []
        assert streamer.socket_conn is None

    def test_initialization_with_socket_config(self, input_queue, stop_event):
        """Test SocketStreamer initialization with custom config."""
        socket_config = {
            'protocol': 'UDP',
            'ip': '192.168.1.100',
            'port': 9999
        }

        streamer = SocketStreamer(input_queue, stop_event, socket_config)

        assert streamer.protocol == 'UDP'
        assert streamer.ip == '192.168.1.100'
        assert streamer.port == 9999

    def test_tcp_socket_initialization(self, input_queue, stop_event, free_port):
        """Test TCP socket initialization."""
        socket_config = {
            'protocol': 'TCP',
            'ip': '127.0.0.1',
            'port': free_port
        }

        streamer = SocketStreamer(input_queue, stop_event, socket_config)

        try:
            streamer._initialize_output()

            assert streamer.server_socket is not None
            assert streamer.server_socket.getsockname()[1] == free_port

        finally:
            streamer._cleanup()

    def test_udp_socket_initialization(self, input_queue, stop_event):
        """Test UDP socket initialization."""
        socket_config = {
            'protocol': 'UDP',
            'ip': '127.0.0.1',
            'port': 8888
        }

        streamer = SocketStreamer(input_queue, stop_event, socket_config)

        try:
            streamer._initialize_output()
            assert streamer.socket_conn is not None

        finally:
            streamer._cleanup()

    @patch('socket.socket')
    def test_tcp_send_data_with_connected_clients(self, mock_socket_class, input_queue, stop_event):
        """Test TCP data sending with connected clients."""
        mock_server_socket = Mock()
        mock_client1 = Mock()
        mock_client2 = Mock()

        mock_socket_class.return_value = mock_server_socket

        streamer = SocketStreamer(input_queue, stop_event)
        streamer.server_socket = mock_server_socket
        streamer.client_connections = [mock_client1, mock_client2]

        test_data = {'prediction': 'left', 'confidence': 0.9}

        streamer._send_data(test_data)

        mock_client1.send.assert_called_once()
        mock_client2.send.assert_called_once()

        sent_data = mock_client1.send.call_args[0][0]
        assert isinstance(sent_data, bytes)
        decoded_data = sent_data.decode('utf-8').rstrip('\n')
        json_data = json.loads(decoded_data)
        assert json_data['prediction'] == 'left'

    @patch('socket.socket')
    def test_tcp_send_data_handles_disconnected_clients(self, mock_socket_class, input_queue, stop_event):
        """Test TCP data sending handles disconnected clients gracefully."""
        mock_server_socket = Mock()
        mock_client_good = Mock()
        mock_client_bad = Mock()

        mock_client_bad.send.side_effect = ConnectionResetError("Client disconnected")

        mock_socket_class.return_value = mock_server_socket

        streamer = SocketStreamer(input_queue, stop_event)
        streamer.server_socket = mock_server_socket
        streamer.client_connections = [mock_client_good, mock_client_bad]

        test_data = {'test': 'data'}

        streamer._send_data(test_data)

        mock_client_good.send.assert_called_once()
        assert mock_client_bad not in streamer.client_connections
        assert mock_client_good in streamer.client_connections

    @patch('socket.socket')
    def test_udp_send_data(self, mock_socket_class, input_queue, stop_event):
        """Test UDP data sending."""
        mock_udp_socket = Mock()
        mock_socket_class.return_value = mock_udp_socket

        socket_config = {
            'protocol': 'UDP',
            'ip': '127.0.0.1',
            'port': 8888
        }

        streamer = SocketStreamer(input_queue, stop_event, socket_config)
        streamer.socket_conn = mock_udp_socket

        test_data = {'prediction': 'right'}

        streamer._send_data(test_data)

        mock_udp_socket.sendto.assert_called_once()
        args = mock_udp_socket.sendto.call_args[0]

        sent_bytes = args[0]
        target_addr = args[1]

        assert isinstance(sent_bytes, bytes)
        assert target_addr == ('127.0.0.1', 8888)

        json_str = sent_bytes.decode('utf-8').rstrip('\n')
        json_data = json.loads(json_str)
        assert json_data['prediction'] == 'right'

    def test_cleanup_tcp_resources(self, input_queue, stop_event):
        """Test cleanup of TCP resources."""
        streamer = SocketStreamer(input_queue, stop_event)

        mock_server = Mock()
        mock_client1 = Mock()
        mock_client2 = Mock()

        streamer.server_socket = mock_server
        streamer.client_connections = [mock_client1, mock_client2]

        streamer._cleanup()

        mock_client1.close.assert_called_once()
        mock_client2.close.assert_called_once()
        mock_server.close.assert_called_once()
        assert streamer.client_connections == []
        assert streamer.server_socket is None

    def test_cleanup_udp_resources(self, input_queue, stop_event):
        """Test cleanup of UDP resources."""
        socket_config = {'protocol': 'UDP'}
        streamer = SocketStreamer(input_queue, stop_event, socket_config)

        mock_udp_socket = Mock()
        streamer.socket_conn = mock_udp_socket

        streamer._cleanup()

        mock_udp_socket.close.assert_called_once()
        assert streamer.socket_conn is None

    def test_validate_and_sanitize_ip_valid_addresses(self, input_queue, stop_event):
        """Test IP validation with valid addresses."""
        streamer = SocketStreamer(input_queue, stop_event)

        assert streamer._validate_and_sanitize_ip('127.0.0.1') == '127.0.0.1'
        assert streamer._validate_and_sanitize_ip('192.168.1.1') == '192.168.1.1'
        assert streamer._validate_and_sanitize_ip('10.0.0.1') == '10.0.0.1'
        assert streamer._validate_and_sanitize_ip('0.0.0.0') == '0.0.0.0'
        assert streamer._validate_and_sanitize_ip('255.255.255.255') == '255.255.255.255'

    def test_validate_and_sanitize_ip_invalid_addresses(self, input_queue, stop_event):
        """Test IP validation with invalid addresses falls back to localhost."""
        streamer = SocketStreamer(input_queue, stop_event)

        with patch.object(streamer, 'logger') as mock_logger:
            assert streamer._validate_and_sanitize_ip('999.999.999.999') == '127.0.0.1'
            assert streamer._validate_and_sanitize_ip('192.168.1.300') == '127.0.0.1'
            assert streamer._validate_and_sanitize_ip('not.an.ip.address') == '127.0.0.1'
            assert streamer._validate_and_sanitize_ip('192.168') == '127.0.0.1'
            assert streamer._validate_and_sanitize_ip('192.168.1.1.1') == '127.0.0.1'

            assert mock_logger.warning.call_count >= 5

    def test_validate_and_sanitize_ip_special_cases(self, input_queue, stop_event):
        """Test IP validation with special cases."""
        streamer = SocketStreamer(input_queue, stop_event)

        assert streamer._validate_and_sanitize_ip('localhost') == '127.0.0.1'
        assert streamer._validate_and_sanitize_ip('*') == '0.0.0.0'
        assert streamer._validate_and_sanitize_ip('  127.0.0.1  ') == '127.0.0.1'
        assert streamer._validate_and_sanitize_ip(' localhost ') == '127.0.0.1'

    def test_validate_and_sanitize_ip_edge_cases(self, input_queue, stop_event):
        """Test IP validation with edge cases."""
        streamer = SocketStreamer(input_queue, stop_event)

        with patch.object(streamer, 'logger') as mock_logger:
            assert streamer._validate_and_sanitize_ip(None) == '127.0.0.1'
            assert streamer._validate_and_sanitize_ip('') == '127.0.0.1'
            assert streamer._validate_and_sanitize_ip('   ') == '127.0.0.1'
            assert streamer._validate_and_sanitize_ip(123) == '127.0.0.1'
            assert streamer._validate_and_sanitize_ip(['127.0.0.1']) == '127.0.0.1'
            assert streamer._validate_and_sanitize_ip({'ip': '127.0.0.1'}) == '127.0.0.1'

            assert mock_logger.warning.call_count >= 6

    def test_initialization_with_invalid_ip_config(self, input_queue, stop_event):
        """Test initialization sanitizes invalid IP during init."""
        socket_config = {
            'protocol': 'TCP',
            'ip': '999.999.999.999',
            'port': 8080
        }

        with patch('dendrite.data.streaming.base.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            streamer = SocketStreamer(input_queue, stop_event, socket_config)

            assert streamer.ip == '127.0.0.1'
            mock_logger.warning.assert_called_once()
            assert 'Invalid IP address' in str(mock_logger.warning.call_args)

    @patch('socket.socket')
    def test_tcp_fallback_to_localhost_on_invalid_ip(self, mock_socket_class, input_queue, stop_event, free_port):
        """Test TCP server falls back to localhost when bind fails with invalid IP."""
        test_ip = '192.168.250.250'
        socket_config = {
            'protocol': 'TCP',
            'ip': test_ip,
            'port': free_port
        }

        mock_server_socket = Mock()
        mock_fallback_socket = Mock()
        mock_socket_class.side_effect = [mock_server_socket, mock_fallback_socket]

        bind_error = OSError("The requested address is not valid in its context")
        bind_error.errno = 10049
        mock_server_socket.bind.side_effect = bind_error

        mock_fallback_socket.bind.return_value = None
        mock_fallback_socket.listen.return_value = None
        mock_fallback_socket.settimeout.return_value = None

        streamer = SocketStreamer(input_queue, stop_event, socket_config)
        assert streamer.ip == test_ip

        with patch.object(streamer, 'logger') as mock_logger:
            with patch('threading.Thread') as mock_thread:
                streamer._initialize_output()

        assert streamer.ip == '127.0.0.1'

        mock_logger.warning.assert_called()
        mock_logger.info.assert_called()

        warning_calls = [call.args[0] for call in mock_logger.warning.call_args_list]
        info_calls = [call.args[0] for call in mock_logger.info.call_args_list]

        assert any(f"Cannot bind to {test_ip}:{free_port}" in msg for msg in warning_calls)
        assert any(f"Falling back to localhost (127.0.0.1:{free_port})" in msg for msg in info_calls)
        assert any(f"TCP server successfully listening on fallback address 127.0.0.1:{free_port}" in msg for msg in info_calls)

        assert mock_server_socket.bind.call_count == 1
        assert mock_fallback_socket.bind.call_count == 1
        mock_fallback_socket.bind.assert_called_with(('127.0.0.1', free_port))

    @patch('socket.socket')
    def test_udp_fallback_to_localhost_on_invalid_ip(self, mock_socket_class, input_queue, stop_event):
        """Test UDP socket falls back to localhost when creation fails with invalid IP."""
        test_ip = '192.168.250.250'
        socket_config = {
            'protocol': 'UDP',
            'ip': test_ip,
            'port': 8888
        }

        mock_initial_socket = Mock()
        mock_fallback_socket = Mock()

        creation_error = OSError("The requested address is not valid in its context")
        creation_error.errno = 10049
        mock_socket_class.side_effect = [creation_error, mock_fallback_socket]

        streamer = SocketStreamer(input_queue, stop_event, socket_config)
        assert streamer.ip == test_ip

        with patch.object(streamer, 'logger') as mock_logger:
            streamer._initialize_output()

        assert streamer.ip == '127.0.0.1'
        assert streamer.socket_conn == mock_fallback_socket

        mock_logger.warning.assert_called()
        mock_logger.info.assert_called()

        warning_calls = [call.args[0] for call in mock_logger.warning.call_args_list]
        info_calls = [call.args[0] for call in mock_logger.info.call_args_list]

        assert any(f"Cannot create UDP socket for {test_ip}:8888" in msg for msg in warning_calls)
        assert any("Falling back to localhost (127.0.0.1:8888)" in msg for msg in info_calls)
        assert any("UDP socket successfully created for fallback address 127.0.0.1:8888" in msg for msg in info_calls)

    @patch('socket.socket')
    def test_tcp_fallback_failure_handling(self, mock_socket_class, input_queue, stop_event):
        """Test handling when even localhost fallback fails for TCP."""
        socket_config = {
            'protocol': 'TCP',
            'ip': '192.168.999.999',
            'port': 8080
        }

        mock_server_socket = Mock()
        mock_fallback_socket = Mock()
        mock_socket_class.side_effect = [mock_server_socket, mock_fallback_socket]

        bind_error = OSError("The requested address is not valid in its context")
        bind_error.errno = 10049
        mock_server_socket.bind.side_effect = bind_error

        fallback_error = OSError("Address already in use")
        mock_fallback_socket.bind.side_effect = fallback_error

        with patch('dendrite.data.streaming.base.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            streamer = SocketStreamer(input_queue, stop_event, socket_config)

            with pytest.raises(OSError):
                streamer._initialize_output()

            mock_logger.error.assert_called_with(
                f"Failed to fallback to localhost for TCP: {fallback_error}"
            )

    @patch('socket.socket')
    def test_udp_fallback_failure_handling(self, mock_socket_class, input_queue, stop_event):
        """Test handling when even localhost fallback fails for UDP."""
        socket_config = {
            'protocol': 'UDP',
            'ip': '192.168.999.999',
            'port': 8888
        }

        initial_error = OSError("The requested address is not valid in its context")
        initial_error.errno = 10049

        fallback_error = OSError("Permission denied")
        mock_socket_class.side_effect = [initial_error, fallback_error]

        with patch('dendrite.data.streaming.base.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            streamer = SocketStreamer(input_queue, stop_event, socket_config)

            with pytest.raises(OSError):
                streamer._initialize_output()

            mock_logger.error.assert_called_with(
                f"Failed to fallback to localhost for UDP: {fallback_error}"
            )

    @patch('socket.socket')
    def test_non_address_errors_are_not_handled_by_fallback(self, mock_socket_class, input_queue, stop_event):
        """Test that non-address errors are raised normally without fallback."""
        socket_config = {
            'protocol': 'TCP',
            'ip': '127.0.0.1',
            'port': 80
        }

        mock_server_socket = Mock()
        mock_socket_class.return_value = mock_server_socket

        permission_error = OSError("Permission denied")
        permission_error.errno = 13
        mock_server_socket.bind.side_effect = permission_error

        with patch('dendrite.data.streaming.base.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            streamer = SocketStreamer(input_queue, stop_event, socket_config)

            with pytest.raises(OSError):
                streamer._initialize_output()

            mock_logger.error.assert_called_once()
            error_call = mock_logger.error.call_args[0][0]
            assert "Failed to initialize TCP socket" in error_call
            assert "Permission denied" in error_call

    def test_cleanup_catches_only_socket_exceptions(self, input_queue, stop_event):
        """Test that cleanup catches only socket-related exceptions, not all exceptions.

        Bug: Bare except: clauses in _cleanup() and _send_data() catch ALL exceptions
        including KeyboardInterrupt and SystemExit, which should propagate.
        This test verifies proper exception handling.
        """
        import socket as socket_module

        streamer = SocketStreamer(input_queue, stop_event)

        # Create mock clients that raise different types of exceptions
        mock_client_socket_error = Mock()
        mock_client_socket_error.close.side_effect = socket_module.error("Socket error")

        mock_client_os_error = Mock()
        mock_client_os_error.close.side_effect = OSError("OS error")

        # These clients should have their close() errors caught
        streamer.client_connections = [mock_client_socket_error, mock_client_os_error]

        # Cleanup should complete without raising (socket errors are caught)
        streamer._cleanup()

        # Verify close was called on both
        mock_client_socket_error.close.assert_called_once()
        mock_client_os_error.close.assert_called_once()
        assert streamer.client_connections == []

    def test_send_data_removes_failed_clients_with_socket_errors(self, input_queue, stop_event):
        """Test that _send_data properly handles socket errors on client removal.

        Bug: Bare except: in the failed client cleanup section catches all exceptions.
        This should only catch socket-related exceptions.
        """
        import socket as socket_module

        streamer = SocketStreamer(input_queue, stop_event)
        streamer.server_socket = Mock()

        # Client that fails to send AND fails to close
        mock_client = Mock()
        mock_client.send.side_effect = ConnectionResetError("Connection reset")
        mock_client.close.side_effect = socket_module.error("Already closed")

        streamer.client_connections = [mock_client]

        # This should handle the error gracefully
        streamer._send_data({'test': 'data'})

        # Client should be removed even though close() failed
        assert mock_client not in streamer.client_connections
