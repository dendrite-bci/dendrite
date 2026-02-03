"""
Socket-based streamer for TCP/UDP output.
"""

import ipaddress
import json
import multiprocessing
import socket
import threading
from multiprocessing.synchronize import Event
from typing import Any

from .base import BaseOutputStreamer


class SocketStreamer(BaseOutputStreamer):
    """
    Socket-based streamer for TCP/UDP output.

    Provides pure socket streaming without LSL dependency.
    """

    def __init__(
        self,
        input_queue: multiprocessing.Queue,
        stop_event: Event | None = None,
        socket_config: dict[str, Any] | None = None,
        shared_state: Any | None = None,
    ) -> None:
        """
        Initialize socket streamer.

        Args:
            input_queue: Queue containing data to be streamed
            stop_event: Event to signal when to stop streaming
            socket_config: Configuration with protocol, ip, port
            shared_state: SharedState instance for exposing streaming metrics
        """
        super().__init__(input_queue, "Socket", stop_event, shared_state)

        # Parse configuration
        config = socket_config or {}
        self.protocol = config.get("protocol", "TCP").upper()
        raw_ip = config.get("ip", "127.0.0.1")  # Default to localhost for security
        self.port = config.get("port", 8080)

        # Validate and sanitize IP address
        self.ip = self._validate_and_sanitize_ip(raw_ip)

        # Socket state
        self.server_socket = None
        self.client_connections = []
        self.socket_conn = None

        self.logger.info(f"SocketStreamer initialized: {self.protocol} on {self.ip}:{self.port}")

    def _validate_and_sanitize_ip(self, ip: str) -> str:
        """
        Validate IP address format and return a safe IP address.

        Args:
            ip: IP address string to validate

        Returns:
            str: Valid IP address (fallback to localhost if invalid)
        """
        if not ip or not isinstance(ip, str):
            self.logger.warning("Empty or invalid IP address provided, using localhost")
            return "127.0.0.1"

        ip = ip.strip()

        # Handle special cases
        if ip in ["localhost", "*"]:
            return "127.0.0.1" if ip == "localhost" else "0.0.0.0"

        # Validate IPv4 address format
        try:
            ipaddress.IPv4Address(ip)
            return ip
        except ipaddress.AddressValueError:
            self.logger.warning(f"Invalid IP address '{ip}' provided, falling back to localhost")
            return "127.0.0.1"

    def _initialize_output(self) -> None:
        """Initialize socket connection."""
        if self.protocol == "TCP":
            self._initialize_tcp_socket()
        elif self.protocol == "UDP":
            self._initialize_udp_socket()

    def _initialize_tcp_socket(self) -> None:
        """Initialize TCP server socket with fallback to localhost on bind error."""
        try:
            # Create TCP server socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.ip, self.port))
            self.server_socket.listen(5)
            self.server_socket.settimeout(1.0)  # Non-blocking accept

            self.logger.info(f"TCP server listening on {self.ip}:{self.port}")

            # Start thread to accept connections
            accept_thread = threading.Thread(
                target=self._accept_tcp_connections, daemon=True, name="TCPAcceptThread"
            )
            accept_thread.start()

        except OSError as e:
            if e.errno == 10049 or "not valid in its context" in str(e):
                # Invalid address error - fallback to localhost
                self.logger.warning(f"Cannot bind to {self.ip}:{self.port} - {e}")
                self.logger.info(f"Falling back to localhost (127.0.0.1:{self.port})")
                self._fallback_to_localhost_tcp()
            else:
                # Other socket errors (port in use, permission denied, etc.)
                self.logger.error(f"Failed to initialize TCP socket on {self.ip}:{self.port}: {e}")
                raise
        except Exception as e:
            self.logger.error(f"Unexpected error initializing TCP socket: {e}")
            raise

    def _initialize_udp_socket(self) -> None:
        """Initialize UDP socket with fallback to localhost on bind error."""
        try:
            # Create UDP socket
            self.socket_conn = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.logger.info(f"UDP socket created for {self.ip}:{self.port}")

        except OSError as e:
            if e.errno == 10049 or "not valid in its context" in str(e):
                # Invalid address error - fallback to localhost
                self.logger.warning(f"Cannot create UDP socket for {self.ip}:{self.port} - {e}")
                self.logger.info(f"Falling back to localhost (127.0.0.1:{self.port})")
                self._fallback_to_localhost_udp()
            else:
                # Other socket errors
                self.logger.error(f"Failed to initialize UDP socket for {self.ip}:{self.port}: {e}")
                raise
        except Exception as e:
            self.logger.error(f"Unexpected error initializing UDP socket: {e}")
            raise

    def _fallback_to_localhost_tcp(self) -> None:
        """Fallback to localhost for TCP when original IP fails."""
        try:
            if self.server_socket:
                self.server_socket.close()

            self.ip = "127.0.0.1"  # Update the IP to localhost
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.ip, self.port))
            self.server_socket.listen(5)
            self.server_socket.settimeout(1.0)

            self.logger.info(
                f"TCP server successfully listening on fallback address {self.ip}:{self.port}"
            )

            # Start thread to accept connections
            accept_thread = threading.Thread(
                target=self._accept_tcp_connections, daemon=True, name="TCPAcceptThread"
            )
            accept_thread.start()

        except Exception as e:
            self.logger.error(f"Failed to fallback to localhost for TCP: {e}")
            raise

    def _fallback_to_localhost_udp(self) -> None:
        """Fallback to localhost for UDP when original IP fails."""
        try:
            if self.socket_conn:
                self.socket_conn.close()

            self.ip = "127.0.0.1"  # Update the IP to localhost
            self.socket_conn = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.logger.info(
                f"UDP socket successfully created for fallback address {self.ip}:{self.port}"
            )

        except Exception as e:
            self.logger.error(f"Failed to fallback to localhost for UDP: {e}")
            raise

    def _accept_tcp_connections(self) -> None:
        """Accept TCP client connections in background."""
        while not (self.stop_event and self.stop_event.is_set()):
            try:
                if self.server_socket:
                    client_socket, client_addr = self.server_socket.accept()
                    self.client_connections.append(client_socket)
                    self.logger.info(f"TCP client connected from {client_addr}")
            except TimeoutError:
                continue
            except Exception as e:
                if not (self.stop_event and self.stop_event.is_set()):
                    self.logger.debug(f"TCP accept error: {e}")

    def _send_data(self, data: Any) -> None:
        """Send data via socket."""
        try:
            # Prepare JSON message
            serializable_data = self._make_json_serializable(data)
            message = json.dumps(serializable_data) + "\n"
            message_bytes = message.encode("utf-8")

            self.bytes_sent += len(message_bytes)

            if self.protocol == "TCP":
                # Send to all connected TCP clients
                if self.client_connections:
                    failed_connections = []
                    for client in self.client_connections[
                        :
                    ]:  # Copy list to avoid modification during iteration
                        try:
                            client.send(message_bytes)
                        except (ConnectionResetError, BrokenPipeError, OSError):
                            # Client disconnected - this is expected, don't log as error
                            failed_connections.append(client)
                        except Exception as e:
                            self.logger.debug(f"Failed to send to TCP client: {e}")
                            failed_connections.append(client)

                    # Remove failed connections
                    for failed_client in failed_connections:
                        try:
                            failed_client.close()
                        except OSError:
                            pass
                        if failed_client in self.client_connections:
                            self.client_connections.remove(failed_client)
                            self.logger.debug("Removed disconnected TCP client")

            elif self.protocol == "UDP" and self.socket_conn:
                # Send UDP packet to configured address
                self.socket_conn.sendto(message_bytes, (self.ip, self.port))

        except Exception as e:
            self.logger.error(f"Error sending {self.protocol} data: {e}")

    def _cleanup(self) -> None:
        """Clean up socket connections."""
        try:
            # Close client connections
            for client in self.client_connections:
                try:
                    client.close()
                except OSError:
                    pass
            self.client_connections = []

            # Close server socket
            if self.server_socket:
                self.server_socket.close()
                self.server_socket = None

            # Close UDP socket
            if self.socket_conn:
                self.socket_conn.close()
                self.socket_conn = None

            self.logger.info(f"{self.protocol} socket cleaned up")

        except Exception as e:
            self.logger.error(f"Error cleaning up socket: {e}")
