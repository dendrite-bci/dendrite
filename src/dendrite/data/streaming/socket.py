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
    """Socket-based streamer for TCP/UDP output."""

    def __init__(
        self,
        input_queue: multiprocessing.Queue,
        stop_event: Event | None = None,
        socket_config: dict[str, Any] | None = None,
        shared_state: Any | None = None,
    ) -> None:
        super().__init__(input_queue, "Socket", stop_event, shared_state)

        config = socket_config or {}
        self.protocol = config.get("protocol", "TCP").upper()
        raw_ip = config.get("ip", "127.0.0.1")
        self.port = config.get("port", 8080)

        self.ip = self._validate_and_sanitize_ip(raw_ip)

        self.server_socket = None
        self.client_connections: list[socket.socket] = []
        self.socket_conn = None

        self.logger.info(f"SocketStreamer initialized: {self.protocol} on {self.ip}:{self.port}")

    def _validate_and_sanitize_ip(self, ip: str) -> str:
        """Validate IP address, fallback to localhost if invalid."""
        if not ip or not isinstance(ip, str):
            self.logger.warning("Empty or invalid IP address provided, using localhost")
            return "127.0.0.1"

        ip = ip.strip()
        if ip in ["localhost", "*"]:
            return "127.0.0.1" if ip == "localhost" else "0.0.0.0"

        try:
            ipaddress.IPv4Address(ip)
            return ip
        except ipaddress.AddressValueError:
            self.logger.warning(f"Invalid IP address '{ip}', falling back to localhost")
            return "127.0.0.1"

    def _initialize_output(self) -> None:
        """Initialize socket connection."""
        if self.protocol == "TCP":
            self._initialize_tcp()
        elif self.protocol == "UDP":
            self._initialize_udp()

    def _initialize_tcp(self) -> None:
        """Initialize TCP server socket with fallback to localhost on bind error."""
        try:
            self._bind_tcp(self.ip)
        except OSError as e:
            if e.errno == 10049 or "not valid in its context" in str(e):
                self.logger.warning(f"Cannot bind to {self.ip}:{self.port} - {e}")
                self.logger.info(f"Falling back to localhost (127.0.0.1:{self.port})")
                self.ip = "127.0.0.1"
                self._bind_tcp(self.ip)
            else:
                self.logger.error(f"Failed to initialize TCP socket: {e}")
                raise

    def _bind_tcp(self, ip: str) -> None:
        """Create, bind, and start accepting on a TCP server socket."""
        if self.server_socket:
            self.server_socket.close()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((ip, self.port))
        self.server_socket.listen(5)
        self.server_socket.settimeout(1.0)
        self.logger.info(f"TCP server listening on {ip}:{self.port}")

        accept_thread = threading.Thread(
            target=self._accept_tcp_connections, daemon=True, name="TCPAcceptThread"
        )
        accept_thread.start()

    def _initialize_udp(self) -> None:
        """Initialize UDP socket with fallback to localhost on bind error."""
        try:
            self._create_udp(self.ip)
        except OSError as e:
            if e.errno == 10049 or "not valid in its context" in str(e):
                self.logger.warning(f"Cannot create UDP socket for {self.ip}:{self.port} - {e}")
                self.logger.info(f"Falling back to localhost (127.0.0.1:{self.port})")
                self.ip = "127.0.0.1"
                self._create_udp(self.ip)
            else:
                self.logger.error(f"Failed to initialize UDP socket: {e}")
                raise

    def _create_udp(self, ip: str) -> None:
        """Create a UDP socket."""
        if self.socket_conn:
            self.socket_conn.close()
        self.socket_conn = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.logger.info(f"UDP socket created for {ip}:{self.port}")

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
            serializable_data = self._make_json_serializable(data)
            message = json.dumps(serializable_data) + "\n"
            message_bytes = message.encode("utf-8")
            self.bytes_sent += len(message_bytes)

            if self.protocol == "TCP":
                self._send_tcp(message_bytes)
            elif self.protocol == "UDP" and self.socket_conn:
                self.socket_conn.sendto(message_bytes, (self.ip, self.port))

        except Exception as e:
            self.logger.error(f"Error sending {self.protocol} data: {e}")

    def _send_tcp(self, message_bytes: bytes) -> None:
        """Send to all connected TCP clients, removing disconnected ones."""
        if not self.client_connections:
            return
        failed = []
        for client in self.client_connections[:]:
            try:
                client.send(message_bytes)
            except (ConnectionResetError, BrokenPipeError, OSError):
                failed.append(client)
            except Exception as e:
                self.logger.debug(f"Failed to send to TCP client: {e}")
                failed.append(client)
        for client in failed:
            try:
                client.close()
            except OSError:
                pass
            if client in self.client_connections:
                self.client_connections.remove(client)
                self.logger.debug("Removed disconnected TCP client")

    def _cleanup(self) -> None:
        """Clean up socket connections."""
        try:
            for client in self.client_connections:
                try:
                    client.close()
                except OSError:
                    pass
            self.client_connections = []

            if self.server_socket:
                self.server_socket.close()
                self.server_socket = None
            if self.socket_conn:
                self.socket_conn.close()
                self.socket_conn = None

            self.logger.info(f"{self.protocol} socket cleaned up")
        except Exception as e:
            self.logger.error(f"Error cleaning up socket: {e}")
