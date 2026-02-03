"""
ZeroMQ-based streamer for high-performance messaging.
"""

import json
import logging
import multiprocessing
from multiprocessing.synchronize import Event
from typing import Any

from .base import BaseOutputStreamer

logger = logging.getLogger(__name__)


def _detect_zmq():
    """Detect if ZeroMQ (pyzmq) is available."""
    try:
        import zmq

        # Test basic ZMQ functionality
        context = zmq.Context()
        context.term()
        return True
    except ImportError:
        logger.info("ZeroMQ not available: pyzmq not installed. Install with: pip install pyzmq")
        return False
    except Exception as e:
        logger.info(f"ZeroMQ not available: {e}")
        return False


HAS_ZMQ = _detect_zmq()


class ZMQStreamer(BaseOutputStreamer):
    """
    ZeroMQ-based streamer for high-performance messaging.

    Provides pure ZMQ publisher functionality without LSL dependency.
    """

    def __init__(
        self,
        input_queue: multiprocessing.Queue,
        stop_event: Event | None = None,
        zmq_config: dict[str, Any] | None = None,
        shared_state: Any | None = None,
    ) -> None:
        """
        Initialize ZMQ streamer.

        Args:
            input_queue: Queue containing data to be streamed
            stop_event: Event to signal when to stop streaming
            zmq_config: Configuration with ip, port, message_format
            shared_state: SharedState instance for exposing streaming metrics
        """
        super().__init__(input_queue, "ZMQ", stop_event, shared_state)

        # Parse configuration
        config = zmq_config or {}
        self.ip = config.get("ip", "127.0.0.1")  # Default to localhost for security
        self.port = config.get("port", 5556)
        self.message_format = config.get("message_format", "JSON")

        # ZMQ state
        self.context = None
        self.publisher = None

        # Check ZMQ availability
        if not HAS_ZMQ:
            self.logger.warning(
                "ZeroMQ not available - ZMQStreamer will not function. Install with: pip install pyzmq"
            )
        else:
            self.logger.info(
                f"ZMQStreamer initialized: tcp://{self.ip}:{self.port} ({self.message_format})"
            )

    def _initialize_output(self) -> None:
        """Initialize ZMQ publisher."""
        if not HAS_ZMQ:
            self.logger.error(
                "ZeroMQ not available - cannot initialize. Install with: pip install pyzmq"
            )
            raise RuntimeError("ZeroMQ not available")

        try:
            import zmq

            # Create ZMQ context and publisher socket
            self.context = zmq.Context()
            self.publisher = self.context.socket(zmq.PUB)

            # Bind to address
            bind_address = f"tcp://{self.ip}:{self.port}"
            self.publisher.bind(bind_address)

            self.logger.info(f"ZMQ publisher bound to {bind_address}")

        except Exception as e:
            self.logger.error(f"Failed to initialize ZMQ: {e}")
            raise

    def _send_data(self, data: Any) -> None:
        """Send data via ZMQ."""
        if self.publisher:
            try:
                # Prepare message based on format
                serializable_data = self._make_json_serializable(data)

                if self.message_format == "JSON":
                    message = json.dumps(serializable_data)
                    self.publisher.send_string(message)
                    self.bytes_sent += len(message.encode("utf-8"))
                else:  # Binary
                    message = json.dumps(serializable_data)
                    self.publisher.send(message.encode("utf-8"))
                    self.bytes_sent += len(message.encode("utf-8"))

            except Exception as e:
                self.logger.error(f"Error sending ZMQ data: {e}")

    def _cleanup(self) -> None:
        """Clean up ZMQ resources."""
        if not HAS_ZMQ:
            return

        try:
            if self.publisher:
                self.publisher.close()
            if self.context:
                self.context.term()

            self.logger.info("ZMQ publisher cleaned up")

        except Exception as e:
            self.logger.error(f"Error cleaning up ZMQ: {e}")
