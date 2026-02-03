"""
Base streamer classes for output streaming.

This module provides the abstract base class for all output streamers
and the LSL-specific base class with Lab Streaming Layer functionality.
"""

import json
import logging
import multiprocessing
import queue
import threading
import time
from abc import ABC, abstractmethod
from multiprocessing.synchronize import Event
from typing import Any

from dendrite.data.lsl_helpers import LSLOutlet, StreamConfig
from dendrite.utils.logger_central import get_logger, setup_logger
from dendrite.utils.serialization import jsonify
from dendrite.utils.state_keys import streamer_metric_key


class BaseOutputStreamer(multiprocessing.Process, ABC):
    """
    Abstract base class for all output streamers.

    This provides the common process-based infrastructure without
    forcing any specific output protocol (like LSL).
    """

    def __init__(
        self,
        input_queue: multiprocessing.Queue,
        stream_name: str,
        stop_event: Event | None = None,
        shared_state: Any | None = None,
    ) -> None:
        """
        Initialize the base output streamer.

        Args:
            input_queue: Queue containing data to be streamed
            stream_name: Name for this streamer
            stop_event: Event to signal when to stop streaming
            shared_state: SharedState instance for exposing streaming metrics
        """
        super().__init__(name=f"{stream_name}Streamer")

        self.input_queue = input_queue
        self.stream_name = stream_name
        self.stop_event = stop_event
        self.shared_state = shared_state

        self.messages_sent = 0
        self.bytes_sent = 0
        self.start_time = None

        # Pre-configure logger
        self.logger = get_logger(f"{stream_name}Streamer")

    def run(self) -> None:
        """Main process entry point."""
        try:
            # Setup logging
            self.logger = setup_logger(f"{self.stream_name}Streamer", level=logging.INFO)
            self.logger.info(f"{self.stream_name} streamer starting...")

            self.start_time = time.time()

            # Initialize the specific output protocol
            self._initialize_output()

            self._process_loop()

        except Exception as e:
            self.logger.error(f"Error in {self.stream_name} streamer: {e}", exc_info=True)
        finally:
            self._cleanup()

    @abstractmethod
    def _initialize_output(self) -> None:
        """Initialize the specific output protocol. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _send_data(self, data: Any) -> None:
        """Send data using the specific output protocol. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _cleanup(self) -> None:
        """Clean up resources. Must be implemented by subclasses."""
        pass

    def _process_loop(self) -> None:
        """Main processing loop to get data from queue and send it."""
        self.logger.info(f"Starting {self.stream_name} processing loop")

        while not (self.stop_event and self.stop_event.is_set()):
            try:
                # Get data from queue with timeout
                try:
                    data = self.input_queue.get(timeout=0.1)

                    if data is None:
                        continue

                    self.messages_sent += 1

                    # Send using the specific protocol
                    self._send_data(data)

                    # Log progress periodically
                    if self.messages_sent % 100 == 0:
                        self._log_statistics()

                except queue.Empty:
                    continue

            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}", exc_info=True)

        self.logger.info(f"{self.stream_name} processing loop stopped")

    def _log_statistics(self) -> None:
        """Log streaming statistics."""
        if self.start_time:
            elapsed = time.time() - self.start_time
            rate = self.messages_sent / elapsed if elapsed > 0 else 0
            self.logger.debug(
                f"Statistics: {self.messages_sent} messages sent, rate: {rate:.1f} msg/s"
            )

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-serializable types to JSON-serializable format."""
        if obj is None:
            return {}  # Return empty dict instead of None for streaming
        return jsonify(obj)


class LSLBaseStreamer(BaseOutputStreamer):
    """
    Base class for streamers that use LSL output.

    This extends BaseOutputStreamer to add LSL-specific functionality.
    """

    def __init__(
        self,
        input_queue: multiprocessing.Queue,
        stream_info: StreamConfig,
        stop_event: Event | None = None,
        bandwidth_reporting_interval: float = 60.0,
        shared_state: Any | None = None,
    ) -> None:
        """
        Initialize LSL base streamer.

        Args:
            input_queue: Queue containing data to be streamed
            stream_info: StreamConfig with LSL stream configuration
            stop_event: Event to signal when to stop streaming
            bandwidth_reporting_interval: Interval for bandwidth reporting
            shared_state: SharedState instance for exposing streaming metrics
        """
        super().__init__(input_queue, stream_info.name, stop_event, shared_state)

        self.stream_info = stream_info
        self.bandwidth_reporting_interval = bandwidth_reporting_interval

        # LSL streamer
        self.streamer = None

        # Bandwidth tracking
        self.bandwidth_lock = None
        self.last_report_time = 0

    def _initialize_output(self) -> None:
        """Initialize LSL output stream."""
        try:
            self.logger.info(f"Setting up LSL stream: {self.stream_info.name}")

            # Initialize bandwidth lock
            self.bandwidth_lock = threading.Lock()
            self.last_report_time = time.time()

            # Create LSL outlet
            self.streamer = LSLOutlet(self.stream_info)

            # Start bandwidth reporting thread
            self._start_bandwidth_reporting()

            self.logger.info(f"LSL stream '{self.stream_info.name}' initialized successfully")

        except Exception as e:
            self.logger.error(f"Error setting up LSL stream: {e}", exc_info=True)
            raise

    def _send_data(self, data: Any) -> None:
        """Send data via LSL."""
        if hasattr(self, "streamer") and self.streamer and data is not None:
            try:
                # Ensure data is JSON serializable
                serializable_data = self._make_json_serializable(data)
                if serializable_data is None:
                    return  # Skip invalid data

                # Convert to JSON string with validation
                json_str = json.dumps(serializable_data)
                if not json_str or json_str == "null":
                    return  # Skip empty/null JSON

                # Send via LSL with proper timestamp
                self.streamer.push_sample([json_str], time.time())

                # Track bandwidth
                with self.bandwidth_lock:
                    self.bytes_sent += len(json_str.encode("utf-8"))

            except Exception as e:
                self.logger.error(f"Error sending LSL data: {e}")

    def _cleanup(self) -> None:
        """Clean up LSL resources."""
        if hasattr(self, "streamer") and self.streamer:
            try:
                # Set to None first to prevent race conditions
                temp_streamer = self.streamer
                self.streamer = None
                del temp_streamer
                self.logger.info("LSL stream cleaned up")
            except Exception as e:
                self.logger.error(f"Error cleaning up LSL stream: {e}")

    def _start_bandwidth_reporting(self) -> None:
        """Start thread for bandwidth reporting."""

        def report_bandwidth() -> None:
            while not (self.stop_event and self.stop_event.is_set()):
                time.sleep(self.bandwidth_reporting_interval)
                self._report_bandwidth()

        bandwidth_thread = threading.Thread(
            target=report_bandwidth, daemon=True, name=f"{self.stream_info.name}BandwidthThread"
        )
        bandwidth_thread.start()

    def _report_bandwidth(self) -> None:
        """Report bandwidth usage."""
        with self.bandwidth_lock:
            current_time = time.time()
            time_diff = current_time - self.last_report_time

            if time_diff > 0:
                bandwidth_bytes = self.bytes_sent / time_diff
                bandwidth_kbps = bandwidth_bytes / 1024
                bandwidth_str = self._format_bandwidth(bandwidth_bytes)
                self.logger.info(f"Bandwidth usage for {self.stream_info.name}: {bandwidth_str}")

                # Report to SharedState if available
                if self.shared_state:
                    self.shared_state.set(
                        streamer_metric_key(self.stream_info.name, "bandwidth_kbps"), bandwidth_kbps
                    )

                # Reset counters
                self.bytes_sent = 0
                self.last_report_time = current_time

    def _format_bandwidth(self, bytes_per_second: float) -> str:
        """Format bandwidth in human-readable units."""
        if bytes_per_second >= 1_048_576:  # >= 1 MB/s
            return f"{bytes_per_second / 1_048_576:.2f} MB/s"
        elif bytes_per_second >= 1024:  # >= 1 KB/s
            return f"{bytes_per_second / 1024:.2f} KB/s"
        else:
            return f"{bytes_per_second:.2f} B/s"
