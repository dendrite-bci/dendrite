"""
Specialized LSL streamer for visualization data.

Handles both raw data from plot queue and mode outputs for dashboard visualization.
"""

import multiprocessing
import queue
import threading
import time
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from typing import Any

from dendrite.data.lsl_helpers import StreamConfig
from dendrite.utils.state_keys import viz_bandwidth_key, viz_consumers_key

from .base import LSLBaseStreamer
from .payloads import ModeHistoryPayload


class VisualizationStreamer(LSLBaseStreamer):
    """
    Specialized LSL streamer for visualization data.

    Handles both raw data from plot queue and classifier outputs.
    Maintains compatibility with existing dashboard visualization.
    """

    def __init__(
        self,
        plot_queue: multiprocessing.Queue,
        stream_info: StreamConfig,
        stop_event: Event | None = None,
        output_queue: Queue | None = None,
        history_length: int = 1000,
        shared_state: Any | None = None,
        channel_labels: dict[str, list[str]] | None = None,
    ) -> None:
        """
        Initialize visualization streamer.

        Args:
            plot_queue: Queue containing raw data for plotting
            stream_info: StreamConfig for LSL
            stop_event: Event to signal when to stop
            output_queue: Single shared queue for all mode classifier outputs
            history_length: Number of samples to keep in history
            shared_state: SharedState instance for exposing streaming health metrics
            channel_labels: Dict mapping modality to list of channel labels (e.g., {'eeg': ['Fp1', 'Fz', ...]})
        """
        super().__init__(plot_queue, stream_info, stop_event)

        # Store additional queues and parameters
        self.plot_queue = plot_queue
        self.output_queue = output_queue
        self.history_length = history_length
        self.shared_state = shared_state

        # Channel labels provided at init from stream config
        self._channel_labels: dict[str, list[str]] = channel_labels or {}

        # History storage for mode outputs (raw data history not needed - dashboard has its own buffer)
        self.mode_outputs = {}
        self.mode_history = {}
        self.history_lock = None

        # Consumer tracking for history sending
        self._previous_consumer_count = 0
        self._consumer_check_interval = 1.0

    def run(self) -> None:
        """Override run to initialize history lock."""
        try:
            # Initialize the history lock before starting processing
            self.history_lock = threading.Lock()

            # Call parent run() to set up LSL and start processing
            super().run()

        except Exception as e:
            self.logger.error(f"Error in visualization streamer: {e}", exc_info=True)

    def _process_loop(self) -> None:
        """Override to handle multiple queues and data types."""
        self.logger.info("Starting visualization processing")

        # Start specialized threads for visualization data
        plot_thread = threading.Thread(
            target=self._process_raw_data, daemon=True, name="RawDataProcessingThread"
        )
        plot_thread.start()

        # Start thread for processing classifier outputs
        if self.output_queue:
            classifier_thread = threading.Thread(
                target=self._process_classifier_outputs,
                daemon=True,
                name="ClassifierOutputProcessingThread",
            )
            classifier_thread.start()

            # Start thread for checking consumers and sending history
            history_thread = threading.Thread(
                target=self._periodically_send_history, daemon=True, name="ConsumerMonitoringThread"
            )
            history_thread.start()

        # Wait for stop signal (keep parent behavior)
        while not (self.stop_event and self.stop_event.is_set()):
            time.sleep(0.1)

        self.logger.info("Visualization processing stopped")

    def _process_raw_data(self) -> None:
        """Process raw data from plot queue.

        The processor sends pre-formatted RawDataPayload with data already
        converted to lists. This method adds channel labels and forwards.
        """
        sample_count = 0

        while not (self.stop_event and self.stop_event.is_set()):
            try:
                try:
                    payload = self.plot_queue.get(timeout=0.1)
                    sample_count += 1
                except queue.Empty:
                    continue

                if payload is None or not isinstance(payload, dict):
                    if payload is not None:
                        self.logger.warning(f"Received non-dictionary data format: {type(payload)}")
                    continue

                payload_data = payload.get("data", {})
                payload["channel_labels"] = self._get_channel_labels(payload_data)
                self._send_data(payload)

                if sample_count % 5000 == 0:
                    self.logger.debug(f"Processed {sample_count} raw data samples")

            except Exception as e:
                self.logger.error(f"Error processing raw data: {e}", exc_info=True)

    def _get_channel_labels(self, payload_data: dict[str, Any]) -> dict[str, list[str]]:
        """Get channel labels for payload data.

        Uses labels from self-describing packets when available,
        generates fallbacks for modalities without labels.
        """
        channel_labels = {}

        for modality, data in payload_data.items():
            if modality in self._channel_labels:
                # Use labels from self-describing packets
                channel_labels[modality] = self._channel_labels[modality]
            elif isinstance(data, list):
                # Generate fallback labels for modalities without labels
                channel_labels[modality] = [f"{modality.upper()}_{i + 1}" for i in range(len(data))]
            else:
                channel_labels[modality] = [modality]

        return channel_labels

    def _process_classifier_outputs(self) -> None:
        """Process classifier outputs from shared queue."""
        sample_count = 0

        while not (self.stop_event and self.stop_event.is_set()):
            try:
                # Single blocking get - wakes immediately when any mode sends data
                output = self.output_queue.get(timeout=0.1)
                sample_count += 1

                if output is None:
                    continue

                if not isinstance(output, dict):
                    self.logger.warning(f"Received non-dict output: {type(output)}")
                    continue

                output_type = output["type"]
                mode_name = output["mode_name"]
                mode_type = output["mode_type"]
                data_content = output["data"]
                data_timestamp = (
                    output["data_timestamp"] if output.get("data_timestamp") is not None else time.time()
                )

                if mode_type is None:
                    self.logger.error(
                        f"Packet from {mode_name} missing required mode_type field - skipping"
                    )
                    continue

                # Create output message (matches ModeOutputPacket structure)
                message = {
                    "type": output_type,
                    "data_timestamp": data_timestamp,
                    "mode_name": mode_name,
                    "mode_type": mode_type,
                    "data": data_content,
                }

                # Store latest output and add to mode history
                with self.history_lock:
                    self.mode_outputs[mode_name] = output
                    self._store_mode_packet(mode_name, message)

                if not (self.stop_event and self.stop_event.is_set()):
                    self._send_data(message)

                if sample_count % 1000 == 0:
                    self.logger.debug(f"Processed {sample_count} classifier outputs")

            except queue.Empty:
                pass  # Timeout already waited, no sleep needed
            except Exception as e:
                self.logger.error(f"Error processing classifier output: {e}")

    def _store_mode_packet(self, mode_name: str, packet: dict[str, Any]) -> None:
        """Store mode packet in history for the given mode."""
        if mode_name not in self.mode_history:
            self.mode_history[mode_name] = []

        self.mode_history[mode_name].append(packet)

        if len(self.mode_history[mode_name]) > self.history_length:
            self.mode_history[mode_name] = self.mode_history[mode_name][-self.history_length :]

    def _check_for_new_consumers(self) -> bool:
        """Check if new consumers have connected to the LSL stream."""
        if not hasattr(self, "streamer") or self.streamer is None:
            return False

        try:
            current_consumer_count = self.streamer.have_consumers()

            if self.shared_state:
                self.shared_state.set(viz_consumers_key(), current_consumer_count)

            if current_consumer_count > self._previous_consumer_count:
                self.logger.info(
                    f"Detected {current_consumer_count - self._previous_consumer_count} new consumer(s)"
                )
                self._previous_consumer_count = current_consumer_count
                return True

            self._previous_consumer_count = current_consumer_count
            return False

        except Exception:
            return False

    def _periodically_send_history(self) -> None:
        """Check for new consumers and send complete history when detected."""
        while not (self.stop_event and self.stop_event.is_set()):
            time.sleep(self._consumer_check_interval)

            try:
                if self._check_for_new_consumers():
                    with self.history_lock:
                        current_time = time.time()

                        for mode_name, packets in self.mode_history.items():
                            if not packets:
                                continue

                            latest_packet = packets[-1]
                            mode_type = latest_packet.get("mode_type")
                            if mode_type is None:
                                self.logger.error(
                                    f"Historical packets for {mode_name} missing required mode_type field - skipping history"
                                )
                                continue

                            history_message: ModeHistoryPayload = {
                                "type": "mode_history",
                                "timestamp": current_time,
                                "mode_name": mode_name,
                                "mode_type": mode_type,
                                "data": {"latest_output": self.mode_outputs.get(mode_name, {})},
                                "packets": packets[-100:],
                                "packet_count": len(packets),
                            }

                            if not (self.stop_event and self.stop_event.is_set()):
                                self._send_data(history_message)

                            self.logger.info(
                                f"Sent mode history for {mode_name} with {len(packets)} packets to new consumer"
                            )

            except Exception as e:
                self.logger.error(f"Error sending history: {e}")

    def _report_bandwidth(self) -> None:
        """Report bandwidth usage and write to SharedState."""
        with self.bandwidth_lock:
            current_time = time.time()
            time_diff = current_time - self.last_report_time

            if time_diff > 0:
                bandwidth_bytes = self.bytes_sent / time_diff
                bandwidth_kbps = bandwidth_bytes / 1024
                bandwidth_str = self._format_bandwidth(bandwidth_bytes)
                self.logger.info(f"Bandwidth usage for {self.stream_info.name}: {bandwidth_str}")

                if self.shared_state:
                    self.shared_state.set(viz_bandwidth_key(), bandwidth_kbps)

                self.bytes_sent = 0
                self.last_report_time = current_time
