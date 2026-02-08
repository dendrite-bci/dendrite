"""
Output Protocol Manager

Manages output protocol streamers (LSL, Socket, ZMQ, ROS2) for prediction distribution.
Extracted from MainWindow to isolate protocol management and make it independently testable.
"""

import copy
import multiprocessing
import queue
import threading
from typing import Any

from dendrite.constants import PREDICTION_STREAM_INFO
from dendrite.data.streaming import LSLStreamer, ROS2Streamer, SocketStreamer, ZMQStreamer
from dendrite.utils.logger_central import get_logger

PREDICTION_FANOUT_TIMEOUT = 0.1
SUPPORTED_PROTOCOLS = ["lsl", "socket", "zmq", "ros2"]


class OutputProtocolManager:
    """Manages output protocol streamers and prediction fan-out.

    Owns:
    - Protocol queue creation
    - Streamer initialization and lifecycle
    - Prediction fan-out thread (distributes predictions to all protocol queues)
    - Protocol status callbacks
    """

    def __init__(
        self,
        stop_event: multiprocessing.Event,
        prediction_queue: multiprocessing.Queue,
        shared_state: Any,
        status_callback=None,
    ):
        self.stop_event = stop_event
        self.prediction_queue = prediction_queue
        self.shared_state = shared_state
        self._status_callback = status_callback

        self.streamer_stop_event = multiprocessing.Event()
        self.output_streamers: dict[str, Any] = {}
        self.protocol_queues: dict[str, multiprocessing.Queue] = {}
        self.prediction_fanout_thread: threading.Thread | None = None

        self.logger = get_logger("OutputProtocolManager")

    def initialize(self, config: dict) -> dict[str, Any]:
        """Initialize all output protocols based on configuration.

        Returns:
            Dict of protocol_name -> streamer process for system process tracking.
        """
        output_config = config.get("output", {}).get("protocols", {})
        self.logger.debug(f"Output configuration: {output_config}")

        enabled_protocols = self._setup_protocol_queues(output_config)

        if not enabled_protocols:
            self.logger.warning("No output protocols enabled - predictions will not be streamed")
            return {}

        self.logger.info(f"Initializing output protocols: {enabled_protocols}")
        self._start_prediction_fanout_thread()

        for protocol in enabled_protocols:
            self._init_streamer(protocol, output_config.get(protocol, {}), config)

        return {p.upper(): s for p, s in self.output_streamers.items()}

    def _setup_protocol_queues(self, output_config: dict) -> list[str]:
        """Set up queues for enabled protocols and return list of enabled protocols."""
        self.protocol_queues = {}
        enabled_protocols = []

        for protocol in SUPPORTED_PROTOCOLS:
            protocol_config = output_config.get(protocol, {})
            is_enabled = protocol_config.get("enabled", False)

            self.logger.debug(f"Protocol {protocol}: enabled={is_enabled}, config={protocol_config}")

            if is_enabled:
                enabled_protocols.append(protocol)
                self.protocol_queues[protocol] = multiprocessing.Queue()

        if not output_config and not enabled_protocols:
            self.logger.info("No output configuration found, defaulting to LSL")
            enabled_protocols = ["lsl"]
            self.protocol_queues["lsl"] = multiprocessing.Queue()

        return enabled_protocols

    def _get_streamer_config(self, protocol: str, protocol_config: dict, full_config: dict):
        """Get streamer class and initialization kwargs for a protocol."""
        base_kwargs = {
            "input_queue": self.protocol_queues[protocol],
            "stop_event": self.streamer_stop_event,
            "shared_state": self.shared_state,
        }

        protocol_map = {
            "lsl": (
                LSLStreamer,
                {
                    **base_kwargs,
                    "stream_info": copy.deepcopy(PREDICTION_STREAM_INFO),
                    "lsl_config": protocol_config.get("config", {}),
                },
            ),
            "socket": (
                SocketStreamer,
                {
                    **base_kwargs,
                    "socket_config": protocol_config.get("config", {}),
                },
            ),
            "zmq": (
                ZMQStreamer,
                {
                    **base_kwargs,
                    "zmq_config": protocol_config.get("config", {}),
                },
            ),
            "ros2": (
                ROS2Streamer,
                {
                    **base_kwargs,
                    "ros2_config": protocol_config.get("config", {}),
                    "stream_name": "BMI_Predictions",
                    "classifier_names": list(full_config.get("mode_instances", {}).keys()),
                },
            ),
        }

        if protocol not in protocol_map:
            raise ValueError(f"Unknown protocol: {protocol}")
        return protocol_map[protocol]

    def _init_streamer(self, protocol: str, protocol_config: dict, full_config: dict):
        """Initialize a single output streamer."""
        try:
            streamer_class, streamer_kwargs = self._get_streamer_config(
                protocol, protocol_config, full_config
            )
            streamer = streamer_class(**streamer_kwargs)
            streamer.daemon = True
            streamer.start()
            self.output_streamers[protocol] = streamer
            self._update_status(True, protocol=protocol)
            self.logger.info(f"{protocol.upper()} streamer started")
        except Exception as e:
            self.logger.error(f"Failed to initialize {protocol} streamer: {e}")
            self._update_status(False, protocol=protocol)

    def _start_prediction_fanout_thread(self):
        """Start thread to distribute predictions to all protocol queues."""

        def fanout_predictions():
            logger = get_logger("PredictionFanOut")
            logger.info(f"Fan-out thread started for: {list(self.protocol_queues.keys())}")

            while not self.stop_event.is_set():
                try:
                    prediction_data = self.prediction_queue.get(timeout=PREDICTION_FANOUT_TIMEOUT)

                    for protocol, protocol_queue in self.protocol_queues.items():
                        try:
                            protocol_queue.put(prediction_data, block=False)
                        except queue.Full:
                            logger.warning(f"{protocol} queue full, dropping prediction")

                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Fan-out error: {e}")
                    break

            logger.info("Fan-out thread stopped")

        self.prediction_fanout_thread = threading.Thread(
            target=fanout_predictions, daemon=True, name="PredictionFanOut"
        )
        self.prediction_fanout_thread.start()

    def _update_status(self, connected: bool, protocol: str = None, all_protocols: bool = False):
        """Update UI status indicators for output protocol connections."""
        if self._status_callback:
            if all_protocols:
                for p in SUPPORTED_PROTOCOLS:
                    self._status_callback(p, connected)
            elif protocol:
                self._status_callback(protocol, connected)

    def get_stop_targets(self) -> list[tuple[str, Any, int]]:
        """Get list of (name, process/thread, timeout) for shutdown sequencing."""
        targets = []
        if self.prediction_fanout_thread and self.prediction_fanout_thread.is_alive():
            targets.append(("Fan-out", self.prediction_fanout_thread, 2))
        for protocol, streamer in self.output_streamers.items():
            if streamer and streamer.is_alive():
                targets.append((protocol, streamer, 2))
        return targets

    def signal_stop(self):
        """Signal all streamers to stop."""
        self.streamer_stop_event.set()

    def cleanup(self):
        """Reset state after shutdown."""
        self.output_streamers = {}
        self.protocol_queues = {}
        self.prediction_fanout_thread = None
        self._update_status(False, all_protocols=True)
