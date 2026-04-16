"""
Output Protocol Manager

Manages output protocol streamers (LSL, Socket, ZMQ, ROS2) for prediction distribution.
"""

import copy
import multiprocessing
import queue
import threading
from multiprocessing.queues import Queue as MpQueue
from multiprocessing.synchronize import Event as MpEvent
from typing import Any

from dendrite.data.stream_schemas import StreamConfig
from dendrite.data.streaming import LSLStreamer, ROS2Streamer, SocketStreamer, ZMQStreamer

PREDICTION_STREAM_INFO = StreamConfig(
    name="PredictionStream",
    type="PredictionStream",
    channel_count=1,
    sample_rate=0,
    channel_format="string",
    labels=["Data"],
    source_id="dendrite_prediction_stream",
)
from dendrite.utils.logger_central import get_logger

PREDICTION_FANOUT_TIMEOUT = 0.1
SUPPORTED_PROTOCOLS = ["lsl", "socket", "zmq", "ros2"]


class OutputProtocolManager:
    """Manages output protocol streamers and prediction fan-out."""

    def __init__(
        self,
        stop_event: MpEvent,
        prediction_queue: MpQueue[Any],
        shared_state: Any,
    ):
        self.stop_event = stop_event
        self.prediction_queue = prediction_queue
        self.shared_state = shared_state

        self.streamer_stop_event = multiprocessing.Event()
        self.output_streamers: dict[str, Any] = {}
        self.protocol_queues: dict[str, MpQueue[Any]] = {}
        self.prediction_fanout_thread: threading.Thread | None = None

        self.logger = get_logger("OutputProtocolManager")

    def initialize(self, config: dict) -> dict[str, Any]:
        """Initialize all output protocols. Returns dict of name -> streamer process."""
        output_config = config.get("output", {}).get("protocols", {})
        enabled_protocols = self._setup_protocol_queues(output_config)

        if not enabled_protocols:
            self.logger.warning("No output protocols enabled")
            return {}

        self.logger.info(f"Initializing output protocols: {enabled_protocols}")
        self._start_prediction_fanout_thread()

        for protocol in enabled_protocols:
            self._init_streamer(protocol, output_config.get(protocol, {}), config)

        return {p.upper(): s for p, s in self.output_streamers.items()}

    def _setup_protocol_queues(self, output_config: dict) -> list[str]:
        self.protocol_queues = {}
        enabled_protocols = []

        for protocol in SUPPORTED_PROTOCOLS:
            protocol_config = output_config.get(protocol, {})
            if protocol_config.get("enabled", False):
                enabled_protocols.append(protocol)
                self.protocol_queues[protocol] = multiprocessing.Queue()

        if not output_config and not enabled_protocols:
            self.logger.info("No output configuration found, defaulting to LSL")
            enabled_protocols = ["lsl"]
            self.protocol_queues["lsl"] = multiprocessing.Queue()

        return enabled_protocols

    def _init_streamer(self, protocol: str, protocol_config: dict, full_config: dict):
        try:
            streamer_class, streamer_kwargs = self._get_streamer_config(
                protocol, protocol_config, full_config
            )
            streamer = streamer_class(**streamer_kwargs)
            streamer.daemon = True
            streamer.start()
            self.output_streamers[protocol] = streamer
            self.logger.info(f"{protocol.upper()} streamer started")
        except Exception as e:
            self.logger.error(f"Failed to initialize {protocol} streamer: {e}")

    def _get_streamer_config(self, protocol: str, protocol_config: dict, full_config: dict):
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

    def _start_prediction_fanout_thread(self):
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

    def get_stop_targets(self) -> list[tuple[str, Any, int]]:
        targets = []
        if self.prediction_fanout_thread and self.prediction_fanout_thread.is_alive():
            targets.append(("Fan-out", self.prediction_fanout_thread, 2))
        for protocol, streamer in self.output_streamers.items():
            if streamer and streamer.is_alive():
                targets.append((protocol, streamer, 2))
        return targets

    def signal_stop(self):
        self.streamer_stop_event.set()

    def cleanup(self):
        self.output_streamers = {}
        self.protocol_queues = {}
        self.prediction_fanout_thread = None
