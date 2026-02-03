"""
Pure LSL streamer for Lab Streaming Layer output.
"""

import multiprocessing
from multiprocessing.synchronize import Event
from typing import Any

from dendrite.data.lsl_helpers import StreamConfig

from .base import LSLBaseStreamer


class LSLStreamer(LSLBaseStreamer):
    """
    Pure LSL-only streamer.

    This provides LSL output functionality with configurable stream settings.
    """

    def __init__(
        self,
        input_queue: multiprocessing.Queue,
        stream_info: StreamConfig,
        stop_event: Event | None = None,
        lsl_config: dict[str, Any] | None = None,
        shared_state: Any | None = None,
    ) -> None:
        """
        Initialize LSL streamer.

        Args:
            input_queue: Queue containing data to be streamed
            stream_info: Default StreamConfig for LSL
            stop_event: Event to signal when to stop streaming
            lsl_config: Optional config with stream_name, stream_type, source_id
            shared_state: SharedState instance for exposing streaming metrics
        """
        # Update stream_info with user configuration if provided
        if lsl_config:
            if "stream_name" in lsl_config:
                stream_info.name = lsl_config["stream_name"]
            if "stream_type" in lsl_config:
                stream_info.type = lsl_config["stream_type"]
            if "source_id" in lsl_config:
                stream_info.source_id = lsl_config["source_id"]

        super().__init__(input_queue, stream_info, stop_event, shared_state=shared_state)

        self.logger.info(f"LSLStreamer initialized with stream: {stream_info.name}")
