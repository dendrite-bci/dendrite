import json
import logging
import queue
import time
from collections import deque
from dataclasses import dataclass
from multiprocessing import Process
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from threading import Thread
from typing import Any

import numpy as np
from pydantic import ValidationError
from pylsl import LostError, StreamInlet, local_clock, proc_clocksync, proc_dejitter, resolve_byprop

from dendrite.constants import (
    LATENCY_EVENT_TYPE,
    LSL_BUFFER_SIZE_SECONDS,
    SAMPLE_PULL_TIMEOUT,
    STREAM_RESOLVE_TIMEOUT,
    THREAD_JOIN_TIMEOUT,
)
from dendrite.data.event_schemas import EventData
from dendrite.data.stream_schemas import StreamMetadata
from dendrite.utils.logger_central import get_logger, setup_logger
from dendrite.utils.state_keys import (
    e2e_latency_key,
    stream_connected_key,
    stream_latency_key,
    stream_timestamp_key,
)


@dataclass
class DataRecord:
    """Data structure for holding samples and metadata."""

    modality: str
    sample: Any
    timestamp: float
    local_timestamp: float


class DataAcquisition(Process):
    """Multi-threaded data acquisition for handling multiple LSL streams at native rates.

    Uses separate threads for each stream type. Each reader sends directly to queue
    at its native sampling rate (no main loop, no rate-ratio downsampling).
    """

    def __init__(
        self,
        data_queue: Queue,
        save_queue: Queue,
        stop_event: Event,
        stream_configs: list[StreamMetadata],
        shared_state=None,
    ) -> None:
        Process.__init__(self)
        self.data_queue = data_queue
        self.save_queue = save_queue
        self.stop_event = stop_event
        self.stream_configs = stream_configs or []
        self.shared_state = shared_state

        self.logger = get_logger("DataAcquisition")

        self.stream_inlets: dict[str, StreamInlet] = {}
        self.stream_info: dict[str, StreamMetadata] = {}
        self.channel_mapping = self._build_channel_mapping()

        self._reader_threads: list[Thread] = []

        # Throttle latency warnings to avoid log spam
        self._last_latency_warning_time: dict[str, float] = {}
        self._latency_warning_interval: float = 10.0

        # Rolling window for P50 latency (smooths noisy per-sample values)
        # Uses deque for O(1) append with automatic eviction
        self._latency_windows: dict[str, deque] = {}
        self._latency_window_size = 100  # ~0.2s at 500Hz
        self._latency_update_interval = 50  # Update P50 every 50 samples (~10Hz)
        self._latency_sample_counts: dict[str, int] = {}

    def _extract_modalities_from_config(self, config: StreamMetadata) -> dict[str, Any]:
        """Extract modality mapping from a stream config."""
        stream_map = {
            "modalities": {},
            "marker_index": None,
            "sample_rate": config.sample_rate,
        }

        channel_types = config.channel_types or []
        if channel_types:
            for i, ch_type in enumerate(channel_types):
                if ch_type.lower() == "markers":
                    stream_map["marker_index"] = i
                else:
                    modality = ch_type.lower()
                    if modality not in stream_map["modalities"]:
                        stream_map["modalities"][modality] = []
                    stream_map["modalities"][modality].append(i)
        else:
            modality = config.type.lower()
            stream_map["modalities"][modality] = list(range(config.channel_count))

        return stream_map

    def _build_channel_mapping(self) -> dict:
        """Build per-stream channel mapping from configs."""
        mapping = {}

        for config in self.stream_configs:
            stream_type = config.type

            if stream_type == "Events":
                continue

            channel_format = config.channel_format or "float32"
            if channel_format == "string":
                mapping[stream_type] = {"is_string": True, "channel_count": config.channel_count}
                self.logger.info(
                    f"Stream {stream_type} has string format - will be saved but not processed"
                )
                continue

            stream_map = self._extract_modalities_from_config(config)
            mapping[stream_type] = stream_map

            for modality, indices in stream_map["modalities"].items():
                self.logger.info(f"  {stream_type} -> {modality}: {len(indices)} channels")
            if stream_map["marker_index"] is not None:
                self.logger.info(f"  {stream_type} -> markers: index {stream_map['marker_index']}")

        return mapping

    def _safe_save(self, record: DataRecord) -> None:
        """Save record to queue, dropping silently if full."""
        try:
            self.save_queue.put_nowait(record)
        except queue.Full:
            self.logger.warning(f"Save queue full, dropped {record.modality} sample")
        except Exception as e:
            self.logger.error(f"Error saving {record.modality}: {e}")

    def _save_data_record(
        self, modality: str, sample: Any, timestamp: float, local_time: float
    ) -> None:
        record = DataRecord(
            modality=modality, sample=sample, timestamp=timestamp, local_timestamp=local_time
        )
        self._safe_save(record)

    def _send_to_data_queue(self, payload: dict[str, Any]) -> None:
        try:
            self.data_queue.put_nowait(payload)
        except queue.Full:
            self.logger.warning("Data queue full, dropped sample")

    def _handle_lost_error(self, stream_type: str) -> None:
        """Handle LSL stream disconnection."""
        self.logger.error(f"{stream_type} stream lost")
        if self.shared_state:
            self.shared_state.set(stream_connected_key(stream_type), False)

    def _handle_reader_error(self, stream_type: str, error: Exception) -> None:
        """Handle generic reader thread error with throttled retry."""
        if not self.stop_event.is_set():
            self.logger.exception(f"{stream_type} reader error: {error}")
            time.sleep(0.1)

    def _update_latency_telemetry(self, stream_type: str, latency_ms: float) -> None:
        """Update shared state with smoothed stream latency (P50, throttled).

        Uses deque for O(1) append and throttles P50 calculation to ~10Hz
        instead of computing on every sample (500Hz).
        """
        # Initialize window for new stream types
        if stream_type not in self._latency_windows:
            self._latency_windows[stream_type] = deque(maxlen=self._latency_window_size)
            self._latency_sample_counts[stream_type] = 0

        # Append to ring buffer (O(1), no allocations)
        self._latency_windows[stream_type].append(latency_ms)
        self._latency_sample_counts[stream_type] += 1

        # Events streams: publish immediately (irregular, low-frequency data)
        # Other streams: throttle to every N samples for efficiency
        is_events = stream_type.lower() == "events"
        min_samples = 1 if is_events else 10
        update_interval = 1 if is_events else self._latency_update_interval

        if self._latency_sample_counts[stream_type] < update_interval:
            return
        self._latency_sample_counts[stream_type] = 0

        # Compute and publish P50 (or single value for Events)
        window = self._latency_windows[stream_type]
        if self.shared_state and len(window) >= min_samples:
            p50 = float(np.median(window))
            self.shared_state.set(stream_latency_key(stream_type), p50)
            self.shared_state.set(stream_timestamp_key(stream_type), time.time())

    def run(self) -> None:
        """
        Main process entry point.

        Sets up logging, runs data collection, and ensures cleanup on exit.
        """
        self.logger = setup_logger("DataAcquisition", level=logging.INFO)
        self.logger.info("Data acquisition process started")

        try:
            self._collect_data()
        except Exception as e:
            self.logger.error(f"DAQ error: {e}", exc_info=True)
        finally:
            self._cleanup()
            self.stop_event.set()

    def _collect_data(self) -> None:
        """
        Main data collection orchestrator.

        Connects streams, starts reader threads. Each reader sends directly to queue.
        No main loop - all streams send independently at native rates.
        """
        if not self._connect_streams():
            return

        self._start_reader_threads()
        self.logger.info("All reader threads started - each stream sends at native rate")

        # Wait for stop event (readers run independently)
        while not self.stop_event.is_set():
            time.sleep(0.1)

    def _connect_streams(self) -> bool:
        """Connect to all configured streams."""
        for config in self.stream_configs:
            stream_type = config.type
            stream_name = config.name

            if stream_type in self.stream_inlets:
                continue

            inlet = self._resolve_stream(stream_name, stream_type)
            if inlet is None:
                if stream_type == "EEG":
                    self.logger.error(
                        f"Required EEG stream not found: '{stream_name}' (type={stream_type})"
                    )
                    return False
                else:
                    self.logger.warning(
                        f"Optional {stream_type} stream not found: '{stream_name}' (type={stream_type})"
                    )
                    continue

            self.stream_inlets[stream_type] = inlet
            self.stream_info[stream_type] = config
            self.logger.info(f"Connected to {stream_type} stream: '{stream_name}'")

            if self.shared_state:
                self.shared_state.set(stream_connected_key(stream_type), True)

            # Add synthetic markers channel if stream has EEG but no markers
            stream_map = self.channel_mapping.get(stream_type, {})
            has_eeg = "eeg" in stream_map.get("modalities", {})
            has_markers = stream_map.get("marker_index") is not None
            if has_eeg and not has_markers:
                orig_count = config.channel_count
                config = config.model_copy(
                    update={
                        "channel_count": orig_count + 1,
                        "labels": (config.labels or [])[:orig_count] + ["Markers"],
                        "channel_types": (config.channel_types or [])[:orig_count] + ["Markers"],
                        "channel_units": (config.channel_units or [])[:orig_count] + ["integer"],
                    }
                )
                self.stream_info[stream_type] = config
                self.logger.info(f"Added synthetic markers channel to {stream_type} stream")

            self._send_stream_metadata(stream_type, inlet)

        return "EEG" in self.stream_inlets

    def _resolve_stream(self, stream_name: str, stream_type: str) -> StreamInlet | None:
        """
        Resolve LSL stream by name and type with retry mechanism.

        Args:
            stream_name: Name of stream to find
            stream_type: LSL type of stream

        Returns:
            StreamInlet if found, None otherwise
        """
        start_time = time.time()

        self.logger.info(f"Resolving stream: name='{stream_name}', type='{stream_type}'")

        while time.time() - start_time < STREAM_RESOLVE_TIMEOUT and not self.stop_event.is_set():
            streams = resolve_byprop("type", stream_type, timeout=1.0)

            if not streams:
                time.sleep(0.1)
                continue

            for stream in streams:
                if stream.name() == stream_name:
                    self.logger.info(f"Stream resolved (exact): {stream_name}")
                    return StreamInlet(
                        stream,
                        max_buflen=LSL_BUFFER_SIZE_SECONDS,
                        processing_flags=proc_clocksync | proc_dejitter,
                    )
            time.sleep(0.1)

        return None

    def _start_reader_threads(self) -> None:
        """
        Start reader threads for all streams.

        Each reader sends directly to data_queue at native rate.
        No buffering - processor handles routing and event distribution.
        """
        # Start data stream readers (EEG, EMG, etc.)
        for stream_type, stream_map in self.channel_mapping.items():
            if stream_type not in self.stream_inlets:
                continue

            # String streams: save only, don't send to processor
            if stream_map.get("is_string"):
                channel_count = stream_map.get("channel_count", 1)
                thread = Thread(
                    target=self._string_reader, args=(stream_type, channel_count), daemon=True
                )
                thread.start()
                self._reader_threads.append(thread)
                self.logger.info(f"Started {stream_type} reader thread (string - save only)")
                continue

            # Numerical streams: extract modalities and send to processor
            thread = Thread(target=self._stream_reader, args=(stream_type,), daemon=True)
            thread.start()
            self._reader_threads.append(thread)
            rate = stream_map.get("sample_rate", "unknown")
            self.logger.info(f"Started {stream_type} reader thread ({rate}Hz)")

        # Start events reader
        if "Events" in self.stream_inlets:
            thread = Thread(target=self._events_reader, daemon=True)
            thread.start()
            self._reader_threads.append(thread)
            self.logger.info("Started events reader thread")
        else:
            self.logger.info("No Events stream found - events will not be processed")

    def _extract_modalities_from_sample(
        self, sample: np.ndarray, stream_map: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        """Extract modality data from a raw sample based on channel mapping."""
        data_dict = {}
        for modality, indices in stream_map["modalities"].items():
            data_dict[modality] = sample[indices].reshape(-1, 1)

        marker_index = stream_map.get("marker_index")
        if marker_index is not None:
            data_dict["markers"] = np.array([[sample[marker_index]]])
        elif "eeg" in data_dict:
            data_dict["markers"] = np.array([[0.0]])

        return data_dict

    def _stream_reader(self, stream_type: str) -> None:
        """Generic reader thread for any data stream."""
        inlet = self.stream_inlets[stream_type]
        stream_map = self.channel_mapping[stream_type]

        while not self.stop_event.is_set():
            try:
                sample, timestamp = inlet.pull_sample(timeout=SAMPLE_PULL_TIMEOUT)
                if sample is None:
                    continue

                local_time = local_clock()
                latency_ms = (local_time - timestamp) * 1000.0
                self._update_latency_telemetry(stream_type, latency_ms)

                sample = np.array(sample, dtype=np.float32)
                data_dict = self._extract_modalities_from_sample(sample, stream_map)
                stream_name = (
                    self.stream_info[stream_type].name
                    if stream_type in self.stream_info
                    else stream_type
                )

                payload = {
                    "data": data_dict,
                    "stream_name": stream_name,
                    "lsl_timestamp": timestamp,
                    "_daq_receive_ns": time.time_ns(),
                    f"{stream_type.lower()}_latency_ms": latency_ms,
                }
                self._send_to_data_queue(payload)
                self._save_data_record(stream_type, sample, timestamp, local_time)

                current_time = time.time()
                if (
                    latency_ms > 50
                    and (current_time - self._last_latency_warning_time.get(stream_type, 0))
                    > self._latency_warning_interval
                ):
                    self.logger.warning(f"{stream_type} high latency: {latency_ms:.2f}ms")
                    self._last_latency_warning_time[stream_type] = current_time

            except LostError:
                self._handle_lost_error(stream_type)
                break
            except Exception as e:
                self._handle_reader_error(stream_type, e)

    def _string_reader(self, stream_type: str, channel_count: int) -> None:
        """
        Reader thread for string streams (save only, not sent to processor).

        Args:
            stream_type: Type identifier for the stream
            channel_count: Number of channels to extract
        """
        inlet = self.stream_inlets[stream_type]

        while not self.stop_event.is_set():
            try:
                sample, timestamp = inlet.pull_sample(timeout=SAMPLE_PULL_TIMEOUT)
                if sample is None:
                    continue

                local_time = local_clock()
                valid_sample = sample[:channel_count]
                self._save_data_record(stream_type, valid_sample, timestamp, local_time)

            except LostError:
                self._handle_lost_error(stream_type)
                break
            except Exception as e:
                self._handle_reader_error(stream_type, e)

    def _parse_event_sample(self, sample: list[str]) -> tuple[dict[str, Any], float] | None:
        """Parse event sample JSON and extract event_id.

        Returns:
            Tuple of (normalized_event_json, event_id) or None if parsing fails
        """
        if not sample:
            return None

        try:
            raw_data = json.loads(sample[0])
            event = EventData.model_validate(raw_data)
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse event JSON: {e}")
            return None
        except ValidationError as e:
            self.logger.warning(f"Invalid event data: {e}")
            return None

        event_json = event.model_dump()

        # Extract E2E latency from task latency_update events
        if event.event_type == LATENCY_EVENT_TYPE:
            latency_raw = event_json.get("latency_ms_raw")
            if latency_raw is not None and self.shared_state:
                self.shared_state.set(e2e_latency_key(), float(latency_raw))

        return event_json, event.event_id

    def _events_reader(self) -> None:
        """Reader thread for event stream."""
        inlet = self.stream_inlets["Events"]

        while not self.stop_event.is_set():
            try:
                sample, timestamp = inlet.pull_sample(timeout=SAMPLE_PULL_TIMEOUT)
                if sample is None:
                    continue

                local_time = local_clock()
                events_latency_ms = (local_time - timestamp) * 1000.0
                self._update_latency_telemetry("Events", events_latency_ms)

                parsed = self._parse_event_sample(sample)
                if parsed is None:
                    continue

                event_json, event_id = parsed
                payload = {
                    "event": {"event_id": event_id, "event_json": event_json},
                    "lsl_timestamp": timestamp,
                    "_daq_receive_ns": time.time_ns(),
                }
                self._send_to_data_queue(payload)
                self._save_data_record("Event", event_json, timestamp, local_time)
                self.logger.debug(f"Event sent to processor: event_id={event_id}")

            except LostError:
                self._handle_lost_error("Events")
                break
            except Exception as e:
                self._handle_reader_error("Events", e)

    def _cleanup(self) -> None:
        """
        Clean up resources on shutdown.

        Joins threads with timeout and closes stream inlets.
        """
        self.logger.info("Cleaning up data acquisition resources")

        for thread in self._reader_threads:
            try:
                thread.join(timeout=THREAD_JOIN_TIMEOUT)
                if thread.is_alive():
                    self.logger.warning(f"Thread {thread.name} did not terminate within timeout")
            except Exception as e:
                self.logger.error(f"Error joining thread {thread.name}: {e}")

        for stream_type, inlet in self.stream_inlets.items():
            try:
                inlet.close_stream()
                self.logger.debug(f"Closed {stream_type} stream inlet")
            except Exception as e:
                self.logger.error(f"Error closing {stream_type} inlet: {e}")

        self.stream_inlets.clear()

        self.logger.info("Data acquisition cleanup complete")

    def _send_stream_metadata(self, stream_type: str, inlet: StreamInlet) -> None:
        """
        Send stream metadata to save queue.

        Extracts metadata from LSL stream and config, then saves as JSON.

        Args:
            stream_type: Stream type identifier
            inlet: LSL stream inlet
        """
        try:
            stream_config = self.stream_info.get(stream_type)
            if not stream_config:
                return

            live_stream_info = inlet.info()
            metadata_object = stream_config.model_copy(
                update={
                    "created_at": live_stream_info.created_at(),
                    "version": live_stream_info.version(),
                }
            )

            metadata_json = json.dumps(metadata_object.model_dump())
            now = local_clock()
            metadata_record = DataRecord(
                modality=f"{stream_type}_Metadata",
                sample=metadata_json,
                timestamp=now,
                local_timestamp=now,
            )

            self._safe_save(metadata_record)
            self.logger.info(f"Metadata for {stream_type} sent")

        except Exception as e:
            self.logger.error(f"Error sending metadata for {stream_type}: {e}")
