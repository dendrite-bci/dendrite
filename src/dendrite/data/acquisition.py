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

from dendrite.constants import SAMPLE_PULL_TIMEOUT

STREAM_RESOLVE_TIMEOUT = 30  # Full LSL stream resolution
LSL_BUFFER_SIZE_SECONDS = 5  # LSL inlet buffer duration
THREAD_JOIN_TIMEOUT = 2.0
LATENCY_EVENT_TYPE = "latency_update"
from dendrite.data.event_schemas import EventData
from dendrite.data.shared_buffers import SharedRingBuffer
from dendrite.data.stream_schemas import StreamMetadata
from dendrite.utils.component_state import ComponentState, ComponentStateMachine
from dendrite.utils.logger_central import get_logger, setup_logger
from dendrite.utils.state_keys import (
    e2e_latency_key,
    stream_connected_key,
    stream_latency_key,
    stream_timestamp_key,
)


@dataclass
class EventRecord:
    """Payload for event queue: rich event data with 3 timestamps."""

    sample: dict
    timestamp: float
    local_timestamp: float
    receive_timestamp: float = 0.0


class DataAcquisition(Process):
    """Multi-threaded data acquisition for handling multiple LSL streams at native rates.

    Uses separate threads for each stream type. Each reader writes to shared ring buffers
    at its native sampling rate (no main loop, no rate-ratio downsampling).
    """

    def __init__(
        self,
        stop_event: Event,
        stream_configs: list[StreamMetadata],
        shared_state=None,
        ring_buffer_names: dict[str, str] | None = None,
        event_queue: Queue | None = None,
    ) -> None:
        Process.__init__(self)
        self.event_queue = event_queue
        self.stop_event = stop_event
        self.stream_configs = stream_configs or []
        self.shared_state = shared_state

        # Shared memory buffer names (connected in run() after spawn)
        self._ring_buffer_names = ring_buffer_names or {}
        self._ring_buffers: dict[str, SharedRingBuffer] = {}

        # Per-stream event deques: events are broadcast to ALL stream readers
        self._pending_events_per_stream: dict[str, deque] = {}

        self.logger = get_logger("DataAcquisition")

        self.stream_inlets: dict[str, StreamInlet] = {}
        self.stream_info: dict[str, StreamMetadata] = {}

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

    def _save_event(self, record: EventRecord) -> None:
        """Save event record to event queue."""
        if self.event_queue is None:
            return
        try:
            self.event_queue.put_nowait(record)
        except queue.Full:
            self.logger.warning("Event queue full, dropped event")
        except Exception as e:
            self.logger.error(f"Error saving event: {e}")

    def _handle_lost_error(self, stream_key: str) -> None:
        """Handle LSL stream disconnection."""
        self.logger.error(f"{stream_key} stream lost")
        if self.shared_state:
            self.shared_state.set(stream_connected_key(stream_key), False)

    def _handle_reader_error(self, stream_key: str, error: Exception) -> None:
        """Handle generic reader thread error with throttled retry."""
        if not self.stop_event.is_set():
            self.logger.exception(f"{stream_key} reader error: {error}")
            time.sleep(0.1)

    def _update_latency_telemetry(self, stream_key: str, latency_ms: float) -> None:
        """Update shared state with smoothed stream latency (P50, throttled).

        Uses deque for O(1) append and throttles P50 calculation to ~10Hz
        instead of computing on every sample (500Hz).
        """
        # Initialize window for new stream keys
        if stream_key not in self._latency_windows:
            self._latency_windows[stream_key] = deque(maxlen=self._latency_window_size)
            self._latency_sample_counts[stream_key] = 0

        # Append to ring buffer (O(1), no allocations)
        self._latency_windows[stream_key].append(latency_ms)
        self._latency_sample_counts[stream_key] += 1

        # Events streams: publish immediately (irregular, low-frequency data)
        # Other streams: throttle to every N samples for efficiency
        is_events = stream_key.lower() == "events"
        min_samples = 1 if is_events else 10
        update_interval = 1 if is_events else self._latency_update_interval

        if self._latency_sample_counts[stream_key] < update_interval:
            return
        self._latency_sample_counts[stream_key] = 0

        # Compute and publish P50 (or single value for Events)
        window = self._latency_windows[stream_key]
        if self.shared_state and len(window) >= min_samples:
            p50 = float(np.median(window))
            self.shared_state.set(stream_latency_key(stream_key), p50)
            self.shared_state.set(stream_timestamp_key(stream_key), time.time())

    def run(self) -> None:
        """
        Main process entry point.

        Sets up logging, runs data collection, and ensures cleanup on exit.
        """
        self.logger = setup_logger("DataAcquisition", level=logging.INFO)
        self._state_machine = ComponentStateMachine("daq", self.shared_state)
        self.logger.info("Data acquisition process started")
        self._state_machine.transition(ComponentState.STARTING)

        # Connect to shared memory buffers (must happen in child process after spawn)
        self._connect_shared_buffers()

        try:
            self._collect_data()
        except Exception as e:
            self.logger.error(f"DAQ error: {e}", exc_info=True)
            if self._state_machine.state not in (ComponentState.STOPPING, ComponentState.STOPPED):
                self._state_machine.set_error(str(e))
        finally:
            self._cleanup()
            self._state_machine.finalize()
            self.stop_event.set()

    def _connect_shared_buffers(self):
        """Connect to shared memory buffers created by orchestrator.

        Must be called in the child process (after spawn on Windows).
        """
        for stream_type, buf_name in self._ring_buffer_names.items():
            try:
                self._ring_buffers[stream_type] = SharedRingBuffer.connect(buf_name)
                self.logger.info(f"Connected to ring buffer: {buf_name} ({stream_type})")
            except Exception as e:
                self.logger.error(f"Failed to connect to ring buffer '{buf_name}': {e}")

    def _collect_data(self) -> None:
        """
        Main data collection orchestrator.

        Connects streams, starts reader threads. Each reader writes to shared ring buffers.
        No main loop - all streams send independently at native rates.
        """
        if not self._connect_streams():
            self._state_machine.set_error("Failed to connect to required streams")
            return

        self._start_reader_threads()
        self._state_machine.transition(ComponentState.RUNNING)
        self.logger.info("All reader threads started - each stream sends at native rate")

        # Wait for stop event (readers run independently)
        while not self.stop_event.is_set():
            time.sleep(0.1)

    def _connect_streams(self) -> bool:
        """Connect to all configured streams."""
        for config in self.stream_configs:
            key = config.stream_key or config.type

            if key in self.stream_inlets:
                continue

            inlet = self._resolve_stream(config.name, config.type)
            if inlet is None:
                self.logger.error(
                    f"Configured stream not found: '{config.name}' (type={config.type})"
                )
                return False

            self.stream_inlets[key] = inlet
            self.stream_info[key] = config
            self.logger.info(f"Connected to {key} stream: '{config.name}'")

            if self.shared_state:
                self.shared_state.set(stream_connected_key(key), True)

        return len(self.stream_inlets) > 0

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

        Numerical readers write to shared ring buffers at native rate.
        String readers save to disk only. Events reader broadcasts to all stream deques.
        """
        # Start data stream readers (EEG, EMG, etc.)
        for config in self.stream_configs:
            key = config.stream_key or config.type
            if config.type.upper() == "EVENTS" or key not in self.stream_inlets:
                continue

            # String streams: save only, no ring buffer
            channel_format = config.channel_format or "float32"
            if channel_format == "string":
                thread = Thread(
                    target=self._string_reader, args=(key, config.channel_count), daemon=True
                )
                thread.start()
                self._reader_threads.append(thread)
                self.logger.info(f"Started {key} reader thread (string - save only)")
                continue

            # Numerical streams: create per-stream event deque and start reader
            self._pending_events_per_stream[key] = deque()
            thread = Thread(target=self._stream_reader, args=(key,), daemon=True)
            thread.start()
            self._reader_threads.append(thread)
            self.logger.info(f"Started {key} reader thread ({config.sample_rate}Hz)")

        # Start events reader — find the Events stream by checking stream_info types
        events_key = next(
            (k for k, info in self.stream_info.items() if info.type.upper() == "EVENTS"), None
        )
        if events_key and events_key in self.stream_inlets:
            thread = Thread(target=self._events_reader, args=(events_key,), daemon=True)
            thread.start()
            self._reader_threads.append(thread)
            self.logger.info("Started events reader thread")
        else:
            self.logger.info("No Events stream found - events will not be processed")

    def _stream_reader(self, stream_key: str) -> None:
        """Generic reader thread for any data stream."""
        inlet = self.stream_inlets[stream_key]
        ring_buffer = self._ring_buffers.get(stream_key)

        # Pre-allocate write buffer: raw channels + 1 markers column
        if ring_buffer is not None:
            rb_buf = np.zeros(ring_buffer.n_channels, dtype=np.float32)
            marker_col = ring_buffer.n_channels - 1

        while not self.stop_event.is_set():
            try:
                sample, timestamp = inlet.pull_sample(timeout=SAMPLE_PULL_TIMEOUT)
                if sample is None or timestamp is None:
                    continue

                local_time = local_clock()
                latency_ms = (local_time - timestamp) * 1000.0
                self._update_latency_telemetry(stream_key, latency_ms)

                sample = np.array(sample, dtype=np.float32)

                # Write to shared ring buffer with markers baked in
                if ring_buffer is not None:
                    n_raw = len(sample)
                    rb_buf[:n_raw] = sample

                    # Inject pending LSL event into markers column
                    marker = 0.0
                    my_events = self._pending_events_per_stream.get(stream_key)
                    if my_events:
                        marker = my_events.popleft()

                    rb_buf[marker_col] = marker
                    ring_buffer.write(rb_buf, timestamp, local_time, time.time_ns())

                current_time = time.time()
                if (
                    latency_ms > 50
                    and (current_time - self._last_latency_warning_time.get(stream_key, 0))
                    > self._latency_warning_interval
                ):
                    self.logger.warning(f"{stream_key} high latency: {latency_ms:.2f}ms")
                    self._last_latency_warning_time[stream_key] = current_time

            except LostError:
                self._handle_lost_error(stream_key)
                break
            except Exception as e:
                self._handle_reader_error(stream_key, e)

    def _string_reader(self, stream_key: str, channel_count: int) -> None:
        """
        Reader thread for string streams (consume only, no ring buffer).

        String streams: no ring buffer, data not archived.
        Must keep reading to prevent LSL buffer overflow.
        """
        inlet = self.stream_inlets[stream_key]

        while not self.stop_event.is_set():
            try:
                sample, timestamp = inlet.pull_sample(timeout=SAMPLE_PULL_TIMEOUT)
                if sample is None or timestamp is None:
                    continue

            except LostError:
                self._handle_lost_error(stream_key)
                break
            except Exception as e:
                self._handle_reader_error(stream_key, e)

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

    def _events_reader(self, stream_key: str) -> None:
        """Reader thread for event stream."""
        inlet = self.stream_inlets[stream_key]

        while not self.stop_event.is_set():
            try:
                sample, timestamp = inlet.pull_sample(timeout=SAMPLE_PULL_TIMEOUT)
                if sample is None or timestamp is None:
                    continue

                local_time = local_clock()
                events_latency_ms = (local_time - timestamp) * 1000.0
                self._update_latency_telemetry(stream_key, events_latency_ms)

                parsed = self._parse_event_sample(sample)
                if parsed is None:
                    continue

                event_json, event_id = parsed

                # Broadcast event to all stream readers' ring buffers
                for dq in self._pending_events_per_stream.values():
                    dq.append(event_id)

                self._save_event(EventRecord(
                    sample=event_json, timestamp=timestamp,
                    local_timestamp=local_time, receive_timestamp=time.time(),
                ))
                self.logger.debug(f"Event: id={event_id}")

            except LostError:
                self._handle_lost_error(stream_key)
                break
            except Exception as e:
                self._handle_reader_error(stream_key, e)

    def _cleanup(self) -> None:
        """
        Clean up resources on shutdown.

        Joins threads with timeout and closes stream inlets and shared buffers.
        """
        self.logger.info("Cleaning up data acquisition resources")

        for thread in self._reader_threads:
            try:
                thread.join(timeout=THREAD_JOIN_TIMEOUT)
                if thread.is_alive():
                    self.logger.warning(f"Thread {thread.name} did not terminate within timeout")
            except Exception as e:
                self.logger.error(f"Error joining thread {thread.name}: {e}")

        for stream_key, inlet in self.stream_inlets.items():
            try:
                inlet.close_stream()
                self.logger.debug(f"Closed {stream_key} stream inlet")
            except Exception as e:
                self.logger.error(f"Error closing {stream_key} inlet: {e}")

        self.stream_inlets.clear()

        # Close shared memory handles (don't unlink — orchestrator owns them)
        for rb in self._ring_buffers.values():
            rb.close()
        self._ring_buffers.clear()

        self.logger.info("Data acquisition cleanup complete")

