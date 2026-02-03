"""Synchronized data distribution to processing modes."""

import logging
import multiprocessing
import queue
import time
from collections import Counter, deque
from typing import TYPE_CHECKING, Any

import numpy as np

from dendrite.constants import (
    DEFAULT_CALIBRATION_DURATION,
    DEFAULT_QUALITY_Z_THRESHOLD,
    DROPPED_SAMPLE_WARNING_INTERVAL,
    PLOT_DECIMATION_FACTOR,
)
from dendrite.data.quality import ChannelQualityDetector
from dendrite.processing.preprocessing.preprocessor import OnlinePreprocessor
from dendrite.utils.logger_central import get_logger, setup_logger
from dendrite.utils.state_keys import channel_quality_key

if TYPE_CHECKING:
    from dendrite.data.stream_schemas import StreamMetadata


def _get_stream_attr(stream: Any, attr: str, default: Any = None) -> Any:
    """Get attribute from stream config, supporting both dict and object access."""
    if isinstance(stream, dict):
        return stream.get(attr, default)
    return getattr(stream, attr, default)


class DataProcessor:
    """Synchronized data processor with fan-out distribution to all modes."""

    def __init__(
        self,
        data_queue: Any,
        plot_queue: Any,
        stop_event: Any,
        mode_configs: dict[str, dict[str, Any]] | None = None,
        stream_configs: list["StreamMetadata"] | None = None,
        preprocessing_config: dict[str, Any] | None = None,
        mode_queue_size: int = 1000,
        shared_state=None,
    ):
        """Initialize processor with queues, modes, and optional preprocessing."""
        # Core components
        self.data_queue = data_queue
        self.plot_queue = plot_queue
        self.stop_event = stop_event
        self.shared_state = shared_state
        self.mode_configs = mode_configs or {}
        self.mode_names = list(self.mode_configs.keys())

        # Stream configs for per-stream preprocessing
        self.stream_configs = stream_configs or []

        # Preprocessing - one preprocessor per stream
        self.preprocessing_config = preprocessing_config
        self.preprocessors: dict[str, OnlinePreprocessor] = {}  # stream_name -> preprocessor

        # Channel quality (one-shot detection at calibration)
        self.quality_detector = None
        self.bad_channels_eeg: list[int] = []  # Fixed for session after calibration

        # Mode-specific queues and routing
        self.mode_queues = {}
        self.mode_required_modalities = {}  # mode_name -> List[str] of required modalities
        self.mode_stream_sources = {}  # mode_name -> {modality: stream_name} for stream-aware routing
        for name in self.mode_names:
            self.mode_queues[name] = multiprocessing.Queue(maxsize=mode_queue_size)
            config = self.mode_configs[name]
            raw_required = config.get("required_modalities", ["eeg"])
            self.mode_required_modalities[name] = [m.lower() for m in raw_required]
            self.mode_stream_sources[name] = config.get("stream_sources", {})

        self.dropped_samples = {name: 0 for name in self.mode_names}

        # Event handling - processor distributes markers to all streams
        self.pending_markers = (
            deque()
        )  # Queue of (event_id, timestamp, streams_pending: set) tuples

        # Visualization decimation (500Hz -> 100Hz for reduced bandwidth)
        self._plot_counter = 0
        self._effective_viz_rate = None  # Set when sample rate known
        self._viz_drops = 0  # Track visualization queue drops
        self._viz_state = {}  # Accumulates latest values from each modality for combined viz

        self.logger = get_logger("DataProcessor")
        self.logger.info("DataProcessor initialized")

    def get_queue_for_mode(self, mode_name: str) -> multiprocessing.Queue:
        """Get the dedicated queue for a specific mode."""
        return self.mode_queues.get(mode_name)

    def start(self) -> multiprocessing.Process:
        """Start the data processor in a separate process."""
        process = multiprocessing.Process(target=self._run_processing_loop, daemon=True)
        process.start()
        return process

    def _run_processing_loop(self):
        """Main data processing and distribution loop."""
        self.logger = setup_logger("DataProcessor", level=logging.DEBUG)
        self.logger.info("DataProcessor starting...")

        first_sample = self._wait_for_first_sample()
        if first_sample is None:
            return

        if self.preprocessing_config:
            self._create_preprocessors()

        self.bad_channels_eeg = self._setup_quality_control()

        self._handle_data_sample(first_sample)

        self.logger.info("Starting main processing loop...")

        while not self.stop_event.is_set():
            try:
                self._process_next_sample()
            except queue.Empty:
                pass  # No sample available, continue
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}", exc_info=True)
                time.sleep(0.1)  # Prevent rapid error retries

        self.logger.info("DataProcessor stopped")

    def _setup_quality_control(self) -> list[int]:
        """Setup channel quality detection and run calibration if enabled.

        Returns:
            List of bad channel indices (empty if quality control disabled)
        """
        quality_control_config = (
            self.preprocessing_config.get("quality_control", {})
            if self.preprocessing_config
            else {}
        )
        quality_control_enabled = quality_control_config.get("enabled", False)
        eeg_channel_count = self._get_eeg_channel_count()

        if not quality_control_enabled or eeg_channel_count == 0:
            return []

        z_threshold = quality_control_config.get("z_threshold", DEFAULT_QUALITY_Z_THRESHOLD)
        calibration_duration = quality_control_config.get(
            "calibration_duration", DEFAULT_CALIBRATION_DURATION
        )

        self.quality_detector = ChannelQualityDetector(z_threshold=z_threshold)
        bad_channels = self._run_calibration(calibration_duration)

        if self.shared_state:
            self.shared_state.set(
                channel_quality_key(),
                {
                    "bad_channels_eeg": bad_channels,
                    "detection_time": time.time(),
                    "total_channels": eeg_channel_count,
                },
            )

        return bad_channels

    def _wait_for_first_sample(self) -> dict[str, Any] | None:
        """Wait for the first data sample (skip event payloads)."""
        self.logger.info("Waiting for first data sample...")
        timeout_seconds = 60
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            if self.stop_event.is_set():
                return None
            try:
                payload = self.data_queue.get(timeout=1.0)
                # Skip event payloads, wait for actual data
                if "event" in payload:
                    self._handle_event(payload)
                    continue
                if "data" in payload:
                    return payload
            except queue.Empty:
                continue

        self.logger.error("Timeout waiting for first data sample")
        return None

    def _get_eeg_channel_count(self) -> int:
        """Get EEG channel count from stream_configs."""
        for stream in self.stream_configs:
            if _get_stream_attr(stream, "type", "").upper() == "EEG":
                return _get_stream_attr(stream, "channel_count", 0)
        return 0

    def _run_calibration(self, duration_sec: float = 3.0) -> list[int]:
        """Collect calibration data and detect bad channels (fixed for session)."""
        self.logger.info(f"Running channel quality calibration ({duration_sec}s)...")
        calibration_samples = []
        start_time = time.time()

        while time.time() - start_time < duration_sec:
            if self.stop_event.is_set():
                self.logger.warning("Calibration interrupted by stop event")
                return []
            try:
                payload = self.data_queue.get(timeout=0.1)

                # Skip event payloads during calibration
                if "event" in payload:
                    continue

                # Extract EEG data from modality dict
                data_dict = payload.get("data", {})
                eeg_data = data_dict.get("eeg")
                if eeg_data is not None:
                    # Flatten single sample: (n_channels, 1) -> (n_channels,)
                    calibration_samples.append(eeg_data.flatten())

            except queue.Empty:
                continue

        if not calibration_samples:
            self.logger.warning("No calibration data collected")
            return []

        # Stack into (n_samples, n_channels), transpose to (n_channels, n_samples)
        cal_data = np.array(calibration_samples).T

        bad_channels = self.quality_detector.detect_from_calibration(cal_data)

        self.logger.info(
            f"Calibration complete: {len(calibration_samples)} samples, "
            f"{len(bad_channels)} bad channels detected"
        )
        return bad_channels

    def _process_next_sample(self):
        """Process incoming payload: event or data sample."""
        payload = self.data_queue.get(timeout=0.1)

        if "event" in payload:
            self._handle_event(payload)
            return

        self._handle_data_sample(payload)

    def _handle_event(self, payload: dict[str, Any]):
        """Queue event marker for distribution to all streams."""
        event_info = payload["event"]
        event_id = event_info.get("event_id")
        lsl_timestamp = payload.get("lsl_timestamp")

        if event_id is not None:
            # Get all data stream names (exclude Events stream)
            data_streams = {
                _get_stream_attr(s, "name")
                for s in self.stream_configs
                if _get_stream_attr(s, "type", "").upper() != "EVENTS"
            }
            self.pending_markers.append((event_id, lsl_timestamp, data_streams.copy()))
            self.logger.debug(f"Event queued: ID={event_id}, pending_streams={data_streams}")

    def _handle_data_sample(self, payload: dict[str, Any]):
        """Process a data sample: preprocess and distribute to modes."""
        data_dict = payload["data"]
        stream_name = payload.get("stream_name")
        lsl_timestamp = payload.get("lsl_timestamp")
        daq_receive_ns = payload.get("_daq_receive_ns")

        # Note: Marker attachment deferred to _distribute_sample (after empty check)
        # to prevent marker loss during downsampling

        preprocessor = self.preprocessors.get(stream_name)
        filtered_dict = (
            preprocessor.process(data_dict, bad_channels_eeg=self.bad_channels_eeg)
            if preprocessor
            else data_dict
        )

        # Add latencies directly to filtered_dict (prefixed with _ for metadata)
        for k, v in payload.items():
            if k.endswith("_latency_ms") and v is not None:
                filtered_dict[f"_{k}"] = v

        self._distribute_sample(filtered_dict, lsl_timestamp, daq_receive_ns, stream_name)

    def _distribute_sample(
        self,
        filtered_dict: dict[str, np.ndarray],
        lsl_timestamp: float,
        daq_receive_ns: int | None,
        stream_name: str | None = None,
    ):
        """Distribute processed sample to all mode queues and plot queue."""
        if not filtered_dict:
            return

        if self._is_empty_sample(filtered_dict):
            return

        # Attach pending marker AFTER empty check (sample will be distributed)
        if self.pending_markers:
            event_id, event_ts, streams_pending = self.pending_markers[0]  # Peek, don't pop yet

            # Check if this stream should receive the marker
            if stream_name in streams_pending:
                marker_array = np.array([[event_id]])
                filtered_dict["markers"] = marker_array
                streams_pending.discard(stream_name)
                self.logger.debug(
                    f"Marker {event_id} attached to stream '{stream_name}', remaining={streams_pending}"
                )

                # Remove marker only when all streams have received it
                if not streams_pending:
                    self.pending_markers.popleft()
                    self.logger.debug(f"Marker {event_id} fully distributed (ts={event_ts})")

        filtered_dict["lsl_timestamp"] = lsl_timestamp
        filtered_dict["_stream_name"] = stream_name  # For stream-aware mode routing
        if daq_receive_ns is not None:
            filtered_dict["_daq_receive_ns"] = daq_receive_ns

        # Accumulate modalities for visualization (after markers attached)
        for key, value in filtered_dict.items():
            if not key.startswith("_"):
                self._viz_state[key] = value

        self._fan_out_sample(filtered_dict)

        # Only send to viz on EEG samples (EEG drives visualization timing)
        if filtered_dict.get("eeg") is not None:
            self._send_to_plot_queue(filtered_dict)

    def _is_empty_sample(self, filtered_dict: dict[str, Any]) -> bool:
        """Check if sample has no data (e.g., during downsampling accumulation)."""
        for key, value in filtered_dict.items():
            if key.startswith("_") or key == "markers":
                continue
            if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] > 0:
                return False
        return True

    def _fan_out_sample(self, sample_dict: dict[str, Any]):
        """Distribute sample to mode queues, filtered by required modalities and stream source."""
        sample_stream = sample_dict.get("_stream_name")

        for mode_name, mode_queue in self.mode_queues.items():
            try:
                # Filter to only required modalities + metadata, respecting stream sources
                required = self.mode_required_modalities.get(mode_name, ["eeg"])
                stream_sources = self.mode_stream_sources.get(mode_name, {})
                filtered_sample = self._filter_sample_for_mode(
                    sample_dict, required, stream_sources, sample_stream
                )

                # Skip if no modalities passed the filter (sample from wrong stream)
                has_modality = any(
                    k
                    for k in filtered_sample
                    if not k.startswith("_") and k not in ("lsl_timestamp", "markers")
                )
                if not has_modality:
                    continue

                mode_queue.put_nowait(filtered_sample)
            except queue.Full:
                self._handle_full_queue(mode_name)
            except Exception as e:
                self.logger.warning(f"Error distributing to mode {mode_name}: {e}")

    def _filter_sample_for_mode(
        self,
        sample_dict: dict[str, Any],
        required_modalities: list[str],
        stream_sources: dict[str, str] | None = None,
        sample_stream: str | None = None,
    ) -> dict[str, Any]:
        """Filter sample to only include required modalities, respecting stream source constraints.

        Args:
            sample_dict: The sample data with modalities and metadata
            required_modalities: List of modality names this mode needs
            stream_sources: Optional mapping of modality -> stream_name for filtering
            sample_stream: The stream name this sample came from

        Returns:
            Filtered sample dict containing only matching modalities and metadata
        """
        filtered = {}
        stream_sources = stream_sources or {}

        for key, value in sample_dict.items():
            # Include metadata (keys starting with _ or special keys)
            if key.startswith("_") or key in ("lsl_timestamp", "markers"):
                filtered[key] = value
            # Include required modalities that match stream source constraints
            elif key.lower() in required_modalities:
                modality = key.lower()
                target_stream = stream_sources.get(modality)

                # Accept if no stream constraint or stream matches
                if target_stream is None or sample_stream == target_stream:
                    filtered[key] = value

        return filtered

    def _handle_full_queue(self, mode_name: str):
        """Handle a full mode queue by dropping the sample (real-time priority)."""
        self.dropped_samples[mode_name] += 1
        if self.dropped_samples[mode_name] % DROPPED_SAMPLE_WARNING_INTERVAL == 1:
            self.logger.warning(
                f"Mode {mode_name} queue full - dropped {self.dropped_samples[mode_name]} samples"
            )

    def _build_viz_payload(self, lsl_timestamp: float) -> dict[str, Any]:
        """Build visualization payload from accumulated _viz_state.

        Uses _viz_state which accumulates all modalities (EEG, EMG, etc.)
        from different streams into a single combined view.
        """
        data = {}
        for key, value in self._viz_state.items():
            if isinstance(value, np.ndarray):
                data[key] = value.flatten().tolist()
            else:
                data[key] = value

        return {
            "type": "raw_data",
            "timestamp": lsl_timestamp,
            "data": data,
            "sample_rate": self._effective_viz_rate,
        }

    def _send_to_plot_queue(self, sample_dict: dict[str, Any]):
        """Send accumulated viz state to plot queue with decimation."""
        self._plot_counter += 1

        # Always send event samples (preserve fidelity), decimate others
        markers = self._viz_state.get("markers")
        has_event = markers is not None and np.any(markers != 0)
        if not has_event and self._plot_counter % PLOT_DECIMATION_FACTOR != 0:
            return  # Skip - decimation

        lsl_timestamp = sample_dict.get("lsl_timestamp")
        payload = self._build_viz_payload(lsl_timestamp)

        # Clear markers after sending to prevent duplicate marker sends
        if has_event:
            self._viz_state.pop("markers", None)

        try:
            self.plot_queue.put_nowait(payload)
        except queue.Full:
            self._viz_drops += 1
            if self._viz_drops % 1000 == 0:
                self.logger.warning(f"Visualization queue: {self._viz_drops} drops total")

    def _create_preprocessors(self):
        """Create one preprocessor per stream using stream_configs metadata."""
        if not self.preprocessing_config:
            self.logger.error("No preprocessing configuration available")
            return

        # Create per-stream preprocessors (stream_configs always provided by GUI)
        for stream in self.stream_configs:
            stream_name = _get_stream_attr(stream, "name")
            stream_type = _get_stream_attr(stream, "type")

            if stream_type.upper() == "EVENTS":
                continue  # Events don't need preprocessing

            modality_preprocessing = self._build_modality_preprocessing_from_stream(stream)
            if modality_preprocessing:
                try:
                    self.preprocessors[stream_name] = OnlinePreprocessor(modality_preprocessing)
                    self.logger.info(
                        f"Created preprocessor for stream '{stream_name}': {list(modality_preprocessing.keys())}"
                    )
                except Exception as e:
                    self.logger.error(f"Failed to create preprocessor for '{stream_name}': {e}")

        # Calculate effective visualization rate from EEG config
        eeg_config = self.preprocessing_config.get("modality_preprocessing", {}).get("eeg", {})
        effective_rate = eeg_config.get("target_sample_rate") or eeg_config.get("sample_rate", 500)
        self._effective_viz_rate = effective_rate // PLOT_DECIMATION_FACTOR
        self.logger.info(f"Visualization rate: {effective_rate}Hz -> {self._effective_viz_rate}Hz")

    def _build_modality_preprocessing_from_stream(self, stream: Any) -> dict[str, dict[str, Any]]:
        """Build modality preprocessing configs from stream config for preprocessor creation.

        Args:
            stream: Stream config (StreamMetadata object or dict)

        Returns:
            Dict mapping modality names to their preprocessing config dicts
        """
        stream_name = _get_stream_attr(stream, "name")
        sample_rate = _get_stream_attr(stream, "sample_rate")
        channel_count = _get_stream_attr(stream, "channel_count")
        channel_types = _get_stream_attr(stream, "channel_types", [])

        base_config = self.preprocessing_config.get("modality_preprocessing", {})

        modality_preprocessing = {}
        if channel_types:
            modality_channel_counts = Counter(ch.lower() for ch in channel_types)
        else:
            modality_channel_counts = {_get_stream_attr(stream, "type").lower(): channel_count}

        for modality, ch_count in modality_channel_counts.items():
            if modality in ("markers", "events"):
                continue

            mod_config = base_config.get(modality, base_config.get(modality.upper(), {})).copy()
            mod_config["num_channels"] = ch_count
            mod_config["sample_rate"] = sample_rate

            target_rate = mod_config.get("target_sample_rate")
            if target_rate and sample_rate and sample_rate > target_rate:
                if sample_rate % target_rate == 0:
                    mod_config["downsample_factor"] = int(sample_rate // target_rate)
                else:
                    mod_config["downsample_factor"] = 1
            else:
                mod_config["downsample_factor"] = mod_config.get("downsample_factor", 1)

            modality_preprocessing[modality] = mod_config
            self.logger.debug(
                f"Stream '{stream_name}' modality '{modality}': {ch_count} channels @ {sample_rate}Hz"
            )

        return modality_preprocessing
