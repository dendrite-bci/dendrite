import logging
import queue
import time
from collections import deque
from typing import Any

import numpy as np

from dendrite.data.shared_buffers import OverrunError, SharedRingBuffer
from dendrite.processing._types import Sample
from dendrite.utils.state_keys import calibration_corr_key


class Buffer:
    """Sliding window buffer for Dendrite modes.

    Uses pre-allocated numpy ring buffers for O(1) window extraction
    """

    def __init__(self, modalities: list[str], buffer_size: int, logger: logging.Logger):
        self.modalities = modalities
        self.buffer_size = buffer_size
        self.logger = logger

        # Ring buffers for data modalities (lazy-init on first sample per modality)
        self._rings: dict[str, np.ndarray] = {}
        self._write_pos = 0
        self._sample_count = 0

        # Markers stay in deque (sparse, variable content)
        self.buffers: dict[str, deque] = {"markers": deque(maxlen=buffer_size)}

        # Track DAQ timestamps for E2E latency measurement
        self.timestamps: deque = deque(maxlen=buffer_size)

        # Step tracking
        self.samples_since_last_step = 0

        self.logger.info(f"Buffer initialized: {modalities}, size={buffer_size}")

    def add_sample(self, sample: dict) -> bool:
        """Add sample to all buffers."""
        for modality in self.modalities:
            if modality not in sample:
                continue
            data = sample[modality]
            if modality not in self._rings:
                n_channels = data.shape[0]
                self._rings[modality] = np.zeros((n_channels, self.buffer_size), dtype=np.float32)
            self._rings[modality][:, self._write_pos] = data[:, 0]

        if "markers" in sample:
            self.buffers["markers"].append(sample["markers"])

        self.timestamps.append(sample.get("_receive_ns"))
        self._write_pos = (self._write_pos + 1) % self.buffer_size
        self._sample_count += 1
        self.samples_since_last_step += 1
        return True

    def get_newest_timestamp(self):
        """Get timestamp of newest sample (when window became ready)."""
        return self.timestamps[-1] if self.timestamps else None

    def _is_full(self) -> bool:
        """Check if buffer has received enough samples to fill."""
        return self._sample_count >= self.buffer_size

    def _extract_slice(self, modality: str, start: int, end: int) -> np.ndarray | None:
        """Extract contiguous data slice from ring buffer.

        Args:
            modality: Data modality key.
            start: Logical start index (0 = oldest sample).
            end: Logical end index (exclusive).
        """
        if modality == "markers" or modality not in self._rings or not self._is_full():
            return None

        length = end - start
        if length <= 0:
            return None

        ring = self._rings[modality]
        # Map logical index 0 (oldest) to ring position
        oldest_pos = self._write_pos  # next write overwrites oldest
        ring_start = (oldest_pos + start) % self.buffer_size
        ring_end = (oldest_pos + end) % self.buffer_size

        if ring_start < ring_end:
            return ring[:, ring_start:ring_end].copy()
        # Wrapped: need two slices
        return np.concatenate([ring[:, ring_start:], ring[:, :ring_end]], axis=1)

    def is_ready_for_step(self, step_size: int) -> bool:
        """Check if ready for step-based processing."""
        if not self.modalities:
            return False
        return (
            self.modalities[0] in self._rings
            and self._is_full()
            and self.samples_since_last_step >= step_size
        )

    def extract_window(self) -> dict[str, np.ndarray] | None:
        """Extract full data window from buffer."""
        if not self._is_full():
            return None

        X_data = {}
        for modality in self.modalities:
            result = self._extract_slice(modality, 0, self.buffer_size)
            if result is not None:
                X_data[modality] = result

        if X_data:
            self.samples_since_last_step = 0
        return X_data if X_data else None

    def extract_epoch_at_event(
        self, start_offset_samples: int, epoch_length_samples: int, event_position_from_end: int = 0
    ) -> dict[str, np.ndarray] | None:
        """Extract epoch data relative to an event position."""
        if not self.modalities or not self._is_full():
            return None

        buffer_length = self.buffer_size
        event_pos = buffer_length - 1 - event_position_from_end
        epoch_start = event_pos + start_offset_samples
        epoch_end = epoch_start + epoch_length_samples

        if epoch_start < 0 or epoch_end > buffer_length:
            self.logger.warning(
                f"Epoch out of bounds: [{epoch_start}:{epoch_end}] vs buffer[0:{buffer_length}]"
            )
            return None

        X_data = {}
        for modality in self.modalities:
            result = self._extract_slice(modality, epoch_start, epoch_end)
            if result is not None and result.shape[1] == epoch_length_samples:
                X_data[modality] = result

        return X_data if X_data else None

    def get_status(self) -> dict:
        """Get buffer status."""
        current_size = min(self._sample_count, self.buffer_size)
        return {
            "buffer_size": self.buffer_size,
            "current_size": current_size,
            "samples_since_last_step": self.samples_since_last_step,
        }


def extract_event_mapping(instance_config: dict[str, Any]) -> dict[int, str]:
    """Extract event mapping {event_id: event_label} from instance config.

    Converts string keys to int (JSON deserializes dict keys as strings).
    """
    raw_mapping = instance_config.get("event_mapping", {})
    return {int(k): v for k, v in raw_mapping.items()}


def extract_event_code(sample: dict) -> int:
    """Extract event code from sample dict, or -1 if no valid marker."""
    event_code = sample.get("markers")
    if event_code is None:
        return -1
    try:
        if isinstance(event_code, np.ndarray):
            return int(event_code.flat[0])
        return int(event_code)
    except (ValueError, TypeError, IndexError):
        return -1


def generate_label_mapping(
    event_mapping: dict[int, str],
) -> tuple[dict[str, int], dict[int, str], dict[int, int]]:
    """Generate label, reverse-label, and index-to-event-code mappings.

    Args:
        event_mapping: {event_code: event_name}, e.g. {7: 'left_hand', 8: 'right_hand'}

    Returns:
        (label_mapping, reverse_label_mapping, index_to_event_code)
    """
    if not event_mapping:
        return {}, {}, {}
    unique_names = sorted(set(event_mapping.values()))
    label_mapping = {name: i for i, name in enumerate(unique_names)}
    reverse_label_mapping = {i: name for name, i in label_mapping.items()}
    # Map each class index to its first event code
    index_to_event_code: dict[int, int] = {}
    for code, name in sorted(event_mapping.items()):
        idx = label_mapping[name]
        if idx not in index_to_event_code:
            index_to_event_code[idx] = code
    return label_mapping, reverse_label_mapping, index_to_event_code


class FanOutQueue:
    """Distributes items to multiple consumer queues without blocking.

    Used for routing mode outputs to multiple destinations (e.g., metrics saver,
    visualization). Drops items silently if any queue is full (real-time priority).
    """

    def __init__(self, queues: list[Any]) -> None:
        self.queues = queues

    def put(self, item: Any) -> None:
        for q in self.queues:
            try:
                q.put_nowait(item)
            except queue.Full:
                pass


# ---------------------------------------------------------------------------
# Ring buffer reader
# ---------------------------------------------------------------------------


class SampleReader:
    """Reads from a SharedRingBuffer and demuxes into per-modality Sample dicts."""

    _MAX_STASH = 1000

    def __init__(self, rb_config: dict[str, Any], logger: logging.Logger) -> None:
        self._config = rb_config
        self._logger = logger
        self._ring_buffer: SharedRingBuffer | None = None
        self._read_pos: int = 0
        self._stash: deque = deque(maxlen=self._MAX_STASH)

    def connect(self) -> None:
        """Connect to shared ring buffer. Must be called in child process."""
        try:
            buf_name = self._config["buffer_name"]
            self._ring_buffer = SharedRingBuffer.connect(buf_name)
            self._read_pos = self._ring_buffer.write_pos
        except Exception as e:
            self._logger.error(f"Failed to connect ring buffer: {e}")
            self._ring_buffer = None

    def read_sample(self) -> Sample | None:
        """Return next sample from ring buffer, or None on timeout."""
        if self._ring_buffer is None:
            return None

        if self._stash:
            return self._stash.popleft()

        try:
            data, timestamps, _local_ts, receive_ns, new_pos = (
                self._ring_buffer.read_new(self._read_pos)
            )
        except OverrunError as e:
            self._logger.warning(f"Ring buffer overrun, skipping ahead: {e}")
            self._read_pos = self._ring_buffer.write_pos
            return None
        except (FileNotFoundError, ValueError, OSError) as e:
            self._logger.error(f"Ring buffer lost: {e}")
            raise RuntimeError(f"Ring buffer lost: {e}") from e

        if len(data) == 0:
            time.sleep(0.0005)
            return None

        self._read_pos = new_pos
        marker_col = self._config["marker_col"]
        modalities = self._config["modalities"]
        stream_name = self._config.get("buffer_name", "")

        for i in range(len(data)):
            sample: dict[str, Any] = {}
            for mod, indices in modalities.items():
                sample[mod] = data[i, indices].reshape(-1, 1)
            sample["markers"] = np.array([[data[i, marker_col]]])
            sample["lsl_timestamp"] = timestamps[i]
            sample["_receive_ns"] = int(receive_ns[i])
            sample["_stream_name"] = stream_name
            self._stash.append(sample)

        return self._stash.popleft()

    def close(self) -> None:
        """Close the shared ring buffer connection."""
        if self._ring_buffer:
            self._ring_buffer.close()
            self._ring_buffer = None


# ---------------------------------------------------------------------------
# Sample preprocessor
# ---------------------------------------------------------------------------


class SamplePreprocessor:
    """Per-mode online preprocessing: bandpass, CAR, downsample, channel selection."""

    _QUALITY_REFRESH_S = 2.0

    def __init__(
        self,
        preproc_config: dict[str, dict],
        sample_rate: float,
        channel_selection: dict[str, list[int]],
        modality_labels: dict[str, list[str]],
        shared_state: Any | None,
        logger: logging.Logger,
    ) -> None:
        self._config = preproc_config
        self._sample_rate = sample_rate
        self._channel_selection = channel_selection
        self._modality_labels = modality_labels
        self._shared_state = shared_state
        self._logger = logger

        self._preprocessor = None  # OnlinePreprocessor, created lazily
        self._stashed_marker: np.ndarray | None = None
        self._bad_channels: dict[str, list[int]] = {}
        self._last_quality_refresh = 0.0
        self._last_interp_version = 0

        self.effective_sample_rate = self._compute_effective_rate()
        self.refresh_bad_channels()

        logger.info(
            f"Preprocessor config ready: effective_sample_rate="
            f"{self.effective_sample_rate}Hz (will init on first sample)"
        )

    def _compute_effective_rate(self) -> float:
        """Compute effective sample rate from preprocessing config."""
        if not self._config:
            return self._sample_rate
        primary = next(iter(self._config), None)
        if not primary:
            return self._sample_rate
        pc = self._config[primary]
        target = pc.get("target_sample_rate")
        if target and self._sample_rate > target and self._sample_rate % target == 0:
            return float(target)
        ds = pc.get("downsample_factor") or 1
        return self._sample_rate / ds

    def refresh_bad_channels(self) -> None:
        now = time.monotonic()
        if now - self._last_quality_refresh < self._QUALITY_REFRESH_S:
            return
        self._last_quality_refresh = now
        if not self._shared_state:
            return
        quality = self._shared_state.get("channel_quality")
        if not quality:
            return
        new_bad = quality.get("bad_channels", {})
        if new_bad != self._bad_channels:
            self._logger.info(f"Bad channels updated: {self._bad_channels} -> {new_bad}")
            self._bad_channels = new_bad

        interp_version = quality.get("interp_version", 0)
        if interp_version > self._last_interp_version:
            self._last_interp_version = interp_version
            if self._preprocessor:
                effective_bad = quality.get("effective_bad", {})
                corr_matrix = self._shared_state.get(calibration_corr_key())
                frozen_auto_bad = quality.get("bad_channels", {})
                for mod, bad_indices in effective_bad.items():
                    proc = self._preprocessor.processors.get(mod)
                    if proc and corr_matrix is not None:
                        bad_at_warmup = frozen_auto_bad.get(mod, [])
                        proc.freeze_interpolation(
                            bad_indices,
                            corr_matrix=corr_matrix,
                            bad_during_warmup=bad_at_warmup,
                        )

    def _ensure_preprocessor(self, data_dict: dict[str, np.ndarray]) -> None:
        """Lazily create OnlinePreprocessor using actual channel counts."""
        if self._preprocessor is not None or not self._config:
            return

        from dendrite.processing.preprocessing.preprocessor import OnlinePreprocessor

        modality_preprocessing = {}
        for modality, config in self._config.items():
            mod_config = {**config}
            if modality in data_dict and isinstance(data_dict[modality], np.ndarray):
                mod_config["num_channels"] = data_dict[modality].shape[0]
            mod_config["sample_rate"] = self._sample_rate

            if modality in self._modality_labels:
                mod_config["channel_labels"] = self._modality_labels[modality]

            target = mod_config.get("target_sample_rate")
            if target and self._sample_rate > target:
                if self._sample_rate % target == 0:
                    mod_config["downsample_factor"] = int(self._sample_rate // target)
                else:
                    mod_config.pop("downsample_factor", None)

            modality_preprocessing[modality] = mod_config

        self._preprocessor = OnlinePreprocessor(modality_preprocessing)
        ch_info = ", ".join(
            f"{m}={c.get('num_channels', '?')}ch" for m, c in modality_preprocessing.items()
        )
        self._logger.info(f"Preprocessor created: {ch_info}")

    def process(self, sample: Sample) -> Sample | None:
        """Preprocess on ALL channels, then apply channel selection.

        Returns None if downsampling is accumulating.
        Raises ValueError if channel selection is out of bounds.
        """
        if not self._config:
            return sample

        self.refresh_bad_channels()

        # Separate data from metadata
        data_keys = {
            k for k in sample
            if not k.startswith("_") and k not in ("lsl_timestamp", "markers")
        }
        data_dict = {k: sample[k] for k in data_keys}
        metadata = {k: v for k, v in sample.items() if k not in data_keys}

        self._ensure_preprocessor(data_dict)
        if not self._preprocessor:
            return sample

        # Stash markers before preprocessing (may be lost during downsample accumulation)
        markers = metadata.pop("markers", None)
        if markers is not None and np.any(markers > 0):
            self._stashed_marker = markers

        # Preprocess ALL channels — bad channels are interpolated
        # inside ModalityProcessor before CAR, giving a clean all-channel reference
        processed = self._preprocessor.process(data_dict)

        has_data = any(
            isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] > 0
            for v in processed.values()
        )
        if not has_data:
            return None  # Accumulating for downsample

        # Apply channel selection AFTER preprocessing
        if self._channel_selection:
            for mod, indices in self._channel_selection.items():
                if mod in processed and isinstance(processed[mod], np.ndarray):
                    n_ch = processed[mod].shape[0]
                    if indices and max(indices) >= n_ch:
                        raise ValueError(
                            f"Channel selection out of bounds for '{mod}': "
                            f"index {max(indices)} >= {n_ch} channels. "
                            f"Stream may have changed"
                        )
                    processed[mod] = processed[mod][indices, :]

        # Reattach stashed markers
        result = {**metadata, **processed}
        if self._stashed_marker is not None:
            result["markers"] = self._stashed_marker
            self._stashed_marker = None

        return result

    def reset_config(self, preproc_config: dict[str, dict]) -> None:
        """Replace preprocessing config and reset internal state.

        Used when a decoder brings new preprocessing parameters.
        """
        self._config = preproc_config
        self._preprocessor = None
        self._stashed_marker = None
        self.effective_sample_rate = self._compute_effective_rate()
        self._logger.info(
            f"Preprocessor config reset: effective_sample_rate="
            f"{self.effective_sample_rate}Hz"
        )


