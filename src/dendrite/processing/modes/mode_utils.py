import logging
from collections import deque
from typing import Any

import numpy as np


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

        self.timestamps.append(sample.get("_daq_receive_ns"))
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


def get_shared_model_path(mode_name: str, file_identifier: str | None = None) -> str:
    """Get relative path for shared model file between sync and async modes.

    Returns relative identifier WITHOUT .json extension.
    Pass to decoder.save() with study_name to save under study's decoders dir.
    """
    if file_identifier:
        return f"shared/{mode_name}_{file_identifier}_latest"
    return f"shared/{mode_name}_latest"
