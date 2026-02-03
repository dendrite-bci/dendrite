import logging
from collections import deque
from itertools import islice
from typing import Any

import numpy as np


class Buffer:
    """Sliding window buffer for Dendrite modes."""

    def __init__(self, modalities: list[str], buffer_size: int, logger: logging.Logger):
        self.modalities = modalities
        self.buffer_size = buffer_size
        self.logger = logger

        # Initialize buffers for each modality + markers
        self.buffers = {modality: deque(maxlen=buffer_size) for modality in modalities}
        self.buffers["markers"] = deque(maxlen=buffer_size)

        # Track DAQ timestamps for E2E latency measurement
        self.timestamps = deque(maxlen=buffer_size)

        # Step tracking
        self.samples_since_last_step = 0

        self.logger.info(f"Buffer initialized: {modalities}, size={buffer_size}")

    def add_sample(self, sample: dict) -> bool:
        """Add sample to all buffers."""
        for modality in self.buffers:
            if modality in sample:
                self.buffers[modality].append(sample[modality])
        self.timestamps.append(sample.get("_daq_receive_ns"))
        self.samples_since_last_step += 1
        return True

    def get_newest_timestamp(self):
        """Get timestamp of newest sample (when window became ready)."""
        return self.timestamps[-1] if self.timestamps else None

    def _is_valid_data_buffer(self, modality: str) -> bool:
        """Check if buffer is valid for data extraction (excludes markers, checks size)."""
        if modality == "markers":
            return False
        buffer = self.buffers.get(modality)
        return buffer is not None and len(buffer) >= self.buffer_size

    def _extract_slice(self, modality: str, start: int, end: int) -> np.ndarray | None:
        """Extract and concatenate data slice from buffer."""
        if not self._is_valid_data_buffer(modality):
            return None
        buffer = self.buffers[modality]
        data_arrays = list(islice(buffer, start, end))
        if data_arrays:
            return np.concatenate(data_arrays, axis=1)
        return None

    def is_ready_for_step(self, step_size: int) -> bool:
        """Check if ready for step-based processing."""
        if not self.modalities:
            return False
        primary_buffer = self.buffers.get(self.modalities[0])
        return (
            primary_buffer is not None
            and len(primary_buffer) >= self.buffer_size
            and self.samples_since_last_step >= step_size
        )

    def extract_window(self) -> dict[str, np.ndarray] | None:
        """Extract full data window from buffer."""
        X_data = {}
        for modality in self.buffers:
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
        if not self.modalities:
            return None

        primary_buffer = self.buffers.get(self.modalities[0])
        if not primary_buffer or len(primary_buffer) < self.buffer_size:
            return None

        # Calculate epoch boundaries
        buffer_length = len(primary_buffer)
        event_pos = buffer_length - 1 - event_position_from_end
        epoch_start = event_pos + start_offset_samples
        epoch_end = epoch_start + epoch_length_samples

        if epoch_start < 0 or epoch_end > buffer_length:
            self.logger.warning(
                f"Epoch out of bounds: [{epoch_start}:{epoch_end}] vs buffer[0:{buffer_length}]"
            )
            return None

        X_data = {}
        for modality in self.buffers:
            result = self._extract_slice(modality, epoch_start, epoch_end)
            if result is not None and result.shape[1] == epoch_length_samples:
                X_data[modality] = result

        return X_data if X_data else None

    def get_status(self) -> dict:
        """Get buffer status."""
        primary_size = len(self.buffers.get(self.modalities[0], [])) if self.modalities else 0
        return {
            "buffer_size": self.buffer_size,
            "current_size": primary_size,
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
    event_code = sample.get("markers", -1)
    try:
        return int(event_code)
    except (ValueError, TypeError):
        return -1


def get_shared_model_path(mode_name: str, file_identifier: str | None = None) -> str:
    """Get relative path for shared model file between sync and async modes.

    Returns relative identifier WITHOUT .json extension.
    Pass to decoder.save() with study_name to save under study's decoders dir.
    """
    if file_identifier:
        return f"shared/{mode_name}_{file_identifier}_latest"
    return f"shared/{mode_name}_latest"
