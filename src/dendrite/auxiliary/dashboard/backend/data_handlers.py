#!/usr/bin/env python
"""
Data Handlers

Manages data buffers, processing, and storage for the visualization dashboard.
Handles raw EEG data, modalities, events, and classifier-specific data.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dendrite.constants import DEFAULT_BUFFER_SIZE


@dataclass
class ModalityInfo:
    """Consolidated metadata for a single modality type."""

    name: str
    channels: list[str] = field(default_factory=list)

    @property
    def channel_count(self) -> int:
        return len(self.channels)


DEFAULT_TIME_WINDOW = 10.0  # seconds - fixed visual time window regardless of sample rate


class DataBufferManager:
    """Manages all data buffers for the dashboard"""

    def __init__(self, sample_rate: int, buffer_size: int = DEFAULT_BUFFER_SIZE):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.initialized = False

        self.num_eeg = 0
        self.eeg_channel_labels: list[str] = []
        self.eeg_buffers: list[deque] = []

        self.modality_info: dict[str, ModalityInfo] = {}
        self.mod_buffers: list[deque] = []

        self.stim_buffer = deque([0.0] * buffer_size, maxlen=buffer_size)
        self.stim_times = deque([0.0] * buffer_size, maxlen=buffer_size)
        self.event_history = deque(maxlen=300)  # 30 seconds of events at ~10Hz max

        self.time_axis = np.linspace(0, (buffer_size - 1) / sample_rate, buffer_size)

        self.last_update_timestamp = 0
        self.data_changed = False

        logging.info(
            f"DataBufferManager initialized with buffer size: {buffer_size} samples ({buffer_size / sample_rate:.1f}s)"
        )

    @property
    def num_modalities(self) -> int:
        return len(self.modality_info)

    @property
    def modality_channel_labels(self) -> list[str]:
        return list(self.modality_info.keys())

    @property
    def modality_channel_names(self) -> dict[str, list[str]]:
        return {name: info.channels for name, info in self.modality_info.items()}

    @property
    def mod_channels_flat(self) -> list[str]:
        return [ch for info in self.modality_info.values() for ch in info.channels]

    def initialize_from_raw_data(self, item: dict[str, Any]) -> bool:
        """Initialize buffers based on the first valid raw_data payload."""
        data = item.get("data", {})
        # Channel labels come from processor via first raw_data payload
        channel_labels_dict = item.get("channel_labels", {})

        if not data:
            logging.warning(
                "Initialization failed: Received raw data payload without 'data' dictionary."
            )
            return False

        # Update sample rate and buffer size from stream (accounts for decimation)
        # Keep fixed TIME window (10s) regardless of sample rate
        # Check _viz_sample_rate first (set by processor decimation), then fallback to sample_rate
        actual_rate = item.get("_viz_sample_rate") or item.get("sample_rate") or self.sample_rate
        if actual_rate != self.sample_rate:
            self.sample_rate = actual_rate
            # Recalculate buffer size to maintain fixed time window
            new_buffer_size = int(DEFAULT_TIME_WINDOW * actual_rate)
            self.buffer_size = new_buffer_size
            self.time_axis = np.linspace(0, DEFAULT_TIME_WINDOW, new_buffer_size)
            # Resize stim buffers (created in __init__)
            self.stim_buffer = deque([0.0] * new_buffer_size, maxlen=new_buffer_size)
            self.stim_times = deque([0.0] * new_buffer_size, maxlen=new_buffer_size)
            logging.info(
                f"Adjusted to {actual_rate}Hz: buffer={new_buffer_size} samples ({DEFAULT_TIME_WINDOW}s window)"
            )

        if "eeg" in data and isinstance(data["eeg"], list):
            self.num_eeg = len(data["eeg"])
            self.eeg_channel_labels = channel_labels_dict.get(
                "eeg", [f"EEG_{i + 1}" for i in range(self.num_eeg)]
            )
            if len(self.eeg_channel_labels) != self.num_eeg:
                logging.warning(
                    f"Mismatch EEG data ({self.num_eeg}) and labels ({len(self.eeg_channel_labels)}). Using default labels."
                )
                self.eeg_channel_labels = [f"EEG_{i + 1}" for i in range(self.num_eeg)]
        else:
            self.num_eeg = 0
            self.eeg_channel_labels = []

        self.eeg_buffers = [
            deque([0.0] * self.buffer_size, maxlen=self.buffer_size) for _ in range(self.num_eeg)
        ]

        # Exclude known non-modality keys and timestamp fields (lowercase for internal data)
        excluded_keys = [
            "eeg",
            "events",
            "markers",
            "_watermark",
            "lsl_timestamp",
            "_daq_receive_ns",
        ]
        modality_names = sorted([key for key in data if key not in excluded_keys])
        self.modality_info.clear()
        self.mod_buffers = []

        for modality in modality_names:
            value = data[modality]
            if isinstance(value, (list, np.ndarray)):
                count = len(value)
                labels = channel_labels_dict.get(
                    modality, [f"{modality}_{i + 1}" for i in range(count)]
                )
                if len(labels) != count:
                    logging.warning(
                        f"Mismatch {modality} data ({count}) and labels ({len(labels)}). Using default labels."
                    )
                    labels = [f"{modality}_{i + 1}" for i in range(count)]
            elif isinstance(value, (int, float)):
                labels = channel_labels_dict.get(modality, [modality])
                if not labels:
                    labels = [modality]
            else:
                logging.warning(
                    f"Ignoring modality '{modality}' due to unexpected data type: {type(value)}"
                )
                continue

            self.modality_info[modality] = ModalityInfo(name=modality, channels=labels)
            for _ in range(len(labels)):
                self.mod_buffers.append(deque([0.0] * self.buffer_size, maxlen=self.buffer_size))

        self.initialized = True
        logging.info(
            f"Initialized: {self.num_eeg} EEG channels, {self.num_modalities} modality types ({len(self.mod_channels_flat)} channels)"
        )
        return True

    @staticmethod
    def _normalize_to_list(data: Any) -> list | np.ndarray:
        """Flatten 2D arrays and wrap scalars for uniform buffer appending."""
        if isinstance(data, np.ndarray) and data.ndim == 2:
            return data.flatten()
        if isinstance(data, (list, np.ndarray)):
            return data
        return [data]

    def append_raw_data(self, item: dict[str, Any]):
        """Append data from a raw_data payload to the buffers."""
        if not self.initialized:
            return

        data = item.get("data", {})
        timestamp = item.get("timestamp", time.time())

        if "eeg" in data:
            eeg_values = self._normalize_to_list(data["eeg"])

            if len(eeg_values) == self.num_eeg:
                for i, value in enumerate(eeg_values):
                    try:
                        self.eeg_buffers[i].append(float(value))
                    except (ValueError, TypeError):
                        self.eeg_buffers[i].append(0.0)

        buffer_idx = 0
        for modality, info in self.modality_info.items():
            num_channels = info.channel_count
            if modality not in data:
                buffer_idx += num_channels
                continue

            mod_values = self._normalize_to_list(data[modality])

            for i in range(num_channels):
                try:
                    val = float(mod_values[i]) if i < len(mod_values) else 0.0
                except (ValueError, TypeError, IndexError):
                    val = 0.0
                if buffer_idx < len(self.mod_buffers):
                    self.mod_buffers[buffer_idx].append(val)
                buffer_idx += 1

            if num_channels > 1 and len(mod_values) < num_channels:
                logging.warning(
                    f"Data mismatch for modality '{modality}': expected {num_channels}, got {len(mod_values)}"
                )

        event_value = 0.0
        if "markers" in data:
            value = data["markers"]
            try:
                # Handle 2D format (1, 1) from processor
                if isinstance(value, np.ndarray) and value.ndim == 2:
                    event_value = float(value.flatten()[0])
                elif isinstance(value, (list, np.ndarray)):
                    event_value = float(value[0])
                else:
                    event_value = float(value)
            except (ValueError, TypeError, IndexError):
                event_value = 0.0

        if event_value != 0.0 or "markers" in data:
            self.stim_buffer.append(event_value)
            self.stim_times.append(timestamp)
            if event_value > 0:
                time_position = len(self.time_axis) - 1
                if time_position >= 0:
                    self.event_history.append(
                        (time.time(), event_value, self.time_axis[time_position])
                    )

        self.data_changed = True
        self.last_update_timestamp = timestamp

    def clear_all_buffers(self):
        """Clear all data buffers and reset state."""
        self.initialized = False
        self.num_eeg = 0
        self.eeg_channel_labels = []

        for buf in self.eeg_buffers:
            buf.clear()
        self.eeg_buffers = []

        self.modality_info.clear()
        for buf in self.mod_buffers:
            buf.clear()
        self.mod_buffers = []

        self.stim_buffer.clear()
        self.stim_times.clear()
        self.event_history.clear()

        logging.info("DataBufferManager: All buffers cleared and state reset.")
