import logging

import numpy as np
from pylsl import StreamInfo, StreamOutlet

from dendrite.data.stream_schemas import StreamConfig


def infer_channel_types_from_labels(labels: list[str], default_type: str = "EEG") -> list[str]:
    """
    Infer channel types based on channel labels using pattern matching.

    This ensures consistent channel type inference across all parts of the system.
    Channels with specific patterns in their labels are automatically assigned
    appropriate types (AUX, EOG, EMG, etc.).

    Args:
        labels: List of channel labels
        default_type: Default type to use if no pattern matches

    Returns:
        List of inferred channel types matching the input labels
    """
    inferred_types = []

    for label in labels:
        label_upper = label.upper().strip()

        # AUX patterns - check first before other patterns
        if any(pattern in label_upper for pattern in ["AUX", "AUXILIARY"]):
            inferred_types.append("AUX")
        # EOG patterns
        elif any(pattern in label_upper for pattern in ["VEOG", "VEOGL", "VEOGR"]):
            inferred_types.append("VEOG")
        elif any(pattern in label_upper for pattern in ["HEOG", "HEOGL", "HEOGR"]):
            inferred_types.append("HEOG")
        elif "EOG" in label_upper:
            inferred_types.append("EOG")
        # Other patterns
        elif any(pattern in label_upper for pattern in ["EMG", "MUSCLE"]):
            inferred_types.append("EMG")
        elif any(pattern in label_upper for pattern in ["ECG", "EKG"]):
            inferred_types.append("ECG")
        elif any(
            pattern in label_upper for pattern in ["STIM", "STI", "TRIGGER", "EVENT", "MARKER"]
        ):
            inferred_types.append("Markers")
        # Default to EEG for electrode-like names or specified default
        elif any(
            pattern in label_upper
            for pattern in ["FP", "F", "C", "P", "O", "T", "AF", "FC", "CP", "PO", "TP", "Z"]
        ):
            inferred_types.append("EEG")
        else:
            inferred_types.append(default_type)

    return inferred_types


class LSLOutlet:
    """LSL stream outlet with channel metadata and lazy initialization."""

    def __init__(self, config: StreamConfig, logger_name: str | None = None) -> None:
        """
        Initialize LSL outlet from stream configuration.

        Args:
            config: Stream configuration (validated by Pydantic)
            logger_name: Optional logger name (defaults to stream name)
        """
        self.config = config
        self.name = config.name
        self.stream_type = config.type
        self.channel_count = config.channel_count
        self.sample_rate = config.sample_rate
        self.channel_format = config.channel_format
        self.outlet = None

        self.logger = logging.getLogger(logger_name or f"LSLOutlet_{self.name}")

        # Apply defaults for missing channel metadata
        labels = config.labels or [f"Ch_{i + 1:02d}" for i in range(config.channel_count)]
        channel_types = config.channel_types or [config.type] * config.channel_count
        channel_units = config.channel_units or ["unknown"] * config.channel_count

        self.info = StreamInfo(
            name=config.name,
            type=config.type,
            channel_count=config.channel_count,
            nominal_srate=config.sample_rate,
            channel_format=config.channel_format,
            source_id=config.source_id or f"nbr_bmi_{config.name.lower()}",
        )

        # Add channel metadata to StreamInfo XML
        channels_section = self.info.desc().append_child("channels")
        for label, ch_type, unit in zip(labels, channel_types, channel_units, strict=False):
            ch = channels_section.append_child("channel")
            ch.append_child_value("label", label)
            ch.append_child_value("type", ch_type)
            ch.append_child_value("unit", unit)

        # Add acquisition info if available
        if config.acquisition_info:
            acq = self.info.desc().append_child("acquisition")
            for key, value in config.acquisition_info.items():
                acq.append_child_value(key, value)

        self.logger.info(
            f"LSL outlet ready: {self.name} ({self.stream_type}), {self.channel_count}ch @ {self.sample_rate}Hz"
        )

    def create_outlet(self) -> None:
        """Create the LSL StreamOutlet."""
        self.outlet = StreamOutlet(self.info)
        self.logger.info(f"Created LSL outlet: {self.name}")

    def push_sample(self, sample: list | np.ndarray, timestamp: float | None = None) -> None:
        """Push a single sample to the stream."""
        if self.outlet is None:
            self.create_outlet()

        if len(sample) != self.channel_count:
            raise ValueError(
                f"Sample length ({len(sample)}) must match channel count ({self.channel_count})"
            )

        if timestamp is None:
            self.outlet.push_sample(sample)
        else:
            self.outlet.push_sample(sample, timestamp)

    def push_chunk(
        self, chunk: list[list] | np.ndarray, timestamps: list[float] | None = None
    ) -> None:
        """Push multiple samples as a chunk [samples x channels]."""
        if self.outlet is None:
            self.create_outlet()

        chunk_array = np.array(chunk)
        if chunk_array.ndim != 2:
            raise ValueError(f"Chunk must be 2D array, got {chunk_array.ndim}D")
        if chunk_array.shape[1] != self.channel_count:
            raise ValueError(
                f"Chunk channels ({chunk_array.shape[1]}) must match channel count ({self.channel_count})"
            )

        if timestamps is None:
            self.outlet.push_chunk(chunk)
        else:
            if len(timestamps) != len(chunk):
                raise ValueError(
                    f"Timestamps ({len(timestamps)}) must match samples ({len(chunk)})"
                )
            self.outlet.push_chunk(chunk, timestamps)

    def close(self) -> None:
        """Close outlet and release resources."""
        if self.outlet is not None:
            self.outlet = None
            self.logger.info(f"Closed LSL outlet: {self.name}")

    def have_consumers(self) -> bool:
        """Check if consumers are connected."""
        return self.outlet is not None and self.outlet.have_consumers()
