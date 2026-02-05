"""Shared utilities for offline streaming GUI."""

from pathlib import Path
from typing import Any

import mne

from dendrite.data.imports import H5Loader
from dendrite.utils.logger_central import get_logger

logger = get_logger("StreamManagerGUI")

# Stream type presets: (channels, sample_rate, default_name, help_text)
STREAM_TYPE_PRESETS = {
    "EEG": (
        64,
        500,
        "EEG_Stream",
        "EEG: Electroencephalography for brain activity. Typically 64 channels at 500-1000 Hz.",
    ),
    "EMG": (
        8,
        1000,
        "EMG_Stream",
        "EMG: Electromyography for muscle activity. Usually 2-16 channels at 1000-2000 Hz.",
    ),
    "ECG/EKG": (
        1,
        500,
        "ECG_Stream",
        "ECG/EKG: Electrocardiography for heart activity. Usually 1-12 leads at 250-1000 Hz.",
    ),
    "Events": (
        1,
        0,
        "Events_Stream",
        "Events: Discrete event markers (irregular timing). Sample rate of 0 means irregular.",
    ),
    "ContinuousEvents": (
        4,
        100,
        "Position_Stream",
        "Continuous Events: Position/torque data (x,y,z,torque). Usually 50-500 Hz.",
    ),
    "GSR/EDA": (
        1,
        50,
        "GSR_Stream",
        "GSR/EDA: Galvanic Skin Response for stress/arousal. Usually 1-4 channels at 10-100 Hz.",
    ),
    "Position": (
        3,
        100,
        "Position_Stream",
        "Position: Spatial coordinates (x,y,z). Usually 3-6 channels at 50-500 Hz.",
    ),
    "Force/Torque": (
        6,
        100,
        "Force_Stream",
        "Force/Torque: 3D force and torque vectors. Usually 6 channels at 100-1000 Hz.",
    ),
    "Acceleration": (
        3,
        200,
        "Accel_Stream",
        "Acceleration: 3D accelerometer data. Usually 3 channels at 100-1000 Hz.",
    ),
    "Gyroscope": (
        3,
        200,
        "Gyro_Stream",
        "Gyroscope: Angular velocity (roll,pitch,yaw). Usually 3 channels at 100-1000 Hz.",
    ),
    "Temperature": (
        1,
        10,
        "Temp_Stream",
        "Temperature: Thermal sensors. Usually 1-4 channels at 1-50 Hz.",
    ),
    "Breathing": (
        1,
        25,
        "Breathing_Stream",
        "Breathing: Respiratory rate/volume. Usually 1-2 channels at 10-100 Hz.",
    ),
    "Custom": (
        1,
        100,
        "Custom_Stream",
        "Custom: Enter your own stream type name. Adjust channels and sample rate as needed.",
    ),
}

# Type normalization mapping for display names to canonical names
TYPE_NORMALIZATION = {
    "ECG/EKG": "ECG",
    "GSR/EDA": "GSR",
    "Force/Torque": "Force",
}


# Module-level cache for file info: {file_path: (duration, events, event_ids)}
_file_info_cache: dict[str, tuple[float, Any, dict]] = {}


def get_source_display(config: dict) -> str:
    """Get human-readable source description from stream config."""
    source_type = config.get("source_type", "generated")
    if source_type == "moabb":
        return f"MOABB ({config.get('preset_name', '?')})"
    elif source_type == "file":
        return "File"
    return "Synthetic"


def get_file_info(file_path: str) -> tuple[float, Any, dict] | None:
    """Load file info once and cache results for reuse.

    Returns:
        Tuple of (duration, events, event_ids) or None on error
    """
    if file_path in _file_info_cache:
        return _file_info_cache[file_path]

    ext = Path(file_path).suffix.lower()
    try:
        if ext == ".fif":
            raw = mne.io.read_raw_fif(file_path, preload=False)
            duration = raw.times[-1] if len(raw.times) > 0 else 0.0
            events, event_ids = mne.events_from_annotations(raw)

        elif ext in H5Loader.EXTENSIONS:
            # H5 format - use lightweight metadata function
            duration, _, _ = H5Loader.get_file_info(file_path)
            events = []
            event_ids = {}
        else:
            return None

        # Cache and return
        file_info = (duration, events, event_ids)
        _file_info_cache[file_path] = file_info
        logger.info(
            f"Cached file info for {Path(file_path).name}: {duration:.1f}s, {len(events)} events"
        )

        return file_info

    except Exception as e:
        logger.warning(f"Could not load file info for {Path(file_path).name}: {e!s}")
        return None
