"""
MNE conversion and FIF export.

Convert H5 recordings to MNE Raw objects and export to FIF format.
"""

import logging
from pathlib import Path

import mne
import numpy as np

from dendrite.constants import DEFAULT_MONTAGE, TIMESTAMP_COLS, UV_TO_V

from .h5_io import load_dataset

logger = logging.getLogger(__name__)


def to_mne_raw(
    h5_path: str | Path,
    sfreq: float,
    dataset: str = "EEG",
    montage: str = DEFAULT_MONTAGE,
) -> mne.io.RawArray:
    """
    Convert H5 dataset to MNE Raw object.

    Args:
        h5_path: Path to H5 file
        dataset: Dataset name to convert
        sfreq: Sampling frequency in Hz
        montage: EEG montage name (e.g., 'standard_1005')

    Returns:
        MNE RawArray with data in Volts and channel types set.

    Note:
        Data is assumed to be stored in microvolts (µV) and is automatically
        converted to Volts (V) for MNE compatibility.
    """
    df = load_dataset(h5_path, dataset)

    # Filter out timestamp columns (case-insensitive for backward compat)
    timestamp_cols_lower = {c.lower() for c in TIMESTAMP_COLS}
    data_cols = [col for col in df.columns if col.lower() not in timestamp_cols_lower]
    data = df[data_cols].values.T * UV_TO_V  # (n_channels, n_samples) in Volts

    # Create MNE info and Raw
    info = mne.create_info(ch_names=data_cols, sfreq=sfreq)
    raw = mne.io.RawArray(data, info)

    # Set channel types
    ch_types = {ch: guess_channel_type(ch) for ch in data_cols}
    raw.set_channel_types(ch_types)

    # Apply montage to EEG channels
    if montage:
        try:
            montage_obj = mne.channels.make_standard_montage(montage)
            raw.set_montage(montage_obj, on_missing="ignore")
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Could not apply montage: {e}")

    logger.info(f"Created MNE Raw: {raw.info['nchan']} channels, {len(raw.times)} samples")
    return raw


def export_to_fif(
    h5_path: str | Path,
    sfreq: float,
    output_path: str | Path | None = None,
    dataset: str = "EEG",
    include_events: bool = True,
    event_dataset: str = "Event",
    overwrite: bool = False,
) -> str:
    """
    Export H5 recording to FIF format.

    Args:
        h5_path: Input H5 file path
        output_path: Output FIF path (auto-generated if None)
        dataset: EEG dataset name
        sfreq: Sampling frequency
        include_events: Whether to attach events as annotations
        event_dataset: Event dataset name
        overwrite: Whether to overwrite existing file

    Returns:
        Path to exported FIF file.
    """
    # Auto-generate output path
    if output_path is None:
        h5_file = Path(h5_path)
        output_path = str(h5_file.with_suffix(".fif"))

    # Create Raw object
    raw = to_mne_raw(h5_path, sfreq, dataset)

    # Attach events
    if include_events:
        try:
            _attach_events(raw, h5_path, dataset, event_dataset, sfreq)
        except (KeyError, ValueError, OSError) as e:
            logger.warning(f"Could not attach events: {e}")

    raw.save(output_path, overwrite=overwrite)
    logger.info(f"Exported to FIF: {output_path}")
    return output_path


def guess_channel_type(name: str) -> str:
    """
    Guess MNE channel type from channel name.

    Uses pattern matching to classify channels into standard MNE types.

    Returns:
        One of: 'eeg', 'emg', 'eog', 'ecg', 'stim', 'misc'
    """
    name_lower = name.lower()

    if name_lower.startswith("stim") or "marker" in name_lower:
        return "stim"
    elif "emg" in name_lower:
        return "emg"
    elif "ecg" in name_lower or "ekg" in name_lower:
        return "ecg"
    elif "eog" in name_lower:
        return "eog"
    elif any(x in name_lower for x in ["gsr", "resp", "breath", "pulse"]) or name_lower.startswith(
        ("exo_", "robot_", "aux_")
    ):
        return "misc"
    else:
        return "eeg"


def _attach_events(
    raw: mne.io.RawArray, h5_path: str, eeg_dataset: str, event_dataset: str, sfreq: float
) -> None:
    """Attach events from H5 file to MNE Raw as annotations."""
    df_eeg = load_dataset(h5_path, eeg_dataset)
    df_events = load_dataset(h5_path, event_dataset)

    # Normalize EEG columns to lowercase for legacy compat (events already normalized)
    df_eeg.columns = df_eeg.columns.str.lower()

    # Find timestamp columns
    eeg_ts_col = "timestamp" if "timestamp" in df_eeg.columns else None
    event_ts_col = "timestamp" if "timestamp" in df_events.columns else None

    if eeg_ts_col is None or event_ts_col is None:
        logger.warning("Could not find timestamp columns for event alignment")
        return

    # Calculate onsets relative to EEG start
    eeg_start = df_eeg[eeg_ts_col].iloc[0]
    onsets = df_events[event_ts_col].values.astype(float) - eeg_start

    # Get event descriptions
    label_col = "event_type" if "event_type" in df_events.columns else df_events.columns[0]
    descriptions = df_events[label_col].astype(str).values

    raw.set_meas_date(None)  # Ensure onsets from 0.0
    annotations = mne.Annotations(
        onset=onsets, duration=np.zeros(len(onsets)), description=descriptions, orig_time=None
    )
    raw.set_annotations(annotations)
    logger.debug(f"Attached {len(onsets)} events")
