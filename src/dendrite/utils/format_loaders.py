"""
Format loaders for offline streaming.

Provides unified interface for loading data from different file formats
(.set, .fif, .h5) for LSL streaming.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LoadedData:
    """Container for loaded file data."""

    data: np.ndarray  # (samples, channels)
    channel_names: list[str]
    channel_types: list[str]
    sample_rate: float
    events: list[tuple[int, int]]  # [(sample_idx, event_code), ...]
    event_id: dict[str, int] | None = None  # {'left_hand': 1, 'right_hand': 2}


# Extension → loader type mapping
SUPPORTED_FORMATS = {
    ".set": "mne",
    ".fif": "mne",
    ".h5": "h5",
    ".hdf5": "h5",
}

# Loader type → function mapping (populated after function definitions)
_LOADERS: dict = {}


def get_file_filter() -> str:
    """Get file dialog filter string for supported formats."""
    return (
        "All Supported (*.set *.fif *.h5 *.hdf5);;"
        "EEGLAB (*.set);;"
        "MNE-Python (*.fif);;"
        "HDF5 (*.h5 *.hdf5);;"
        "All Files (*)"
    )


def load_file(file_path: str) -> LoadedData:
    """Load data from file, auto-detecting format by extension."""
    ext = Path(file_path).suffix.lower()
    loader_type = SUPPORTED_FORMATS.get(ext)

    loader = _LOADERS.get(loader_type) if loader_type else None
    if not loader:
        raise ValueError(f"Unsupported format: {ext}. Supported: {list(SUPPORTED_FORMATS.keys())}")
    return loader(file_path)


def get_h5_file_info(file_path: str) -> tuple[float, int, list[str]]:
    """Get H5 file metadata without loading all data.

    Returns:
        Tuple of (duration_seconds, n_channels, channel_names)
    """
    import h5py  # Lazy import: h5py is heavy and only needed for H5 files

    with h5py.File(file_path, "r") as f:
        # Find main data dataset (not Event*)
        data_datasets = [
            k for k in f.keys() if not k.startswith("Event") and isinstance(f[k], h5py.Dataset)
        ]
        if not data_datasets:
            raise ValueError("No data datasets in H5 file")

        ds = f[data_datasets[0]]
        n_samples = ds.shape[0]

        # Get sample rate
        sample_rate = float(ds.attrs.get("sampling_frequency", ds.attrs.get("sample_rate", 500.0)))
        duration = n_samples / sample_rate

        # Get channel names (case-insensitive filter for backward compat)
        if "channel_labels" in ds.attrs:
            channel_names = _decode_labels(ds.attrs["channel_labels"])
            channel_names = [
                l for l in channel_names if l.lower() not in ["timestamp", "local_timestamp"]
            ]
        elif ds.dtype.names:
            channel_names = [
                n for n in ds.dtype.names if n.lower() not in ["timestamp", "local_timestamp"]
            ]
        else:
            channel_names = []

        return duration, len(channel_names), channel_names


def _load_mne(file_path: str) -> LoadedData:
    """Load via MNE (supports .set, .fif)."""
    import mne  # Lazy import: MNE is heavy (~500MB) and only needed for .set/.fif files

    ext = Path(file_path).suffix.lower()

    logger.info(f"Loading {ext} file via MNE: {file_path}")

    if ext == ".set":
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
    else:  # .fif
        raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)

    # Get data - MNE returns (channels, samples), we want (samples, channels)
    data = raw.get_data().T
    channel_names = list(raw.ch_names)
    channel_types = raw.get_channel_types()
    sample_rate = raw.info["sfreq"]

    # Extract events from annotations
    events = []
    event_id = None
    try:
        events_array, event_id = mne.events_from_annotations(raw)
        events = [(int(e[0]), int(e[2])) for e in events_array]
        logger.info(f"Found {len(events)} events in file, event_id: {event_id}")
    except (ValueError, KeyError, RuntimeError):
        logger.info("No events found in file")

    logger.info(f"Loaded: {data.shape[0]} samples, {data.shape[1]} channels @ {sample_rate} Hz")

    return LoadedData(
        data=data,
        channel_names=channel_names,
        channel_types=channel_types,
        sample_rate=sample_rate,
        events=events,
        event_id=event_id,
    )


def _load_h5(file_path: str) -> LoadedData:
    """Load internal Dendrite H5 format."""
    import h5py  # Lazy import: h5py is heavy and only needed for H5 files

    logger.info(f"Loading H5 file: {file_path}")

    with h5py.File(file_path, "r") as f:
        # Find data datasets (EEG, EMG - skip Event* datasets)
        data_datasets = [
            k for k in f.keys() if not k.startswith("Event") and isinstance(f[k], h5py.Dataset)
        ]

        if not data_datasets:
            raise ValueError("No data datasets found in H5 file")

        # Usually just one main dataset (EEG)
        ds_name = data_datasets[0]
        ds = f[ds_name]
        ds_data = ds[()]

        # Get channel labels from attribute
        if "channel_labels" in ds.attrs:
            all_labels = _decode_labels(ds.attrs["channel_labels"])
        elif ds_data.dtype.names:
            all_labels = list(ds_data.dtype.names)
        else:
            raise ValueError("Cannot determine channel labels")

        # Separate actual channels from metadata (case-insensitive for backward compat)
        data_labels = [
            label for label in all_labels if label.lower() not in ["timestamp", "local_timestamp"]
        ]
        has_markers = "Markers" in data_labels

        # Extract channel data
        channel_data = []
        channel_names = []
        for label in data_labels:
            if label in ds_data.dtype.names:
                channel_data.append(ds_data[label].astype(np.float32))
                channel_names.append(label)

        data = np.column_stack(channel_data)

        # Channel types - dataset type for all except Markers
        channel_types = [ds_name.lower() if l != "Markers" else "markers" for l in channel_names]

        # Sample rate
        sample_rate = float(ds.attrs.get("sampling_frequency", ds.attrs.get("sample_rate", 500.0)))

        # Extract events from Markers column (already in data)
        # Valid events are positive integers; 0 and -1 mean "no event"
        events = []
        if has_markers:
            markers_idx = channel_names.index("Markers")
            markers = data[:, markers_idx]
            for i, m in enumerate(markers):
                if m > 0:
                    events.append((i, int(m)))
            if events:
                logger.info(f"Found {len(events)} events in Markers channel")

        logger.info(
            f"Loaded: {data.shape[0]} samples, {len(channel_names)} channels @ {sample_rate} Hz"
        )

        return LoadedData(
            data=data,
            channel_names=channel_names,
            channel_types=channel_types,
            sample_rate=sample_rate,
            events=events,
        )


# Register loaders after function definitions
_LOADERS.update({"mne": _load_mne, "h5": _load_h5})


def _decode_labels(labels) -> list[str]:
    """Decode channel labels from H5 attributes."""
    # Handle bytes/str input
    if isinstance(labels, (bytes, str)):
        text = labels.decode("utf-8") if isinstance(labels, bytes) else labels
        try:
            import ast

            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return [text]

    # Handle iterable input
    return [l.decode("utf-8") if isinstance(l, bytes) else str(l) for l in labels]
