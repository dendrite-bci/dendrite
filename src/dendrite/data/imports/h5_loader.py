"""H5 file data loader.

H5DatasetLoader: Unified loader for Dendrite H5/HDF5 files.
Static methods for streaming/preview, instance methods for training.
"""

import ast
import logging

import h5py
import mne
import numpy as np

from ._event_utils import apply_preprocessing
from ._types import LoadedData
from .base_loader import FileDatasetLoader

logger = logging.getLogger(__name__)

# Columns to filter out from channel data
_METADATA_COLUMNS = {"timestamp", "local_timestamp"}


def _find_data_datasets(h5_file: h5py.File) -> list[str]:
    """Find data datasets (not Event* datasets) in H5 file."""
    return [k for k in h5_file.keys() if not k.startswith("Event") and isinstance(h5_file[k], h5py.Dataset)]


def _detect_datasets(h5_path: str) -> tuple[str, str | None]:
    """Detect data and event dataset names in an H5 file.

    Returns:
        Tuple of (data_dataset_name, event_dataset_name_or_None)
    """
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        data_datasets = [k for k in keys if not k.startswith("Event") and isinstance(f[k], h5py.Dataset)]
        event_datasets = [k for k in keys if k.startswith("Event") and isinstance(f[k], h5py.Dataset)]

    if not data_datasets:
        raise ValueError(f"No data datasets found in H5 file: {h5_path}")

    data_name = data_datasets[0]
    event_name = event_datasets[0] if event_datasets else None
    return data_name, event_name


class H5DatasetLoader(FileDatasetLoader):
    """Unified loader for Dendrite H5/HDF5 files.

    Static methods (load_file, get_file_info) for streaming/preview.
    Instance methods (load_raw) for training via FileDatasetLoader.
    """

    EXTENSIONS = {".h5", ".hdf5"}

    @staticmethod
    def load_file(file_path: str) -> LoadedData:
        """Load data from internal Dendrite H5 format."""
        logger.info(f"Loading H5 file: {file_path}")

        with h5py.File(file_path, "r") as f:
            data_datasets = _find_data_datasets(f)

            if not data_datasets:
                raise ValueError("No data datasets found in H5 file")

            ds_name = data_datasets[0]
            ds = f[ds_name]
            ds_data = ds[()]

            # Get channel labels
            if "channel_labels" in ds.attrs:
                all_labels = H5DatasetLoader._decode_labels(ds.attrs["channel_labels"])
            elif ds_data.dtype.names:
                all_labels = list(ds_data.dtype.names)
            else:
                raise ValueError("Cannot determine channel labels")

            # Filter out metadata columns
            data_labels = [label for label in all_labels if label.lower() not in _METADATA_COLUMNS]
            has_markers = "Markers" in data_labels

            # Extract channel data
            channel_data = []
            channel_names = []
            for label in data_labels:
                if label in ds_data.dtype.names:
                    channel_data.append(ds_data[label].astype(np.float32))
                    channel_names.append(label)

            data = np.column_stack(channel_data)
            channel_types = [ds_name.lower() if name != "Markers" else "markers" for name in channel_names]
            sample_rate = float(ds.attrs.get("sampling_frequency", ds.attrs.get("sample_rate", 500.0)))

            # Extract events from Markers column (legacy format)
            events: list[tuple[int, int]] = []
            event_id: dict[str, int] | None = None
            if has_markers:
                markers_idx = channel_names.index("Markers")
                markers = data[:, markers_idx]
                for i, m in enumerate(markers):
                    if m > 0:
                        events.append((i, int(m)))

            # If no Markers events, try separate Event datasets (Dendrite recording format)
            if not events:
                events, event_id = H5DatasetLoader._extract_h5_events(
                    f, ds_data, sample_rate
                )

            logger.info(f"Loaded: {data.shape[0]} samples, {len(channel_names)} channels @ {sample_rate} Hz")
            return LoadedData(data, channel_names, channel_types, sample_rate, events, event_id)

    @staticmethod
    def get_file_info(file_path: str) -> tuple[float, int, list[str]]:
        """Get H5 file metadata without loading all data.

        Args:
            file_path: Path to .h5 or .hdf5 file

        Returns:
            Tuple of (duration_seconds, n_channels, channel_names)
        """
        with h5py.File(file_path, "r") as f:
            data_datasets = _find_data_datasets(f)
            if not data_datasets:
                raise ValueError("No data datasets in H5 file")

            ds = f[data_datasets[0]]
            n_samples = ds.shape[0]

            # Get sample rate
            sample_rate = float(ds.attrs.get("sampling_frequency", ds.attrs.get("sample_rate", 500.0)))
            duration = n_samples / sample_rate

            # Get channel names (case-insensitive filter for backward compat)
            if "channel_labels" in ds.attrs:
                channel_names = H5DatasetLoader._decode_labels(ds.attrs["channel_labels"])
                channel_names = [name for name in channel_names if name.lower() not in _METADATA_COLUMNS]
            elif ds.dtype.names:
                channel_names = [name for name in ds.dtype.names if name.lower() not in _METADATA_COLUMNS]
            else:
                channel_names = []

            return duration, len(channel_names), channel_names

    @staticmethod
    def _extract_h5_events(
        h5_file: h5py.File,
        ds_data: np.ndarray,
        sample_rate: float,
    ) -> tuple[list[tuple[int, int]], dict[str, int] | None]:
        """Extract events from separate Event datasets in H5 file.

        Dendrite recordings store events in dedicated datasets (e.g., Event_EEG_Stream)
        with structured arrays containing event_type and timestamp fields.
        Uses the data dataset's timestamp to align event sample indices.
        """
        event_ds_names = [
            k for k in h5_file.keys()
            if k.startswith("Event") and isinstance(h5_file[k], h5py.Dataset)
        ]
        if not event_ds_names:
            return [], None

        event_data = h5_file[event_ds_names[0]][()]
        if not event_data.dtype.names:
            return [], None

        # Find field names (case-insensitive)
        field_map = {n.lower(): n for n in event_data.dtype.names}
        if "timestamp" not in field_map or "event_type" not in field_map:
            return [], None

        # Get data start timestamp for alignment (timestamp lives in the structured dtype)
        if ds_data.dtype.names is None:
            return [], None
        ts_label = next((n for n in ds_data.dtype.names if n.lower() == "timestamp"), None)
        if ts_label is None:
            return [], None

        data_start = float(ds_data[ts_label][0])
        event_timestamps = event_data[field_map["timestamp"]].astype(float)

        # Decode event types (may be byte strings)
        raw_types = event_data[field_map["event_type"]]
        event_types = [
            et.decode("utf-8") if isinstance(et, bytes) else str(et)
            for et in raw_types
        ]

        # Build event_id mapping from unique types
        unique_types = sorted(set(event_types))
        event_id = {name: i + 1 for i, name in enumerate(unique_types)}

        # Build events list with sample indices
        n_samples = len(ds_data)
        events = []
        for ts, etype in zip(event_timestamps, event_types):
            sample_idx = int((ts - data_start) * sample_rate)
            if 0 <= sample_idx < n_samples:
                events.append((sample_idx, event_id[etype]))

        return events, event_id

    @staticmethod
    def _decode_labels(labels) -> list[str]:
        """Decode channel labels from H5 attributes."""
        if isinstance(labels, (bytes, str)):
            text = labels.decode("utf-8") if isinstance(labels, bytes) else labels
            try:
                return ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return [text]

        return [l.decode("utf-8") if isinstance(l, bytes) else str(l) for l in labels]

    def load_raw(
        self,
        subject_id: int,
        preprocess: bool = True,
        session: str | None = None,
        run: str | None = None,
    ) -> mne.io.Raw:
        """Load H5 recording as MNE Raw with optional preprocessing."""
        cache_key = (subject_id, preprocess)
        if cache_key in self._raw_cache:
            return self._raw_cache[cache_key]

        # Local import to avoid circular dependency: mne_export -> constants -> data
        from dendrite.data.io.mne_export import _attach_events, to_mne_raw

        data_dataset, event_dataset = _detect_datasets(str(self._file_path))
        sfreq = self.get_sample_rate()

        raw = to_mne_raw(str(self._file_path), sfreq=sfreq, dataset=data_dataset)

        # Attach events as annotations for epoch extraction
        if event_dataset:
            try:
                _attach_events(raw, str(self._file_path), data_dataset, event_dataset, sfreq)
            except (KeyError, ValueError, OSError) as e:
                logger.warning(f"Could not attach events from H5: {e}")

        if preprocess:
            lowcut, highcut, rereference, target_sample_rate = self._get_preproc_params()
            if lowcut or highcut or target_sample_rate:
                raw = apply_preprocessing(raw, lowcut, highcut, rereference, target_sample_rate)

        self._raw_cache[cache_key] = raw
        return raw
