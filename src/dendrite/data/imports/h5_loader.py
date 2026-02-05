"""H5 file data loader.

Loads data from Dendrite H5/HDF5 files with channel labels and events.
"""

import logging

import h5py
import numpy as np

from ._types import LoadedData

logger = logging.getLogger(__name__)

# Columns to filter out from channel data
_METADATA_COLUMNS = {"timestamp", "local_timestamp"}


def _find_data_datasets(h5_file: h5py.File) -> list[str]:
    """Find data datasets (not Event* datasets) in H5 file."""
    return [k for k in h5_file.keys() if not k.startswith("Event") and isinstance(h5_file[k], h5py.Dataset)]


class H5Loader:
    """Loader for Dendrite H5/HDF5 files.

    Provides static methods for loading H5 data and metadata.
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
                all_labels = H5Loader._decode_labels(ds.attrs["channel_labels"])
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

            # Extract events from Markers column
            events = []
            if has_markers:
                markers_idx = channel_names.index("Markers")
                markers = data[:, markers_idx]
                for i, m in enumerate(markers):
                    if m > 0:
                        events.append((i, int(m)))

            logger.info(f"Loaded: {data.shape[0]} samples, {len(channel_names)} channels @ {sample_rate} Hz")
            return LoadedData(data, channel_names, channel_types, sample_rate, events)

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
                channel_names = H5Loader._decode_labels(ds.attrs["channel_labels"])
                channel_names = [name for name in channel_names if name.lower() not in _METADATA_COLUMNS]
            elif ds.dtype.names:
                channel_names = [name for name in ds.dtype.names if name.lower() not in _METADATA_COLUMNS]
            else:
                channel_names = []

            return duration, len(channel_names), channel_names

    @staticmethod
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
