"""H5 file data loader."""

import ast
import logging

import h5py
import numpy as np

from ._types import RawData

logger = logging.getLogger(__name__)

_METADATA_COLUMNS = {"timestamp", "local_timestamp", "receive_timestamp"}


def _filter_metadata(labels: list[str]) -> list[str]:
    return [name for name in labels if name.lower() not in _METADATA_COLUMNS]


def _find_event_datasets(h5_file: h5py.File) -> list[str]:
    return [k for k in h5_file.keys() if k.startswith("Event") and isinstance(h5_file[k], h5py.Dataset)]


class RawH5Loader:
    """Loader for Dendrite H5/HDF5 files."""

    EXTENSIONS = {".h5", ".hdf5"}

    def __init__(self, file_path: str, swmr: bool = False):
        self._file_path = file_path
        self._swmr = swmr

    def load(self, *, modality: str | None = None) -> RawData:
        """Load data from internal Dendrite H5 format.

        Args:
            modality: If provided, prefer the dataset whose name matches
                      (e.g. 'emg' selects the 'EMG' dataset over 'EEG').
        """
        logger.info(f"Loading H5 file: {self._file_path}")

        with h5py.File(self._file_path, "r", swmr=self._swmr) as f:
            from dendrite.data.io.h5_explorer import find_dataset

            ds_name = find_dataset(f, modality)
            if ds_name is None:
                raise ValueError(
                    "No data dataset found"
                    + (f" for modality '{modality}'" if modality else "")
                )
            ds = f[ds_name]
            if self._swmr:
                ds.refresh()
            ds_data = ds[()]

            if "channel_labels" in ds.attrs:
                all_labels = _decode_labels(ds.attrs["channel_labels"])
            elif ds_data.dtype.names:
                all_labels = list(ds_data.dtype.names)
            else:
                raise ValueError("Cannot determine channel labels")

            data_labels = _filter_metadata(all_labels)
            has_markers = "Markers" in data_labels

            channel_data = []
            channel_names = []
            for label in data_labels:
                if label in ds_data.dtype.names:
                    channel_data.append(ds_data[label].astype(np.float32))
                    channel_names.append(label)

            data = np.array(channel_data)  # (channels, samples)

            # Read channel_types from attrs if available, else infer from type attr
            # Always lowercase to match MNE convention ('eeg', 'eog', etc.)
            fallback_type = str(ds.attrs.get("type", ds_name)).lower()
            raw_types = ds.attrs.get("channel_types", None)
            if raw_types is not None:
                all_types = [t.lower() for t in _decode_labels(raw_types)]
                # Map label→type (all_labels includes timestamp etc., all_types aligns with it)
                type_map = (
                    dict(zip(all_labels, all_types))
                    if len(all_types) == len(all_labels)
                    else {}
                )
                channel_types = [type_map.get(name, fallback_type) for name in channel_names]
            else:
                channel_types = [
                    fallback_type if name != "Markers" else "markers"
                    for name in channel_names
                ]
            sample_rate = float(ds.attrs.get("sampling_frequency", ds.attrs.get("sample_rate", 500.0)))

            events: list[tuple[int, int]] = []
            event_id: dict[str, int] | None = None
            if has_markers:
                markers_idx = channel_names.index("Markers")
                markers = data[markers_idx]
                for i, m in enumerate(markers):
                    if m > 0:
                        events.append((i, int(m)))
                if events:
                    event_id = _extract_event_id_mapping(f)

            if not events:
                events, event_id = _extract_h5_events(f, ds_data, sample_rate)

            logger.info(f"Loaded: {data.shape[1]} samples, {len(channel_names)} channels @ {sample_rate} Hz")
            return RawData(data, channel_names, channel_types, sample_rate, events, event_id)

def _extract_event_id_mapping(h5_file: h5py.File) -> dict[str, int] | None:
    """Extract event name→code mapping from Event datasets."""
    event_ds_names = _find_event_datasets(h5_file)
    if not event_ds_names:
        return None
    event_data = h5_file[event_ds_names[0]][()]
    if not event_data.dtype.names:
        return None
    field_map = {n.lower(): n for n in event_data.dtype.names}
    if "event_type" not in field_map or "event_id" not in field_map:
        return None
    mapping: dict[str, int] = {}
    for row in event_data:
        name = row[field_map["event_type"]]
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        code = int(row[field_map["event_id"]])
        if name and name not in mapping:
            mapping[name] = code
    return mapping if mapping else None


def _extract_h5_events(
    h5_file: h5py.File,
    ds_data: np.ndarray,
    sample_rate: float,
) -> tuple[list[tuple[int, int]], dict[str, int] | None]:
    """Extract events from separate Event datasets in H5 file."""
    event_ds_names = _find_event_datasets(h5_file)
    if not event_ds_names:
        return [], None

    event_data = h5_file[event_ds_names[0]][()]
    if not event_data.dtype.names:
        return [], None

    field_map = {n.lower(): n for n in event_data.dtype.names}
    if "timestamp" not in field_map or "event_type" not in field_map:
        return [], None

    if ds_data.dtype.names is None:
        return [], None
    ts_label = next((n for n in ds_data.dtype.names if n.lower() == "timestamp"), None)
    if ts_label is None:
        return [], None

    if len(ds_data) == 0:
        return [], None
    data_start = float(ds_data[ts_label][0])
    event_timestamps = event_data[field_map["timestamp"]].astype(float)

    raw_types = event_data[field_map["event_type"]]
    event_types = [
        et.decode("utf-8", errors="replace") if isinstance(et, bytes) else str(et)
        for et in raw_types
    ]

    # Build name→code mapping from actual event_id field
    raw_codes = event_data[field_map["event_id"]] if "event_id" in field_map else None
    event_id: dict[str, int] = {}
    if raw_codes is not None:
        for etype, code in zip(event_types, raw_codes, strict=True):
            if etype not in event_id:
                event_id[etype] = int(code)
    else:
        unique_types = sorted(set(event_types))
        event_id = {name: i + 1 for i, name in enumerate(unique_types)}

    n_samples = len(ds_data)
    events = []
    for ts, etype in zip(event_timestamps, event_types, strict=True):
        sample_idx = int(np.searchsorted(ds_data[ts_label], ts))
        if 0 <= sample_idx < n_samples:
            events.append((sample_idx, event_id[etype]))

    return events, event_id


def _decode_labels(labels) -> list[str]:
    """Decode channel labels from H5 attributes."""
    if isinstance(labels, (bytes, str)):
        text = labels.decode("utf-8") if isinstance(labels, bytes) else labels
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return [text]
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in labels]
