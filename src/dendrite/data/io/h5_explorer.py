"""
Core H5 I/O operations.

Simple, focused functions for loading and inspecting H5 files.
"""

import ast
import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from dendrite.utils.serialization import jsonify

logger = logging.getLogger(__name__)


def find_dataset(h5f: h5py.File, stream_type: str | None = None) -> str | None:
    """Find a dataset name by stream type, using stream_index or type attributes.

    Args:
        h5f: Open HDF5 file handle.
        stream_type: Stream type to find (e.g. "eeg", "emg"). Case-insensitive.
            If None, returns the first non-Event data dataset.

    Returns:
        Dataset name if found, None otherwise.
    """
    # 1. Try stream_index root attr (fast path, new files)
    raw_index = h5f.attrs.get("stream_index")
    if raw_index is not None:
        index = json.loads(raw_index) if isinstance(raw_index, str) else raw_index
        if stream_type:
            st = stream_type.lower()
            matches = [k for k, v in index.items() if v.lower() == st]
            if matches:
                return matches[0]
        else:
            non_event = [k for k, v in index.items() if not v.upper().startswith("EVENT")]
            if non_event:
                return non_event[0]

    # 2. Fallback: check type attr on each dataset (old files)
    for name in h5f.keys():
        obj = h5f[name]
        if not isinstance(obj, h5py.Dataset):
            continue
        ds_type = str(obj.attrs.get("type", name))
        if stream_type and ds_type.lower() == stream_type.lower():
            return name
        if not stream_type and not name.startswith("Event"):
            return name

    return None


def load_dataset(path: str, name: str) -> pd.DataFrame:
    """
    Load a dataset from an H5 file.

    Args:
        path: Path to H5 file
        name: Dataset name (e.g., 'EEG', 'EMG', 'Event')

    Returns:
        DataFrame with data. For numeric data, columns are channel labels
        (preserving original case). For structured data (events), column names
        are normalized to lowercase (some recordings use PascalCase column
        names).
    """
    with h5py.File(path, "r", swmr=True) as h5f:
        if name not in h5f:
            available = list(h5f.keys())
            raise KeyError(f"Dataset '{name}' not found. Available: {available}")

        dataset = h5f[name]
        data = dataset[()]

        if data.dtype.names:  # Structured array (events)
            df = pd.DataFrame(data)
            df = _decode_bytes(df)
            df.columns = df.columns.str.lower()  # Normalize event fields
        else:  # Regular array (EEG/EMG)
            labels = get_channel_labels(dataset, data.shape[1])
            df = pd.DataFrame(data, columns=labels)

    logger.debug(f"Loaded {name}: {df.shape}")
    return df


def get_h5_info(path: str) -> dict[str, Any]:
    """
    Get H5 file structure information.

    Returns:
        Dict with:
        - datasets: {name: {shape, dtype, attributes}}
        - groups: {name: {attributes, children}}
        - root_attributes: file-level metadata
        - file_size_mb: file size in MB
    """
    info = {"datasets": {}, "groups": {}, "root_attributes": {}, "file_size_mb": 0}

    with h5py.File(path, "r", swmr=True) as h5f:
        # Root attributes
        info["root_attributes"] = {k: jsonify(v) for k, v in h5f.attrs.items()}

        # Datasets and groups
        for name in h5f.keys():
            obj = h5f[name]
            if isinstance(obj, h5py.Dataset):
                info["datasets"][name] = {
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "attributes": {k: jsonify(v) for k, v in obj.attrs.items()},
                }
            elif isinstance(obj, h5py.Group):
                info["groups"][name] = {
                    "attributes": {k: jsonify(v) for k, v in obj.attrs.items()},
                    "children": list(obj.keys()),
                }

    info["file_size_mb"] = Path(path).stat().st_size / (1024 * 1024)
    return info


def get_h5_metadata(path: str) -> dict[str, Any]:
    """
    Extract root-level metadata from H5 file.

    Returns:
        Dict with file attributes (study_name, subject_id, session_id, etc.)
    """
    with h5py.File(path, "r", swmr=True) as h5f:
        return {key: jsonify(value) for key, value in h5f.attrs.items()}


def get_channel_info(path: str, dataset: str | None = None) -> dict[str, Any]:
    """
    Get channel information from a dataset.

    Args:
        path: Path to H5 file.
        dataset: Dataset name. If None, auto-detects the first data dataset.

    Returns:
        Dict with labels, count, and sample_rate if available.
    """
    with h5py.File(path, "r", swmr=True) as h5f:
        if dataset is None:
            dataset = find_dataset(h5f)
        if dataset is None or dataset not in h5f:
            raise KeyError(f"No data dataset found (requested: {dataset!r})")

        ds = h5f[dataset]
        n_channels = ds.shape[1] if len(ds.shape) > 1 else 1
        labels = get_channel_labels(ds, n_channels)

        info = {
            "labels": labels,
            "count": n_channels,
            "n_samples": ds.shape[0],
        }

        # Sample rate if stored
        if "sample_rate" in ds.attrs:
            info["sample_rate"] = float(ds.attrs["sample_rate"])

        return info


def save_dataset(
    path: str, name: str, data: pd.DataFrame, overwrite: bool = False, **attrs
) -> None:
    """
    Save DataFrame to H5 dataset.

    Args:
        path: Path to H5 file
        name: Dataset name
        data: DataFrame to save
        overwrite: Whether to overwrite existing dataset
        **attrs: Additional attributes to store
    """
    # Convert DataFrame for H5 storage
    if data.select_dtypes(include=["object"]).shape[1] > 0:
        data_for_h5 = _prepare_structured_data(data)
    else:
        data_for_h5 = data.values

    with h5py.File(path, "r+") as h5f:
        if name in h5f:
            if not overwrite:
                raise ValueError(f"Dataset '{name}' exists. Use overwrite=True")
            del h5f[name]

        ds = h5f.create_dataset(name, data=data_for_h5)

        # Store channel labels for numeric data
        if data.select_dtypes(include=["object"]).shape[1] == 0:
            ds.attrs["channel_labels"] = [str(col) for col in data.columns]

        for key, value in attrs.items():
            ds.attrs[key] = value

    logger.debug(f"Saved {name}: {data.shape}")


def load_events(
    path: str, dataset: str = "Event", save: bool = True, target: str = "Event_Clean"
) -> pd.DataFrame:
    """
    Load and clean event data from H5 file.

    Automatically decodes byte strings and parses JSON in extra_vars column.

    Args:
        path: Path to H5 file
        dataset: Source dataset name
        save: If True, save cleaned events to target dataset (default: True)
        target: Target dataset name when saving

    Returns:
        DataFrame with event_type, timestamp, and parsed extra variables.
        Column names are lowercase (normalized by load_dataset).
    """
    df = load_dataset(path, dataset)

    # Parse JSON in extra_vars if present
    if "extra_vars" in df.columns:
        json_data = df["extra_vars"].apply(_parse_json)
        extra_df = pd.json_normalize(json_data).add_prefix("extra_")
        df = pd.concat([df.drop(columns=["extra_vars"]), extra_df], axis=1)

    if save:
        save_dataset(path, target, df, overwrite=True, cleaned=True, source=dataset)

    return df


def _decode_value(value: Any) -> Any:
    """Decode bytes to str, pass through other types."""
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _decode_bytes(df: pd.DataFrame) -> pd.DataFrame:
    """Decode byte string columns to UTF-8."""
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(_decode_value)
    return df


def get_channel_labels(dataset: "h5py.Dataset", n_channels: int) -> list[str]:
    """Extract channel labels from dataset attributes."""
    if "channel_labels" not in dataset.attrs:
        return [f"ch_{i}" for i in range(n_channels)]

    labels = dataset.attrs["channel_labels"]

    # Handle various formats
    if isinstance(labels, bytes):
        labels = labels.decode("utf-8")
    if isinstance(labels, str):
        try:
            labels = ast.literal_eval(labels)
        except (ValueError, SyntaxError):
            labels = [labels]

    # Decode byte strings in list
    labels = [str(_decode_value(label)) for label in labels]

    # Ensure correct length
    if len(labels) < n_channels:
        labels.extend([f"ch_{i}" for i in range(len(labels), n_channels)])
    elif len(labels) > n_channels:
        labels = labels[:n_channels]

    return labels


def _parse_json(value: Any) -> dict:
    """Safely parse JSON string."""
    if pd.isna(value) or value == "":
        return {}
    try:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return {}


# String length bounds for H5 byte string storage
_MIN_STRING_LEN = 10
_DEFAULT_STRING_LEN = 100
_MAX_STRING_LEN = 10000


def _prepare_structured_data(df: pd.DataFrame) -> np.ndarray:
    """Convert DataFrame to structured array for H5 storage."""
    df = df.copy()

    # Convert object columns to fixed-size byte strings
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("").astype(str)
        observed_len = df[col].str.len().max()
        if pd.isna(observed_len) or observed_len <= 0:
            string_len = _DEFAULT_STRING_LEN
        else:
            string_len = min(max(int(observed_len), _MIN_STRING_LEN), _MAX_STRING_LEN)
        df[col] = df[col].astype(f"S{string_len}")

    # Handle nullable dtypes
    for col in df.columns:
        if str(df[col].dtype) == "Int64":
            df[col] = df[col].fillna(0).astype("int64")
        elif str(df[col].dtype) == "Float64":
            df[col] = df[col].fillna(0.0).astype("float64")

    # Create structured array
    dtype_list = [(col, df[col].dtype) for col in df.columns]
    structured_array = np.zeros(len(df), dtype=dtype_list)

    for col in df.columns:
        structured_array[col] = df[col]

    return structured_array
