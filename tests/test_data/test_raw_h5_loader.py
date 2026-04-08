"""Tests for RawH5Loader event_id extraction."""

import numpy as np
import pytest
import h5py

from dendrite.data.loaders.raw_h5_loader import (
    RawH5Loader,
    _extract_event_id_mapping,
    _extract_h5_events,
)

EVENT_DTYPE = np.dtype([
    ("event_id", np.int32),
    ("event_type", h5py.string_dtype(encoding="utf-8")),
    ("timestamp", np.float64),
    ("local_timestamp", np.float64),
    ("extra_vars", h5py.string_dtype(encoding="utf-8")),
])


def _make_data_dtype(channel_names: list[str]) -> np.dtype:
    """Build a structured dtype matching DataSaver format."""
    fields = [("timestamp", np.float64), ("local_timestamp", np.float64)]
    fields += [(name, np.float32) for name in channel_names]
    return np.dtype(fields)


def _write_h5_with_events(
    path: str,
    channel_names: list[str],
    n_samples: int,
    sample_rate: float,
    events: list[tuple[str, int, float]],
    *,
    include_markers: bool = False,
):
    """Create an H5 file in DataSaver format with Event dataset.

    Args:
        events: List of (event_type, event_code, timestamp) tuples.
        include_markers: If True, add a Markers column to the data.
    """
    all_channels = list(channel_names)
    if include_markers:
        all_channels.append("Markers")

    dtype = _make_data_dtype(all_channels)
    data = np.zeros(n_samples, dtype=dtype)

    timestamps = np.arange(n_samples, dtype=np.float64) / sample_rate
    data["timestamp"] = timestamps

    if include_markers:
        for etype, ecode, ts in events:
            sample_idx = int(ts * sample_rate)
            if 0 <= sample_idx < n_samples:
                data["Markers"][sample_idx] = ecode

    with h5py.File(path, "w") as f:
        ds = f.create_dataset("eeg", data=data)
        ds.attrs["sampling_frequency"] = sample_rate

        if events:
            event_data = np.array(
                [(code, etype, ts, ts, "") for etype, code, ts in events],
                dtype=EVENT_DTYPE,
            )
            f.create_dataset("Event_eeg", data=event_data)


def _write_h5_markers_only(
    path: str,
    channel_names: list[str],
    n_samples: int,
    sample_rate: float,
    marker_events: list[tuple[int, int]],
):
    """Create an H5 file with Markers column but no Event dataset.

    Args:
        marker_events: List of (sample_index, event_code) tuples.
    """
    all_channels = list(channel_names) + ["Markers"]
    dtype = _make_data_dtype(all_channels)
    data = np.zeros(n_samples, dtype=dtype)
    data["timestamp"] = np.arange(n_samples, dtype=np.float64) / sample_rate

    for sample_idx, code in marker_events:
        if 0 <= sample_idx < n_samples:
            data["Markers"][sample_idx] = code

    with h5py.File(path, "w") as f:
        ds = f.create_dataset("eeg", data=data)
        ds.attrs["sampling_frequency"] = sample_rate


class TestExtractEventIdMapping:
    def test_reads_actual_codes(self, tmp_path):
        """Event dataset event_id field should be used, not sequential codes."""
        h5_path = str(tmp_path / "test.h5")
        _write_h5_with_events(
            h5_path, ["C3", "C4"], 1000, 250.0,
            events=[
                ("left_hand", 10, 0.5),
                ("right_hand", 20, 1.0),
                ("left_hand", 10, 1.5),
            ],
        )
        with h5py.File(h5_path, "r") as f:
            mapping = _extract_event_id_mapping(f)

        assert mapping == {"left_hand": 10, "right_hand": 20}

    def test_returns_none_without_event_dataset(self, tmp_path):
        """No Event dataset → returns None."""
        h5_path = str(tmp_path / "test.h5")
        _write_h5_markers_only(
            h5_path, ["C3", "C4"], 1000, 250.0,
            marker_events=[(100, 1), (200, 2)],
        )
        with h5py.File(h5_path, "r") as f:
            mapping = _extract_event_id_mapping(f)

        assert mapping is None

    def test_returns_none_for_empty_file(self, tmp_path):
        """H5 with no Event datasets returns None."""
        h5_path = str(tmp_path / "test.h5")
        dtype = _make_data_dtype(["C3"])
        data = np.zeros(100, dtype=dtype)
        with h5py.File(h5_path, "w") as f:
            ds = f.create_dataset("eeg", data=data)
            ds.attrs["sampling_frequency"] = 250.0

        with h5py.File(h5_path, "r") as f:
            assert _extract_event_id_mapping(f) is None


class TestExtractH5Events:
    def test_uses_actual_event_codes(self, tmp_path):
        """_extract_h5_events should use actual event_id field, not sequential."""
        h5_path = str(tmp_path / "test.h5")
        _write_h5_with_events(
            h5_path, ["C3", "C4"], 1000, 250.0,
            events=[
                ("right_hand", 20, 0.5),
                ("left_hand", 10, 1.0),
            ],
        )
        with h5py.File(h5_path, "r") as f:
            ds_data = f[_find_first_data(f)][()]
            events, event_id = _extract_h5_events(f, ds_data, 250.0)

        assert event_id == {"right_hand": 20, "left_hand": 10}
        # Verify event tuples use real codes
        codes = [code for _, code in events]
        assert set(codes) == {10, 20}

    def test_no_event_datasets(self, tmp_path):
        """No Event datasets → empty events, None event_id."""
        h5_path = str(tmp_path / "test.h5")
        dtype = _make_data_dtype(["C3"])
        data = np.zeros(100, dtype=dtype)
        data["timestamp"] = np.arange(100) / 250.0
        with h5py.File(h5_path, "w") as f:
            ds = f.create_dataset("eeg", data=data)
            ds.attrs["sampling_frequency"] = 250.0

        with h5py.File(h5_path, "r") as f:
            ds_data = f["eeg"][()]
            events, event_id = _extract_h5_events(f, ds_data, 250.0)

        assert events == []
        assert event_id is None


class TestRawH5LoaderEventId:
    def test_event_dataset_correct_mapping(self, tmp_path):
        """Full load with Event dataset: event_id uses actual codes."""
        h5_path = str(tmp_path / "test.h5")
        _write_h5_with_events(
            h5_path, ["C3", "C4"], 1000, 250.0,
            events=[
                ("left_hand", 10, 0.5),
                ("right_hand", 20, 1.5),
            ],
        )
        loaded = RawH5Loader(h5_path).load()

        assert loaded.event_id == {"left_hand": 10, "right_hand": 20}
        assert len(loaded.events) == 2

    def test_markers_with_event_dataset(self, tmp_path):
        """Markers + Event dataset: names come from Event dataset, not class_N."""
        h5_path = str(tmp_path / "test.h5")
        _write_h5_with_events(
            h5_path, ["C3", "C4"], 1000, 250.0,
            events=[
                ("left_hand", 10, 0.5),
                ("right_hand", 20, 1.5),
            ],
            include_markers=True,
        )
        loaded = RawH5Loader(h5_path).load()

        assert loaded.event_id is not None
        assert "left_hand" in loaded.event_id
        assert "right_hand" in loaded.event_id
        assert loaded.event_id["left_hand"] == 10
        assert loaded.event_id["right_hand"] == 20
        # Must NOT have class_N keys
        assert not any(k.startswith("class_") for k in loaded.event_id)

    def test_markers_without_event_dataset(self, tmp_path):
        """Markers only, no Event dataset: event_id is None."""
        h5_path = str(tmp_path / "test.h5")
        _write_h5_markers_only(
            h5_path, ["C3", "C4"], 1000, 250.0,
            marker_events=[(125, 1), (375, 2)],
        )
        loaded = RawH5Loader(h5_path).load()

        assert loaded.event_id is None
        assert len(loaded.events) == 2

    def test_no_events_at_all(self, tmp_path):
        """No Markers, no Event dataset: event_id is None, events empty."""
        h5_path = str(tmp_path / "test.h5")
        dtype = _make_data_dtype(["C3", "C4"])
        data = np.zeros(100, dtype=dtype)
        data["timestamp"] = np.arange(100) / 250.0
        with h5py.File(h5_path, "w") as f:
            ds = f.create_dataset("eeg", data=data)
            ds.attrs["sampling_frequency"] = 250.0

        loaded = RawH5Loader(h5_path).load()

        assert loaded.event_id is None
        assert loaded.events == []


def _find_first_data(h5_file: h5py.File) -> str:
    """Find first non-Event dataset name."""
    for k in h5_file.keys():
        if not k.startswith("Event") and isinstance(h5_file[k], h5py.Dataset):
            return k
    raise ValueError("No data dataset found")
