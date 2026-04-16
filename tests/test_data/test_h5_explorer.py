"""Tests for H5 explorer operations."""

import numpy as np
import pandas as pd
import pytest

from dendrite.data.io.h5_explorer import (
    get_channel_info,
    get_h5_info,
    get_h5_metadata,
    load_dataset,
    load_events,
    save_dataset,
)


@pytest.fixture
def sample_h5(tmp_path):
    """Create a small H5 file with EEG + Event datasets."""
    import h5py

    path = str(tmp_path / "test.h5")
    rng = np.random.default_rng(42)

    with h5py.File(path, "w") as f:
        # Root attrs
        f.attrs["study_name"] = "unit_test"
        f.attrs["subject_id"] = 1

        # EEG dataset: 100 samples x 3 channels
        eeg = rng.standard_normal((100, 3)).astype(np.float64)
        ds = f.create_dataset("EEG", data=eeg)
        ds.attrs["channel_labels"] = ["Fp1", "Cz", "O2"]
        ds.attrs["sample_rate"] = 250.0

        # Event dataset (structured dtype)
        event_dt = np.dtype([
            ("event_type", "S20"),
            ("timestamp", "f8"),
        ])
        events = np.array([
            (b"left", 0.1),
            (b"right", 0.5),
            (b"left", 0.9),
        ], dtype=event_dt)
        f.create_dataset("Event", data=events)

    return path


class TestH5IO:
    def test_save_and_load_roundtrip(self, sample_h5):
        """save_dataset then load_dataset preserves data."""

        df_out = pd.DataFrame(
            {"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]},
        )
        # File already exists — open r+ to add dataset
        save_dataset(sample_h5, "Test", df_out)
        df_in = load_dataset(sample_h5, "Test")
        assert df_in.shape == (3, 2)
        np.testing.assert_array_almost_equal(df_in.values, df_out.values)

    def test_get_h5_info(self, sample_h5):
        """Structure dict includes dataset name, shape, dtype."""
        info = get_h5_info(sample_h5)
        assert "EEG" in info["datasets"]
        assert info["datasets"]["EEG"]["shape"] == [100, 3]

    def test_get_h5_metadata(self, sample_h5):
        """Root attributes are returned."""
        meta = get_h5_metadata(sample_h5)
        assert meta["study_name"] == "unit_test"
        assert meta["subject_id"] == 1

    def test_get_channel_info(self, sample_h5):
        """Channel labels, count, sample_rate from attrs."""
        info = get_channel_info(sample_h5, "EEG")
        assert info["labels"] == ["Fp1", "Cz", "O2"]
        assert info["count"] == 3
        assert info["sample_rate"] == 250.0

    def test_load_dataset(self, sample_h5):
        """Load EEG dataset — columns match channel_labels."""
        df = load_dataset(sample_h5, "EEG")
        assert list(df.columns) == ["Fp1", "Cz", "O2"]
        assert df.shape == (100, 3)

    def test_load_events(self, sample_h5):
        """Load events with byte string decoding."""
        df = load_events(sample_h5, "Event", save=False)
        assert "event_type" in df.columns
        # Byte strings should be decoded
        assert df["event_type"].iloc[0] == "left"
        assert len(df) == 3
