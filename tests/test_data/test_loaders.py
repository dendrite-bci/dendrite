"""Tests for format detection and FIF loader."""

import numpy as np
import pytest

from dendrite.data.loaders import FIFLoader, RawData, is_supported_format, load_file


class TestFormatDetection:
    @pytest.mark.parametrize("path,expected", [
        ("recording.fif", True),
        ("data.h5", True),
        ("data.hdf5", True),
        ("data.csv", False),
        ("data.edf", False),
    ])
    def test_is_supported(self, path, expected):
        assert is_supported_format(path) is expected


class TestFIFLoader:
    def test_load_file_roundtrip(self, tmp_path):
        """Create synthetic Raw, save to FIF, load via FIFLoader, verify fields."""
        import mne

        n_channels, sfreq, duration = 3, 250, 1.0
        rng = np.random.default_rng(42)
        data = rng.standard_normal((n_channels, int(sfreq * duration)))
        ch_names = ["Fp1", "Cz", "O2"]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info)

        # Add 2 annotations
        raw.set_annotations(mne.Annotations(
            onset=[0.1, 0.5],
            duration=[0, 0],
            description=["left", "right"],
        ))

        fif_path = str(tmp_path / "test_raw.fif")
        raw.save(fif_path, overwrite=True)

        loaded = FIFLoader(fif_path).load()
        assert loaded.data.shape == (3, 250)
        assert loaded.sample_rate == 250
        assert len(loaded.events) == 2

    def test_load_file_dispatch(self, tmp_path):
        """load_file() dispatches .fif to FIFLoader."""
        import mne

        data = np.random.default_rng(0).standard_normal((2, 100))
        info = mne.create_info(ch_names=["C3", "C4"], sfreq=250, ch_types="eeg")
        raw = mne.io.RawArray(data, info)
        fif_path = str(tmp_path / "dispatch_raw.fif")
        raw.save(fif_path, overwrite=True)

        loaded = load_file(fif_path)
        assert loaded.data.shape[0] == 2
        assert loaded.sample_rate == 250


class TestRawDataUnits:
    def test_units_default(self):
        """RawData defaults to volts."""
        ld = RawData(
            data=np.zeros((10, 2)),
            channel_names=["C3", "C4"],
            channel_types=["eeg", "eeg"],
            sample_rate=250.0,
            events=[],
        )
        assert ld.units == "V"

    def test_fif_load_file_units(self, tmp_path):
        """FIFLoader() returns RawData with units='V'."""
        import mne

        data = np.random.default_rng(0).standard_normal((2, 100))
        info = mne.create_info(ch_names=["C3", "C4"], sfreq=250, ch_types="eeg")
        raw = mne.io.RawArray(data, info)
        fif_path = str(tmp_path / "units_raw.fif")
        raw.save(fif_path, overwrite=True)

        loaded = FIFLoader(fif_path).load()
        assert loaded.units == "V"
