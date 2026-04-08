"""Tests for MNE export utilities."""

import numpy as np
import pytest

from dendrite.data.io.mne_export import guess_channel_type, to_mne_raw


class TestGuessChannelType:
    def test_eeg_names(self):
        """Standard 10-20 names classify as EEG."""
        assert guess_channel_type("Fp1") == "eeg"
        assert guess_channel_type("Cz") == "eeg"
        assert guess_channel_type("O2") == "eeg"

    def test_emg_eog_ecg(self):
        """EMG/EOG/ECG prefixes detected."""
        assert guess_channel_type("EMG1") == "emg"
        assert guess_channel_type("VEOG") == "eog"
        assert guess_channel_type("ECG") == "ecg"

    def test_unknown_defaults_to_misc(self):
        """Names with misc-like prefixes classify as misc."""
        assert guess_channel_type("AUX_1") == "misc"


class TestToMneRaw:
    def test_creates_raw_array(self, tmp_path):
        """Creates RawArray with correct shape and channel names."""
        import h5py

        path = str(tmp_path / "test.h5")
        rng = np.random.default_rng(0)
        data = rng.standard_normal((100, 3)).astype(np.float64)

        with h5py.File(path, "w") as f:
            ds = f.create_dataset("EEG", data=data)
            ds.attrs["channel_labels"] = ["Fp1", "Cz", "O2"]

        raw = to_mne_raw(path, sfreq=250.0, dataset="EEG", montage=None)
        assert raw.get_data().shape == (3, 100)
        assert list(raw.ch_names) == ["Fp1", "Cz", "O2"]

    def test_channel_types_assigned(self, tmp_path):
        """EEG channel names get type 'eeg'."""
        import h5py

        path = str(tmp_path / "test.h5")
        rng = np.random.default_rng(0)
        data = rng.standard_normal((100, 3)).astype(np.float64)

        with h5py.File(path, "w") as f:
            ds = f.create_dataset("EEG", data=data)
            ds.attrs["channel_labels"] = ["Fp1", "Cz", "O2"]

        raw = to_mne_raw(path, sfreq=250.0, dataset="EEG", montage=None)
        ch_types = raw.get_channel_types()
        assert all(t == "eeg" for t in ch_types)
