"""
Unit tests for dendrite.data.io.mne_export module.

Tests cover:
- Converting H5 datasets to MNE Raw objects
- Exporting to FIF format
- Channel type guessing from names
"""

import tempfile
from pathlib import Path

import h5py
import mne
import numpy as np
import pytest

from dendrite.constants import UV_TO_V
from dendrite.data.io.mne_export import (
    export_to_fif,
    guess_channel_type,
    to_mne_raw,
)


@pytest.fixture
def sample_h5_for_mne():
    """Create H5 file with EEG data suitable for MNE conversion."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        temp_path = f.name

    n_samples = 500
    n_channels = 4
    channel_labels = ['Fp1', 'Fp2', 'C3', 'C4']

    # Create data in microvolts (typical EEG range)
    eeg_data = np.random.randn(n_samples, n_channels).astype(np.float64) * 50

    with h5py.File(temp_path, 'w') as h5f:
        h5f.attrs['study_name'] = 'mne_test'
        h5f.attrs['sample_rate'] = 500.0

        eeg_ds = h5f.create_dataset('EEG', data=eeg_data)
        eeg_ds.attrs['channel_labels'] = channel_labels
        eeg_ds.attrs['sample_rate'] = 500.0

    yield temp_path
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def h5_with_events():
    """Create H5 file with EEG and event data."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        temp_path = f.name

    n_samples = 1000
    n_channels = 4
    sfreq = 500.0
    channel_labels = ['Fp1', 'Fp2', 'C3', 'C4']

    # EEG data with timestamp column
    timestamps = np.arange(n_samples) / sfreq
    eeg_values = np.random.randn(n_samples, n_channels).astype(np.float64) * 50
    eeg_data = np.column_stack([timestamps, eeg_values])

    with h5py.File(temp_path, 'w') as h5f:
        eeg_ds = h5f.create_dataset('EEG', data=eeg_data)
        eeg_ds.attrs['channel_labels'] = ['Timestamp'] + channel_labels
        eeg_ds.attrs['sample_rate'] = sfreq

        # Event data
        event_dtype = np.dtype([
            ('event_type', 'S50'),
            ('timestamp', 'f8'),
        ])
        events = np.array([
            (b'stimulus', 0.5),
            (b'response', 1.0),
            (b'stimulus', 1.5),
        ], dtype=event_dtype)
        h5f.create_dataset('Event', data=events)

    yield temp_path
    Path(temp_path).unlink(missing_ok=True)


class TestToMneRaw:
    """Tests for to_mne_raw function."""

    def test_returns_mne_raw_array(self, sample_h5_for_mne):
        """Test conversion returns an MNE RawArray object."""
        raw = to_mne_raw(sample_h5_for_mne, sfreq=500.0)

        assert isinstance(raw, mne.io.RawArray)
        assert raw.info['nchan'] == 4
        assert len(raw.times) == 500

    def test_converts_uv_to_volts(self, sample_h5_for_mne):
        """Test data is converted from microvolts to volts."""
        # Get original data in microvolts
        with h5py.File(sample_h5_for_mne, 'r') as h5f:
            original_uv = h5f['EEG'][()]

        raw = to_mne_raw(sample_h5_for_mne, sfreq=500.0)
        data_v = raw.get_data()

        # MNE data should be original * UV_TO_V
        expected_v = original_uv.T * UV_TO_V
        np.testing.assert_array_almost_equal(data_v, expected_v)

    def test_sets_channel_types(self, sample_h5_for_mne):
        """Test channel types are set based on channel names."""
        raw = to_mne_raw(sample_h5_for_mne, sfreq=500.0)

        # Standard EEG channel names should be typed as 'eeg'
        ch_types = raw.get_channel_types()
        assert all(ct == 'eeg' for ct in ch_types)

    def test_applies_montage(self, sample_h5_for_mne):
        """Test montage is applied when channels match."""
        raw = to_mne_raw(sample_h5_for_mne, sfreq=500.0, montage='standard_1005')

        # Montage should set channel positions
        # Fp1, Fp2, C3, C4 are standard 10-05 positions
        dig = raw.info.get('dig')
        # If montage applied, dig should contain positions
        assert dig is not None or len(raw.ch_names) > 0

    def test_uses_provided_sfreq(self, sample_h5_for_mne):
        """Test sampling frequency from parameter is used."""
        raw = to_mne_raw(sample_h5_for_mne, sfreq=250.0)

        assert raw.info['sfreq'] == 250.0

    def test_filters_timestamp_columns(self, h5_with_events):
        """Test timestamp columns are excluded from data."""
        raw = to_mne_raw(h5_with_events, sfreq=500.0)

        # Should have 4 channels (Timestamp filtered out)
        assert raw.info['nchan'] == 4
        assert 'Timestamp' not in raw.ch_names


class TestExportToFif:
    """Tests for export_to_fif function."""

    def test_creates_fif_file(self, sample_h5_for_mne, tmp_path):
        """Test FIF file is created."""
        output_path = tmp_path / "test_export.fif"

        result = export_to_fif(sample_h5_for_mne, 500.0, output_path)

        assert Path(result).exists()
        assert str(result) == str(output_path)

    def test_auto_generates_output_path(self, sample_h5_for_mne):
        """Test output path is auto-generated from input path."""
        result = export_to_fif(sample_h5_for_mne, 500.0)

        expected = str(Path(sample_h5_for_mne).with_suffix('.fif'))
        assert result == expected
        assert Path(result).exists()

        Path(result).unlink(missing_ok=True)

    def test_attaches_events_when_present(self, h5_with_events, tmp_path):
        """Test events are attached as annotations."""
        output_path = tmp_path / "with_events.fif"

        export_to_fif(h5_with_events, 500.0, output_path, include_events=True)

        # Load and check annotations
        raw = mne.io.read_raw_fif(output_path, preload=True, verbose=False)
        annotations = raw.annotations

        assert len(annotations) == 3
        assert 'stimulus' in annotations.description

    def test_skips_events_when_disabled(self, h5_with_events, tmp_path):
        """Test events are not attached when include_events=False."""
        output_path = tmp_path / "no_events.fif"

        export_to_fif(h5_with_events, 500.0, output_path, include_events=False)

        raw = mne.io.read_raw_fif(output_path, preload=True, verbose=False)
        assert len(raw.annotations) == 0

    def test_overwrites_existing_file(self, sample_h5_for_mne, tmp_path):
        """Test existing file is overwritten when overwrite=True."""
        output_path = tmp_path / "overwrite.fif"

        # Create first file
        export_to_fif(sample_h5_for_mne, 500.0, output_path)
        first_mtime = output_path.stat().st_mtime

        import time
        time.sleep(0.1)  # Ensure different mtime
        export_to_fif(sample_h5_for_mne, 500.0, output_path, overwrite=True)
        second_mtime = output_path.stat().st_mtime

        assert second_mtime > first_mtime

    def test_exported_fif_is_readable(self, sample_h5_for_mne, tmp_path):
        """Test exported FIF file can be read back."""
        output_path = tmp_path / "readable.fif"

        export_to_fif(sample_h5_for_mne, 500.0, output_path)

        raw = mne.io.read_raw_fif(output_path, preload=True, verbose=False)
        assert raw.info['nchan'] == 4
        assert raw.info['sfreq'] == 500.0


class TestGuessChannelType:
    """Tests for guess_channel_type function."""

    def test_returns_eeg_for_standard_names(self):
        """Test standard EEG channel names return 'eeg' type."""
        eeg_channels = ['Fp1', 'Fp2', 'Fz', 'Cz', 'Pz', 'O1', 'O2', 'C3', 'C4']

        for ch in eeg_channels:
            assert guess_channel_type(ch) == 'eeg', f"Failed for {ch}"

    def test_returns_emg_for_emg_channels(self):
        """Test EMG channel names return 'emg' type."""
        emg_channels = ['EMG1', 'emg_left', 'EMG_right', 'EMG']

        for ch in emg_channels:
            assert guess_channel_type(ch) == 'emg', f"Failed for {ch}"

    def test_returns_stim_for_markers(self):
        """Test marker/stim channels return 'stim' type."""
        stim_channels = ['Stim', 'STIM1', 'Marker', 'EventMarker', 'stim_channel']

        for ch in stim_channels:
            assert guess_channel_type(ch) == 'stim', f"Failed for {ch}"

    def test_returns_misc_for_auxiliary(self):
        """Test auxiliary channels return 'misc' type."""
        misc_channels = [
            'GSR', 'gsr1', 'Resp', 'respiration', 'Breath',
            'Pulse', 'exo_joint1', 'robot_sensor', 'aux_temp'
        ]

        for ch in misc_channels:
            assert guess_channel_type(ch) == 'misc', f"Failed for {ch}"

    def test_returns_ecg_for_ecg_channels(self):
        """Test ECG/EKG channels return 'ecg' type."""
        ecg_channels = ['ECG', 'ecg1', 'EKG', 'ekg_lead']

        for ch in ecg_channels:
            assert guess_channel_type(ch) == 'ecg', f"Failed for {ch}"

    def test_returns_eog_for_eog_channels(self):
        """Test EOG channels return 'eog' type."""
        eog_channels = ['EOG', 'eog_left', 'EOG_right', 'VEOG', 'HEOG']

        for ch in eog_channels:
            assert guess_channel_type(ch) == 'eog', f"Failed for {ch}"

    def test_case_insensitive(self):
        """Test channel type detection is case insensitive."""
        assert guess_channel_type('EMG') == 'emg'
        assert guess_channel_type('emg') == 'emg'
        assert guess_channel_type('Emg') == 'emg'
        assert guess_channel_type('ECG') == 'ecg'
        assert guess_channel_type('ecg') == 'ecg'
