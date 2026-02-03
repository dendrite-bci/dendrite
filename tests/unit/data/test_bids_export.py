"""
Unit tests for dendrite.data.io.bids_export module.

Tests cover:
- Exporting recordings to BIDS format
- Generating BIDS metadata files
- BIDS directory structure creation
"""

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from dendrite.constants import BIDS_VERSION
from dendrite.data.io.bids_export import (
    export_recording_to_bids,
    generate_channels_tsv,
    generate_dataset_description,
    generate_events_tsv,
    generate_participants_tsv,
    generate_sidecar_json,
)


@pytest.fixture
def sample_recording_h5():
    """Create a sample H5 recording file with full metadata."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        temp_path = f.name

    n_samples = 500
    n_channels = 4
    sfreq = 500.0
    channel_labels = ['Fp1', 'Fp2', 'C3', 'C4']

    # EEG data with timestamps
    timestamps = np.arange(n_samples) / sfreq + 1000.0  # Offset for realistic timestamps
    eeg_values = np.random.randn(n_samples, n_channels).astype(np.float64) * 50
    eeg_data = np.column_stack([timestamps, eeg_values])

    with h5py.File(temp_path, 'w') as h5f:
        # Root metadata
        h5f.attrs['study_name'] = 'test_study'
        h5f.attrs['subject_id'] = '001'
        h5f.attrs['session_id'] = '01'
        h5f.attrs['recording_name'] = 'motor_imagery'
        h5f.attrs['sample_rate'] = sfreq

        # EEG dataset
        eeg_ds = h5f.create_dataset('EEG', data=eeg_data)
        eeg_ds.attrs['channel_labels'] = ['Timestamp'] + channel_labels
        eeg_ds.attrs['sample_rate'] = sfreq

        # Event dataset
        event_dtype = np.dtype([
            ('event_type', 'S50'),
            ('timestamp', 'f8'),
            ('extra_vars', 'S200')
        ])
        events = np.array([
            (b'left_hand', 1000.2, b'{"trial": 1}'),
            (b'right_hand', 1000.5, b'{"trial": 2}'),
            (b'rest', 1000.8, b''),
        ], dtype=event_dtype)
        h5f.create_dataset('Event', data=events)

    yield temp_path
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def minimal_h5():
    """Create minimal H5 file with just EEG data (no events)."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        temp_path = f.name

    with h5py.File(temp_path, 'w') as h5f:
        h5f.attrs['sample_rate'] = 500.0
        eeg_data = np.random.randn(100, 4).astype(np.float64)
        eeg_ds = h5f.create_dataset('EEG', data=eeg_data)
        eeg_ds.attrs['channel_labels'] = ['Ch1', 'Ch2', 'Ch3', 'Ch4']
        eeg_ds.attrs['sample_rate'] = 500.0

    yield temp_path
    Path(temp_path).unlink(missing_ok=True)


class TestGenerateDatasetDescription:
    """Tests for generate_dataset_description function."""

    def test_returns_valid_bids_structure(self):
        """Test returned dict has required BIDS fields."""
        desc = generate_dataset_description('my_study')

        assert desc['Name'] == 'my_study'
        assert desc['BIDSVersion'] == BIDS_VERSION
        assert desc['DatasetType'] == 'raw'
        assert 'Authors' in desc
        assert 'GeneratedBy' in desc

    def test_includes_generator_info(self):
        """Test GeneratedBy includes software info."""
        desc = generate_dataset_description('study')

        gen = desc['GeneratedBy'][0]
        assert 'Name' in gen
        assert 'Version' in gen


class TestGenerateParticipantsTsv:
    """Tests for generate_participants_tsv function."""

    def test_formats_participant_ids(self):
        """Test participant IDs are formatted with sub- prefix."""
        result = generate_participants_tsv(['001', '002', '003'])

        assert len(result) == 3
        assert result[0]['participant_id'] == 'sub-001'
        assert result[1]['participant_id'] == 'sub-002'
        assert result[2]['participant_id'] == 'sub-003'

    def test_sorts_participants(self):
        """Test participants are sorted."""
        result = generate_participants_tsv(['003', '001', '002'])

        ids = [r['participant_id'] for r in result]
        assert ids == ['sub-001', 'sub-002', 'sub-003']


class TestGenerateSidecarJson:
    """Tests for generate_sidecar_json function."""

    def test_includes_required_fields(self, sample_recording_h5):
        """Test sidecar includes required BIDS fields."""
        bids_info = {'task': 'motor_imagery', 'sample_rate': 500.0}
        sidecar = generate_sidecar_json(sample_recording_h5, bids_info)

        assert 'TaskName' in sidecar
        assert 'SamplingFrequency' in sidecar
        assert 'EEGChannelCount' in sidecar
        assert 'RecordingType' in sidecar

    def test_uses_bids_info_task_name(self, sample_recording_h5):
        """Test task name from bids_info is used."""
        bids_info = {'task': 'custom_task'}
        sidecar = generate_sidecar_json(sample_recording_h5, bids_info)

        assert sidecar['TaskName'] == 'custom_task'

    def test_includes_source_file(self, sample_recording_h5):
        """Test source file name is included."""
        bids_info = {}
        sidecar = generate_sidecar_json(sample_recording_h5, bids_info)

        assert 'SourceFile' in sidecar
        assert sidecar['SourceFile'].endswith('.h5')


class TestGenerateChannelsTsv:
    """Tests for generate_channels_tsv function."""

    def test_returns_channel_rows(self, sample_recording_h5):
        """Test returns list of channel dicts."""
        channels = generate_channels_tsv(sample_recording_h5)

        # Should have 5 channels (Timestamp + 4 EEG)
        assert len(channels) == 5
        assert all('name' in ch for ch in channels)
        assert all('type' in ch for ch in channels)
        assert all('units' in ch for ch in channels)

    def test_includes_channel_types(self, sample_recording_h5):
        """Test channel types are guessed from names."""
        channels = generate_channels_tsv(sample_recording_h5)

        # Filter to EEG channels (Fp1, Fp2, C3, C4)
        eeg_channels = [ch for ch in channels if ch['name'] in ['Fp1', 'Fp2', 'C3', 'C4']]
        assert all(ch['type'] == 'EEG' for ch in eeg_channels)

    def test_returns_empty_for_missing_dataset(self, minimal_h5):
        """Test returns empty list when dataset doesn't exist."""
        channels = generate_channels_tsv(minimal_h5, dataset='NonExistent')

        assert channels == []


class TestGenerateEventsTsv:
    """Tests for generate_events_tsv function."""

    def test_returns_event_rows(self, sample_recording_h5):
        """Test returns list of event dicts."""
        events = generate_events_tsv(sample_recording_h5)

        assert len(events) == 3
        assert all('onset' in e for e in events)
        assert all('duration' in e for e in events)
        assert all('trial_type' in e for e in events)

    def test_calculates_onset_relative_to_eeg(self, sample_recording_h5):
        """Test onset times are relative to EEG start."""
        events = generate_events_tsv(sample_recording_h5)

        # First event should be shortly after EEG start
        first_onset = events[0]['onset']
        assert first_onset >= 0
        assert first_onset < 1.0  # Should be < 1 second from start

    def test_includes_trial_type(self, sample_recording_h5):
        """Test trial_type contains event_type."""
        events = generate_events_tsv(sample_recording_h5)

        types = [e['trial_type'] for e in events]
        assert 'left_hand' in types
        assert 'right_hand' in types
        assert 'rest' in types

    def test_returns_empty_for_no_events(self, minimal_h5):
        """Test returns empty list when no events exist."""
        events = generate_events_tsv(minimal_h5)

        assert events == []


@pytest.mark.slow
class TestExportRecordingToBids:
    """Tests for export_recording_to_bids function."""

    def test_creates_bids_directory_structure(self, sample_recording_h5, tmp_path):
        """Test BIDS directory structure is created."""
        export_recording_to_bids(sample_recording_h5, tmp_path)

        # Check study directory
        study_dir = tmp_path / 'test_study'
        assert study_dir.exists()

        # Check subject/session/eeg structure
        eeg_dir = study_dir / 'sub-001' / 'ses-01' / 'eeg'
        assert eeg_dir.exists()

    def test_exports_fif_file(self, sample_recording_h5, tmp_path):
        """Test FIF file is exported."""
        result = export_recording_to_bids(sample_recording_h5, tmp_path)

        assert Path(result).exists()
        assert result.suffix == '.fif'

    def test_creates_sidecar_json(self, sample_recording_h5, tmp_path):
        """Test sidecar JSON is created alongside FIF."""
        result = export_recording_to_bids(sample_recording_h5, tmp_path)

        json_path = Path(result).with_name(
            Path(result).name.replace('_eeg.fif', '_eeg.json')
        )
        assert json_path.exists()

        with open(json_path) as f:
            sidecar = json.load(f)
        assert 'TaskName' in sidecar

    def test_creates_channels_tsv(self, sample_recording_h5, tmp_path):
        """Test channels.tsv is created."""
        result = export_recording_to_bids(sample_recording_h5, tmp_path)

        channels_path = Path(result).with_name(
            Path(result).name.replace('_eeg.fif', '_channels.tsv')
        )
        assert channels_path.exists()

        # Verify TSV content
        with open(channels_path) as f:
            lines = f.readlines()
        assert len(lines) > 1  # Header + data

    def test_creates_events_tsv_when_present(self, sample_recording_h5, tmp_path):
        """Test events.tsv is created when events exist."""
        result = export_recording_to_bids(sample_recording_h5, tmp_path)

        events_path = Path(result).with_name(
            Path(result).name.replace('_eeg.fif', '_events.tsv')
        )
        assert events_path.exists()

        with open(events_path) as f:
            lines = f.readlines()
        assert len(lines) == 4  # Header + 3 events

    def test_creates_dataset_description(self, sample_recording_h5, tmp_path):
        """Test dataset_description.json is created."""
        export_recording_to_bids(sample_recording_h5, tmp_path)

        desc_path = tmp_path / 'test_study' / 'dataset_description.json'
        assert desc_path.exists()

        with open(desc_path) as f:
            desc = json.load(f)
        assert desc['Name'] == 'test_study'
        assert desc['BIDSVersion'] == BIDS_VERSION

    def test_creates_participants_tsv(self, sample_recording_h5, tmp_path):
        """Test participants.tsv is created."""
        export_recording_to_bids(sample_recording_h5, tmp_path)

        participants_path = tmp_path / 'test_study' / 'participants.tsv'
        assert participants_path.exists()

        with open(participants_path) as f:
            content = f.read()
        assert 'sub-001' in content

    def test_copies_to_sourcedata_when_enabled(self, sample_recording_h5, tmp_path):
        """Test original H5 is copied to sourcedata."""
        export_recording_to_bids(
            sample_recording_h5, tmp_path,
            include_sourcedata=True
        )

        sourcedata_dir = tmp_path / 'test_study' / 'sourcedata' / 'sub-001' / 'ses-01'
        h5_files = list(sourcedata_dir.glob('*.h5'))
        assert len(h5_files) == 1

    def test_skips_sourcedata_when_disabled(self, sample_recording_h5, tmp_path):
        """Test sourcedata is not created when disabled."""
        export_recording_to_bids(
            sample_recording_h5, tmp_path,
            include_sourcedata=False
        )

        sourcedata_dir = tmp_path / 'test_study' / 'sourcedata'
        assert not sourcedata_dir.exists()

    def test_uses_custom_study_name(self, sample_recording_h5, tmp_path):
        """Test custom study name overrides H5 metadata."""
        export_recording_to_bids(
            sample_recording_h5, tmp_path,
            study_name='custom_study'
        )

        custom_dir = tmp_path / 'custom_study'
        assert custom_dir.exists()

    def test_handles_minimal_h5(self, minimal_h5, tmp_path):
        """Test export works with minimal metadata."""
        result = export_recording_to_bids(minimal_h5, tmp_path)

        assert Path(result).exists()
        # Should use defaults for missing metadata

    def test_uses_existing_bids_dataset(self, sample_recording_h5, tmp_path):
        """Test export into existing BIDS dataset root."""
        # Create existing BIDS dataset
        bids_root = tmp_path / 'existing_bids'
        bids_root.mkdir()
        (bids_root / 'dataset_description.json').write_text(
            json.dumps({'Name': 'existing', 'BIDSVersion': BIDS_VERSION})
        )

        result = export_recording_to_bids(sample_recording_h5, bids_root)

        # Should export directly to existing root, not create subdirectory
        assert Path(result).parent.parent.parent.parent == bids_root


class TestBidsFilenameExtraction:
    """Tests for BIDS filename parsing."""

    def test_extracts_from_bids_filename(self, tmp_path):
        """Test BIDS fields are extracted from properly named file."""
        # Create file with BIDS naming
        h5_path = tmp_path / 'sub-002_ses-02_task-rest_run-03_eeg.h5'

        with h5py.File(h5_path, 'w') as h5f:
            h5f.attrs['sample_rate'] = 500.0
            data = np.random.randn(100, 4)
            ds = h5f.create_dataset('EEG', data=data)
            ds.attrs['channel_labels'] = ['A', 'B', 'C', 'D']
            ds.attrs['sample_rate'] = 500.0

        result = export_recording_to_bids(h5_path, tmp_path / 'output')

        # Check extracted values are in output path
        assert 'sub-002' in str(result)
        assert 'ses-02' in str(result)
        assert 'task-rest' in str(result)
        assert 'run-03' in str(result)

        h5_path.unlink()

    def test_uses_defaults_for_non_bids_filename(self, minimal_h5, tmp_path):
        """Test defaults are used when filename is not BIDS format."""
        result = export_recording_to_bids(minimal_h5, tmp_path)

        # Should use default values
        assert 'sub-001' in str(result)
        assert 'ses-01' in str(result)
