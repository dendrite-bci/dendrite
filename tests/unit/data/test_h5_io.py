"""
Unit tests for dendrite.data.io.h5_io module.

Tests cover:
- Loading numeric and structured datasets from H5 files
- Getting H5 file info, metadata, and channel info
- Saving datasets to H5 files
- Loading and cleaning event data
"""

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from dendrite.data.io.h5_io import (
    get_channel_info,
    get_h5_info,
    get_h5_metadata,
    load_dataset,
    load_events,
    save_dataset,
)


@pytest.fixture
def sample_h5_file():
    """Create a temporary H5 file with EEG and Event datasets."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        temp_path = f.name

    n_samples = 100
    n_channels = 8
    channel_labels = [f'EEG_{i+1}' for i in range(n_channels)]

    with h5py.File(temp_path, 'w') as h5f:
        # Root attributes
        h5f.attrs['study_name'] = 'test_study'
        h5f.attrs['subject_id'] = '001'
        h5f.attrs['sample_rate'] = 500.0

        # EEG dataset (numeric)
        eeg_data = np.random.randn(n_samples, n_channels).astype(np.float64)
        eeg_ds = h5f.create_dataset('EEG', data=eeg_data)
        eeg_ds.attrs['channel_labels'] = channel_labels
        eeg_ds.attrs['sample_rate'] = 500.0

        # Event dataset (structured)
        event_dtype = np.dtype([
            ('event_type', 'S50'),
            ('timestamp', 'f8'),
            ('extra_vars', 'S200')
        ])
        events = np.array([
            (b'stimulus', 1.0, b'{"trial": 1}'),
            (b'response', 1.5, b'{"correct": true}'),
            (b'stimulus', 2.0, b'{"trial": 2}'),
        ], dtype=event_dtype)
        h5f.create_dataset('Event', data=events)

        # Create a group for testing
        grp = h5f.create_group('metadata')
        grp.attrs['version'] = '1.0'

    yield temp_path

    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def empty_h5_file():
    """Create an empty H5 file."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        temp_path = f.name

    with h5py.File(temp_path, 'w') as h5f:
        h5f.attrs['study_name'] = 'empty_study'

    yield temp_path
    Path(temp_path).unlink(missing_ok=True)


class TestLoadDataset:
    """Tests for load_dataset function."""

    def test_load_numeric_dataset_returns_dataframe(self, sample_h5_file):
        """Test loading a numeric dataset returns a DataFrame with correct shape."""
        df = load_dataset(sample_h5_file, 'EEG')

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (100, 8)
        assert list(df.columns) == [f'EEG_{i+1}' for i in range(8)]

    def test_load_structured_dataset_decodes_bytes(self, sample_h5_file):
        """Test loading a structured dataset decodes byte strings to UTF-8."""
        df = load_dataset(sample_h5_file, 'Event')

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        # Byte strings should be decoded
        assert df['event_type'].iloc[0] == 'stimulus'
        assert df['event_type'].iloc[1] == 'response'

    def test_raises_key_error_for_missing_dataset(self, sample_h5_file):
        """Test KeyError is raised when dataset doesn't exist."""
        with pytest.raises(KeyError) as exc_info:
            load_dataset(sample_h5_file, 'NonExistent')

        assert 'NonExistent' in str(exc_info.value)
        assert 'Available' in str(exc_info.value)

    def test_generates_fallback_labels_when_missing(self, empty_h5_file):
        """Test fallback channel labels are generated when not in attributes."""
        # Add a dataset without channel_labels attribute
        with h5py.File(empty_h5_file, 'r+') as h5f:
            data = np.random.randn(10, 4)
            h5f.create_dataset('NoLabels', data=data)

        df = load_dataset(empty_h5_file, 'NoLabels')
        assert list(df.columns) == ['ch_0', 'ch_1', 'ch_2', 'ch_3']


class TestGetH5Info:
    """Tests for get_h5_info function."""

    def test_returns_datasets_and_groups(self, sample_h5_file):
        """Test info contains datasets and groups."""
        info = get_h5_info(sample_h5_file)

        assert 'datasets' in info
        assert 'groups' in info
        assert 'EEG' in info['datasets']
        assert 'Event' in info['datasets']
        assert 'metadata' in info['groups']

    def test_includes_root_attributes(self, sample_h5_file):
        """Test info contains root-level attributes."""
        info = get_h5_info(sample_h5_file)

        assert 'root_attributes' in info
        assert info['root_attributes']['study_name'] == 'test_study'
        assert info['root_attributes']['subject_id'] == '001'

    def test_includes_file_size(self, sample_h5_file):
        """Test info contains file size in MB."""
        info = get_h5_info(sample_h5_file)

        assert 'file_size_mb' in info
        assert isinstance(info['file_size_mb'], float)
        assert info['file_size_mb'] > 0

    def test_includes_dataset_shape_and_dtype(self, sample_h5_file):
        """Test dataset info includes shape and dtype."""
        info = get_h5_info(sample_h5_file)

        eeg_info = info['datasets']['EEG']
        assert eeg_info['shape'] == (100, 8)
        assert 'float64' in eeg_info['dtype']


class TestGetH5Metadata:
    """Tests for get_h5_metadata function."""

    def test_returns_root_attributes(self, sample_h5_file):
        """Test metadata returns root-level attributes."""
        metadata = get_h5_metadata(sample_h5_file)

        assert metadata['study_name'] == 'test_study'
        assert metadata['subject_id'] == '001'
        assert metadata['sample_rate'] == 500.0

    def test_decodes_bytes_to_strings(self):
        """Test byte attributes are decoded to UTF-8 strings."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_path = f.name

        try:
            with h5py.File(temp_path, 'w') as h5f:
                h5f.attrs['byte_attr'] = b'byte_value'
                h5f.attrs['string_attr'] = 'string_value'

            metadata = get_h5_metadata(temp_path)

            assert metadata['byte_attr'] == 'byte_value'
            assert metadata['string_attr'] == 'string_value'
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestGetChannelInfo:
    """Tests for get_channel_info function."""

    def test_returns_labels_and_count(self, sample_h5_file):
        """Test channel info returns labels and count."""
        info = get_channel_info(sample_h5_file, 'EEG')

        assert info['labels'] == [f'EEG_{i+1}' for i in range(8)]
        assert info['count'] == 8

    def test_returns_sample_rate_if_present(self, sample_h5_file):
        """Test sample_rate is included when present in attributes."""
        info = get_channel_info(sample_h5_file, 'EEG')

        assert 'sample_rate' in info
        assert info['sample_rate'] == 500.0

    def test_raises_key_error_for_missing_dataset(self, sample_h5_file):
        """Test KeyError is raised for non-existent dataset."""
        with pytest.raises(KeyError) as exc_info:
            get_channel_info(sample_h5_file, 'NonExistent')

        assert 'NonExistent' in str(exc_info.value)

    def test_returns_n_samples(self, sample_h5_file):
        """Test n_samples is included in channel info."""
        info = get_channel_info(sample_h5_file, 'EEG')

        assert 'n_samples' in info
        assert info['n_samples'] == 100


class TestLoadEvents:
    """Tests for load_events function."""

    @pytest.fixture
    def h5_with_events(self):
        """Create H5 file with event data."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_path = f.name

        event_dtype = np.dtype([
            ('Event_Type', 'S50'),
            ('Timestamp', 'f8'),
            ('Extra_Vars', 'S200')
        ])
        events = np.array([
            (b'left_hand', 1.0, b'{"trial": 1, "correct": true}'),
            (b'right_hand', 2.0, b'{"trial": 2, "correct": false}'),
            (b'rest', 3.0, b''),
        ], dtype=event_dtype)

        with h5py.File(temp_path, 'w') as h5f:
            h5f.create_dataset('Event', data=events)

        yield temp_path
        Path(temp_path).unlink(missing_ok=True)

    def test_loads_and_cleans_events(self, h5_with_events):
        """Test events are loaded and cleaned."""
        df = load_events(h5_with_events)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        # Bytes should be decoded
        assert df['event_type'].iloc[0] == 'left_hand'

    def test_parses_extra_vars_json(self, h5_with_events):
        """Test extra_vars JSON is parsed and expanded."""
        df = load_events(h5_with_events)

        # JSON should be parsed into extra_ prefixed columns
        assert 'extra_trial' in df.columns
        assert 'extra_correct' in df.columns
        assert df['extra_trial'].iloc[0] == 1
        assert df['extra_correct'].iloc[0] is True

    def test_normalizes_columns_to_lowercase(self, h5_with_events):
        """Test column names are normalized to lowercase."""
        df = load_events(h5_with_events)

        # PascalCase should be converted to lowercase
        assert 'event_type' in df.columns
        assert 'timestamp' in df.columns
        assert 'Event_Type' not in df.columns

    def test_handles_empty_extra_vars(self, h5_with_events):
        """Test empty extra_vars are handled gracefully."""
        df = load_events(h5_with_events)

        # Third event has empty extra_vars - should produce empty dict
        # NaN or empty values for that row
        assert len(df) == 3


class TestChannelLabelParsing:
    """Tests for _get_channel_labels helper function edge cases."""

    def test_handles_bytes_channel_labels(self):
        """Test channel labels stored as bytes are decoded."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_path = f.name

        try:
            with h5py.File(temp_path, 'w') as h5f:
                data = np.random.randn(10, 3)
                ds = h5f.create_dataset('Test', data=data)
                ds.attrs['channel_labels'] = [b'Ch1', b'Ch2', b'Ch3']

            df = load_dataset(temp_path, 'Test')
            assert list(df.columns) == ['Ch1', 'Ch2', 'Ch3']
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_handles_string_representation_of_list(self):
        """Test channel labels stored as string representation of list."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_path = f.name

        try:
            with h5py.File(temp_path, 'w') as h5f:
                data = np.random.randn(10, 3)
                ds = h5f.create_dataset('Test', data=data)
                ds.attrs['channel_labels'] = "['A', 'B', 'C']"

            df = load_dataset(temp_path, 'Test')
            assert list(df.columns) == ['A', 'B', 'C']
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_extends_short_label_list(self):
        """Test fallback labels are added when list is too short."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_path = f.name

        try:
            with h5py.File(temp_path, 'w') as h5f:
                data = np.random.randn(10, 5)
                ds = h5f.create_dataset('Test', data=data)
                ds.attrs['channel_labels'] = ['A', 'B']  # Only 2, need 5

            df = load_dataset(temp_path, 'Test')
            assert list(df.columns) == ['A', 'B', 'ch_2', 'ch_3', 'ch_4']
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_truncates_long_label_list(self):
        """Test extra labels are truncated when list is too long."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_path = f.name

        try:
            with h5py.File(temp_path, 'w') as h5f:
                data = np.random.randn(10, 2)
                ds = h5f.create_dataset('Test', data=data)
                ds.attrs['channel_labels'] = ['A', 'B', 'C', 'D', 'E']  # 5, need 2

            df = load_dataset(temp_path, 'Test')
            assert list(df.columns) == ['A', 'B']
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestSaveDataset:
    """Tests for save_dataset function."""

    def test_saves_numeric_dataframe(self, empty_h5_file):
        """Test saving a numeric DataFrame creates dataset with correct data."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [4.0, 5.0, 6.0],
        })

        save_dataset(empty_h5_file, 'TestData', df)

        # Verify data was saved correctly
        with h5py.File(empty_h5_file, 'r') as h5f:
            assert 'TestData' in h5f
            saved_data = h5f['TestData'][()]
            np.testing.assert_array_equal(saved_data, df.values)

    def test_stores_channel_labels_as_attribute(self, empty_h5_file):
        """Test channel labels from DataFrame columns are stored as attribute."""
        df = pd.DataFrame({
            'EEG_1': [1.0, 2.0],
            'EEG_2': [3.0, 4.0],
            'EEG_3': [5.0, 6.0],
        })

        save_dataset(empty_h5_file, 'EEG', df)

        with h5py.File(empty_h5_file, 'r') as h5f:
            labels = list(h5f['EEG'].attrs['channel_labels'])
            assert labels == ['EEG_1', 'EEG_2', 'EEG_3']

    def test_raises_if_dataset_exists_without_overwrite(self, sample_h5_file):
        """Test ValueError when dataset exists and overwrite=False."""
        df = pd.DataFrame({'A': [1.0, 2.0]})

        with pytest.raises(ValueError) as exc_info:
            save_dataset(sample_h5_file, 'EEG', df, overwrite=False)

        assert "exists" in str(exc_info.value)
        assert "overwrite=True" in str(exc_info.value)

    def test_overwrites_existing_dataset(self, sample_h5_file):
        """Test dataset is replaced when overwrite=True."""
        new_data = pd.DataFrame({
            'New_1': [100.0, 200.0],
            'New_2': [300.0, 400.0],
        })

        save_dataset(sample_h5_file, 'EEG', new_data, overwrite=True)

        with h5py.File(sample_h5_file, 'r') as h5f:
            assert h5f['EEG'].shape == (2, 2)
            labels = list(h5f['EEG'].attrs['channel_labels'])
            assert labels == ['New_1', 'New_2']

    def test_stores_custom_attributes(self, empty_h5_file):
        """Test custom attributes are stored on dataset."""
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0]})

        save_dataset(
            empty_h5_file, 'Data', df,
            sample_rate=500.0,
            description='test data',
            version=2
        )

        with h5py.File(empty_h5_file, 'r') as h5f:
            attrs = dict(h5f['Data'].attrs)
            assert attrs['sample_rate'] == 500.0
            assert attrs['description'] == 'test data'
            assert attrs['version'] == 2

    def test_saves_structured_data_with_object_columns(self, empty_h5_file):
        """Test DataFrame with string columns is saved as structured array."""
        df = pd.DataFrame({
            'event_type': ['stimulus', 'response', 'stimulus'],
            'timestamp': [1.0, 1.5, 2.0],
            'label': ['A', 'B', 'C'],
        })

        save_dataset(empty_h5_file, 'Events', df)

        with h5py.File(empty_h5_file, 'r') as h5f:
            assert 'Events' in h5f
            saved = h5f['Events'][()]
            # Structured array has named fields
            assert saved.dtype.names is not None
            assert 'event_type' in saved.dtype.names
            assert 'timestamp' in saved.dtype.names
            assert 'label' in saved.dtype.names
            # Data should be retrievable
            assert len(saved) == 3
