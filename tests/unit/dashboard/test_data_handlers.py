"""Tests for Dashboard DataBufferManager."""

import pytest

from dendrite.auxiliary.dashboard.backend.data_handlers import DataBufferManager


class TestDataBufferManager:
    """Test suite for DataBufferManager class."""

    def test_initialize_uses_payload_labels(self):
        """Test that initialization uses channel labels from payload."""
        manager = DataBufferManager(sample_rate=500)

        # Initialize with sample that has channel labels in payload
        sample_data = {
            'data': {
                'eeg': [1.0, 2.0, 3.0],
                'emg': [0.5]
            },
            'channel_labels': {
                'eeg': ['C3', 'C4', 'Cz'],
                'emg': ['EMG1']
            },
            'sample_rate': 500
        }

        result = manager.initialize_from_raw_data(sample_data)

        assert result is True
        assert manager.initialized is True
        assert manager.eeg_channel_labels == ['C3', 'C4', 'Cz']

    def test_initialize_generates_default_labels_when_none_provided(self):
        """Test that default labels are generated when no channel labels in payload."""
        manager = DataBufferManager(sample_rate=500)

        sample_data = {
            'data': {
                'eeg': [1.0, 2.0, 3.0],
            },
            'sample_rate': 500
        }

        result = manager.initialize_from_raw_data(sample_data)

        assert result is True
        assert manager.eeg_channel_labels == ['EEG_1', 'EEG_2', 'EEG_3']

    def test_clear_all_buffers_resets_state(self):
        """Test that clear_all_buffers resets all state."""
        manager = DataBufferManager(sample_rate=500)
        manager.initialized = True
        manager.eeg_channel_labels = ['C3', 'C4']

        manager.clear_all_buffers()

        assert manager.initialized is False
        assert manager.eeg_channel_labels == []
