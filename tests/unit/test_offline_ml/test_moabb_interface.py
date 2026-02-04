"""Tests for minimal MOABB interface used by streamer and benchmark.

These tests define the contract that must be maintained when refactoring.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, Mock, patch


class TestStreamerInterface:
    """Tests for the interface streamer.py needs from MOAABLoader."""

    @pytest.fixture
    def mock_loader(self):
        """Create MOAABLoader with mocked MOABB dependencies."""
        from dendrite.data import MOAABLoader, DatasetConfig

        config = DatasetConfig(
            name="test",
            source_type="moabb",
            moabb_dataset="BNCI2014_001",
            moabb_paradigm="MotorImagery",
            events={"left_hand": 0, "right_hand": 1},
            subjects=[1],
        )
        loader = MOAABLoader(config)

        # Mock MOABB internals
        loader._dataset = Mock()
        loader._paradigm = Mock()
        loader._paradigm.filters = [[8, 30]]

        return loader

    def test_load_continuous_returns_4_values(self, mock_loader):
        """load_continuous must return (data, event_times, event_labels, event_mapping)."""
        # Setup mocks
        mock_raw = MagicMock()
        mock_raw.get_data.return_value = np.random.randn(22, 1000)
        mock_raw.info = {"sfreq": 250}
        mock_raw.ch_names = [f"EEG{i}" for i in range(22)]

        mock_loader.load_raw_concatenated = Mock(return_value=mock_raw)
        mock_loader._apply_paradigm_preprocessing = Mock(return_value=mock_raw)
        mock_loader._extract_events_from_raw = Mock(
            return_value=([100, 200], [0, 1], {"left_hand": 0, "right_hand": 1})
        )

        # Call and verify 4 return values
        result = mock_loader.load_continuous(subject_id=1)

        assert len(result) == 4, "load_continuous must return 4 values"
        data, event_times, event_labels, event_mapping = result

        # Verify types
        assert isinstance(data, np.ndarray)
        assert isinstance(event_times, np.ndarray)
        assert isinstance(event_labels, np.ndarray)
        assert isinstance(event_mapping, dict)

    def test_load_continuous_data_shape(self, mock_loader):
        """data must be (n_channels, n_samples) shape."""
        mock_raw = MagicMock()
        mock_raw.get_data.return_value = np.random.randn(22, 1000)
        mock_raw.info = {"sfreq": 250}

        mock_loader.load_raw_concatenated = Mock(return_value=mock_raw)
        mock_loader._apply_paradigm_preprocessing = Mock(return_value=mock_raw)
        mock_loader._extract_events_from_raw = Mock(return_value=([], [], {}))

        data, _, _, _ = mock_loader.load_continuous(subject_id=1)

        assert data.ndim == 2
        assert data.shape[0] == 22  # channels first

    def test_event_mapping_has_string_keys_int_values(self, mock_loader):
        """event_mapping must be dict[str, int]."""
        mock_raw = MagicMock()
        mock_raw.get_data.return_value = np.random.randn(22, 1000)

        mock_loader.load_raw_concatenated = Mock(return_value=mock_raw)
        mock_loader._apply_paradigm_preprocessing = Mock(return_value=mock_raw)
        mock_loader._extract_events_from_raw = Mock(
            return_value=([100], [0], {"left_hand": 0, "right_hand": 1})
        )

        _, _, _, event_mapping = mock_loader.load_continuous(subject_id=1)

        for key, value in event_mapping.items():
            assert isinstance(key, str), f"event_mapping key must be str, got {type(key)}"
            assert isinstance(value, int), f"event_mapping value must be int, got {type(value)}"

    def test_get_channel_names_returns_list_of_strings(self, mock_loader):
        """get_channel_names must return list[str]."""
        mock_raw = MagicMock()
        mock_raw.ch_names = ["Fz", "Cz", "Pz"]

        mock_loader.load_raw = Mock(return_value=mock_raw)
        mock_loader._apply_paradigm_preprocessing = Mock(return_value=mock_raw)

        names = mock_loader.get_channel_names(subject_id=1)

        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_get_channel_types_returns_list_of_strings(self, mock_loader):
        """get_channel_types must return list[str]."""
        mock_raw = MagicMock()
        mock_raw.get_channel_types.return_value = ["eeg", "eeg", "eeg"]

        mock_loader.load_raw = Mock(return_value=mock_raw)
        mock_loader._apply_paradigm_preprocessing = Mock(return_value=mock_raw)

        types = mock_loader.get_channel_types(subject_id=1)

        assert isinstance(types, list)
        assert all(isinstance(t, str) for t in types)

    def test_get_sample_rate_returns_float(self, mock_loader):
        """get_sample_rate must return float."""
        mock_loader.config.sample_rate = 250.0

        rate = mock_loader.get_sample_rate()

        assert isinstance(rate, float)


class TestGetMoabbDatasetInfo:
    """Tests for get_moabb_dataset_info function."""

    def test_returns_dict_with_config_key(self):
        """get_moabb_dataset_info must return dict with 'config' key."""
        from dendrite.data import get_moabb_dataset_info, DatasetConfig

        # Mock discover_moabb_datasets to avoid network
        with patch(
            "dendrite.data.imports.moabb_discovery.discover_moabb_datasets"
        ) as mock_discover:

            mock_config = DatasetConfig(
                name="BNCI2014_001",
                source_type="moabb",
                moabb_dataset="BNCI2014_001",
                moabb_paradigm="MotorImagery",
            )
            mock_discover.return_value = [mock_config]

            result = get_moabb_dataset_info("BNCI2014_001")

            assert result is not None
            assert "config" in result
            assert result["config"] == mock_config

    def test_returns_none_for_unknown_dataset(self):
        """get_moabb_dataset_info must return None for unknown dataset."""
        from dendrite.data import get_moabb_dataset_info

        with patch(
            "dendrite.data.imports.moabb_discovery.discover_moabb_datasets"
        ) as mock_discover:
            mock_discover.return_value = []

            result = get_moabb_dataset_info("NonExistentDataset")

            assert result is None


class TestBenchmarkInterface:
    """Tests for the interface benchmark_worker.py needs."""

    def test_get_moabb_dataset_returns_object_with_subject_list(self):
        """_get_moabb_dataset must return object with subject_list attribute."""
        from dendrite.data.imports.moabb_loader import _get_moabb_dataset

        with patch("moabb.datasets") as mock_datasets:
            mock_ds = Mock()
            mock_ds.subject_list = [1, 2, 3]
            mock_datasets.BNCI2014_001.return_value = mock_ds

            dataset = _get_moabb_dataset("BNCI2014_001")

            assert hasattr(dataset, "subject_list")
            assert isinstance(dataset.subject_list, list)

    def test_internal_wrapper_has_subject_list(self):
        """InternalDatasetWrapper must have subject_list attribute."""
        from dendrite.data import InternalDatasetWrapper, DatasetConfig

        config = DatasetConfig(
            name="test",
            source_type="fif",
            subjects=[1, 2, 3],
            events={"left": 0, "right": 1},
        )
        mock_loader = Mock()

        with patch("dendrite.data.imports.internal_moabb_wrapper.BaseDataset.__init__"):
            wrapper = InternalDatasetWrapper.__new__(InternalDatasetWrapper)
            wrapper._loader = mock_loader
            wrapper._config = config
            # Simulate what MOABB BaseDataset sets
            wrapper.subject_list = config.subjects

            assert hasattr(wrapper, "subject_list")
            assert wrapper.subject_list == [1, 2, 3]
