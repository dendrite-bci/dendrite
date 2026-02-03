"""Tests for MOAABLoader - regression tests for bug fixes."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestLoadContinuous:
    """Tests for MOAABLoader.load_continuous method."""

    def test_load_continuous_no_name_error(self):
        """Regression test: load_continuous should not raise NameError for events_array.

        Bug: Line 469 references 'events_array' which is only defined inside
        _extract_events_from_raw(), causing NameError when events are found.
        """
        from dendrite.auxiliary.ml_workbench.datasets.moabb_loader import MOAABLoader
        from dendrite.auxiliary.ml_workbench.datasets.config import DatasetConfig

        # Create minimal config
        config = DatasetConfig(
            name="test",
            source_type="moabb",
            moabb_dataset="BNCI2014_001",
            moabb_paradigm="MotorImagery",
            events={"left_hand": 0, "right_hand": 1},
            subjects=[1],
        )

        # Create loader and mock its internal state to avoid network calls
        loader = MOAABLoader(config)

        # Mock internal state (lazy-loaded properties)
        loader._dataset = Mock()
        loader._paradigm = Mock()
        loader._paradigm.filters = [[8, 30]]

        # Create mock raw object
        mock_raw = MagicMock()
        mock_raw.get_data.return_value = np.random.randn(22, 1000)
        mock_raw.info = {"sfreq": 250}
        mock_raw.ch_names = [f"EEG{i}" for i in range(22)]

        # Mock the methods that load_continuous calls
        loader.load_raw_concatenated = Mock(return_value=mock_raw)
        loader._apply_paradigm_preprocessing = Mock(return_value=mock_raw)

        # Return events that trigger the "else" branch (non-empty events)
        # This is where the NameError occurs
        loader._extract_events_from_raw = Mock(
            return_value=([100, 200, 300], [0, 1, 0])
        )

        # This should NOT raise NameError
        data, event_times, event_labels = loader.load_continuous(subject_id=1)

        # Basic assertions to verify it worked
        assert data.shape[0] == 22  # Has channels
        assert len(event_times) == 3
        assert len(event_labels) == 3
