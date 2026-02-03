"""Tests that mode outputs conform to expected schema for LSL serialization.

These tests ensure mode outputs are JSON-serializable dicts, preventing
the ModeOutputPacket dataclass serialization bug from recurring.
"""

import json
from dataclasses import is_dataclass
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from dendrite.processing.modes.base_mode import ModeOutputPacket


def assert_valid_mode_output(output: Any) -> None:
    """Validate mode output conforms to expected schema for LSL streaming.

    Args:
        output: The output from a mode's queue

    Raises:
        AssertionError: If output doesn't conform to schema
    """
    # Must be dict, not dataclass
    assert isinstance(output, dict), f"Output must be dict, got {type(output).__name__}"
    assert not is_dataclass(output), "Output must not be a dataclass instance"

    # Required keys
    required = {"type", "mode_name", "mode_type", "data"}
    missing = required - output.keys()
    assert not missing, f"Missing required keys: {missing}"

    # Must be JSON-serializable
    try:
        json.dumps(output)
    except (TypeError, ValueError) as e:
        raise AssertionError(f"Output not JSON-serializable: {e}")


class TestModeOutputSchema:
    """Test suite for mode output schema validation."""

    def test_valid_dict_output_passes(self):
        """Valid dict output passes validation."""
        output = {
            "type": "prediction",
            "mode_name": "test_mode",
            "mode_type": "synchronous",
            "data": {"prediction": 1, "confidence": 0.9},
            "data_timestamp": 1234567890.123,
        }
        assert_valid_mode_output(output)  # Should not raise

    def test_dataclass_output_fails(self):
        """Dataclass output fails validation."""
        output = ModeOutputPacket(
            type="prediction",
            mode_name="test_mode",
            mode_type="synchronous",
            data={"prediction": 1},
            data_timestamp=None,
        )
        with pytest.raises(AssertionError, match="must be dict"):
            assert_valid_mode_output(output)

    def test_missing_required_keys_fails(self):
        """Missing required keys fails validation."""
        output = {
            "type": "prediction",
            "mode_name": "test_mode",
            # Missing 'mode_type' and 'data'
        }
        with pytest.raises(AssertionError, match="Missing required keys"):
            assert_valid_mode_output(output)

    def test_non_serializable_data_fails(self):
        """Non-JSON-serializable data fails validation."""
        output = {
            "type": "prediction",
            "mode_name": "test_mode",
            "mode_type": "synchronous",
            "data": {"callback": lambda x: x},  # Functions aren't serializable
        }
        with pytest.raises(AssertionError, match="not JSON-serializable"):
            assert_valid_mode_output(output)

    def test_numpy_arrays_fail_without_conversion(self):
        """Raw numpy arrays aren't JSON-serializable."""
        output = {
            "type": "erp",
            "mode_name": "sync_mode",
            "mode_type": "synchronous",
            "data": {"eeg_data": np.array([1, 2, 3])},
        }
        with pytest.raises(AssertionError, match="not JSON-serializable"):
            assert_valid_mode_output(output)

    def test_numpy_arrays_converted_to_list_passes(self):
        """Numpy arrays converted to lists pass validation."""
        output = {
            "type": "erp",
            "mode_name": "sync_mode",
            "mode_type": "synchronous",
            "data": {"eeg_data": [1, 2, 3]},  # Converted from np.array
        }
        assert_valid_mode_output(output)  # Should not raise

    def test_none_data_timestamp_is_valid(self):
        """None data_timestamp is valid (optional field)."""
        output = {
            "type": "prediction",
            "mode_name": "test_mode",
            "mode_type": "synchronous",
            "data": {},
            "data_timestamp": None,
        }
        assert_valid_mode_output(output)  # Should not raise

    def test_nested_dict_data_is_valid(self):
        """Nested dict structures are valid."""
        output = {
            "type": "neurofeedback",
            "mode_name": "nf_mode",
            "mode_type": "neurofeedback",
            "data": {
                "channel_powers": {
                    "C3": {"alpha": 0.15, "beta": 0.08},
                    "C4": {"alpha": 0.12, "beta": 0.10},
                },
                "target_bands": {"alpha": [8, 12], "beta": [13, 30]},
            },
            "data_timestamp": 1234567890.123,
        }
        assert_valid_mode_output(output)  # Should not raise
