"""
Behavior-focused unit tests for BMI mode utilities.

This module provides unit tests for the utilities in BMI.modes.mode_utils,
focusing on public behavior and integration testing rather than
internal implementation details.

Tests cover:
- Buffer behavior and extraction methods
- Configuration utilities (event mapping, decoder config)
- Shared model path utilities
- Output queue management and data structures

Follows red-green-refactor principles by testing what utilities do,
not how they do it internally.
"""

import sys
import os
import pytest
import numpy as np
import time
import logging
from unittest.mock import Mock, MagicMock
from collections import deque
from typing import Dict, List, Any

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import current utilities
from dendrite.processing.modes.mode_utils import (
    Buffer,
    extract_event_mapping,
    get_shared_model_path
)


class TestConfigurationUtilities:
    """Test suite for configuration utility functions."""
    
    def test_extract_event_mapping_with_valid_config(self):
        """Test event mapping extraction from valid configuration."""
        instance_config = {
            'event_mapping': {1: 'left_hand', 2: 'right_hand', 3: 'rest'}
        }

        result = extract_event_mapping(instance_config)

        expected = {1: 'left_hand', 2: 'right_hand', 3: 'rest'}
        assert result == expected

    def test_extract_event_mapping_with_empty_config(self):
        """Test event mapping extraction with empty or missing config."""
        # Empty dict
        result = extract_event_mapping({'event_mapping': {}})
        assert result == {}

        # Missing key
        result = extract_event_mapping({})
        assert result == {}

    def test_extract_event_mapping_returns_direct_value(self):
        """Test event mapping returns the dict directly."""
        instance_config = {
            'event_mapping': {1: 'valid', 3: 'also_valid'}
        }

        result = extract_event_mapping(instance_config)

        # Should return the dict directly
        expected = {1: 'valid', 3: 'also_valid'}
        assert result == expected

    def test_extract_event_mapping_converts_string_keys_to_int(self):
        """Test JSON-deserialized configs with string keys are converted to int."""
        # JSON deserializes dict keys as strings
        instance_config = {
            'event_mapping': {'1': 'left_hand', '2': 'right_hand', '3': 'rest'}
        }

        result = extract_event_mapping(instance_config)

        # Keys should be integers, not strings
        assert result == {1: 'left_hand', 2: 'right_hand', 3: 'rest'}
        assert all(isinstance(k, int) for k in result.keys())


class TestSharedModelPath:
    """Test suite for shared model path utility."""

    def test_shared_model_path_without_file_identifier(self):
        """Test path generation without file identifier."""
        path = get_shared_model_path("sync_mode_1")
        assert path == "shared/sync_mode_1_latest"

    def test_shared_model_path_with_file_identifier(self):
        """Test path generation with file identifier."""
        path = get_shared_model_path("sync_mode_1", "exp_001")
        assert path == "shared/sync_mode_1_exp_001_latest"


class TestBufferExtraction:
    """Test suite for Buffer extraction methods."""
    
    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return Mock(spec=logging.Logger)
    
    @pytest.fixture
    def new_buffer(self, mock_logger):
        """Create optimized buffer with lowercase modalities (as provided by base_mode)."""
        return Buffer(['eeg', 'emg'], 100, mock_logger)

    def test_optimized_extraction_methods_exist(self, new_buffer):
        """Test that optimized extraction methods are available."""
        # Test extraction methods exist
        assert hasattr(new_buffer, 'extract_window')
        assert hasattr(new_buffer, '_extract_slice')

        # Test extract_window returns None with empty buffer
        result = new_buffer.extract_window()
        assert result is None
    
    def test_markers_still_collected_but_excluded(self, new_buffer):
        """Test that markers are still collected but excluded from extraction."""
        # Add sample with markers (lowercase keys as per normalized modality convention)
        sample = {
            'eeg': np.array([[1.0]]),
            'markers': 42
        }
        new_buffer.add_sample(sample)

        # Test markers buffer has data (lowercase key)
        assert len(new_buffer.buffers['markers']) == 1
        assert new_buffer.buffers['markers'][0] == 42

        # Fill buffer for extraction test
        for i in range(new_buffer.buffer_size):
            sample = {'eeg': np.array([[i]]), 'markers': i}
            new_buffer.add_sample(sample)

        result = new_buffer.extract_window()

        # Test markers excluded from extraction result
        assert result is not None
        assert 'Markers' not in result
        assert 'markers' not in result

        # But markers buffer should still have data
        assert len(new_buffer.buffers['markers']) == new_buffer.buffer_size


if __name__ == '__main__':
    pytest.main([__file__])