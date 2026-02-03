"""
Unit tests for processing_pipeline configuration setup.

Tests cover:
- Effective sample rate calculation (EEG-based)

Note: downsample_factor calculation is handled by DataProcessor, not pipeline.
"""

import pytest
from dendrite.processing.pipeline import _get_effective_sample_rate


class TestGetEffectiveSampleRate:
    """Test suite for _get_effective_sample_rate function."""

    def test_no_resampling(self):
        """Test when no target sample rate specified."""
        config = {
            'modality_preprocessing': {
                'eeg': {'num_channels': 32, 'sample_rate': 500, 'target_sample_rate': None}
            }
        }

        effective_rate = _get_effective_sample_rate(config, sample_rate=500)
        assert effective_rate == 500

    def test_with_resampling(self):
        """Test when target sample rate specified."""
        config = {
            'modality_preprocessing': {
                'eeg': {'num_channels': 32, 'sample_rate': 500, 'target_sample_rate': 250}
            }
        }

        effective_rate = _get_effective_sample_rate(config, sample_rate=500)
        assert effective_rate == 250

    def test_multi_stream_uses_eeg(self):
        """Test that EEG's effective rate is used regardless of other modalities."""
        config = {
            'modality_preprocessing': {
                'eeg': {'num_channels': 32, 'sample_rate': 500, 'target_sample_rate': 250},
                'emg': {'num_channels': 8, 'sample_rate': 2000, 'target_sample_rate': 500},
            }
        }

        effective_rate = _get_effective_sample_rate(config, sample_rate=500)
        # Should use EEG's effective rate
        assert effective_rate == 250

    def test_no_eeg_modality_uses_system_rate(self):
        """Test fallback to system rate when no EEG configured."""
        config = {
            'modality_preprocessing': {
                'emg': {'num_channels': 8, 'sample_rate': 2000, 'target_sample_rate': 500}
            }
        }

        effective_rate = _get_effective_sample_rate(config, sample_rate=500)
        # No EEG, uses system rate
        assert effective_rate == 500

    def test_empty_modality_preprocessing(self):
        """Test with empty modality configs."""
        config = {
            'modality_preprocessing': {}
        }

        effective_rate = _get_effective_sample_rate(config, sample_rate=500)
        assert effective_rate == 500

    def test_missing_modality_preprocessing(self):
        """Test with missing modality_preprocessing key."""
        config = {}

        effective_rate = _get_effective_sample_rate(config, sample_rate=500)
        assert effective_rate == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
