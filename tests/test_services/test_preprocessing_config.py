"""Tests for preprocessing config schemas and validation flow."""

import pytest
from pydantic import ValidationError

from dendrite.processing.preprocessing.preprocessing_schemas import (
    PreprocessingConfig,
    ModalityPreprocessing,
)


# --- ModalityPreprocessing (user-configurable, optional runtime context) ---


def test_signal_filter_accepts_filter_params_only():
    """ModalityPreprocessing should work with just filter parameters."""
    config = ModalityPreprocessing(lowcut=0.5, highcut=50.0)
    assert config.lowcut == 0.5
    assert config.highcut == 50.0


def test_signal_filter_accepts_empty():
    """ModalityPreprocessing should accept no arguments (all fields optional)."""
    config = ModalityPreprocessing()
    assert config.lowcut is None
    assert config.highcut is None


def test_signal_filter_rejects_bad_filters():
    """highcut must be greater than lowcut."""
    with pytest.raises(ValidationError, match="greater than lowcut"):
        ModalityPreprocessing(lowcut=50.0, highcut=10.0)


def test_signal_filter_rejects_extra_fields():
    """extra='forbid' catches typos in config fields."""
    with pytest.raises(ValidationError):
        ModalityPreprocessing(lowcut=0.5, highcut=50.0, quality_monitoring=True)


# --- ModalityPreprocessing with runtime context (num_channels, sample_rate) ---


def test_signal_filter_validates_nyquist_when_sample_rate_present():
    """Nyquist check: highcut must be < sample_rate / 2."""
    with pytest.raises(ValidationError, match="Nyquist"):
        ModalityPreprocessing(num_channels=8, sample_rate=100.0, highcut=60.0)


def test_signal_filter_accepts_valid_with_runtime():
    """Valid config with all fields including runtime context."""
    config = ModalityPreprocessing(
        num_channels=8, sample_rate=500.0, lowcut=0.5, highcut=50.0
    )
    assert config.num_channels == 8
    assert config.sample_rate == 500.0


def test_signal_filter_no_nyquist_without_sample_rate():
    """Without sample_rate, Nyquist check is skipped."""
    config = ModalityPreprocessing(lowcut=0.5, highcut=200.0)
    assert config.highcut == 200.0


def test_signal_filter_inherits_filter_validation():
    """highcut > lowcut check works with runtime context too."""
    with pytest.raises(ValidationError, match="greater than lowcut"):
        ModalityPreprocessing(num_channels=8, sample_rate=500.0, lowcut=50.0, highcut=10.0)


# --- ModalityPreprocessing.from_user_config ---


def test_from_user_config_apply_rereferencing():
    """apply_rereferencing passes through from_user_config."""
    sf = ModalityPreprocessing.from_user_config(
        {"lowcut": 1.0, "highcut": 40.0, "apply_rereferencing": True}, n_channels=8, sample_rate=500.0,
    )
    assert sf.apply_rereferencing is True
    assert sf.num_channels == 8
    assert sf.sample_rate == 500.0


def test_from_user_config_computes_downsample():
    """target_sample_rate is converted to downsample_factor."""
    sf = ModalityPreprocessing.from_user_config(
        {"target_sample_rate": 250}, n_channels=8, sample_rate=500.0,
    )
    assert sf.downsample_factor == 2


def test_from_user_config_skips_downsample_when_not_divisible():
    """No downsample if sample_rate is not evenly divisible."""
    sf = ModalityPreprocessing.from_user_config(
        {"target_sample_rate": 300}, n_channels=8, sample_rate=500.0,
    )
    assert sf.downsample_factor is None


# --- PreprocessingConfig (full config, uses ModalityPreprocessing) ---


def test_preprocessing_config_builds_without_num_channels():
    """PreprocessingConfig should accept ModalityPreprocessing (no num_channels)."""
    config = PreprocessingConfig(
        modality_preprocessing={
            "eeg": {"lowcut": 0.5, "highcut": 50.0, "apply_rereferencing": True},
            "emg": {"lowcut": 20.0, "highcut": 200.0},
        },
    )
    assert "eeg" in config.modality_preprocessing
    assert "emg" in config.modality_preprocessing


def test_preprocessing_config_default_empty():
    """PreprocessingConfig with defaults."""
    config = PreprocessingConfig()
    assert config.modality_preprocessing == {}
