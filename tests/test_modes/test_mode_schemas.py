"""Tests for Pydantic validation schemas in mode_schemas.py."""

import pytest
from pydantic import ValidationError

from dendrite.processing.modes.mode_schemas import (
    AsynchronousInstanceConfig,
    BaseModeInstanceConfig,
    NeurofeedbackInstanceConfig,
    SynchronousInstanceConfig,
    _get_system_shapes,
    validate_mode_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_FIELDS = {
    "name": "test_instance",
    "mode": "synchronous",
    "channel_selection": {"eeg": [0, 1, 2, 3]},
}

SYNC_FIELDS = {
    **BASE_FIELDS,
    "event_mapping": {1: "left", 2: "right"},
}

ASYNC_FIELDS = {
    "name": "async_test",
    "mode": "asynchronous",
    "channel_selection": {"eeg": [0, 1, 2, 3]},
    "decoder_source": "database",
    "decoder_config": {
        "decoder_type": "Decoder",
        "decoder_path": "/fake/model.json",
        "model_config": {"model_type": "EEGNet", "num_classes": 2},
    },
}

NFB_FIELDS = {
    "name": "nfb_test",
    "mode": "neurofeedback",
    "channel_selection": {"eeg": [0, 1, 2, 3]},
    "feature_config": {
        "target_bands": {"alpha": [8.0, 12.0]},
        "use_relative_power": True,
    },
}


# ---------------------------------------------------------------------------
# BaseModeInstanceConfig
# ---------------------------------------------------------------------------


class TestBaseModeInstanceConfig:
    def test_valid_minimal(self):
        cfg = BaseModeInstanceConfig(**BASE_FIELDS)
        assert cfg.name == "test_instance"
        assert cfg.mode == "synchronous"

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            BaseModeInstanceConfig(**{**BASE_FIELDS, "name": ""})

    def test_invalid_mode_raises(self):
        with pytest.raises(ValidationError):
            BaseModeInstanceConfig(**{**BASE_FIELDS, "mode": "bogus"})

    def test_mode_case_insensitive(self):
        cfg = BaseModeInstanceConfig(**{**BASE_FIELDS, "mode": "Synchronous"})
        assert cfg.mode == "synchronous"

    def test_channel_selection_empty_allowed(self):
        """Empty channel selection is allowed at creation time; preflight validates later."""
        cfg = BaseModeInstanceConfig(**{**BASE_FIELDS, "channel_selection": {}})
        assert cfg.channel_selection == {}

    def test_channel_selection_no_channels_raises(self):
        with pytest.raises(ValidationError):
            BaseModeInstanceConfig(**{**BASE_FIELDS, "channel_selection": {"eeg": []}})

    def test_channel_selection_multi_modality_raises(self):
        with pytest.raises(ValidationError):
            BaseModeInstanceConfig(
                **{**BASE_FIELDS, "channel_selection": {"eeg": [0], "emg": [1]}}
            )

    def test_preprocessing_aligns_to_modality(self):
        cfg = BaseModeInstanceConfig(**{**BASE_FIELDS, "channel_selection": {"emg": [0, 1]}})
        assert "emg" in cfg.mode_preprocessing

    def test_channel_selection_keys_lowercased(self):
        """Schema is the single source of truth for modality-key normalization."""
        cfg = BaseModeInstanceConfig(
            **{**BASE_FIELDS, "channel_selection": {"EEG": [0, 1, 2]}}
        )
        assert cfg.channel_selection == {"eeg": [0, 1, 2]}


# ---------------------------------------------------------------------------
# SynchronousInstanceConfig
# ---------------------------------------------------------------------------


class TestSynchronousInstanceConfig:
    def test_valid_config(self):
        cfg = SynchronousInstanceConfig(**SYNC_FIELDS)
        assert cfg.name == "test_instance"
        assert cfg.event_mapping == {1: "left", 2: "right"}

    def test_event_mapping_string_keys_converted(self):
        fields = {**SYNC_FIELDS, "event_mapping": {"1": "left", "2": "right"}}
        cfg = SynchronousInstanceConfig(**fields)
        assert cfg.event_mapping == {1: "left", 2: "right"}

    def test_event_mapping_empty_gets_defaults(self):
        """Empty event mapping gets sensible defaults for new instances."""
        cfg = SynchronousInstanceConfig(**{**SYNC_FIELDS, "event_mapping": {}})
        assert cfg.event_mapping == {1: "Left", 2: "Right"}

    def test_event_mapping_one_class_raises(self):
        with pytest.raises(ValidationError):
            SynchronousInstanceConfig(**{**SYNC_FIELDS, "event_mapping": {1: "left"}})

    def test_event_mapping_non_int_key_raises(self):
        with pytest.raises(ValidationError):
            SynchronousInstanceConfig(
                **{**SYNC_FIELDS, "event_mapping": {"abc": "left", "def": "right"}}
            )

    def test_event_mapping_empty_label_raises(self):
        with pytest.raises(ValidationError):
            SynchronousInstanceConfig(
                **{**SYNC_FIELDS, "event_mapping": {1: "", 2: "right"}}
            )

    def test_epoch_tmax_must_be_after_tmin(self):
        with pytest.raises(ValidationError):
            SynchronousInstanceConfig(
                **{**SYNC_FIELDS, "epoch_tmin": 1.0, "epoch_tmax": 0.5}
            )

    def test_training_interval_min_one(self):
        with pytest.raises(ValidationError):
            SynchronousInstanceConfig(**{**SYNC_FIELDS, "training_interval": 0})

    def test_decoder_config_defaults(self):
        cfg = SynchronousInstanceConfig(**SYNC_FIELDS)
        assert cfg.decoder_config["decoder_type"] == "Decoder"
        assert "model_config" in cfg.decoder_config


# ---------------------------------------------------------------------------
# AsynchronousInstanceConfig
# ---------------------------------------------------------------------------


class TestAsynchronousInstanceConfig:
    def test_valid_database_config(self):
        cfg = AsynchronousInstanceConfig(**ASYNC_FIELDS)
        assert cfg.decoder_source == "database"

    def test_database_missing_path_allowed(self):
        """Missing decoder path is allowed at creation time; preflight validates later."""
        fields = {
            **ASYNC_FIELDS,
            "decoder_config": {
                "decoder_type": "Decoder",
                "model_config": {"model_type": "EEGNet", "num_classes": 2},
            },
        }
        cfg = AsynchronousInstanceConfig(**fields)
        assert cfg.decoder_source == "database"

    def test_invalid_decoder_source_raises(self):
        with pytest.raises(ValidationError):
            AsynchronousInstanceConfig(**{**ASYNC_FIELDS, "decoder_source": "unknown"})

    def test_window_length_positive(self):
        with pytest.raises(ValidationError):
            AsynchronousInstanceConfig(**{**ASYNC_FIELDS, "window_length_sec": 0})

    def test_step_size_positive(self):
        with pytest.raises(ValidationError):
            AsynchronousInstanceConfig(**{**ASYNC_FIELDS, "step_size_ms": 0})

    def test_invalid_model_type_raises(self):
        fields = {
            **ASYNC_FIELDS,
            "decoder_config": {
                "decoder_type": "Decoder",
                "decoder_path": "/fake/model.json",
                "model_config": {"model_type": "NonExistentModel", "num_classes": 2},
            },
        }
        with pytest.raises(ValidationError):
            AsynchronousInstanceConfig(**fields)


# ---------------------------------------------------------------------------
# NeurofeedbackInstanceConfig
# ---------------------------------------------------------------------------


class TestNeurofeedbackInstanceConfig:
    def test_valid_config(self):
        cfg = NeurofeedbackInstanceConfig(**NFB_FIELDS)
        assert cfg.name == "nfb_test"

    def test_missing_bands_raises(self):
        fields = {**NFB_FIELDS, "feature_config": {"use_relative_power": True}}
        with pytest.raises(ValidationError):
            NeurofeedbackInstanceConfig(**fields)

    def test_band_wrong_length_raises(self):
        fields = {
            **NFB_FIELDS,
            "feature_config": {"target_bands": {"alpha": [8.0]}},
        }
        with pytest.raises(ValidationError):
            NeurofeedbackInstanceConfig(**fields)

    def test_band_low_ge_high_raises(self):
        fields = {
            **NFB_FIELDS,
            "feature_config": {"target_bands": {"alpha": [12.0, 8.0]}},
        }
        with pytest.raises(ValidationError):
            NeurofeedbackInstanceConfig(**fields)

    def test_cluster_mode_must_be_bool(self):
        fields = {
            **NFB_FIELDS,
            "feature_config": {
                "target_bands": {"alpha": [8.0, 12.0]},
                "use_cluster_mode": "yes",
            },
        }
        with pytest.raises(ValidationError):
            NeurofeedbackInstanceConfig(**fields)


# ---------------------------------------------------------------------------
# validate_mode_config
# ---------------------------------------------------------------------------


class TestValidateModeConfig:
    def test_valid_sync_returns_true(self):
        ok, errors, validated = validate_mode_config(SYNC_FIELDS)
        assert ok is True
        assert errors == []
        assert isinstance(validated, dict)

    def test_valid_async_returns_true(self):
        ok, errors, validated = validate_mode_config(ASYNC_FIELDS)
        assert ok is True
        assert errors == []

    def test_valid_nfb_returns_true(self):
        ok, errors, validated = validate_mode_config(NFB_FIELDS)
        assert ok is True
        assert errors == []

    def test_unknown_mode_returns_false(self):
        ok, errors, validated = validate_mode_config({"mode": "bogus", "name": "x"})
        assert ok is False
        assert len(errors) > 0
        assert validated is None

    def test_validation_error_returns_errors(self):
        # Invalid sync config (single event class)
        ok, errors, validated = validate_mode_config({**BASE_FIELDS, "event_mapping": {1: "only_one"}})
        assert ok is False
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# _get_system_shapes
# ---------------------------------------------------------------------------


class TestGetSystemShapes:
    def test_sync_calculates_from_offsets(self):
        config = {
            "mode": "synchronous",
            "channel_selection": {"eeg": [0, 1, 2, 3]},
            "epoch_tmin": 0.0,
            "epoch_tmax": 2.0,
        }
        shapes = _get_system_shapes(config, {"sample_rate": 250})
        assert shapes == {"eeg": [4, 500]}

    def test_async_uses_window_length(self):
        config = {
            "mode": "asynchronous",
            "channel_selection": {"eeg": [0, 1, 2]},
            "window_length_sec": 1.0,
        }
        shapes = _get_system_shapes(config, {"sample_rate": 500})
        assert shapes == {"eeg": [3, 500]}

    def test_async_with_decoder_path_omits_time(self):
        config = {
            "mode": "asynchronous",
            "channel_selection": {"eeg": [0, 1]},
            "decoder_config": {"decoder_path": "/some/path.json"},
        }
        shapes = _get_system_shapes(config, {"sample_rate": 250})
        assert shapes == {"eeg": [2]}

    def test_empty_channel_selection(self):
        config = {"mode": "synchronous", "channel_selection": {}}
        assert _get_system_shapes(config, {"sample_rate": 250}) == {}
