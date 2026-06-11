"""Tests for online training: schema validation, async mode decoder reload, REST endpoint."""

from unittest.mock import MagicMock, patch

import pytest

from dendrite.processing.modes.mode_schemas import (
    AsynchronousInstanceConfig,
    validate_mode_config,
)


def _make_async_mode(**overrides):
    """Create an AsynchronousMode instance without running __init__."""
    from dendrite.processing.modes.asynchronous_mode import AsynchronousMode

    mode = AsynchronousMode.__new__(AsynchronousMode)
    mode.epoch_length_samples = 250
    mode.samples_per_prediction_step = 25
    mode.decoder_config = {}
    mode.decoder_source = "online"
    mode.channel_selection = {"eeg": [0, 1, 2, 3]}
    mode.decoder = None
    mode.logger = MagicMock()
    mode._last_decoder_check_ts = 0.0
    mode._source_mode = None
    mode._pending_decoder_load = None
    mode._sample_preprocessor = None
    mode._reader = None
    mode.sample_rate = 500.0
    mode.effective_sample_rate = 500.0
    mode.modalities = ["eeg"]
    mode.label_mapping = {"left": 0, "right": 1}
    mode.reverse_label_mapping = {0: "left", 1: "right"}
    mode.window_length_sec = 0.5
    mode.step_size_ms = 100
    mode.metrics_manager = None
    mode.buffer = None
    mode.shared_state = MagicMock()
    mode.shared_state.get.return_value = None
    for k, v in overrides.items():
        setattr(mode, k, v)
    return mode


def _make_mock_decoder(*, n_channels=4, n_times=250, preprocessing=None):
    """Create a mock decoder with configurable shapes and preprocessing."""
    decoder = MagicMock()
    decoder.input_shapes = {"eeg": [n_channels, n_times]}
    decoder.config.model_type = "CSP+LDA"
    decoder.config.preprocessing_config = preprocessing
    decoder.is_fitted = True
    return decoder


# ---------------------------------------------------------------------------
# Schema: decoder_source "online" passes validation without decoder_path
# ---------------------------------------------------------------------------


class TestOnlineDecoderSource:
    """Test that decoder_source='online' is valid without decoder_path."""

    def test_online_source_valid_without_decoder_path(self):
        config = AsynchronousInstanceConfig(
            name="async_online",
            mode="asynchronous",
            channel_selection={"eeg": [0, 1, 2, 3]},
            decoder_source="online",
            decoder_config={
                "decoder_type": "Decoder",
                "model_config": {"model_type": "EEGNet", "num_classes": 2},
            },
        )
        assert config.decoder_source == "online"

    def test_online_source_via_validate_mode_config(self):
        ok, errors, validated = validate_mode_config({
            "name": "async_online",
            "mode": "asynchronous",
            "channel_selection": {"eeg": [0, 1, 2, 3]},
            "decoder_source": "online",
            "decoder_config": {
                "decoder_type": "Decoder",
                "model_config": {"model_type": "EEGNet", "num_classes": 2},
            },
        })
        assert ok, f"Validation failed: {errors}"
        assert validated["decoder_source"] == "online"

    def test_online_source_no_decoder_path_needed(self):
        """Online source should not require decoder_path."""
        config = AsynchronousInstanceConfig(
            name="async_online",
            mode="asynchronous",
            channel_selection={"eeg": [0, 1, 2, 3]},
            decoder_source="online",
            decoder_config={
                "decoder_type": "Decoder",
                "model_config": {"model_type": "EEGNet", "num_classes": 2},
            },
        )
        assert config.decoder_config.get("decoder_path") is None

    def test_database_source_works(self):
        config = AsynchronousInstanceConfig(
            name="async_db",
            mode="asynchronous",
            channel_selection={"eeg": [0, 1, 2, 3]},
            decoder_source="database",
            decoder_config={
                "decoder_type": "Decoder",
                "decoder_path": "/fake/model.json",
                "model_config": {"model_type": "EEGNet", "num_classes": 2},
            },
        )
        assert config.decoder_source == "database"

    def test_invalid_source_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AsynchronousInstanceConfig(
                name="async_bad",
                mode="asynchronous",
                channel_selection={"eeg": [0, 1, 2, 3]},
                decoder_source="invalid_source",
            )


# ---------------------------------------------------------------------------
# AsyncMode: validates online source skips decoder requirement
# ---------------------------------------------------------------------------


class TestAsyncModeOnlineInit:
    """Test AsynchronousMode initialization with decoder_source='online'."""

    def test_validate_configuration_online_no_decoder_config(self):
        """Online mode should pass validation even without decoder_config."""
        from dendrite.processing.modes.asynchronous_mode import AsynchronousMode

        mode = AsynchronousMode.__new__(AsynchronousMode)
        mode.epoch_length_samples = 250
        mode.samples_per_prediction_step = 25
        mode.decoder_config = {}
        mode.decoder_source = "online"
        mode.channel_selection = {"eeg": [0, 1, 2, 3]}
        mode.logger = MagicMock()

        assert mode._validate_configuration() is True

    def test_validate_configuration_database_requires_decoder_config(self):
        """Database mode should fail validation without decoder_config."""
        from dendrite.processing.modes.asynchronous_mode import AsynchronousMode

        mode = AsynchronousMode.__new__(AsynchronousMode)
        mode.epoch_length_samples = 250
        mode.samples_per_prediction_step = 25
        mode.decoder_config = {}
        mode.decoder_source = "database"
        mode.channel_selection = {"eeg": [0, 1, 2, 3]}
        mode.logger = MagicMock()

        assert mode._validate_configuration() is False


# ---------------------------------------------------------------------------
# AsyncMode: _check_for_trained_decoder polls and loads
# ---------------------------------------------------------------------------


class TestAsyncModeDecoderReload:
    """Test async mode SharedState polling for trained decoder."""

    def test_check_for_trained_decoder_loads_new(self):
        """When SharedState has a new decoder, it should start bg loading."""
        mode = _make_async_mode()
        decoder_info = {
            "path": "/fake/decoder.json",
            "timestamp": 100.0,
            "n_epochs": 20,
            "elapsed": 5.0,
            "source_mode": "SyncMode1",
        }
        mode.shared_state.get.return_value = decoder_info

        with patch.object(mode, "_start_background_decoder_load") as mock_bg:
            mode._check_for_trained_decoder()
            mock_bg.assert_called_once_with("/fake/decoder.json", decoder_info)
            assert mode._last_decoder_check_ts == 100.0

    def test_check_for_trained_decoder_skips_old(self):
        """Should not reload if timestamp hasn't changed."""
        mode = _make_async_mode(_last_decoder_check_ts=100.0)
        mode.shared_state.get.return_value = {
            "path": "/fake/decoder.json",
            "timestamp": 100.0,
        }

        with patch.object(mode, "_start_background_decoder_load") as mock_bg:
            mode._check_for_trained_decoder()
            mock_bg.assert_not_called()

    def test_check_for_trained_decoder_no_result(self):
        """Should do nothing when SharedState has no decoder."""
        mode = _make_async_mode()
        mode.shared_state.get.return_value = None

        with patch.object(mode, "_start_background_decoder_load") as mock_bg:
            mode._check_for_trained_decoder()
            mock_bg.assert_not_called()

    def test_check_decoder_uses_source_mode_key(self):
        """When linked, should prefer mode-specific SharedState key."""
        mode = _make_async_mode(_source_mode="Sync1")
        decoder_info = {
            "path": "/fake/decoder.json",
            "timestamp": 100.0,
            "source_mode": "Sync1",
        }

        def get_key(key):
            if key == "Sync1:trained_decoder":
                return decoder_info
            return None

        mode.shared_state.get.side_effect = get_key

        with patch.object(mode, "_start_background_decoder_load") as mock_bg:
            mode._check_for_trained_decoder()
            mock_bg.assert_called_once()
            # Should have read from mode-specific key
            mode.shared_state.get.assert_any_call("Sync1:trained_decoder")

    def test_check_decoder_logs_source_mismatch(self):
        """Should log debug message when source_mode doesn't match."""
        mode = _make_async_mode(_source_mode="Sync1")
        mode.shared_state.get.side_effect = [
            None,  # mode-specific key returns None
            {"path": "/fake/decoder.json", "timestamp": 100.0, "source_mode": "OtherSync"},
        ]

        with patch.object(mode, "_start_background_decoder_load") as mock_bg:
            mode._check_for_trained_decoder()
            mock_bg.assert_not_called()
            mode.logger.debug.assert_called_once()
            assert "Ignoring decoder" in mode.logger.debug.call_args[0][0]


# ---------------------------------------------------------------------------
# AsyncMode: _activate_decoder preprocessing preservation
# ---------------------------------------------------------------------------


class TestActivateDecoderPreprocessing:
    """Test that _activate_decoder preserves all preprocessing fields."""

    def _make_preprocessor(self, mode, config):
        from dendrite.processing.modes.mode_utils import SamplePreprocessor

        return SamplePreprocessor(
            preproc_config=config,
            sample_rate=mode.sample_rate,
            channel_selection=mode.channel_selection,
            modality_labels={},
            shared_state=mode.shared_state,
            logger=mode.logger,
        )

    def test_preserves_all_preproc_fields(self):
        """Decoder with target_sample_rate should preserve it in preprocessor config."""
        from dendrite.processing.preprocessing.preprocessing_schemas import (
            ModalityPreprocessing,
            PreprocessingConfig,
        )

        mode = _make_async_mode()
        mode._sample_preprocessor = self._make_preprocessor(
            mode, {"eeg": {"lowcut": 8.0, "highcut": 30.0}},
        )

        preproc = PreprocessingConfig(modality_preprocessing={
            "eeg": ModalityPreprocessing(
                lowcut=8.0, highcut=30.0, apply_rereferencing=True,
                target_sample_rate=250.0,
            ),
        })
        decoder = _make_mock_decoder(preprocessing=preproc)
        mode._activate_decoder(decoder)

        eeg_config = mode._sample_preprocessor._config["eeg"]
        assert eeg_config.get("target_sample_rate") == 250.0
        assert eeg_config.get("lowcut") == 8.0
        assert eeg_config.get("highcut") == 30.0
        assert eeg_config.get("apply_rereferencing") is True

    def test_skips_preprocessor_reset_when_unchanged(self):
        """Should NOT reset preprocessor when config is identical."""
        from dendrite.processing.preprocessing.preprocessing_schemas import (
            ModalityPreprocessing,
            PreprocessingConfig,
        )

        mode = _make_async_mode()
        mode._sample_preprocessor = self._make_preprocessor(
            mode, {"eeg": {"lowcut": 8.0, "highcut": 30.0, "apply_rereferencing": True,
                           "apply_eog_correction": False, "filter_order": 4}},
        )
        sentinel = object()
        mode._sample_preprocessor._preprocessor = sentinel

        preproc = PreprocessingConfig(modality_preprocessing={
            "eeg": ModalityPreprocessing(
                lowcut=8.0, highcut=30.0, apply_rereferencing=True,
            ),
        })
        decoder = _make_mock_decoder(preprocessing=preproc)
        mode._activate_decoder(decoder)

        # Internal preprocessor should NOT have been reset
        assert mode._sample_preprocessor._preprocessor is sentinel

    def test_resets_preprocessor_when_changed(self):
        """Should reset preprocessor when filter settings differ."""
        from dendrite.processing.preprocessing.preprocessing_schemas import (
            ModalityPreprocessing,
            PreprocessingConfig,
        )

        mode = _make_async_mode()
        mode._sample_preprocessor = self._make_preprocessor(
            mode, {"eeg": {"lowcut": 0.5, "highcut": 50.0, "apply_rereferencing": True}},
        )
        mode._sample_preprocessor._preprocessor = object()

        preproc = PreprocessingConfig(modality_preprocessing={
            "eeg": ModalityPreprocessing(
                lowcut=8.0, highcut=30.0, apply_rereferencing=True,
            ),
        })
        decoder = _make_mock_decoder(preprocessing=preproc)
        mode._activate_decoder(decoder)

        assert mode._sample_preprocessor._preprocessor is None

    def test_recalculates_effective_sample_rate(self):
        """Should update effective_sample_rate when target_sample_rate changes."""
        from dendrite.processing.preprocessing.preprocessing_schemas import (
            ModalityPreprocessing,
            PreprocessingConfig,
        )

        mode = _make_async_mode()
        mode.sample_rate = 500.0
        mode.effective_sample_rate = 500.0
        mode._sample_preprocessor = self._make_preprocessor(
            mode, {"eeg": {"lowcut": 8.0, "highcut": 30.0}},
        )

        preproc = PreprocessingConfig(modality_preprocessing={
            "eeg": ModalityPreprocessing(
                lowcut=8.0, highcut=30.0, apply_rereferencing=True,
                target_sample_rate=250.0,
            ),
        })
        decoder = _make_mock_decoder(preprocessing=preproc)
        mode._activate_decoder(decoder)

        assert mode.effective_sample_rate == 250.0

    def test_channel_mismatch_rejects_decoder(self):
        """Should reject decoder with wrong channel count."""
        mode = _make_async_mode(channel_selection={"eeg": [0, 1, 2, 3]})
        decoder = _make_mock_decoder(n_channels=8)

        result = mode._activate_decoder(decoder)

        assert result is False
        assert mode.decoder is None


