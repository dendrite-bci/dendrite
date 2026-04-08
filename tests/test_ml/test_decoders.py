"""Tests for Decoder lifecycle: creation, fit, predict, save/load, config."""

import numpy as np
import pytest

from dendrite.ml import (
    Decoder,
    create_decoder,
    get_available_decoders,
    load_decoder,
)
from dendrite.ml.decoders import validate_decoder_file
from dendrite.ml.decoders.decoder_schemas import DecoderConfig

# Fast neural models for parametrized tests
FAST_NEURAL_MODELS = ["EEGNet", "LinearEEG"]
# Slower models gated behind @pytest.mark.slow
SLOW_NEURAL_MODELS = ["BDEEGNet", "TransformerEEG"]


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------


class TestDecoderCreation:
    def test_create_decoder_default(self):
        decoder = create_decoder(
            model_type="EEGNet",
            num_classes=2,
            input_shapes={"eeg": [8, 64]},
        )
        assert isinstance(decoder, Decoder)
        assert decoder.num_classes == 2

    @pytest.mark.parametrize("model_type", get_available_decoders())
    def test_create_decoder_all_types(self, model_type):
        decoder = create_decoder(
            model_type=model_type,
            num_classes=2,
            input_shapes={"eeg": [8, 64]},
        )
        assert isinstance(decoder, Decoder)

    def test_create_decoder_invalid_type_raises(self):
        with pytest.raises(ValueError):
            create_decoder(
                model_type="NonexistentModel",
                num_classes=2,
                input_shapes={"eeg": [8, 64]},
            )


# ---------------------------------------------------------------------------
# Fit / Predict
# ---------------------------------------------------------------------------


class TestDecoderFitPredict:
    @pytest.mark.parametrize("model_type", FAST_NEURAL_MODELS)
    def test_fit_marks_fitted(self, model_type, eeg_data_2class, fast_decoder_config):
        X, y = eeg_data_2class
        cfg = {**fast_decoder_config, "model_type": model_type, "input_shapes": {"eeg": [8, 64]}}
        decoder = create_decoder(**cfg)
        decoder.fit(X, y)
        assert decoder.is_fitted is True

    @pytest.mark.parametrize("model_type", FAST_NEURAL_MODELS)
    def test_fit_returns_self(self, model_type, eeg_data_2class, fast_decoder_config):
        X, y = eeg_data_2class
        cfg = {**fast_decoder_config, "model_type": model_type, "input_shapes": {"eeg": [8, 64]}}
        decoder = create_decoder(**cfg)
        result = decoder.fit(X, y)
        assert result is decoder

    @pytest.mark.parametrize("model_type", FAST_NEURAL_MODELS)
    def test_predict_shape(self, model_type, eeg_data_2class, fast_decoder_config):
        X, y = eeg_data_2class
        cfg = {**fast_decoder_config, "model_type": model_type, "input_shapes": {"eeg": [8, 64]}}
        decoder = create_decoder(**cfg)
        decoder.fit(X, y)
        preds = decoder.predict(X)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (20,)
        assert all(p in [0, 1] for p in preds)

    @pytest.mark.parametrize("model_type", FAST_NEURAL_MODELS)
    def test_predict_proba_shape(self, model_type, eeg_data_2class, fast_decoder_config):
        X, y = eeg_data_2class
        cfg = {**fast_decoder_config, "model_type": model_type, "input_shapes": {"eeg": [8, 64]}}
        decoder = create_decoder(**cfg)
        decoder.fit(X, y)
        proba = decoder.predict_proba(X)
        assert proba.shape == (20, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    @pytest.mark.parametrize("model_type", FAST_NEURAL_MODELS)
    def test_predict_sample_returns_tuple(self, model_type, eeg_data_2class, fast_decoder_config):
        X, y = eeg_data_2class
        cfg = {**fast_decoder_config, "model_type": model_type, "input_shapes": {"eeg": [8, 64]}}
        decoder = create_decoder(**cfg)
        decoder.fit(X, y)
        pred, conf = decoder.predict_sample(X[0])
        assert isinstance(pred, int)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_predict_before_fit_raises(self):
        decoder = create_decoder(
            model_type="EEGNet", num_classes=2, input_shapes={"eeg": [8, 64]}
        )
        X = np.random.randn(5, 8, 64).astype(np.float32)
        with pytest.raises(RuntimeError):
            decoder.predict_proba(X)

    @pytest.mark.parametrize("model_type", FAST_NEURAL_MODELS)
    def test_score_returns_float(self, model_type, eeg_data_2class, fast_decoder_config):
        X, y = eeg_data_2class
        cfg = {**fast_decoder_config, "model_type": model_type, "input_shapes": {"eeg": [8, 64]}}
        decoder = create_decoder(**cfg)
        decoder.fit(X, y)
        score = decoder.score(X, y)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_fit_predict_3class(self, eeg_data_3class, fast_decoder_config):
        X, y = eeg_data_3class
        cfg = {
            **fast_decoder_config,
            "model_type": "EEGNet",
            "num_classes": 3,
            "input_shapes": {"eeg": [8, 64]},
        }
        decoder = create_decoder(**cfg)
        decoder.fit(X, y)
        preds = decoder.predict(X)
        assert isinstance(preds, np.ndarray)
        assert all(p in [0, 1, 2] for p in preds)
        proba = decoder.predict_proba(X)
        assert proba.shape == (30, 3)

    @pytest.mark.slow
    @pytest.mark.parametrize("model_type", SLOW_NEURAL_MODELS)
    def test_fit_predict_slow_models(self, model_type, eeg_data_2class, fast_decoder_config):
        X, y = eeg_data_2class
        cfg = {**fast_decoder_config, "model_type": model_type, "input_shapes": {"eeg": [8, 64]}}
        decoder = create_decoder(**cfg)
        decoder.fit(X, y)
        assert decoder.is_fitted
        preds = decoder.predict(X)
        assert preds.shape == (20,)


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


class TestDecoderSaveLoad:
    @pytest.mark.parametrize("model_type", FAST_NEURAL_MODELS)
    def test_save_creates_files(self, model_type, eeg_data_2class, fast_decoder_config, tmp_path):
        X, y = eeg_data_2class
        cfg = {**fast_decoder_config, "model_type": model_type, "input_shapes": {"eeg": [8, 64]}}
        decoder = create_decoder(**cfg)
        decoder.fit(X, y)

        # Monkey-patch get_study_paths to use tmp_path
        import dendrite.ml.decoders.decoder as dec_mod

        original_fn = dec_mod.get_study_paths
        dec_mod.get_study_paths = lambda study: {"decoders": tmp_path}
        try:
            json_path = decoder.save("test_decoder")
        finally:
            dec_mod.get_study_paths = original_fn

        import os

        assert os.path.exists(json_path)
        # Neural models save .pt, classical save .joblib
        assert os.path.exists(json_path.replace(".json", ".pt"))

    @pytest.mark.parametrize("model_type", FAST_NEURAL_MODELS)
    def test_load_roundtrip(self, model_type, eeg_data_2class, fast_decoder_config, tmp_path):
        X, y = eeg_data_2class
        cfg = {**fast_decoder_config, "model_type": model_type, "input_shapes": {"eeg": [8, 64]}}
        decoder = create_decoder(**cfg)
        decoder.fit(X, y)

        import dendrite.ml.decoders.decoder as dec_mod

        original_fn = dec_mod.get_study_paths
        dec_mod.get_study_paths = lambda study: {"decoders": tmp_path}
        try:
            json_path = decoder.save("roundtrip_test")
        finally:
            dec_mod.get_study_paths = original_fn

        loaded = load_decoder(json_path)
        assert loaded.is_fitted

        # Predictions should match
        orig_proba = decoder.predict_proba(X)
        loaded_proba = loaded.predict_proba(X)
        np.testing.assert_allclose(orig_proba, loaded_proba, atol=1e-5)

    def test_load_preserves_metadata(self, eeg_data_2class, fast_decoder_config, tmp_path):
        X, y = eeg_data_2class
        cfg = {
            **fast_decoder_config,
            "input_shapes": {"eeg": [8, 64]},
            "event_mapping": {7: "left", 8: "right"},
            "label_mapping": {"left": 0, "right": 1},
            "sample_rate": 256.0,
        }
        decoder = create_decoder(**cfg)
        decoder.fit(X, y)

        import dendrite.ml.decoders.decoder as dec_mod

        original_fn = dec_mod.get_study_paths
        dec_mod.get_study_paths = lambda study: {"decoders": tmp_path}
        try:
            json_path = decoder.save("meta_test")
        finally:
            dec_mod.get_study_paths = original_fn

        loaded = load_decoder(json_path)
        assert loaded.event_mapping == {7: "left", 8: "right"}
        assert loaded.label_mapping == {"left": 0, "right": 1}
        assert loaded.config.sample_rate == 256.0
        assert loaded.input_shapes == {"eeg": [8, 64]}

    def test_load_preserves_config(self, eeg_data_2class, fast_decoder_config, tmp_path):
        X, y = eeg_data_2class
        cfg = {
            **fast_decoder_config,
            "input_shapes": {"eeg": [8, 64]},
            "learning_rate": 0.005,
        }
        decoder = create_decoder(**cfg)
        decoder.fit(X, y)

        import dendrite.ml.decoders.decoder as dec_mod

        original_fn = dec_mod.get_study_paths
        dec_mod.get_study_paths = lambda study: {"decoders": tmp_path}
        try:
            json_path = decoder.save("config_test")
        finally:
            dec_mod.get_study_paths = original_fn

        loaded = load_decoder(json_path)
        assert loaded.config.learning_rate == 0.005
        assert loaded.config.epochs == 2

    def test_validate_decoder_file(self, eeg_data_2class, fast_decoder_config, tmp_path):
        X, y = eeg_data_2class
        cfg = {**fast_decoder_config, "input_shapes": {"eeg": [8, 64]}}
        decoder = create_decoder(**cfg)
        decoder.fit(X, y)

        import dendrite.ml.decoders.decoder as dec_mod

        original_fn = dec_mod.get_study_paths
        dec_mod.get_study_paths = lambda study: {"decoders": tmp_path}
        try:
            json_path = decoder.save("validate_test")
        finally:
            dec_mod.get_study_paths = original_fn

        # Valid file, matching shapes
        metadata, issues = validate_decoder_file(
            json_path, expected_shapes={"eeg": (8, 64)}
        )
        assert metadata is not None
        assert len(issues) == 0

        # Mismatched channels should report issue
        metadata2, issues2 = validate_decoder_file(
            json_path, expected_shapes={"eeg": (16, 64)}
        )
        assert len(issues2) > 0

    def test_save_unfitted_raises(self):
        decoder = create_decoder(
            model_type="EEGNet", num_classes=2, input_shapes={"eeg": [8, 64]}
        )
        with pytest.raises(ValueError):
            decoder.save("should_fail")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestDecoderConfig:
    def test_config_defaults(self):
        config = DecoderConfig()
        assert config.model_type == "EEGNet"
        assert config.num_classes == 2
        assert config.epochs == 100
        assert config.learning_rate == 0.001
        assert config.batch_size == 32
        assert config.validation_split == 0.2

    def test_config_validation_epochs(self):
        with pytest.raises(Exception):
            DecoderConfig(epochs=0)

    def test_config_validation_lr(self):
        with pytest.raises(Exception):
            DecoderConfig(learning_rate=0)
        with pytest.raises(Exception):
            DecoderConfig(learning_rate=-0.01)

    def test_effective_sample_rate(self):
        config = DecoderConfig(sample_rate=500.0, target_sample_rate=250.0)
        assert config.effective_sample_rate == 250.0

        config2 = DecoderConfig(sample_rate=500.0, target_sample_rate=None)
        assert config2.effective_sample_rate == 500.0

    def test_check_compatibility_match(self):
        config = DecoderConfig(
            input_shapes={"eeg": [8, 64]}, sample_rate=250.0
        )
        issues = config.check_compatibility(
            system_shapes={"eeg": [8, 128]}, system_sample_rate=250.0
        )
        assert len(issues) == 0  # channels match, time samples don't matter

    def test_check_compatibility_channel_mismatch(self):
        config = DecoderConfig(
            input_shapes={"eeg": [8, 64]}, sample_rate=250.0
        )
        issues = config.check_compatibility(system_shapes={"eeg": [16, 64]})
        assert any("channels" in i.lower() or "channel" in i.lower() for i in issues)

    def test_check_compatibility_rate_mismatch(self):
        config = DecoderConfig(
            input_shapes={"eeg": [8, 64]}, sample_rate=250.0
        )
        issues = config.check_compatibility(
            system_shapes={"eeg": [8, 64]}, system_sample_rate=500.0
        )
        assert any("sample rate" in i.lower() for i in issues)


# ---------------------------------------------------------------------------
# Classical decoders
# ---------------------------------------------------------------------------


class TestClassicalDecoders:
    def test_csp_lda_fit_predict(self, eeg_data_2class):
        X, y = eeg_data_2class
        decoder = create_decoder(
            model_type="CSP+LDA",
            num_classes=2,
            input_shapes={"eeg": [8, 64]},
        )
        decoder.fit(X, y)
        assert decoder.is_fitted
        preds = decoder.predict(X)
        assert isinstance(preds, (int, np.integer, np.ndarray))

    def test_csp_svm_fit_predict(self, eeg_data_2class):
        X, y = eeg_data_2class
        decoder = create_decoder(
            model_type="CSP+SVM",
            num_classes=2,
            input_shapes={"eeg": [8, 64]},
        )
        decoder.fit(X, y)
        assert decoder.is_fitted
        preds = decoder.predict(X)
        assert isinstance(preds, (int, np.integer, np.ndarray))

    def test_classical_save_load(self, eeg_data_2class, tmp_path):
        X, y = eeg_data_2class
        decoder = create_decoder(
            model_type="CSP+LDA",
            num_classes=2,
            input_shapes={"eeg": [8, 64]},
        )
        decoder.fit(X, y)

        import dendrite.ml.decoders.decoder as dec_mod
        import os

        original_fn = dec_mod.get_study_paths
        dec_mod.get_study_paths = lambda study: {"decoders": tmp_path}
        try:
            json_path = decoder.save("classical_test")
        finally:
            dec_mod.get_study_paths = original_fn

        assert os.path.exists(json_path)
        assert os.path.exists(json_path.replace(".json", ".joblib"))

        loaded = load_decoder(json_path)
        assert loaded.is_fitted

        orig_proba = decoder.predict_proba(X)
        loaded_proba = loaded.predict_proba(X)
        np.testing.assert_allclose(orig_proba, loaded_proba, atol=1e-5)
