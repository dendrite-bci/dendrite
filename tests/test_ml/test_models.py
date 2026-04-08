"""Tests for model registry and forward pass validation."""

import pytest
import torch

from dendrite.ml.models import MODEL_REGISTRY, create_model, get_available_models
from dendrite.ml.decoders.registry import (
    get_available_decoders,
    get_decoder_capabilities,
    get_decoder_entry,
    check_decoder_compatibility,
)

# Neural models only (exclude classical classifiers which are sklearn estimators)
NEURAL_MODEL_NAMES = [
    name for name, entry in MODEL_REGISTRY.items() if not entry.get("classical")
]

# Models that require longer time series (n_times > 64)
LARGE_INPUT_MODELS = {"BDEEGConformer", "BDShallowNet"}

# Models that work with small input (8, 64)
SMALL_INPUT_MODELS = [n for n in NEURAL_MODEL_NAMES if n not in LARGE_INPUT_MODELS]


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------


class TestModelRegistry:
    def test_registry_not_empty(self):
        models = get_available_models()
        assert len(models) > 0

    def test_all_models_have_info(self):
        for name, entry in MODEL_REGISTRY.items():
            model_class = entry["class"]
            if hasattr(model_class, "get_model_info"):
                info = model_class.get_model_info()
                assert "description" in info or "model_type" in info
                assert "modalities" in info

    @pytest.mark.parametrize("model_type", SMALL_INPUT_MODELS)
    def test_create_model_small_input(self, model_type):
        model = create_model(model_type, num_classes=2, input_shape=(8, 64))
        assert isinstance(model, torch.nn.Module)

    @pytest.mark.parametrize("model_type", list(LARGE_INPUT_MODELS))
    def test_create_model_large_input(self, model_type):
        model = create_model(model_type, num_classes=2, input_shape=(8, 256))
        assert isinstance(model, torch.nn.Module)

    @pytest.mark.parametrize("model_type", SMALL_INPUT_MODELS)
    def test_model_output_shape(self, model_type):
        model = create_model(model_type, num_classes=2, input_shape=(8, 64))
        model.eval()
        with torch.no_grad():
            x = torch.randn(1, 8, 64)
            out = model(x)
        assert out.shape == (1, 2)

    @pytest.mark.parametrize("model_type", list(LARGE_INPUT_MODELS))
    def test_model_output_shape_large_input(self, model_type):
        model = create_model(model_type, num_classes=2, input_shape=(8, 256))
        model.eval()
        with torch.no_grad():
            x = torch.randn(1, 8, 256)
            out = model(x)
        assert out.shape == (1, 2)

    @pytest.mark.parametrize("n_channels", [4, 8, 16, 32])
    def test_model_variable_channels(self, n_channels):
        model = create_model("EEGNet", num_classes=2, input_shape=(n_channels, 64))
        model.eval()
        with torch.no_grad():
            x = torch.randn(1, n_channels, 64)
            out = model(x)
        assert out.shape == (1, 2)

    @pytest.mark.parametrize("n_classes", [2, 3, 4])
    def test_model_variable_classes(self, n_classes):
        model = create_model("EEGNet", num_classes=n_classes, input_shape=(8, 64))
        model.eval()
        with torch.no_grad():
            x = torch.randn(1, 8, 64)
            out = model(x)
        assert out.shape == (1, n_classes)

    def test_model_invalid_type_raises(self):
        with pytest.raises(ValueError):
            create_model("FakeModel", num_classes=2, input_shape=(8, 64))


# ---------------------------------------------------------------------------
# Decoder Registry
# ---------------------------------------------------------------------------


class TestDecoderRegistry:
    def test_decoder_registry_includes_classical(self):
        decoders = get_available_decoders()
        assert "CSP+LDA" in decoders
        assert "CSP+SVM" in decoders

    def test_decoder_capabilities_eeg(self):
        caps = get_decoder_capabilities("EEGNet")
        assert "eeg" in caps

    def test_decoder_capabilities_any(self):
        caps = get_decoder_capabilities("LinearEEG")
        assert "any" in caps

    def test_check_compatibility_compatible(self):
        is_ok, unsupported = check_decoder_compatibility("EEGNet", ["eeg"])
        assert is_ok is True
        assert unsupported == []

    def test_check_compatibility_incompatible(self):
        is_ok, unsupported = check_decoder_compatibility("EEGNet", ["emg"])
        assert is_ok is False
        assert "emg" in unsupported

    def test_check_compatibility_any_modality(self):
        is_ok, unsupported = check_decoder_compatibility("LinearEEG", ["emg"])
        assert is_ok is True
        assert unsupported == []

    def test_all_decoders_have_entry(self):
        for name in get_available_decoders():
            entry = get_decoder_entry(name)
            assert entry is not None
            assert "pipeline_builder" in entry
            assert "description" in entry
            assert "modalities" in entry
