"""
Unit tests for BMI models.

Tests verify that all models:
- Accept arbitrary channel counts
- Produce correct output shapes
- Handle edge cases (1 channel, many channels)
"""

import pytest
import torch
import numpy as np

from dendrite.ml.models import (
    MODEL_REGISTRY, create_model, get_available_models,
    EEGNet, EEGNetPP, LinearEEG, TransformerEEG,
    # Braindecode models
    BDEEGNet, BDEEGConformer, BDShallowNet, BDDeep4Net, BDATCNet, BDTCN, BDEEGInception,
    # Classical models
    CSPModel, LDAModel, SVMModel,
)


# All time-series models use 3D input: (batch, n_channels, n_times)
MODELS_3D = ['EEGNet', 'EEGNetPP', 'LinearEEG', 'TransformerEEG']

# Braindecode models also use 3D input: (batch, n_channels, n_times)
MODELS_BD = ['BDEEGNet', 'BDEEGConformer', 'BDShallowNet', 'BDDeep4Net', 'BDATCNet', 'BDTCN', 'BDEEGInception']

# Classical ML models (sklearn-compatible, not nn.Module)
MODELS_CLASSICAL = ['CSP', 'LDA', 'SVM']

# All neural network models (used for create_model tests)
MODELS_NEURAL = MODELS_3D + MODELS_BD


class TestModelChannelVariability:
    """Test that all models support various channel counts."""

    @pytest.mark.parametrize("n_channels", [1, 2, 4, 8, 16, 32, 64])
    @pytest.mark.parametrize("model_name", MODELS_3D)
    def test_3d_models_various_channels(self, model_name, n_channels):
        """Test native 3D input models with various channel counts."""
        n_times = 250
        n_classes = 2
        batch_size = 2

        model = create_model(model_name, num_classes=n_classes, input_shape=(n_channels, n_times))
        model.eval()

        # 3D input: (batch, n_channels, n_times)
        x = torch.randn(batch_size, n_channels, n_times)

        with torch.no_grad():
            out = model(x)

        assert out.shape == (batch_size, n_classes), f"{model_name} output shape mismatch"
        assert not torch.isnan(out).any(), f"{model_name} produced NaN"
        assert not torch.isinf(out).any(), f"{model_name} produced Inf"

    @pytest.mark.parametrize("n_channels", [2, 4, 8, 16, 32])
    @pytest.mark.parametrize("model_name", MODELS_BD)
    def test_braindecode_models_various_channels(self, model_name, n_channels):
        """Test braindecode models with various channel counts."""
        n_times = 250
        n_classes = 2
        batch_size = 2

        model = create_model(model_name, num_classes=n_classes, input_shape=(n_channels, n_times))
        model.eval()

        # Braindecode expects 3D input: (batch, n_channels, n_times)
        x = torch.randn(batch_size, n_channels, n_times)

        with torch.no_grad():
            out = model(x)

        assert out.shape == (batch_size, n_classes), f"{model_name} output shape mismatch"
        assert not torch.isnan(out).any(), f"{model_name} produced NaN"
        assert not torch.isinf(out).any(), f"{model_name} produced Inf"


class TestModelRegistry:
    """Test unified model registry."""

    def test_registry_has_all_models(self):
        """Test that MODEL_REGISTRY contains all expected models."""
        expected_neural = [
            # Native models
            'EEGNet', 'EEGNetPP', 'LinearEEG', 'TransformerEEG',
            # Braindecode models
            'BDEEGNet', 'BDEEGConformer', 'BDShallowNet', 'BDDeep4Net',
            'BDATCNet', 'BDTCN', 'BDEEGInception',
        ]
        expected_classical = ['CSP', 'LDA', 'SVM']
        for model_name in expected_neural + expected_classical:
            assert model_name in MODEL_REGISTRY, f"{model_name} missing from MODEL_REGISTRY"

    def test_registry_entries_have_required_fields(self):
        """Test that all registry entries have class and config key."""
        for model_name, entry in MODEL_REGISTRY.items():
            assert 'class' in entry, f"{model_name} missing 'class'"
            assert 'config' in entry, f"{model_name} missing 'config'"
            # Modalities are sourced from model's get_model_info() method
            model_class = entry['class']
            assert hasattr(model_class, 'get_model_info'), f"{model_name} missing get_model_info()"
            info = model_class.get_model_info()
            assert 'modalities' in info, f"{model_name} get_model_info() missing 'modalities'"

    def test_neural_models_have_config_class(self):
        """Test that neural network models have Pydantic config classes."""
        for model_name in MODELS_NEURAL:
            entry = MODEL_REGISTRY[model_name]
            assert entry['config'] is not None, f"{model_name} should have a config class"

    def test_classical_models_have_no_config_class(self):
        """Test that classical models don't need Pydantic configs (use sklearn defaults)."""
        for model_name in MODELS_CLASSICAL:
            entry = MODEL_REGISTRY[model_name]
            assert entry['config'] is None, f"{model_name} should not have a config class"

    def test_get_available_models_returns_all(self):
        """Test that get_available_models() returns all models from MODEL_REGISTRY."""
        available = get_available_models()
        for model in MODEL_REGISTRY.keys():
            assert model in available, f"{model} missing from get_available_models()"
        assert len(available) == len(MODEL_REGISTRY)


class TestModelCreation:
    """Test neural network model creation via factory function."""

    @pytest.mark.parametrize("model_name", MODELS_NEURAL)
    def test_create_model_neural_types(self, model_name):
        """Test that create_model works for all neural network model types."""
        n_channels = 8
        n_times = 250
        n_classes = 2

        model = create_model(model_name, num_classes=n_classes, input_shape=(n_channels, n_times))

        assert model is not None
        assert hasattr(model, 'forward')
        assert hasattr(model, 'n_channels')
        assert model.n_channels == n_channels


class TestClassicalModels:
    """Test classical ML model wrappers."""

    def test_csp_model_info(self):
        """Test CSPModel has correct get_model_info()."""
        info = CSPModel.get_model_info()
        assert info['model_type'] == 'CSP'
        assert info['modalities'] == ['eeg']
        assert info['component_type'] == 'feature_extractor'
        assert 'n_components' in info['default_parameters']

    def test_lda_model_info(self):
        """Test LDAModel has correct get_model_info()."""
        info = LDAModel.get_model_info()
        assert info['model_type'] == 'LDA'
        assert info['modalities'] == ['any']
        assert info['component_type'] == 'classifier'
        assert 'shrinkage' in info['default_parameters']

    def test_svm_model_info(self):
        """Test SVMModel has correct get_model_info()."""
        info = SVMModel.get_model_info()
        assert info['model_type'] == 'SVM'
        assert info['modalities'] == ['any']
        assert info['component_type'] == 'classifier'
        assert 'kernel' in info['default_parameters']

    def test_csp_model_instantiation(self):
        """Test CSPModel can be instantiated with default params."""
        csp = CSPModel()
        assert csp.n_components == 8

    def test_lda_model_instantiation(self):
        """Test LDAModel can be instantiated with default params."""
        lda = LDAModel()
        assert lda.shrinkage == 'auto'
        assert lda.solver == 'lsqr'

    def test_svm_model_instantiation(self):
        """Test SVMModel can be instantiated with default params."""
        svm = SVMModel()
        assert svm.kernel == 'rbf'
        assert svm.C == 1.0
        assert svm.probability is True

    @pytest.mark.parametrize("model_name", MODELS_CLASSICAL)
    def test_create_model_classical_types(self, model_name):
        """Test that create_model works for classical models (ignores shape params)."""
        model = create_model(model_name, num_classes=2, input_shape=(32, 250))
        assert model is not None
