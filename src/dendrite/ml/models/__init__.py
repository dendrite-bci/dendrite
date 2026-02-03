"""
Models for EEG/EMG classification.

This module provides all model classes (neural networks and classical ML) used in
brain-computer and muscle-computer interfaces. All models are registered in
MODEL_REGISTRY which is the single source of truth for model classes and configs.

Neural network models:
- BDEEGNet: Compact CNN for EEG (braindecode EEGNetv4)
- BDEEGConformer: CNN-Transformer hybrid (braindecode)
- BDShallowNet: Fast FBCSP-inspired baseline
- BDDeep4Net: Deep ConvNet with 4 blocks
- BDATCNet: Attention TCN (SOTA on MI)
- BDTCN: Temporal Convolutional Network
- BDEEGInception: Inception for Motor Imagery
- EEGNet: Native compact CNN implementation
- EEGNetPP: EEGNet with SE attention and residual connections
- LinearEEG: Simple linear model for few-shot learning (any modality)
- TransformerEEG: Transformer-based architecture

Classical ML models:
- CSP: Common Spatial Patterns feature extractor
- LDA: Linear Discriminant Analysis classifier
- SVM: Support Vector Machine classifier

Use get_available_models() to see all available models.
Use get_available_decoders() for pipeline configurations (e.g., CSP+LDA).
"""

from .base import ModelBase
from .braindecode_adapter import (
    BDTCN,
    BDATCNet,
    BDDeep4Net,
    BDEEGConformer,
    BDEEGInception,
    BDEEGNet,
    BDShallowNet,
)
from .classical import CSPModel, LDAModel, SVMModel
from .eegnet import EEGNet
from .eegnet_plus import EEGNetPP
from .linear import LinearEEG
from .model_configs import (
    BDATCNetConfig,
    BDDeep4NetConfig,
    BDEEGConformerConfig,
    BDEEGInceptionConfig,
    BDEEGNetConfig,
    BDShallowNetConfig,
    BDTCNConfig,
    EEGNetConfig,
    EEGNetPPConfig,
    LinearEEGConfig,
    TransformerEEGConfig,
    get_model_config_class,
    validate_model_config,
)
from .registry import MODEL_REGISTRY
from .transformer import TransformerEEG

__all__ = [
    "create_model",
    "get_available_models",
]


def create_model(model_type: str, num_classes: int, input_shape: tuple[int, int], **kwargs):
    """Create a neural network model of the specified type.

    Args:
        model_type: Type of model to create. Use get_available_models() to see all options.
        num_classes: Number of output classes.
        input_shape: Shape of input data (n_channels, n_times).
        **kwargs: Additional model-specific parameters.

    Returns:
        Initialized PyTorch model (torch.nn.Module).

    Raises:
        ValueError: If model_type is not recognized.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. Available models: {list(MODEL_REGISTRY.keys())}"
        )

    model_info = MODEL_REGISTRY[model_type]
    model_class = model_info["class"]

    # Classical models (config=None) don't use shape params
    if model_info["config"] is None:
        return model_class(**kwargs)

    n_channels, n_times = input_shape
    return model_class(n_classes=num_classes, n_channels=n_channels, n_times=n_times, **kwargs)


def get_available_models() -> list[str]:
    """Get a list of all available model types.

    Returns all models from MODEL_REGISTRY including neural networks
    (EEGNet, etc.) and classical ML models (CSP, LDA, SVM).

    For pipeline configurations (CSP+LDA, CSP+SVM), use get_available_decoders().

    Returns:
        List of supported model type names.
    """
    return list(MODEL_REGISTRY.keys())
