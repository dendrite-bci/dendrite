"""
Model-specific Pydantic configuration schemas for BMI neural networks.

This module defines validated configuration classes for each model type,
ensuring type safety and parameter validation for model-specific settings.
"""

import warnings
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class BaseModelConfig(BaseModel):
    """Base configuration shared by all neural network models."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class EEGNetBaseConfig(BaseModelConfig):
    """Shared configuration fields for EEGNet-family models."""

    F1: int = Field(
        default=8,
        ge=1,
        le=256,
        description="Number of temporal filters",
        json_schema_extra={"hpo": {"type": "int", "low": 8, "high": 32, "step": 4}},
    )
    D: int = Field(
        default=2,
        ge=1,
        le=8,
        description="Depth multiplier for spatial filters",
        json_schema_extra={"hpo": {"type": "int", "low": 1, "high": 4}},
    )
    F2: int = Field(default=16, ge=1, le=512, description="Number of pointwise filters")
    kern_length: int = Field(
        default=64,
        ge=1,
        description="Length of temporal convolution kernel",
        json_schema_extra={"hpo": {"type": "int", "low": 32, "high": 256, "step": 16}},
    )
    norm_rate: float = Field(default=0.25, ge=0.0, le=1.0, description="Max norm constraint rate")
    dropout_rate: float = Field(default=0.5, ge=0.0, le=1.0, description="Dropout rate")

    @field_validator("F2")
    @classmethod
    def validate_f2_relationship(cls, v, info):
        """Validate F2 = F1 * D relationship (warning only)."""
        if info.data.get("F1") and info.data.get("D"):
            expected = info.data["F1"] * info.data["D"]
            if v != expected:
                warnings.warn(
                    f"F2 ({v}) should equal F1 * D ({info.data['F1']} * {info.data['D']} = {expected}) "
                    f"for optimal performance as per original paper",
                    stacklevel=2,
                )
        return v


class EEGNetConfig(EEGNetBaseConfig):
    """Configuration for EEGNet model."""

    dropout_type: Literal["Dropout", "SpatialDropout2D"] = Field(
        default="Dropout", description="Type of dropout to use"
    )


class EEGNetPPConfig(EEGNetBaseConfig):
    """Configuration for EEGNet++ model (EEGNet with SE attention and residuals)."""

    se_reduction: int = Field(
        default=4,
        ge=1,
        le=16,
        description="SE block channel reduction ratio",
        json_schema_extra={"hpo": {"type": "categorical", "choices": [2, 4, 8]}},
    )
    use_residual: bool = Field(
        default=True, description="Use residual connection around separable conv"
    )


class TransformerEEGConfig(BaseModelConfig):
    """Configuration for TransformerEEG model."""

    embed_dim: int = Field(
        default=64,
        ge=16,
        le=1024,
        description="Embedding dimension",
        json_schema_extra={"hpo": {"type": "categorical", "choices": [32, 64, 128]}},
    )
    num_heads: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of attention heads",
        json_schema_extra={"hpo": {"type": "categorical", "choices": [2, 4, 8]}},
    )
    num_layers: int = Field(
        default=2,
        ge=1,
        le=24,
        description="Number of transformer layers",
        json_schema_extra={"hpo": {"type": "int", "low": 2, "high": 6}},
    )
    positional_encoding: bool = Field(default=True, description="Use positional encoding")
    dropout_rate: float = Field(default=0.2, ge=0.0, le=1.0, description="Dropout rate")

    @field_validator("num_heads")
    @classmethod
    def validate_heads_divisible(cls, v, info):
        """Ensure num_heads divides embed_dim evenly."""
        if info.data.get("embed_dim"):
            if info.data["embed_dim"] % v != 0:
                raise ValueError(
                    f"embed_dim ({info.data['embed_dim']}) must be divisible by num_heads ({v})"
                )
        return v


class LinearEEGConfig(BaseModelConfig):
    """Configuration for LinearEEG model."""

    hidden_size: int = Field(
        default=32,
        ge=1,
        le=2048,
        description="Hidden layer size",
        json_schema_extra={"hpo": {"type": "categorical", "choices": [16, 32, 64, 128]}},
    )
    dropout_rate: float = Field(default=0.2, ge=0.0, le=1.0, description="Dropout rate")
    pool_size: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Temporal pooling output size",
        json_schema_extra={"hpo": {"type": "categorical", "choices": [4, 8, 16]}},
    )


class BDEEGNetConfig(BaseModelConfig):
    """Configuration for BDEEGNet (EEGNetv4)."""

    F1: int = Field(
        default=8,
        ge=1,
        le=256,
        description="Number of temporal filters",
        json_schema_extra={"hpo": {"type": "int", "low": 8, "high": 32, "step": 4}},
    )
    D: int = Field(
        default=2,
        ge=1,
        le=8,
        description="Depth multiplier for spatial filters",
        json_schema_extra={"hpo": {"type": "int", "low": 1, "high": 4}},
    )
    F2: int = Field(default=16, ge=1, le=512, description="Number of pointwise filters")
    kernel_length: int = Field(
        default=64,
        ge=1,
        description="Length of temporal convolution kernel",
        json_schema_extra={"hpo": {"type": "int", "low": 32, "high": 128, "step": 16}},
    )
    drop_prob: float = Field(default=0.25, ge=0.0, le=1.0, description="Dropout probability")


class BDEEGConformerConfig(BaseModelConfig):
    """Configuration for BDEEGConformer (EEGConformer)."""

    n_filters_time: int = Field(
        default=40,
        ge=1,
        le=256,
        description="Number of temporal filters",
        json_schema_extra={"hpo": {"type": "categorical", "choices": [20, 40, 60]}},
    )
    filter_time_length: int = Field(default=25, ge=1, description="Length of temporal filter")
    pool_time_length: int = Field(default=75, ge=1, description="Pooling window length")
    pool_time_stride: int = Field(default=15, ge=1, description="Pooling stride")
    num_layers: int = Field(
        default=6,
        ge=1,
        le=24,
        description="Number of transformer layers",
        json_schema_extra={"hpo": {"type": "int", "low": 2, "high": 8}},
    )
    num_heads: int = Field(
        default=10,
        ge=1,
        le=32,
        description="Number of attention heads",
        json_schema_extra={"hpo": {"type": "categorical", "choices": [4, 8, 10]}},
    )
    drop_prob: float = Field(default=0.5, ge=0.0, le=1.0, description="Dropout probability")
    att_drop_prob: float = Field(default=0.5, ge=0.0, le=1.0, description="Attention dropout")


class BDShallowNetConfig(BaseModelConfig):
    """Configuration for BDShallowNet (ShallowFBCSPNet)."""

    n_filters_time: int = Field(
        default=40,
        ge=1,
        le=256,
        description="Number of temporal filters",
        json_schema_extra={"hpo": {"type": "categorical", "choices": [20, 40, 60]}},
    )
    filter_time_length: int = Field(default=25, ge=1, description="Length of temporal filter")
    n_filters_spat: int = Field(
        default=40,
        ge=1,
        le=256,
        description="Number of spatial filters",
        json_schema_extra={"hpo": {"type": "categorical", "choices": [20, 40, 60]}},
    )
    pool_time_length: int = Field(default=75, ge=1, description="Pooling window length")
    pool_time_stride: int = Field(default=15, ge=1, description="Pooling stride")
    drop_prob: float = Field(default=0.5, ge=0.0, le=1.0, description="Dropout probability")
    final_conv_length: int | str = Field(
        default="auto", description="Length of final conv layer ('auto' or int)"
    )
    split_first_layer: bool = Field(
        default=True, description="Split first layer into temporal and spatial"
    )
    batch_norm: bool = Field(default=True, description="Use batch normalization")
    batch_norm_alpha: float = Field(default=0.1, ge=0.0, le=1.0, description="Batch norm momentum")


class BDDeep4NetConfig(BaseModelConfig):
    """Configuration for BDDeep4Net (Deep4Net)."""

    n_filters_time: int = Field(
        default=25,
        ge=1,
        le=256,
        description="Number of temporal filters",
        json_schema_extra={"hpo": {"type": "categorical", "choices": [15, 25, 35]}},
    )
    n_filters_spat: int = Field(
        default=25,
        ge=1,
        le=256,
        description="Number of spatial filters",
        json_schema_extra={"hpo": {"type": "categorical", "choices": [15, 25, 35]}},
    )
    filter_time_length: int = Field(default=10, ge=1, description="Length of temporal filter")
    pool_time_length: int = Field(default=3, ge=1, description="Pooling window length")
    pool_time_stride: int = Field(default=3, ge=1, description="Pooling stride")
    n_filters_2: int = Field(default=50, ge=1, le=512, description="Filters in 2nd block")
    filter_length_2: int = Field(default=10, ge=1, description="Filter length in 2nd block")
    n_filters_3: int = Field(default=100, ge=1, le=512, description="Filters in 3rd block")
    filter_length_3: int = Field(default=10, ge=1, description="Filter length in 3rd block")
    n_filters_4: int = Field(default=200, ge=1, le=512, description="Filters in 4th block")
    filter_length_4: int = Field(default=10, ge=1, description="Filter length in 4th block")
    drop_prob: float = Field(default=0.5, ge=0.0, le=1.0, description="Dropout probability")


class BDATCNetConfig(BaseModelConfig):
    """Configuration for BDATCNet (ATCNet)."""

    n_windows: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of sliding windows",
        json_schema_extra={"hpo": {"type": "int", "low": 3, "high": 7}},
    )
    head_dim: int = Field(default=8, ge=1, le=64, description="Attention head dimension")
    num_heads: int = Field(
        default=2,
        ge=1,
        le=16,
        description="Number of attention heads",
        json_schema_extra={"hpo": {"type": "categorical", "choices": [1, 2, 4]}},
    )
    tcn_depth: int = Field(
        default=2,
        ge=1,
        le=8,
        description="TCN depth",
        json_schema_extra={"hpo": {"type": "int", "low": 1, "high": 4}},
    )
    tcn_kernel_size: int = Field(default=4, ge=2, le=16, description="TCN kernel size")
    conv_block_dropout: float = Field(default=0.3, ge=0.0, le=1.0, description="Conv block dropout")
    att_drop_prob: float = Field(default=0.5, ge=0.0, le=1.0, description="Attention dropout")
    tcn_drop_prob: float = Field(default=0.3, ge=0.0, le=1.0, description="TCN dropout")


class BDTCNConfig(BaseModelConfig):
    """Configuration for BDTCN (Temporal Convolutional Network)."""

    n_filters: int = Field(
        default=30,
        ge=1,
        le=256,
        description="Number of filters per layer",
        json_schema_extra={"hpo": {"type": "categorical", "choices": [15, 30, 45]}},
    )
    n_blocks: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of TCN blocks",
        json_schema_extra={"hpo": {"type": "int", "low": 2, "high": 5}},
    )
    kernel_size: int = Field(
        default=5,
        ge=2,
        le=16,
        description="Kernel size",
        json_schema_extra={"hpo": {"type": "categorical", "choices": [3, 5, 7]}},
    )
    drop_prob: float = Field(default=0.5, ge=0.0, le=1.0, description="Dropout probability")


class BDEEGInceptionConfig(BaseModelConfig):
    """Configuration for BDEEGInception (EEGInceptionMI)."""

    n_convs: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of convolution layers",
        json_schema_extra={"hpo": {"type": "int", "low": 3, "high": 7}},
    )
    n_filters: int = Field(
        default=48,
        ge=8,
        le=256,
        description="Number of filters",
        json_schema_extra={"hpo": {"type": "categorical", "choices": [24, 48, 72]}},
    )


def get_model_config_class(model_type: str):
    """Get the configuration class for a given model type."""
    from .registry import MODEL_REGISTRY

    entry = MODEL_REGISTRY.get(model_type)
    return entry["config"] if entry else None


def validate_model_config(model_type: str, config_dict: dict[str, Any]) -> dict[str, Any]:
    """Validate and clean model configuration parameters.

    Args:
        model_type: The model type identifier.
        config_dict: Raw configuration dictionary.

    Returns:
        Validated configuration dictionary with only valid parameters.

    Raises:
        ValueError: If validation fails.
    """
    config_class = get_model_config_class(model_type)

    if config_class is None:
        # Unknown model type, return as-is
        warnings.warn(f"No configuration class found for model type: {model_type}", stacklevel=2)
        return config_dict

    try:
        # Create config instance to validate
        config = config_class(**config_dict)
        # Return validated dict
        return config.model_dump()
    except ValidationError as e:
        raise ValueError(f"Invalid configuration for {model_type}: {e}") from e
