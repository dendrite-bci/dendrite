"""
Pydantic configuration schemas for Dendrite decoders.

This module defines the valid parameters for each classifier type using Pydantic models,
replacing the scattered manual parameter filtering with automatic validation.
"""

from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator

from dendrite.ml.models import get_model_config_class, validate_model_config
from dendrite.processing.preprocessing.preprocessing_schemas import PreprocessingConfig


class NeuralNetConfig(BaseModel):
    """Configuration schema for neural network classifiers."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    # Core parameters
    model_type: str = Field(default="EEGNet", description="Type of neural network model")
    num_classes: int = Field(default=2, ge=2, description="Number of output classes")
    learning_rate: float = Field(
        default=0.001, gt=0.0, le=1.0, description="Learning rate for optimizer"
    )
    device: str = Field(
        default="auto", description="Device for model execution (auto/cpu/cuda/mps)"
    )

    # Optimizer selection (AdamW recommended for better weight decay handling)
    optimizer_type: Literal["Adam", "AdamW"] = Field(
        default="Adam",  # Keep Adam as default for backwards compatibility
        description="Optimizer type (AdamW has better weight decay, recommended for few-shot)",
    )

    # Training parameters with sensible defaults
    epochs: int = Field(default=100, ge=1, le=1000, description="Number of training epochs")
    batch_size: int = Field(default=32, ge=1, le=512, description="Batch size for training")
    validation_split: float = Field(
        default=0.2, ge=0.0, le=0.5, description="Validation split ratio"
    )

    # Early stopping parameters with defaults
    use_early_stopping: bool = Field(default=True, description="Enable early stopping")
    early_stopping_patience: int = Field(
        default=10, ge=1, le=100, description="Early stopping patience (epochs)"
    )
    early_stopping_min_delta: float = Field(
        default=1e-4, ge=0.0, description="Minimum change to qualify as improvement"
    )

    # Regularization parameters with defaults
    weight_decay: float = Field(
        default=0.0, ge=0.0, description="Weight decay for regularization (use with AdamW)"
    )

    # Max norm constraint (EEGNet-style regularization - clamps classifier weights after each step)
    max_norm_constraint: float | None = Field(
        default=0.25,
        ge=0.0,
        description="Max norm for classifier weights (None to disable). Applied after each optimizer step.",
    )

    # Label smoothing (prevents overconfidence, helps few-shot generalization)
    label_smoothing_factor: float = Field(
        default=0.0,
        ge=0.0,
        le=0.3,
        description="Label smoothing factor (0.1 recommended for few-shot, 0 = disabled)",
    )

    # Data augmentation parameters (moved from mode level)
    use_augmentation: bool = Field(
        default=False, description="Enable data augmentation during training"
    )
    aug_strategy: str = Field(
        default="moderate", description="Augmentation strategy (light/moderate/aggressive)"
    )

    # Class imbalance handling with defaults
    use_class_weights: bool = Field(
        default=True, description="Use class weights for imbalanced data"
    )
    class_weight_strategy: Literal["balanced", "inverse", "equal"] = Field(
        default="balanced", description="Strategy for calculating class weights"
    )

    # Learning rate scheduling with defaults
    use_lr_scheduler: bool = Field(default=True, description="Enable learning rate scheduling")
    lr_scheduler_type: Literal["ReduceLROnPlateau", "StepLR", "CosineAnnealingLR", "OneCycleLR"] = (
        Field(
            default="OneCycleLR",
            description="Type of LR scheduler (OneCycleLR recommended for few-shot)",
        )
    )
    lr_patience: int = Field(default=5, ge=1, description="LR scheduler patience")
    lr_factor: float = Field(default=0.5, gt=0.0, lt=1.0, description="LR reduction factor")
    lr_min: float = Field(default=1e-6, gt=0.0, description="Minimum learning rate")
    lr_step_size: int = Field(default=30, ge=1, description="Step size for StepLR scheduler")

    # LR warmup (critical for Transformers, helpful for all models)
    use_lr_warmup: bool = Field(default=False, description="Enable learning rate warmup")
    warmup_epochs: int = Field(default=5, ge=1, le=50, description="Number of warmup epochs")
    warmup_start_factor: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description="Starting LR factor during warmup (fraction of base LR)",
    )

    # OneCycleLR specific parameters
    onecycle_max_lr: float | None = Field(
        default=None, gt=0.0, description="Max LR for OneCycleLR (None = 10x learning_rate)"
    )
    onecycle_pct_start: float = Field(
        default=0.3, gt=0.0, lt=1.0, description="Fraction of cycle spent increasing LR"
    )

    # Mixup/CutMix augmentation (creates virtual training examples)
    mixup_alpha: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Beta distribution alpha for mixup (0 = disabled, 0.2 recommended when enabled)",
    )
    mixup_type: Literal["mixup", "cutmix", "both"] = Field(
        default="mixup",
        description="Type of mixing augmentation (cutmix better for preserving local structure)",
    )

    # Loss function selection
    loss_type: Literal["cross_entropy", "focal"] = Field(
        default="cross_entropy", description="Loss function type (focal helps with hard examples)"
    )
    focal_gamma: float = Field(
        default=2.0,
        ge=0.5,
        le=5.0,
        description="Focal loss gamma (higher = more focus on hard examples)",
    )

    # SWA (Stochastic Weight Averaging) - averages weights for better generalization
    use_swa: bool = Field(default=False, description="Enable Stochastic Weight Averaging")
    swa_start_epoch: float = Field(
        default=0.75,
        gt=0.0,
        le=1.0,
        description="Start SWA at this fraction of total epochs (e.g., 0.75 = last 25%)",
    )
    swa_lr: float | None = Field(
        default=None, gt=0.0, description="SWA learning rate (None = use current LR)"
    )

    # MC Dropout for uncertainty estimation
    use_mc_dropout: bool = Field(
        default=False, description="Enable MC Dropout for prediction uncertainty estimation"
    )
    mc_dropout_samples: int = Field(
        default=10,
        ge=2,
        le=50,
        description="Number of MC forward passes for uncertainty estimation",
    )

    # Model-specific parameters (passed directly to model constructor)
    model_params: dict[str, Any] = Field(
        default_factory=dict, description="Additional model-specific parameters"
    )

    # Input shapes for different modalities (needed for training)
    input_shapes: dict[str, list[int]] | None = Field(
        default=None, description="Input shapes for different modalities"
    )

    # Pipeline preprocessing
    use_scaler: bool = Field(
        default=True, description="Apply per-channel z-score normalization before model"
    )

    # Pipeline structure - explicit list of pipeline components
    # When specified, overrides use_scaler. Examples: ['scaler', 'classifier'], ['csp', 'lda']
    pipeline_steps: list[str] | None = Field(
        default=None,
        description="Explicit pipeline steps. When None, uses use_scaler to determine steps.",
    )

    @field_validator("model_params")
    @classmethod
    def validate_model_params(cls, v, info):
        """Validate model_params against model-specific config if available."""
        if not isinstance(v, dict):
            v = {}

        # Try to validate against model-specific config
        model_type = info.data.get("model_type")
        if model_type and v:
            try:
                config_class = get_model_config_class(model_type)

                if config_class:
                    # Validate the parameters
                    validated = validate_model_config(model_type, v)
                    # Return validated params
                    return validated
            except ValueError as e:
                # Raise error for invalid model parameters - don't silently ignore
                raise ValueError(f"Invalid model parameters for {model_type}: {e}") from e

        return v

    def get_device(self) -> torch.device:
        """Get torch device from config."""
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.device)

    def get_model_specific_params(self) -> dict[str, Any]:
        """Extract parameters specific to the configured model type.

        Returns model_params dict. Each model's default_parameters (from
        get_model_info) handles its own dropout parameter naming.

        Returns:
            Dictionary of model-specific parameters
        """
        return self.model_params.copy() if self.model_params else {}


class DecoderConfig(NeuralNetConfig):
    """Complete decoder configuration - used for training AND metadata storage."""

    # Decoder metadata fields
    event_mapping: dict[int, str] | None = Field(
        default=None, description="Event ID to event name mapping"
    )
    label_mapping: dict[str, int] | None = Field(
        default=None, description="Event name to class index mapping"
    )
    sample_rate: float | None = Field(
        default=500.0, gt=0.0, description="Original data sample rate in Hz"
    )
    target_sample_rate: float | None = Field(
        default=None, gt=0.0, description="Sample rate after resampling (None = no resampling)"
    )
    modality: str | None = Field(default=None, description="Signal modality (EEG, EMG, etc.)")

    # Runtime metadata
    input_shapes: dict[str, list[int]] | None = Field(
        default=None, description="Input shapes for each modality"
    )
    channel_labels: dict[str, list[str]] | None = Field(
        default=None, description="Channel labels per modality {'eeg': ['Fp1', 'Fp2', ...]}"
    )

    # Training reproducibility
    preprocessing_config: PreprocessingConfig | None = Field(
        default=None, description="Preprocessing config used during training for reproducibility"
    )

    @field_validator("event_mapping", mode="before")
    @classmethod
    def convert_event_mapping_keys(cls, v):
        """Convert string keys to integers for JSON compatibility.

        JSON serialization always converts object keys to strings, but we need
        integer event IDs for proper decoder functionality.
        """
        if v is None:
            return v

        if isinstance(v, dict):
            converted = {}
            for key, value in v.items():
                try:
                    # Convert string keys to integers
                    int_key = int(key) if isinstance(key, str) else key
                    converted[int_key] = value
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Event mapping key '{key}' must be convertible to integer"
                    ) from None
            return converted

        return v

    @field_validator("label_mapping")
    @classmethod
    def validate_label_mapping_consistency(cls, v, info):
        """Ensure label_mapping is consistent with num_classes if both are provided."""
        num_classes = info.data.get("num_classes")
        if v is not None and num_classes is not None:
            actual_classes = len(set(v.values()))
            if actual_classes != num_classes:
                raise ValueError(
                    f"label_mapping has {actual_classes} unique classes, but num_classes={num_classes}"
                )
        return v

    @field_validator("input_shapes")
    @classmethod
    def validate_input_shapes(cls, v):
        """Validate and normalize input shapes."""
        if v is None:
            return v

        # Normalize keys to lowercase (backward compat with old decoders)
        normalized = {k.lower(): shape for k, shape in v.items()}

        for modality, shape in normalized.items():
            if not isinstance(shape, (list, tuple)) or len(shape) < 2:
                raise ValueError(f"Input shape for '{modality}' must have at least 2 dimensions")
            if any(dim <= 0 for dim in shape):
                raise ValueError(f"Input shape dimensions must be positive for '{modality}'")

        return normalized

    @field_validator("channel_labels")
    @classmethod
    def normalize_channel_labels(cls, v):
        """Normalize channel_labels keys to lowercase."""
        if v is None:
            return v
        return {k.lower(): labels for k, labels in v.items()}

    @property
    def effective_sample_rate(self) -> float:
        """Get the rate at which model was trained (target if resampled, else original)."""
        return self.target_sample_rate or self.sample_rate or 500.0

    def _check_shapes(
        self,
        decoder_shapes: dict[str, list[int]],
        system_shapes: dict[str, list[int]],
    ) -> list[str]:
        """Check channel counts match (time samples are freely configurable)."""
        issues = []
        for modality, system_shape in system_shapes.items():
            decoder_shape = decoder_shapes.get(modality)
            if not decoder_shape:
                continue

            if decoder_shape[0] != system_shape[0]:
                issues.append(
                    f"{modality.upper()}: decoder needs {decoder_shape[0]} channels, "
                    f"system has {system_shape[0]}"
                )
        return issues

    def _check_channel_labels(
        self,
        system_labels: dict[str, list[str]],
    ) -> list[str]:
        """Check channel labels match between decoder and system."""
        if not self.channel_labels:
            return []

        issues = []
        system_labels_lower = {k.lower(): v for k, v in system_labels.items()}

        for modality, labels in system_labels_lower.items():
            decoder_labels = self.channel_labels.get(modality)
            if not decoder_labels or decoder_labels == labels:
                continue

            mismatches = [
                (i, d, s)
                for i, (d, s) in enumerate(zip(decoder_labels, labels, strict=False))
                if d != s
            ]
            if mismatches:
                idx, dec_lbl, sys_lbl = mismatches[0]
                issues.append(
                    f"{modality.upper()}: channel label mismatch at index {idx} "
                    f"(decoder='{dec_lbl}', system='{sys_lbl}')"
                )
        return issues

    def _check_sample_rate(self, system_sample_rate: float) -> list[str]:
        """Check sample rate compatibility."""
        expected_rate = self.effective_sample_rate
        if abs(expected_rate - system_sample_rate) > 0.1:
            return [
                f"Sample rate mismatch: decoder trained at {expected_rate:.0f}Hz, "
                f"system is {system_sample_rate:.0f}Hz (resampling may be needed)"
            ]
        return []

    def check_compatibility(
        self,
        system_shapes: dict[str, list[int]],
        system_labels: dict[str, list[str]] | None = None,
        system_sample_rate: float | None = None,
    ) -> list[str]:
        """Check if this decoder is compatible with the given system configuration.

        Args:
            system_shapes: Expected input shapes per modality {modality: [channels, time_samples]}
            system_labels: Channel labels per modality {modality: ['ch1', 'ch2', ...]}
            system_sample_rate: System sample rate in Hz

        Returns:
            List of compatibility issues (empty if compatible)
        """
        issues = []
        decoder_shapes = self.input_shapes or {}
        system_shapes_lower = {k.lower(): v for k, v in system_shapes.items()}

        # Check modalities exist
        for modality in system_shapes_lower:
            if modality not in decoder_shapes:
                issues.append(f"Decoder missing modality: {modality.upper()}")

        issues.extend(self._check_shapes(decoder_shapes, system_shapes_lower))

        if system_labels:
            issues.extend(self._check_channel_labels(system_labels))

        if system_sample_rate:
            issues.extend(self._check_sample_rate(system_sample_rate))

        return issues
