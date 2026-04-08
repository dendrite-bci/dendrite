"""
Pydantic configuration schemas for Dendrite decoders.

DecoderConfig is the single configuration class for all decoders — training
hyperparameters, model architecture params, and decoder metadata.
"""

from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator

from dendrite.ml.models import get_model_config_class, validate_model_config
from dendrite.processing.preprocessing.preprocessing_schemas import PreprocessingConfig


class DecoderConfig(BaseModel):
    """Complete decoder configuration — training, architecture, and metadata.

    Used for training, inference, and serialization. Extra keys are accepted
    so callers can pass through additional context without manual extraction.
    """

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    # ── Core ──────────────────────────────────────────────────────────────
    model_type: str = Field(default="EEGNet", description="Model/decoder type")
    num_classes: int = Field(default=2, ge=2, description="Number of output classes")
    device: str = Field(default="auto", description="Device (auto/cpu/cuda/mps)")

    # ── Training hyperparameters ──────────────────────────────────────────
    # Fields with json_schema_extra={"hpo": ...} are searchable via Optuna.
    learning_rate: float = Field(
        default=0.001, gt=0.0, le=1.0,
        json_schema_extra={"hpo": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True}},
    )
    optimizer_type: Literal["Adam", "AdamW"] = Field(
        default="Adam",
        json_schema_extra={"hpo": {"type": "categorical", "choices": ["Adam", "AdamW"]}},
    )
    epochs: int = Field(default=100, ge=1, le=1000)
    batch_size: int = Field(
        default=32, ge=1, le=512,
        json_schema_extra={"hpo": {"type": "categorical", "choices": [8, 16, 32, 64, 128]}},
    )
    seed: int = Field(default=42, ge=0)
    validation_split: float = Field(default=0.2, ge=0.0, le=0.5)
    weight_decay: float = Field(
        default=0.0, ge=0.0,
        json_schema_extra={"hpo": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True}},
    )
    max_norm_constraint: float | None = Field(default=0.25, ge=0.0)
    label_smoothing_factor: float = Field(
        default=0.0, ge=0.0, le=0.3,
        json_schema_extra={"hpo": {"type": "float", "low": 0.0, "high": 0.2}},
    )

    # Early stopping
    use_early_stopping: bool = Field(default=True)
    early_stopping_patience: int = Field(
        default=10, ge=1, le=100,
        json_schema_extra={"hpo": {"type": "int", "low": 5, "high": 20}},
    )
    early_stopping_min_delta: float = Field(default=1e-4, ge=0.0)

    # Augmentation
    use_augmentation: bool = Field(default=False)
    aug_strategy: str = Field(default="moderate")
    use_class_weights: bool = Field(default=True)
    class_weight_strategy: Literal["balanced", "inverse"] = Field(default="balanced")
    mixup_alpha: float = Field(
        default=0.0, ge=0.0, le=1.0,
        json_schema_extra={"hpo": {"type": "float", "low": 0.0, "high": 0.4}},
    )
    mixup_type: Literal["mixup", "cutmix"] = Field(default="mixup")

    # Loss
    loss_type: Literal["cross_entropy", "focal"] = Field(
        default="cross_entropy",
        json_schema_extra={"hpo": {"type": "categorical", "choices": ["cross_entropy", "focal"]}},
    )
    focal_gamma: float = Field(default=2.0, ge=0.5, le=5.0)

    # LR scheduling
    use_lr_scheduler: bool = Field(default=True)
    lr_scheduler_type: Literal["ReduceLROnPlateau", "StepLR", "CosineAnnealingLR", "OneCycleLR"] = (
        Field(
            default="OneCycleLR",
            json_schema_extra={"hpo": {"type": "categorical", "choices": [
                "OneCycleLR", "ReduceLROnPlateau", "CosineAnnealingLR", "StepLR",
            ]}},
        )
    )
    lr_patience: int = Field(default=5, ge=1)
    lr_factor: float = Field(default=0.5, gt=0.0, lt=1.0)
    lr_min: float = Field(default=1e-6, gt=0.0)
    lr_step_size: int = Field(default=30, ge=1)
    use_lr_warmup: bool = Field(default=False)
    warmup_epochs: int = Field(default=5, ge=1, le=50)
    warmup_start_factor: float = Field(default=0.1, gt=0.0, le=1.0)
    onecycle_max_lr: float | None = Field(default=None, gt=0.0)
    onecycle_pct_start: float = Field(default=0.3, gt=0.0, lt=1.0)

    # SWA
    use_swa: bool = Field(default=False)
    swa_start_epoch: float = Field(default=0.75, gt=0.0, le=1.0)
    swa_lr: float | None = Field(default=None, gt=0.0)

    # ── Model architecture ────────────────────────────────────────────────
    model_params: dict[str, Any] = Field(
        default_factory=dict, description="Model-specific parameters"
    )
    input_shapes: dict[str, list[int]] | None = Field(default=None)
    pipeline_steps: list[str] | None = Field(default=None)

    # ── Decoder metadata ──────────────────────────────────────────────────
    event_mapping: dict[int, str] | None = Field(default=None)
    label_mapping: dict[str, int] | None = Field(default=None)
    sample_rate: float | None = Field(default=500.0, gt=0.0)
    target_sample_rate: float | None = Field(default=None, gt=0.0)
    modality: str | None = Field(default=None)
    channel_labels: dict[str, list[str]] | None = Field(default=None)

    # Training reproducibility / provenance
    epoch_tmin: float | None = Field(default=None)
    epoch_tmax: float | None = Field(default=None)
    preprocessing_config: PreprocessingConfig | None = Field(default=None)
    training_recording_ids: list[int] | None = Field(default=None)
    training_file_identifier: str | None = Field(default=None)

    # ── Validators ────────────────────────────────────────────────────────

    @field_validator("model_params")
    @classmethod
    def validate_model_params(cls, v, info):
        """Validate model_params against model-specific config if available."""
        if not isinstance(v, dict):
            v = {}
        model_type = info.data.get("model_type")
        if model_type and v:
            try:
                config_class = get_model_config_class(model_type)
                if config_class:
                    return validate_model_config(model_type, v)
            except ValueError as e:
                raise ValueError(f"Invalid model parameters for {model_type}: {e}") from e
        return v

    @field_validator("event_mapping", mode="before")
    @classmethod
    def convert_event_mapping_keys(cls, v):
        """Convert string keys to integers (JSON always stringifies object keys)."""
        if not isinstance(v, dict):
            return v
        converted = {}
        for key, value in v.items():
            try:
                converted[int(key) if isinstance(key, str) else key] = value
            except (ValueError, TypeError):
                raise ValueError(
                    f"Event mapping key '{key}' must be convertible to integer"
                ) from None
        return converted

    @field_validator("label_mapping")
    @classmethod
    def validate_label_mapping_consistency(cls, v, info):
        """Ensure label_mapping class count matches num_classes."""
        if not v:
            return None
        num_classes = info.data.get("num_classes")
        if num_classes is not None and len(set(v.values())) != num_classes:
            raise ValueError(
                f"label_mapping has {len(set(v.values()))} unique classes, "
                f"but num_classes={num_classes}"
            )
        return v

    @field_validator("input_shapes")
    @classmethod
    def validate_input_shapes(cls, v):
        """Validate input shapes have positive dimensions."""
        if v is None:
            return v
        for modality, shape in v.items():
            if not isinstance(shape, (list, tuple)) or len(shape) < 2:
                raise ValueError(f"Input shape for '{modality}' must have at least 2 dimensions")
            if any(dim <= 0 for dim in shape):
                raise ValueError(f"Input shape dimensions must be positive for '{modality}'")
        return v

    # ── Helpers ───────────────────────────────────────────────────────────

    def get_device(self) -> torch.device:
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.device)

    def get_model_specific_params(self) -> dict[str, Any]:
        return self.model_params.copy() if self.model_params else {}

    @property
    def effective_sample_rate(self) -> float:
        return self.target_sample_rate or self.sample_rate or 500.0

    def _check_shapes(self, decoder_shapes, system_shapes):
        issues = []
        for modality, system_shape in system_shapes.items():
            decoder_shape = decoder_shapes.get(modality)
            if decoder_shape and decoder_shape[0] != system_shape[0]:
                issues.append(
                    f"{modality.upper()}: decoder needs {decoder_shape[0]} channels, "
                    f"system has {system_shape[0]}"
                )
        return issues

    def _check_channel_labels(self, system_labels):
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
                examples = ", ".join(f"idx {i}: '{d}'->'{s}'" for i, d, s in mismatches[:3])
                suffix = f" (and {len(mismatches) - 3} more)" if len(mismatches) > 3 else ""
                issues.append(
                    f"{modality.upper()}: {len(mismatches)} channel label mismatch(es): "
                    f"{examples}{suffix}"
                )
        return issues

    def _check_sample_rate(self, system_sample_rate):
        expected_rate = self.effective_sample_rate
        if abs(expected_rate - system_sample_rate) > 0.1:
            return [
                f"Sample rate mismatch: decoder trained at {expected_rate:.0f}Hz, "
                f"system is {system_sample_rate:.0f}Hz"
            ]
        return []

    def check_compatibility(
        self,
        system_shapes: dict[str, list[int]],
        system_labels: dict[str, list[str]] | None = None,
        system_sample_rate: float | None = None,
    ) -> list[str]:
        issues = []
        decoder_shapes = self.input_shapes or {}
        system_shapes_lower = {k.lower(): v for k, v in system_shapes.items()}
        for modality in system_shapes_lower:
            if modality not in decoder_shapes:
                issues.append(f"Decoder missing modality: {modality.upper()}")
        issues.extend(self._check_shapes(decoder_shapes, system_shapes_lower))
        if system_labels:
            issues.extend(self._check_channel_labels(system_labels))
        if system_sample_rate:
            issues.extend(self._check_sample_rate(system_sample_rate))
        return issues
