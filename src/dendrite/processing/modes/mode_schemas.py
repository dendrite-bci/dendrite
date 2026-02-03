"""
Pydantic validation schemas for Dendrite mode instance configurations.
"""

import copy
import logging
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from dendrite.ml.models import get_available_models
from dendrite.utils.modality import normalize_modality

logger = logging.getLogger(__name__)


VALID_MODES = {"synchronous", "asynchronous", "neurofeedback"}
VALID_DECODER_TYPE = "Decoder"
VALID_MODEL_TYPES = set(get_available_models())
VALID_MODEL_SOURCES = {"pretrained", "sync_mode", "database"}

# Default decoder configurations (training hyperparameters come from NeuralNetConfig)
DEFAULT_SYNC_DECODER_CONFIG = {"decoder_type": "Decoder", "model_config": {"model_type": "EEGNet"}}

DEFAULT_ASYNC_DECODER_CONFIG = {
    "decoder_type": "Decoder",
    "decoder_path": None,
    "model_config": {"model_type": "EEGNet", "num_classes": 2},
}


def _validate_decoder_type(v: dict) -> None:
    """Validate decoder_type field."""
    if v.get("decoder_type", VALID_DECODER_TYPE) != VALID_DECODER_TYPE:
        raise ValueError(f"decoder_type must be '{VALID_DECODER_TYPE}'")


class BaseModeInstanceConfig(BaseModel):
    """Base validation schema for all mode instance configurations."""

    model_config = ConfigDict(
        extra="allow", validate_assignment=True, str_strip_whitespace=True, validate_default=True
    )

    name: str = Field(min_length=1, description="Unique instance name")
    mode: str = Field(description="Mode type: synchronous/asynchronous/neurofeedback")
    channel_selection: dict[str, list[int]] = Field(
        description="Channel indices per modality: {'eeg': [0,1,2,3]}"
    )
    required_modalities: list[str] = Field(
        default_factory=lambda: ["eeg"],
        description="Modalities this mode receives from processor routing",
    )
    stream_sources: dict[str, str] = Field(
        default_factory=dict,
        description="Stream name per modality: {'eeg': 'BioSemi', 'emg': 'EMGDevice'}",
    )
    modality_labels: dict[str, list[str]] = Field(
        default_factory=dict, description="Channel labels per modality for decoder validation"
    )

    event_mapping: dict[int, str] = Field(
        default_factory=dict, description="Event ID to label mapping: {1: 'left', 2: 'right'}"
    )
    file_identifier: str | None = Field(
        default=None, description="Unique identifier for this recording session"
    )
    study_name: str = Field(default="default_study", description="Study name for data organization")

    @field_validator("name", mode="after")
    @classmethod
    def validate_name_not_empty(cls, v):
        if not v:
            raise ValueError("Instance name cannot be empty or whitespace")
        return v

    @field_validator("mode")
    @classmethod
    def validate_mode_type(cls, v):
        if v.lower() not in VALID_MODES:
            raise ValueError(f"Mode must be one of: {VALID_MODES}")
        return v.lower()

    @field_validator("channel_selection")
    @classmethod
    def validate_channel_selection(cls, v):
        if not v:
            raise ValueError("At least one modality with channels must be configured")

        # Enforce single modality per mode instance
        if len(v) > 1:
            raise ValueError(
                f"Only one modality allowed per mode. Got: {list(v.keys())}. "
                "Configure separate mode instances for each modality."
            )

        total_channels = sum(len(ch) for ch in v.values() if ch)
        if total_channels == 0:
            raise ValueError(
                "No channels selected. Please select at least one channel for processing."
            )

        return v

    @field_validator("required_modalities")
    @classmethod
    def validate_required_modalities(cls, v):
        if not v:
            return ["eeg"]  # Default for backward compatibility
        return [normalize_modality(m) for m in v]

    @model_validator(mode="after")
    def validate_model_modality_compatibility(self):
        """Check model supports selected modalities."""
        if not hasattr(self, "decoder_config") or not self.channel_selection:
            return self

        from dendrite.ml.decoders import check_decoder_compatibility, get_decoder_capabilities

        model_type = self.decoder_config.get("model_config", {}).get("model_type", "EEGNet")
        is_compatible, unsupported = check_decoder_compatibility(
            model_type, list(self.channel_selection.keys())
        )

        if not is_compatible:
            raise ValueError(
                f"Model '{model_type}' doesn't support modalities: {', '.join(m.upper() for m in unsupported)}. "
                f"Supported: {', '.join(m.upper() for m in get_decoder_capabilities(model_type))}."
            )
        return self


class SynchronousInstanceConfig(BaseModeInstanceConfig):
    """Validation schema for synchronous mode instances."""

    decoder_config: dict[str, Any] = Field(
        default_factory=lambda: copy.deepcopy(DEFAULT_SYNC_DECODER_CONFIG),
        description="Complete decoder and model configuration",
    )

    start_offset: float = Field(default=0.0, description="Epoch start offset in seconds")
    end_offset: float = Field(default=2.0, gt=0, description="Epoch end offset in seconds")
    training_interval: int = Field(default=10, ge=1, description="Train every N epochs")

    @field_validator("decoder_config")
    @classmethod
    def validate_decoder_config(cls, v):
        # Local import to avoid circular dependency
        from dendrite.ml.decoders.decoder_schemas import NeuralNetConfig

        _validate_decoder_type(v)
        model_config = v.get("model_config", {})
        if model_config:
            try:
                validated = NeuralNetConfig(**model_config)
                v["model_config"] = validated.model_dump()
            except ValidationError as e:
                raise ValueError(f"Invalid model_config: {e}") from e
        return v

    @field_validator("event_mapping", mode="before")
    @classmethod
    def validate_event_mapping(cls, v):
        if not v:
            raise ValueError("Synchronous mode requires at least 2 event classes")

        converted = {}
        for event_id, event_label in v.items():
            try:
                int_id = int(event_id)
            except (ValueError, TypeError):
                raise ValueError(f"Event ID '{event_id}' must be convertible to integer") from None

            if not isinstance(event_label, str) or not event_label.strip():
                raise ValueError(f"Event label for ID {int_id} must be a non-empty string")
            converted[int_id] = event_label

        if len(converted) < 2:
            raise ValueError("Synchronous mode requires at least 2 event classes")

        return converted

    @field_validator("end_offset")
    @classmethod
    def validate_end_after_start(cls, v: float, info: ValidationInfo) -> float:
        if "start_offset" in info.data and v <= info.data["start_offset"]:
            raise ValueError("end_offset must be greater than start_offset")
        return v


class EvaluationConfig(BaseModel):
    """Evaluation configuration for async mode."""

    model_config = ConfigDict(extra="allow")

    background_class: int | None = Field(default=None, ge=0, le=10)


class AsynchronousInstanceConfig(BaseModeInstanceConfig):
    """Validation schema for asynchronous mode instances."""

    decoder_config: dict[str, Any] = Field(
        default_factory=lambda: copy.deepcopy(DEFAULT_ASYNC_DECODER_CONFIG),
        description="Minimal decoder configuration for inference-only async mode",
    )

    window_length_sec: float = Field(
        default=1.0, gt=0, description="Analysis window length in seconds"
    )
    step_size_ms: int = Field(default=100, gt=0, description="Step size between predictions in ms")

    decoder_source: str = Field(
        default="pretrained", description="Decoder source: pretrained/sync_mode/database"
    )
    source_sync_mode: str = Field(default="", description="Source synchronous mode instance name")

    evaluation_config: EvaluationConfig = Field(
        default_factory=EvaluationConfig, description="Evaluation configuration"
    )

    @field_validator("decoder_config")
    @classmethod
    def validate_decoder_config(cls, v):
        _validate_decoder_type(v)
        model_config = v.get("model_config", {})
        if model_config:
            model_type = model_config.get("model_type", "EEGNet")
            if model_type not in VALID_MODEL_TYPES:
                raise ValueError(f"model_type must be one of: {VALID_MODEL_TYPES}")
            num_classes = model_config.get("num_classes", 2)
            if not isinstance(num_classes, int) or num_classes < 2:
                raise ValueError("num_classes must be an integer >= 2")
        decoder_path = v.get("decoder_path")
        if decoder_path and not isinstance(decoder_path, str):
            raise ValueError("decoder_path must be a string")
        return v

    @field_validator("decoder_source")
    @classmethod
    def validate_decoder_source(cls, v):
        if v not in VALID_MODEL_SOURCES:
            raise ValueError(f"decoder_source must be one of: {VALID_MODEL_SOURCES}")
        return v

    @field_validator("evaluation_config", mode="before")
    @classmethod
    def validate_evaluation_config(cls, v):
        if isinstance(v, dict):
            return EvaluationConfig(**v)
        return v

    @model_validator(mode="after")
    def validate_decoder_source_requirements(self):
        if self.decoder_source in ("pretrained", "database"):
            decoder_path = self.decoder_config.get("decoder_path")
            if not decoder_path or not decoder_path.strip():
                raise ValueError("Decoder file path is required")
        elif self.decoder_source == "sync_mode":
            if not self.source_sync_mode:
                raise ValueError("Source synchronous mode instance must be specified")
        return self


class NeurofeedbackInstanceConfig(BaseModeInstanceConfig):
    """Validation schema for neurofeedback mode instances."""

    window_length_sec: float = Field(
        default=1.0, gt=0, description="Analysis window length in seconds"
    )
    step_size_ms: int = Field(default=250, gt=0, description="Step size between features in ms")

    feature_config: dict[str, Any] = Field(
        default_factory=lambda: {
            "target_bands": {"alpha": [8.0, 12.0]},
            "use_relative_power": True,
        },
        description="Feature extraction configuration",
    )

    @field_validator("feature_config")
    @classmethod
    def validate_feature_config(cls, v):
        if not isinstance(v, dict):
            raise ValueError("feature_config must be a dictionary")

        if "use_cluster_mode" in v and not isinstance(v["use_cluster_mode"], bool):
            raise ValueError("use_cluster_mode must be a boolean")

        def validate_band(name: str, freq_range):
            if not isinstance(freq_range, list) or len(freq_range) != 2:
                raise ValueError(f"Band '{name}' must have exactly 2 frequencies [low, high]")
            if freq_range[0] >= freq_range[1]:
                raise ValueError(f"Band '{name}': low frequency must be < high frequency")

        bands_to_validate = {}
        if "target_bands" in v and isinstance(v["target_bands"], dict):
            bands_to_validate.update(v["target_bands"])
        if "target_band" in v and isinstance(v["target_band"], list):
            bands_to_validate["target_band"] = v["target_band"]

        if not bands_to_validate:
            raise ValueError("feature_config must contain 'target_bands' or 'target_band'")

        for band_name, freq_range in bands_to_validate.items():
            validate_band(band_name, freq_range)

        return v


def _get_system_shapes(config: dict, stream_context: dict) -> dict[str, list[int]]:
    """Build input shapes from config and stream context."""
    channel_selection = config.get("channel_selection", {})
    if not channel_selection:
        return {}

    mode_type = config.get("mode", "").lower()
    sample_rate = stream_context.get("sample_rate", 500)

    # Async with decoder: only validate channel counts
    if mode_type == "asynchronous" and config.get("decoder_config", {}).get("decoder_path"):
        return {m.lower(): [len(ch)] for m, ch in channel_selection.items() if ch}

    # Calculate time samples from mode-specific window
    if mode_type == "synchronous":
        window_sec = config.get("end_offset", 2.0) - config.get("start_offset", 0.0)
    elif mode_type in ("asynchronous", "neurofeedback"):
        window_sec = config.get("window_length_sec", 1.0)
    else:
        return {}

    time_samples = int(window_sec * sample_rate)
    return {m.lower(): [len(ch), time_samples] for m, ch in channel_selection.items() if ch}


def _check_decoder_compatibility(config: dict, stream_context: dict) -> list[str]:
    """Check decoder compatibility with current stream config."""
    decoder_path = config.get("decoder_config", {}).get("decoder_path")
    if not decoder_path:
        return []

    system_shapes = _get_system_shapes(config, stream_context)
    if not system_shapes:
        return []

    try:
        from dendrite.ml.decoders import get_decoder_metadata
        from dendrite.ml.decoders.decoder_schemas import DecoderConfig

        metadata = get_decoder_metadata(decoder_path)
        decoder_cfg = DecoderConfig(**metadata)
        return decoder_cfg.check_compatibility(system_shapes)
    except FileNotFoundError:
        return [f"Decoder file not found: {decoder_path}"]
    except ValidationError as e:
        return [f"Invalid decoder metadata: {e}"]


def validate_mode_config(
    config: dict, stream_context: dict | None = None
) -> tuple[bool, list[str], dict | None]:
    """Validate mode config. Returns (is_valid, errors, validated_config)."""
    try:
        mode_type = config.get("mode", "").lower()
        if mode_type not in VALID_MODES:
            return False, [f"Unknown mode: '{mode_type}'. Must be: {VALID_MODES}"], None

        schema_map = {
            "synchronous": SynchronousInstanceConfig,
            "asynchronous": AsynchronousInstanceConfig,
            "neurofeedback": NeurofeedbackInstanceConfig,
        }
        validated = schema_map[mode_type](**config)
        validated_dict = validated.model_dump()

        if stream_context:
            compat_errors = _check_decoder_compatibility(config, stream_context)
            if compat_errors:
                return False, compat_errors, validated_dict

        return True, [], validated_dict

    except ValidationError as e:
        errors = [f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in e.errors()]
        return False, errors, None
    except Exception as e:
        return False, [f"Validation error: {e!s}"], None
