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

from dendrite.ml.decoders.registry import get_available_decoders
from dendrite.ml.models import get_available_models
from dendrite.processing.preprocessing.preprocessing_schemas import ModalityPreprocessing
from dendrite.utils.modality import normalize_modality

logger = logging.getLogger(__name__)


VALID_MODES = {"synchronous", "asynchronous", "neurofeedback"}
VALID_MODEL_TYPES = set(get_available_models()) | set(get_available_decoders())
VALID_MODEL_SOURCES = {"database", "online"}

DEFAULT_DECODER_CONFIG: dict[str, Any] = {
    "decoder_type": "Decoder",
    "model_config": {"model_type": "EEGNet", "num_classes": 2},
}


class BaseModeInstanceConfig(BaseModel):
    """Base validation schema for all mode instance configurations."""

    model_config = ConfigDict(
        extra="ignore", str_strip_whitespace=True
    )

    name: str = Field(min_length=1, description="Unique instance name")
    mode: str = Field(description="Mode type: synchronous/asynchronous/neurofeedback")
    enabled: bool = Field(default=True, description="Whether this mode is active for pipeline runs")
    channel_selection: dict[str, list[int]] = Field(
        default_factory=dict,
        description="Channel indices per modality: {'eeg': [0,1,2,3]}",
    )
    stream_sources: dict[str, str] = Field(
        default_factory=dict,
        description="Stream name per modality: {'eeg': 'BioSemi', 'emg': 'EMGDevice'}",
    )
    modality_labels: dict[str, list[str]] = Field(
        default_factory=dict, description="Channel labels per modality for decoder validation"
    )
    source_stream: str | None = Field(
        default=None,
        description="Preferred source stream key for ring buffer selection",
    )
    mode_preprocessing: dict[str, ModalityPreprocessing] = Field(
        default_factory=dict,
        description="Per-mode preprocessing config per modality. "
        "e.g., {'eeg': {'lowcut': 1.0, 'highcut': 45.0, 'apply_rereferencing': True}}",
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
            return v  # Allow empty — preflight validates before pipeline start

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

    @model_validator(mode="after")
    def align_preprocessing_to_modality(self):
        """Ensure mode_preprocessing matches the selected modality."""
        if not self.channel_selection:
            return self
        primary = normalize_modality(next(iter(self.channel_selection)))
        if primary not in self.mode_preprocessing:
            self.mode_preprocessing = {primary: ModalityPreprocessing.default_for(primary)}
        return self

    @model_validator(mode="after")
    def validate_model_modality_compatibility(self):
        """Check model supports selected modalities."""
        decoder_config = getattr(self, "decoder_config", None)
        if decoder_config is None or not self.channel_selection:
            return self

        from dendrite.ml.decoders import check_decoder_compatibility, get_decoder_capabilities

        model_type = decoder_config.get("model_config", {}).get("model_type", "EEGNet")
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
        default_factory=lambda: copy.deepcopy(DEFAULT_DECODER_CONFIG),
        description="Complete decoder and model configuration",
    )

    epoch_tmin: float = Field(default=0.0, description="Epoch start time in seconds (relative to event)")
    epoch_tmax: float = Field(default=2.0, gt=0, description="Epoch end time in seconds (relative to event)")
    training_interval: int = Field(default=10, ge=1, description="Train every N epochs")
    use_epoch_qc: bool = Field(default=True, description="Filter bad epochs during training")
    include_background: bool = Field(
        default=False, description="Train with rest class from inter-trial gaps",
    )
    use_study_history: bool = Field(
        default=False,
        description="Augment live training data with compatible recordings from the study",
    )
    study_history_recording_ids: list[int] | None = Field(
        default=None,
        description="Specific recording IDs to use for study history augmentation",
    )

    @field_validator("decoder_config")
    @classmethod
    def validate_decoder_config(cls, v):
        from dendrite.ml.decoders.decoder_schemas import DecoderConfig

        model_config = v.get("model_config", {})
        if model_config:
            try:
                validated = DecoderConfig(**model_config)
                v["model_config"] = validated.model_dump()
            except ValidationError as e:
                raise ValueError(f"Invalid model_config: {e}") from e
        return v

    @field_validator("event_mapping", mode="before")
    @classmethod
    def validate_event_mapping(cls, v):
        if not v:
            # Provide sensible defaults for new instances
            return {1: "Left", 2: "Right"}

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

    @field_validator("epoch_tmax")
    @classmethod
    def validate_end_after_start(cls, v: float, info: ValidationInfo) -> float:
        if "epoch_tmin" in info.data and v <= info.data["epoch_tmin"]:
            raise ValueError("epoch_tmax must be greater than epoch_tmin")
        return v


class AsynchronousInstanceConfig(BaseModeInstanceConfig):
    """Validation schema for asynchronous mode instances."""

    decoder_config: dict[str, Any] = Field(
        default_factory=lambda: copy.deepcopy(DEFAULT_DECODER_CONFIG),
        description="Minimal decoder configuration for inference-only async mode",
    )

    window_length_sec: float = Field(
        default=1.0, gt=0, description="Analysis window length in seconds"
    )
    step_size_ms: int = Field(default=100, gt=0, description="Step size between predictions in ms")

    decoder_source: str = Field(
        default="database", description="Decoder source: database/online"
    )
    source_mode: str | None = Field(
        default=None,
        description="Paired sync mode name — filters online decoders to this source only",
    )

    @field_validator("decoder_config")
    @classmethod
    def validate_decoder_config(cls, v):
        from dendrite.ml.decoders.decoder_schemas import DecoderConfig

        model_config = v.get("model_config") or {}
        if model_config:
            model_type = model_config.get("model_type", "EEGNet")
            if model_type not in VALID_MODEL_TYPES:
                raise ValueError(f"model_type must be one of: {VALID_MODEL_TYPES}")
            try:
                validated = DecoderConfig(**model_config)
                v["model_config"] = validated.model_dump()
            except ValidationError as e:
                raise ValueError(f"Invalid model_config: {e}") from e
        return v

    @field_validator("decoder_source")
    @classmethod
    def validate_decoder_source(cls, v):
        if v not in VALID_MODEL_SOURCES:
            raise ValueError(f"decoder_source must be one of: {VALID_MODEL_SOURCES}")
        return v

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

        # IAF calibration (optional)
        if "iaf_event_id" in v:
            eid = v["iaf_event_id"]
            if not isinstance(eid, int) or eid < 1:
                raise ValueError("iaf_event_id must be a positive integer")
            iaf_sec = v.get("iaf_baseline_sec", 5.0)
            if not isinstance(iaf_sec, (int, float)) or iaf_sec <= 0:
                raise ValueError("iaf_baseline_sec must be a positive number")
            iaf_range = v.get("iaf_range", [7.0, 14.0])
            if not isinstance(iaf_range, list) or len(iaf_range) != 2:
                raise ValueError("iaf_range must be [low, high]")
            if iaf_range[0] >= iaf_range[1]:
                raise ValueError("iaf_range: low must be < high")

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
        window_sec = config.get("epoch_tmax", 2.0) - config.get("epoch_tmin", 0.0)
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

        # Modality check: decoder's training modalities vs config's channel_selection
        config_modalities = {k.lower() for k in config.get("channel_selection", {})}
        decoder_modalities = set()
        input_shapes = metadata.get("input_shapes")
        if input_shapes and isinstance(input_shapes, dict):
            decoder_modalities = {k.lower() for k in input_shapes}
        elif metadata.get("modality"):
            decoder_modalities = {metadata["modality"].lower()}

        if decoder_modalities and config_modalities and not (config_modalities & decoder_modalities):
            return [
                f"Decoder trained on {', '.join(sorted(decoder_modalities)).upper()}, "
                f"mode uses {', '.join(sorted(config_modalities)).upper()}"
            ]

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
        validated_dict = validated.model_dump(exclude_none=True)

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
