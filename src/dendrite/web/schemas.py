"""
Pydantic models for API request/response schemas.
"""

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dendrite.data.stream_schemas import StreamMetadata

# --- BIDS Validation ---

INVALID_PATH_CHARS = re.compile(r'[<>:"/\\|?*]')
BIDS_LABEL_PATTERN = re.compile(r"^[a-zA-Z0-9]+$")


def validate_no_path_traversal(path: str) -> str:
    """Reject paths containing '..' components."""
    from pathlib import PurePath

    if ".." in PurePath(path).parts:
        raise ValueError("Path traversal ('..') not allowed")
    return path


class StudyConfig(BaseModel):
    """BIDS-compliant study configuration with validation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    study_name: str = Field(default="default_study", min_length=1, max_length=100)
    subject_id: str = Field(default="01", min_length=1, max_length=50)
    session_id: str = Field(default="01", min_length=1, max_length=50)
    recording_name: str = Field(default="task", min_length=1, max_length=100)

    @field_validator("study_name", "recording_name")
    @classmethod
    def validate_path_safe(cls, v: str) -> str:
        if INVALID_PATH_CHARS.search(v):
            raise ValueError('Contains invalid characters: < > : " / \\ | ? *')
        return v

    @field_validator("subject_id", "session_id")
    @classmethod
    def validate_bids_label(cls, v: str) -> str:
        if not BIDS_LABEL_PATTERN.match(v):
            raise ValueError("Must be alphanumeric only (a-z, A-Z, 0-9)")
        return v


# --- Pipeline ---

class PipelineStartRequest(BaseModel):
    """Full configuration for starting the pipeline."""
    study_name: str = "default_study"
    subject_id: str = ""
    session_id: str = ""
    recording_name: str = "recording"
    sample_rate: float = 500.0
    mode_instances: dict = {}
    stream_configs: list = []
    modalities_by_stream: dict = {}
    output: dict = {}
    experiment_description: str = ""


class PipelineStatusResponse(BaseModel):
    """Current pipeline status with per-component states."""
    recording: bool
    recording_id: int | None = None
    elapsed_seconds: float = 0.0
    log_file: str | None = None
    mode_pids: dict[str, int] = {}
    system_pids: dict[str, int] = {}
    component_states: dict[str, str] = {}


# --- Config ---

class GeneralConfigRequest(BaseModel):
    study_name: str = "default_study"
    subject_id: str = ""
    session_id: str = ""
    recording_name: str = "recording"


class OutputConfigRequest(BaseModel):
    protocols: dict[str, Any] = {}


# --- Streams ---

class StreamMetadataResponse(BaseModel):
    """Explicit API response for stream metadata. No extra fields leak to the frontend."""

    uid: str = ""
    name: str
    type: str
    channel_count: int
    sample_rate: float
    channel_format: str = "float32"
    source_id: str = ""
    labels: list[str] = Field(default_factory=list)
    channel_types: list[str] = Field(default_factory=list)
    channel_units: list[str] = Field(default_factory=list)
    stream_key: str = ""
    has_metadata_issues: bool = False
    metadata_issues: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_stream_metadata(cls, meta: StreamMetadata) -> "StreamMetadataResponse":
        return cls.model_validate(meta.model_dump())


class StreamConfigureRequest(BaseModel):
    selected_uids: list[str]
    channel_overrides: dict[str, dict[str, Any]] = {}


# --- Modes ---

class ModeInstanceRequest(BaseModel):
    name: str | None = None
    mode: str = "synchronous"
    config: dict[str, Any] = {}


class ModeRenameRequest(BaseModel):
    new_name: str


# --- Preflight ---

class PreflightCheck(BaseModel):
    id: str
    label: str
    passed: bool
    required: bool = True
    detail: str | None = None


class PreflightResult(BaseModel):
    ready: bool
    checks: list[PreflightCheck]


# --- Data Explorer ---

class StudyCreateRequest(BaseModel):
    study_name: str
    description: str | None = None


class StudyUpdateRequest(BaseModel):
    description: str | None = None


# --- ML Workbench ---

class TrainingStartRequest(BaseModel):
    study_id: int | None = None
    recording_id: int | None = None
    file_path: str | None = None
    model_type: str
    pipeline_steps: list[str] | None = None
    num_classes: int = 2
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    holdout_ratio: float = 0.0
    optuna_enabled: bool = False
    optuna_n_trials: int | None = None
    search_categories: list[str] | None = None
    event_mapping: dict[int, str] | None = None
    label_mapping: dict[str, int] | None = None
    model_params: dict[str, Any] = {}
    optimizer_type: str = "Adam"
    weight_decay: float = 0.0
    use_augmentation: bool = False
    aug_strategy: str = "moderate"
    use_class_weights: bool = True
    use_lr_scheduler: bool = True
    lr_scheduler_type: str = "OneCycleLR"
    loss_type: str = "cross_entropy"
    label_smoothing_factor: float = 0.0
    mixup_alpha: float = 0.0
    use_loaded_data: bool = False
    channel_indices: list[int] | None = None
    selected_events: list[str] | None = None
    lowcut: float | None = None
    highcut: float | None = None
    apply_rereferencing: bool = False
    epoch_tmin: float = 0.0
    epoch_tmax: float = 2.0
    use_epoch_qc: bool = True
    include_background: bool = False




class SaveDecoderRequest(BaseModel):
    decoder_name: str
    description: str | None = None


# --- ML Data Loading ---

class MoabbLoadRequest(BaseModel):
    dataset_code: str
    subject: int = 1
    paradigm: str = "MotorImagery"
    lowcut: float | None = None
    highcut: float | None = None
    apply_rereferencing: bool = False


class RecordingLoadRequest(BaseModel):
    recording_id: int | None = None
    recording_ids: list[int] | None = None
    eval_recording_ids: list[int] | None = None
    eval_split: float = 0.2
    lowcut: float | None = None
    highcut: float | None = None
    apply_rereferencing: bool = False
    epoch_tmin: float = -0.2
    epoch_tmax: float = 0.8
    selected_events: list[str] | None = None
    channel_indices: list[int] | None = None
    use_epoch_qc: bool = True
    include_background: bool = False


# --- ML Evaluation ---

class EvaluationStartRequest(BaseModel):
    job_id: int
    mode: str = "epoch"
    step_size_ms: int = 100
    detection_strategy: str = "dwell"
    dwell_n: int = 3
    confidence_threshold: float = 0.0


# --- ML Benchmark ---

class BenchmarkStartRequest(BaseModel):
    model_types: list[str]
    n_folds: int = 5
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001


# --- Generic ---

class ErrorResponse(BaseModel):
    detail: str
