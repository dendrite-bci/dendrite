"""Typed pipeline configuration contract.

PipelineConfig is the validated shape returned by ConfigService.build_configuration().
Validates at build time so errors surface before subprocess spawn.
For cross-process serialization, call .model_dump().
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dendrite.data.stream_schemas import StreamMetadata


class PipelineConfig(BaseModel):
    """Full pipeline configuration validated at build time."""

    model_config = ConfigDict(extra="forbid")

    # General
    study_name: str = "default_study"
    subject_id: str = ""
    session_id: str = ""
    recording_name: str = "recording"
    experiment_description: str = ""

    # Streams
    stream_configs: list[StreamMetadata] = Field(default_factory=list)
    modalities_by_stream: dict[str, Any] = Field(default_factory=dict)

    # Modes (individual entries already validated by mode_schemas.py)
    mode_instances: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # Output protocols
    output: dict[str, Any] = Field(default_factory=dict)
