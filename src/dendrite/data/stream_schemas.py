"""
Pydantic validation schemas for stream configuration and metadata.

StreamConfig: Core stream configuration for creating LSL outlets.
StreamMetadata: Extended metadata from LSL discovery with validation tracking.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class StreamConfig(BaseModel):
    """
    Core stream configuration for creating LSL outlets.

    Contains the essential fields needed to create and configure an LSL stream.
    Used by LSLOutlet and can be extended by StreamMetadata for input validation.
    """

    model_config = ConfigDict(extra="allow", validate_assignment=True, str_strip_whitespace=True)

    name: str = Field(..., min_length=1, description="Stream name")
    type: str = Field(
        ...,
        description="Stream type (EEG, EMG, Events, etc). Stored as-is; "
        "call .lower() for modality matching.",
    )
    channel_count: int = Field(..., gt=0, description="Number of channels")
    sample_rate: float = Field(
        ..., ge=0, description="Sampling rate in Hz (0 for irregular rate streams)"
    )

    channel_format: str = Field(
        default="float32", description="Data format (float32, int32, string, etc)"
    )
    source_id: str = Field(default="", description="Stream source identifier")

    labels: list[str] = Field(default_factory=list, description="Channel labels")
    channel_types: list[str] = Field(
        default_factory=list, description="Channel types (EEG, EMG, Markers, etc)"
    )
    channel_units: list[str] = Field(
        default_factory=list, description="Channel units (uV, mV, etc)"
    )

    stream_key: str = Field(
        default="",
        description="Unique key for pipeline dict indexing. Assigned during configuration: "
        "first stream of a type gets the type as key (e.g. 'EEG'), duplicates get '_2', '_3' suffix.",
    )

    acquisition_info: dict[str, str] = Field(
        default_factory=dict, description="Hardware/acquisition metadata"
    )

    @field_validator("labels", "channel_types", "channel_units")
    @classmethod
    def validate_channel_list_length(cls, v: list[str], info: ValidationInfo) -> list[str]:
        """Ensure channel metadata list matches channel count if provided."""
        if v and "channel_count" in info.data:
            if len(v) != info.data["channel_count"]:
                raise ValueError(
                    f"{info.field_name} length ({len(v)}) must match "
                    f"channel_count ({info.data['channel_count']})"
                )
        return v



class StreamMetadata(StreamConfig):
    """
    Validated stream metadata from external sources (LSL, files, network, etc).

    Extends StreamConfig with runtime metadata from LSL discovery and
    validation tracking fields. Used when extracting metadata from
    incoming streams to catch malformed data early.
    """

    uid: str = Field(default="", description="Unique stream identifier")
    created_at: float = Field(default=0.0, description="Stream creation timestamp")
    version: int = Field(default=0, description="Stream version number")
    has_metadata_issues: bool = Field(
        default=False, description="Flag indicating metadata extraction had issues"
    )
    metadata_issues: dict[str, Any] = Field(
        default_factory=dict, description="Details of any metadata extraction issues"
    )

    @property
    def stable_key(self) -> str:
        """Stable identity key that survives streaming app restarts."""
        return self.source_id if self.source_id else f"{self.name}:{self.type}"
