"""
Pydantic validation schemas for stream configuration and metadata.

StreamConfig: Core stream configuration for creating LSL outlets.
StreamMetadata: Extended metadata from LSL discovery with validation tracking.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


def _validate_channel_list_length(v: list[str], info: ValidationInfo, field_name: str) -> list[str]:
    """Validate that a channel metadata list matches channel_count if provided."""
    if v and "channel_count" in info.data:
        if len(v) != info.data["channel_count"]:
            raise ValueError(
                f"{field_name} length ({len(v)}) must match channel_count ({info.data['channel_count']})"
            )
    return v


class StreamConfig(BaseModel):
    """
    Core stream configuration for creating LSL outlets.

    Contains the essential fields needed to create and configure an LSL stream.
    Used by LSLOutlet and can be extended by StreamMetadata for input validation.
    """

    model_config = ConfigDict(extra="allow", validate_assignment=True, str_strip_whitespace=True)

    name: str = Field(..., min_length=1, description="Stream name")
    type: str = Field(..., description="Stream type (EEG, EMG, Events, etc)")
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

    acquisition_info: dict[str, str] = Field(
        default_factory=dict, description="Hardware/acquisition metadata"
    )

    @field_validator("labels")
    @classmethod
    def validate_labels_length(cls, v: list[str], info: ValidationInfo) -> list[str]:
        """Ensure labels list matches channel count if provided."""
        return _validate_channel_list_length(v, info, "Channel labels")

    @field_validator("channel_types")
    @classmethod
    def validate_channel_types_length(cls, v: list[str], info: ValidationInfo) -> list[str]:
        """Ensure channel_types list matches channel count if provided."""
        return _validate_channel_list_length(v, info, "Channel types")

    @field_validator("channel_units")
    @classmethod
    def validate_channel_units_length(cls, v: list[str], info: ValidationInfo) -> list[str]:
        """Ensure channel_units list matches channel count if provided."""
        return _validate_channel_list_length(v, info, "Channel units")

    @field_validator("name")
    @classmethod
    def validate_name_not_empty(cls, v):
        """Ensure name is not just whitespace."""
        if not v.strip():
            raise ValueError("Stream name cannot be empty or whitespace")
        return v.strip()


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
