"""
Pydantic validation schema for LSL event data.

EventData: Validates and normalizes event payloads from LSL streams.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EventData(BaseModel):
    """Validated event data from LSL stream.

    Handles key normalization (PascalCase to lowercase) and type coercion
    for event_id. Extra fields are allowed and preserved for custom event data.
    """

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )

    event_id: float = Field(..., description="Event identifier (numeric)")
    event_type: str = Field(..., description="Event type label")

    @field_validator("event_id", mode="before")
    @classmethod
    def coerce_event_id(cls, v: Any) -> float:
        """Coerce event_id to float, raising clear error on failure."""
        try:
            return float(v)
        except (TypeError, ValueError) as e:
            raise ValueError(f"event_id must be numeric, got: {v!r}") from e

    @model_validator(mode="before")
    @classmethod
    def normalize_keys(cls, data: dict) -> dict:
        """Normalize keys to lowercase."""
        if isinstance(data, dict):
            return {k.lower(): v for k, v in data.items()}
        return data
