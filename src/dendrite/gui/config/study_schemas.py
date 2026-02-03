"""
Pydantic schemas for study/session configuration.

StudyConfig: BIDS-compliant study parameters with filesystem and naming validation.
"""

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Filesystem-unsafe characters (Windows/Unix)
INVALID_PATH_CHARS = re.compile(r'[<>:"/\\|?*]')

# BIDS spec: labels must be alphanumeric only
BIDS_LABEL_PATTERN = re.compile(r"^[a-zA-Z0-9]+$")


class StudyConfig(BaseModel):
    """
    BIDS-compliant study configuration with validation.

    Validates:
    - study_name, recording_name: filesystem-safe (no special chars)
    - subject_id, session_id: BIDS-compliant (alphanumeric only)
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    study_name: str = Field(default="default_study", min_length=1, max_length=100)
    subject_id: str = Field(default="001", min_length=1, max_length=50)
    session_id: str = Field(default="01", min_length=1, max_length=50)
    recording_name: str = Field(default="task", min_length=1, max_length=100)

    @field_validator("study_name", "recording_name")
    @classmethod
    def validate_path_safe(cls, v: str) -> str:
        """Validate filesystem-safe characters."""
        if INVALID_PATH_CHARS.search(v):
            raise ValueError('Contains invalid characters: < > : " / \\ | ? *')
        return v

    @field_validator("subject_id", "session_id")
    @classmethod
    def validate_bids_label(cls, v: str) -> str:
        """Validate BIDS-compliant label (alphanumeric only)."""
        if not BIDS_LABEL_PATTERN.match(v):
            raise ValueError("Must be alphanumeric only (a-z, A-Z, 0-9)")
        return v
