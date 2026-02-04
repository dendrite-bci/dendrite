"""Dataset configuration dataclass."""

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DatasetConfig:
    """Configuration for a dataset.

    Defines all parameters needed to load and process data from a dataset.
    Can be serialized to/from JSON for persistence.
    """

    name: str
    description: str = ""
    data_root: str = ""
    file_pattern: str = "*.fif"
    source_type: str = "fif"  # "fif", "hdf5", "moabb", "edf"
    subjects: list[int] = field(default_factory=list)

    # Events - mapping from event name to label
    events: dict[str, int] = field(default_factory=dict)

    # Block structure (optional) - mapping from split name to block numbers
    blocks: dict[str, list[int]] | None = None

    # Signal parameters
    sample_rate: float = 500.0
    channels: list[str] | None = None  # None = all EEG channels

    # Epoch parameters
    epoch_tmin: float = 0.0
    epoch_tmax: float = 0.5

    # Preprocessing (applied on load)
    preproc_lowcut: float | None = None
    preproc_highcut: float | None = None
    preproc_rereference: bool = False

    # Dataset-specific event name patterns (for flexible event matching)
    event_patterns: dict[str, dict[str, list[str]]] | None = None

    # MOABB-specific fields (only used when source_type="moabb")
    moabb_dataset: str | None = None  # e.g., "BNCI2014_001", "PhysionetMI"
    moabb_paradigm: str | None = None  # e.g., "MotorImagery", "P300"
    moabb_n_classes: int | None = None  # Override paradigm's default n_classes
    moabb_events: list[str] | None = None  # Specific events to select

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetConfig":
        """Create from dictionary."""
        return cls(**data)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "DatasetConfig":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @property
    def window_samples(self) -> int:
        """Epoch window size in samples."""
        return round((self.epoch_tmax - self.epoch_tmin) * self.sample_rate) + 1

    def validate(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues = []

        if not self.name:
            issues.append("name is required")

        # MOABB datasets don't need data_root (data is auto-downloaded)
        if self.source_type == "moabb":
            if not self.moabb_dataset:
                issues.append("moabb_dataset is required for MOABB source type")
            if not self.moabb_paradigm:
                issues.append("moabb_paradigm is required for MOABB source type")
        else:
            if not self.data_root:
                issues.append("data_root is required")

        if not self.subjects:
            issues.append("subjects list is empty")

        # Events can be auto-generated for MOABB
        if not self.events and self.source_type != "moabb":
            issues.append("events mapping is empty")

        if self.sample_rate <= 0:
            issues.append("sample_rate must be positive")

        if self.epoch_tmax <= self.epoch_tmin:
            issues.append("epoch_tmax must be greater than epoch_tmin")

        return issues
