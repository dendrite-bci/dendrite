"""Shared types for data loaders."""

from dataclasses import dataclass

import numpy as np


@dataclass
class LoadedData:
    """Standardized output from file loaders."""

    data: np.ndarray  # (samples, channels)
    channel_names: list[str]
    channel_types: list[str]
    sample_rate: float
    events: list[tuple[int, int]]  # [(sample_idx, event_code), ...]
    event_id: dict[str, int] | None = None
