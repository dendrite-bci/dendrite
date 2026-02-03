"""
Data payload type definitions for streamer output formats.

These TypedDicts define the structure of data sent via LSL and other streaming protocols.
"""

from typing import Any, Literal, TypedDict


class RawDataPayload(TypedDict):
    """Raw sensor data payload for unified LSL stream."""

    type: Literal["raw_data"]
    timestamp: float
    data: dict[str, list[float]]  # e.g., {"EEG": [...], "EMG": [...]}
    channel_labels: dict[str, list[str]]  # e.g., {"EEG": ["C3", "C4"], "EMG": ["EMG1"]}
    sample_rate: int  # Effective sample rate after decimation (for dashboard time axis)


class ModeHistoryPayload(TypedDict):
    """Historical mode data for dashboard initialization."""

    type: Literal["mode_history"]
    timestamp: float
    mode_name: str
    mode_type: str
    data: dict[str, Any]
    packets: list[dict[str, Any]]  # Historical packets
    packet_count: int
