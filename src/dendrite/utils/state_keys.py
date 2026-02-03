"""
Centralized metric registry for SharedMetrics.

Provides type-safe key generation for all shared metrics, eliminating
scattered f-string patterns across the codebase.
"""

from enum import Enum
from typing import NamedTuple


class MetricType(Enum):
    """Metric data types."""

    FLOAT = "float"
    DICT = "dict"


class Metric(NamedTuple):
    """Metric definition."""

    name: str
    metric_type: MetricType
    unit: str = ""
    description: str = ""


# Static metrics (always present)
STATIC_METRICS = {
    # System latency
    "e2e_latency_ms": Metric("e2e_latency_ms", MetricType.FLOAT, "ms", "End-to-end latency"),
    # Visualization bandwidth
    "viz_stream_bandwidth_kbps": Metric("viz_stream_bandwidth_kbps", MetricType.FLOAT, "kbps"),
    "viz_stream_consumers": Metric("viz_stream_consumers", MetricType.FLOAT, "", "Consumer count"),
    "bmi_visualization_bandwidth_kbps": Metric(
        "bmi_visualization_bandwidth_kbps", MetricType.FLOAT, "kbps"
    ),
    "predictionstream_bandwidth_kbps": Metric(
        "predictionstream_bandwidth_kbps", MetricType.FLOAT, "kbps"
    ),
    "channel_info": Metric("channel_info", MetricType.DICT, "", "Channel metadata"),
    "channel_quality": Metric("channel_quality", MetricType.DICT, "", "Quality control results"),
}


def stream_latency_key(stream_type: str) -> str:
    """Generate latency metric key for stream type.

    Args:
        stream_type: Stream type (e.g., 'EEG', 'Events', 'EMG')

    Returns:
        Metric key (e.g., 'eeg_latency_p50')
    """
    return f"{stream_type.lower()}_latency_p50"


def stream_timestamp_key(stream_type: str) -> str:
    """Generate timestamp metric key for stream type.

    Args:
        stream_type: Stream type (e.g., 'EEG', 'Events', 'EMG')

    Returns:
        Metric key (e.g., 'eeg_latency_ts')
    """
    return f"{stream_type.lower()}_latency_ts"


def mode_metric_key(mode_name: str, metric: str) -> str:
    """Generate metric key for mode instance.

    Args:
        mode_name: Mode instance name (e.g., 'sync_mode_1')
        metric: Metric suffix (e.g., 'internal_ms', 'inference_ms', 'gpu_mb')

    Returns:
        Metric key (e.g., 'sync_mode_1_internal_ms')
    """
    return f"{mode_name}_{metric}"


def streamer_metric_key(stream_name: str, metric: str) -> str:
    """Generate metric key for streamer.

    Args:
        stream_name: Streamer name (e.g., 'LSL', 'Visualization', 'Dendrite_Visualization')
        metric: Metric suffix (e.g., 'bandwidth_kbps', 'msg_rate')

    Returns:
        Metric key (e.g., 'lsl_bandwidth_kbps', 'dendrite_visualization_bandwidth_kbps')
    """
    # Normalize: lowercase and replace spaces with underscores
    normalized = stream_name.lower().replace(" ", "_")
    return f"{normalized}_{metric}"


def e2e_latency_key() -> str:
    """Get key for end-to-end latency metric."""
    return STATIC_METRICS["e2e_latency_ms"].name


def channel_quality_key() -> str:
    """Get key for channel quality dict metric."""
    return STATIC_METRICS["channel_quality"].name


def viz_consumers_key() -> str:
    """Get key for visualization stream consumer count."""
    return STATIC_METRICS["viz_stream_consumers"].name


def viz_bandwidth_key() -> str:
    """Get key for visualization stream bandwidth."""
    return STATIC_METRICS["viz_stream_bandwidth_kbps"].name


def stream_connected_key(stream_type: str) -> str:
    """Generate connection status key for stream type.

    Args:
        stream_type: Stream type (e.g., 'EEG', 'Events', 'EMG')

    Returns:
        Metric key (e.g., 'eeg_connected')
    """
    return f"{stream_type.lower()}_connected"
