"""Centralized metric key helpers for SharedMetrics."""


def stream_latency_key(stream_type: str) -> str:
    """e.g. 'EEG' -> 'eeg_latency_p50'"""
    return f"{stream_type.lower()}_latency_p50"


def stream_timestamp_key(stream_type: str) -> str:
    """e.g. 'EEG' -> 'eeg_latency_ts'"""
    return f"{stream_type.lower()}_latency_ts"


def mode_metric_key(mode_name: str, metric: str) -> str:
    """e.g. ('sync_mode_1', 'inference_ms') -> 'sync_mode_1_inference_ms'"""
    return f"{mode_name}_{metric}"


def streamer_metric_key(stream_name: str, metric: str) -> str:
    """e.g. ('Dendrite Visualization', 'bandwidth_kbps') -> 'dendrite_visualization_bandwidth_kbps'"""
    normalized = stream_name.lower().replace(" ", "_")
    return f"{normalized}_{metric}"


def e2e_latency_key() -> str:
    return "e2e_latency_ms"


def channel_quality_key() -> str:
    return "channel_quality"


def manual_bad_channels_key() -> str:
    return "manual_bad_channels"


def calibration_corr_key() -> str:
    return "calibration_corr"


def stream_connected_key(stream_type: str) -> str:
    """e.g. 'EEG' -> 'eeg_connected'"""
    return f"{stream_type.lower()}_connected"


# --- Component lifecycle keys ---


def component_state_key(component_id: str) -> str:
    """e.g. 'daq' -> '_cstate_daq'"""
    return f"_cstate_{component_id}"


def component_error_key(component_id: str) -> str:
    """e.g. 'daq' -> '_cerror_daq'"""
    return f"_cerror_{component_id}"
