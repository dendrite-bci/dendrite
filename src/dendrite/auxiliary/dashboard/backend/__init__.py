"""Backend logic for the Dashboard app."""

from .data_handlers import DataBufferManager
from .lsl_receiver import DEFAULT_STREAM_NAME, OptimizedLSLReceiver
from .mode_manager import ModeManager
from .signal_quality import SignalQualityAnalyzer

__all__ = [
    "OptimizedLSLReceiver",
    "DEFAULT_STREAM_NAME",
    "DataBufferManager",
    "ModeManager",
    "SignalQualityAnalyzer",
]
