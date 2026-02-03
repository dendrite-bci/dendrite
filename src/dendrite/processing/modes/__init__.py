"""
Processing modes for Dendrite operation.

For mode classes, import directly:
- `from dendrite.processing.modes.synchronous_mode import SynchronousMode`
"""

from dendrite.processing.modes.asynchronous_mode import AsynchronousMode
from dendrite.processing.modes.neurofeedback_mode import NeurofeedbackMode
from dendrite.processing.modes.synchronous_mode import SynchronousMode

_AVAILABLE_MODES = {
    "synchronous": SynchronousMode,
    "asynchronous": AsynchronousMode,
    "neurofeedback": NeurofeedbackMode,
}

__all__ = ["create_mode", "get_available_modes"]


def create_mode(mode_type, **kwargs):
    """Create a mode instance by type. Raises ValueError if unknown."""
    mode_type_lower = mode_type.lower()
    if mode_type_lower in _AVAILABLE_MODES:
        return _AVAILABLE_MODES[mode_type_lower](**kwargs)
    raise ValueError(f"Unknown mode type: {mode_type}. Available: {list(_AVAILABLE_MODES.keys())}")


def get_available_modes():
    """Get list of available mode type names."""
    return list(_AVAILABLE_MODES.keys())
