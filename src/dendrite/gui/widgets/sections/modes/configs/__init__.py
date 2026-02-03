"""
Mode-Specific Configuration Widgets

Configuration widgets for different Dendrite processing modes.
"""

from .asynchronous import AsynchronousModeConfig
from .base import BaseModeConfig, DecoderSource
from .factory import ModeConfigRegistry, ModeWidgetFactory
from .neurofeedback import NeurofeedbackModeConfig
from .synchronous import SynchronousModeConfig

# Register all mode configurations
ModeConfigRegistry.register_mode("Synchronous", SynchronousModeConfig)
ModeConfigRegistry.register_mode("Asynchronous", AsynchronousModeConfig)
ModeConfigRegistry.register_mode("Neurofeedback", NeurofeedbackModeConfig)

__all__ = [
    # Registry and base classes
    "ModeConfigRegistry",
    "BaseModeConfig",
    "ModeWidgetFactory",
    "DecoderSource",
    # Specific mode configs
    "SynchronousModeConfig",
    "AsynchronousModeConfig",
    "NeurofeedbackModeConfig",
]
