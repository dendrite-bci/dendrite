"""
Section-Specific Widgets

Widgets organized by functional sections matching the GUI navigation.
"""

from .general import GeneralParametersWidget, StreamConfigurationWidget
from .modes import ModeInstanceBadge, ModeInstanceConfigDialog
from .output import OutputWidget
from .preprocessing import PreprocessingWidget

__all__ = [
    # General section
    "GeneralParametersWidget",
    "StreamConfigurationWidget",
    # Modes section
    "ModeInstanceBadge",
    "ModeInstanceConfigDialog",
    # Preprocessing section
    "PreprocessingWidget",
    # Output section
    "OutputWidget",
]
