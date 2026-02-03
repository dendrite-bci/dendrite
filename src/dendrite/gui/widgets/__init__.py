"""
Organized UI components and custom widgets for Dendrite GUI.

Structure:
- common/    - Reusable widgets used across sections
- sections/  - Section-specific widgets (general, modes, preprocessing, output)
- display/   - Display and monitoring widgets
"""

from .common import CompactResourceBar, ControlButtonsWidget, PillNavigation, TopNavigationBar
from .display import LogDisplayWidget, TelemetryWidget
from .sections import (
    GeneralParametersWidget,
    ModeInstanceBadge,
    ModeInstanceConfigDialog,
    OutputWidget,
    PreprocessingWidget,
    StreamConfigurationWidget,
)

__all__ = [
    "CompactResourceBar",
    "ControlButtonsWidget",
    "GeneralParametersWidget",
    "LogDisplayWidget",
    "ModeInstanceBadge",
    "ModeInstanceConfigDialog",
    "OutputWidget",
    "PillNavigation",
    "PreprocessingWidget",
    "StreamConfigurationWidget",
    "TelemetryWidget",
    "TopNavigationBar",
]
