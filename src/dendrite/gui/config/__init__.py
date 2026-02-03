"""
Dendrite GUI Configuration Module

Configuration management utilities for the Dendrite GUI system.
"""

from .study_schemas import StudyConfig
from .system_config_manager import SystemConfigurationManager

__all__ = [
    "StudyConfig",
    "SystemConfigurationManager",
]
