"""Dendrite GUI Styles - widget styling system with builder methods."""

from .design_tokens import (
    ACCENT,
    ACCENT_HOVER,
    BG_ELEVATED,
    BG_INPUT,
    BG_MAIN,
    BG_PANEL,
    BORDER,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_WARN,
    TEXT_DISABLED,
    TEXT_LABEL,
    TEXT_MAIN,
)
from .widget_styles import WidgetStyles, apply_app_styles


def apply_global_stylesheet(app) -> None:
    """Apply global styles to the application."""
    apply_app_styles(app)


__all__ = [
    "WidgetStyles",
    "apply_app_styles",
    "apply_global_stylesheet",
    "BG_MAIN",
    "BG_PANEL",
    "BG_ELEVATED",
    "BG_INPUT",
    "TEXT_MAIN",
    "TEXT_LABEL",
    "TEXT_DISABLED",
    "BORDER",
    "ACCENT",
    "ACCENT_HOVER",
    "STATUS_OK",
    "STATUS_WARN",
    "STATUS_ERROR",
]
