"""GUI utility functions."""

import logging
from pathlib import Path

from PyQt6 import QtGui, QtWidgets

logger = logging.getLogger(__name__)

# Static assets directory
_STATIC_DIR = Path(__file__).parent / "static"


def load_icon(icon_name: str) -> QtGui.QIcon:
    """Load an icon from the static directory."""
    icon_path = _STATIC_DIR / icon_name
    if icon_path.exists():
        return QtGui.QIcon(str(icon_path))
    logger.warning(f"Icon not found: {icon_path}")
    return QtGui.QIcon()


def set_app_icon(app: QtWidgets.QApplication, icon_name: str) -> bool:
    """Set the application window icon."""
    icon = load_icon(icon_name)
    if not icon.isNull():
        app.setWindowIcon(icon)
        return True
    return False


__all__ = [
    "load_icon",
    "set_app_icon",
]
