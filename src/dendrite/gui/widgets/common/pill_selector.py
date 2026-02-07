"""
Pill Selector Widget

Segmented pill-style button selector with exclusive selection.
"""

from PyQt6 import QtCore, QtWidgets

from dendrite.gui.styles.design_tokens import (
    ACCENT,
    BG_INPUT,
    BG_PANEL,
    BORDER,
    TEXT_LABEL,
    TEXT_MAIN,
)
from dendrite.gui.styles.widget_styles import FONTS, LAYOUT


class PillSelector(QtWidgets.QWidget):
    """Segmented pill-style button selector with exclusive selection.

    Example:
        selector = PillSelector([
            ("train", "Train", "Load data for training"),
            ("eval", "Eval", "Load data for evaluation"),
        ])
        selector.selection_changed.connect(on_role_changed)
    """

    selection_changed = QtCore.pyqtSignal(str)

    def __init__(self, options: list[tuple[str, str, str]], parent=None):
        """
        Args:
            options: List of (value, label, tooltip) tuples
        """
        super().__init__(parent)
        self._options = options
        self._current = options[0][0] if options else ""
        self._buttons: list[QtWidgets.QPushButton] = []
        self._setup_ui()

    @property
    def current(self) -> str:
        """Get the currently selected value."""
        return self._current

    def set_current(self, value: str):
        """Set the current selection by value."""
        for i, (val, _, _) in enumerate(self._options):
            if val == value:
                self._buttons[i].setChecked(True)
                self._current = value
                break

    def _setup_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._button_group = QtWidgets.QButtonGroup(self)
        self._button_group.setExclusive(True)

        for i, (_value, label, tooltip) in enumerate(self._options):
            btn = QtWidgets.QPushButton(label)
            btn.setCheckable(True)
            btn.setToolTip(tooltip)
            if i == 0:
                btn.setChecked(True)
            self._button_group.addButton(btn, i)
            self._buttons.append(btn)
            layout.addWidget(btn)

        layout.addStretch()
        self._apply_styles()
        self._button_group.idClicked.connect(self._on_button_clicked)

    def _apply_styles(self):
        """Apply pill styling with joined buttons and rounded ends."""
        base_style = f"""
            QPushButton {{
                padding: {LAYOUT["padding"]}px {LAYOUT["spacing_lg"]}px;
                border: 1px solid {BORDER};
                background: {BG_PANEL};
                color: {TEXT_LABEL};
                font-weight: 500;
                font-size: {FONTS["size"]}px;
            }}
            QPushButton:checked {{
                background: {ACCENT};
                border-color: {ACCENT};
                color: {TEXT_MAIN};
            }}
            QPushButton:hover:!checked {{
                background: {BG_INPUT};
            }}
        """
        for i, btn in enumerate(self._buttons):
            if i == 0:
                # First button: rounded left corners
                btn.setStyleSheet(
                    base_style
                    + f"""
                    QPushButton {{
                        border-top-left-radius: {LAYOUT["radius"]}px;
                        border-bottom-left-radius: {LAYOUT["radius"]}px;
                        border-right: none;
                    }}
                """
                )
            elif i == len(self._buttons) - 1:
                # Last button: rounded right corners
                btn.setStyleSheet(
                    base_style
                    + f"""
                    QPushButton {{
                        border-top-right-radius: {LAYOUT["radius"]}px;
                        border-bottom-right-radius: {LAYOUT["radius"]}px;
                    }}
                """
                )
            else:
                # Middle buttons: no rounded corners
                btn.setStyleSheet(
                    base_style
                    + """
                    QPushButton { border-right: none; }
                """
                )

    def _on_button_clicked(self, button_id: int):
        """Handle button click and emit selection changed signal."""
        self._current = self._options[button_id][0]
        self.selection_changed.emit(self._current)
