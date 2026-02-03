"""
Pill Navigation Widget

Reusable pill-style tab navigation with sliding highlight animation.
"""

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import QEasingCurve, QPropertyAnimation

from dendrite.gui.styles.design_tokens import (
    BG_ELEVATED,
    BG_MAIN,
    BG_PANEL,
    TEXT_LABEL,
    TEXT_MAIN,
    TEXT_MUTED,
    TEXT_MUTED_DARK,
)
from dendrite.gui.styles.widget_styles import FONTS, LAYOUT


class _PillNavigationBase(QtWidgets.QWidget):
    """Base class for pill-style navigation widgets with sliding highlight."""

    section_changed = QtCore.pyqtSignal(str)

    def __init__(self, tabs: list[tuple[str, str]] | None = None, parent=None):
        """
        Initialize pill navigation base.

        Args:
            tabs: List of (tab_id, tab_label) tuples
            parent: Parent widget
        """
        super().__init__(parent)
        self.tabs = tabs or []
        self.current_index = 0
        self.buttons = {}
        self._highlight = None
        self._slide_anim = None
        self._container = None

    def _on_button_clicked(self, index: int, tab_id: str):
        """Handle button click with animation."""
        if index == self.current_index:
            return

        target_btn, _ = self.buttons[tab_id]
        self._slide_anim.stop()
        self._slide_anim.setStartValue(self._highlight.geometry())
        self._slide_anim.setEndValue(target_btn.geometry())
        self._slide_anim.start()

        self.current_index = index
        self.section_changed.emit(tab_id)

    def get_current_index(self) -> int:
        """Get the current tab index."""
        return self.current_index

    def set_current_index(self, index: int):
        """Programmatically set the current tab."""
        if index < 0 or index >= len(self.tabs):
            return

        tab_id = self.tabs[index][0]
        if tab_id in self.buttons:
            button, _ = self.buttons[tab_id]
            button.setChecked(True)
            self._highlight.setGeometry(button.geometry())
            self.current_index = index

    def set_current_tab(self, tab_id: str):
        """Programmatically set the current tab by ID."""
        if tab_id in self.buttons:
            _, index = self.buttons[tab_id]
            self.set_current_index(index)

    def showEvent(self, event):
        """Position highlight when widget is shown."""
        super().showEvent(event)
        self._update_highlight_position()

    def resizeEvent(self, event):
        """Update highlight position on resize."""
        super().resizeEvent(event)
        self._update_highlight_position()

    def _update_highlight_position(self):
        """Update highlight to match current button position."""
        if self.tabs and self._highlight:
            current_tab_id = self.tabs[self.current_index][0]
            button, _ = self.buttons[current_tab_id]
            self._highlight.setGeometry(button.geometry())


class PillNavigation(_PillNavigationBase):
    """Horizontal pill-shaped navigation with sliding highlight animation."""

    def __init__(self, tabs: list[tuple[str, str]] | None = None, size: str = "large", parent=None):
        """
        Initialize pill navigation.

        Args:
            tabs: List of (tab_id, tab_label) tuples
            size: Button size - 'normal' or 'large' (default)
            parent: Parent widget
        """
        super().__init__(tabs, parent)
        self.size = size
        self.setup_ui()

    def setup_ui(self):
        """Set up the navigation UI."""
        self._container = QtWidgets.QWidget()
        self._container.setStyleSheet(f"""
            background: {BG_MAIN};
            border-radius: {LAYOUT["radius_lg"]}px;
        """)

        container_layout = QtWidgets.QHBoxLayout(self._container)
        container_layout.setContentsMargins(2, 2, 2, 2)
        container_layout.setSpacing(2)

        self._highlight = QtWidgets.QWidget(self._container)
        self._highlight.setStyleSheet(f"""
            background-color: {BG_ELEVATED};
            border-radius: {LAYOUT["radius"]}px;
        """)
        self._highlight.lower()

        self._slide_anim = QPropertyAnimation(self._highlight, b"geometry")
        self._slide_anim.setDuration(200)
        self._slide_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        self.button_group = QtWidgets.QButtonGroup(self)
        self.button_group.setExclusive(True)

        for idx, (tab_id, tab_label) in enumerate(self.tabs):
            button = QtWidgets.QPushButton(tab_label)
            button.setCheckable(True)
            button.clicked.connect(
                lambda checked, i=idx, tid=tab_id: self._on_button_clicked(i, tid)
            )
            button.setStyleSheet(self._button_style())
            self.button_group.addButton(button)
            container_layout.addWidget(button)
            self.buttons[tab_id] = (button, idx)

        if self.tabs:
            first_button = self.buttons[self.tabs[0][0]][0]
            first_button.setChecked(True)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self._container)

    def _button_style(self) -> str:
        """Get button style without background (highlight provides it)."""
        if self.size == "large":
            padding = f"{LAYOUT['padding_md']}px {LAYOUT['spacing_xxl']}px"
            font_size = 14
            min_width = 105
        else:
            padding = f"{LAYOUT['padding']}px {LAYOUT['spacing_xxl']}px"
            font_size = 13
            min_width = 95

        return f"""
            QPushButton {{
                background-color: transparent;
                color: {TEXT_MUTED_DARK};
                border: none;
                padding: {padding};
                border-radius: {LAYOUT["radius"]}px;
                font-size: {font_size}px;
                font-weight: 500;
                min-width: {min_width}px;
            }}
            QPushButton:hover {{
                color: {TEXT_LABEL};
            }}
            QPushButton:checked {{
                color: {TEXT_MAIN};
                font-weight: 600;
            }}
        """


class VerticalPillNavigation(_PillNavigationBase):
    """Vertical sidebar tab navigation with sliding highlight."""

    def __init__(self, tabs: list[tuple[str, str]] | None = None, parent=None):
        """
        Initialize vertical pill navigation.

        Args:
            tabs: List of (tab_id, tab_label) tuples
            parent: Parent widget
        """
        super().__init__(tabs, parent)
        self.setup_ui()

    def setup_ui(self):
        """Set up the navigation UI."""
        self.setFixedWidth(130)

        self._container = QtWidgets.QWidget()
        self._container.setStyleSheet(f"""
            background: {BG_PANEL};
            border-radius: 0;
        """)

        container_layout = QtWidgets.QVBoxLayout(self._container)
        container_layout.setContentsMargins(2, 2, 2, 2)
        container_layout.setSpacing(2)

        self._highlight = QtWidgets.QWidget(self._container)
        self._highlight.setStyleSheet(f"""
            background-color: {BG_ELEVATED};
            border-radius: {LAYOUT["radius"]}px;
        """)
        self._highlight.lower()

        self._slide_anim = QPropertyAnimation(self._highlight, b"geometry")
        self._slide_anim.setDuration(200)
        self._slide_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        self.button_group = QtWidgets.QButtonGroup(self)
        self.button_group.setExclusive(True)

        for idx, (tab_id, tab_label) in enumerate(self.tabs):
            button = QtWidgets.QPushButton(tab_label)
            button.setCheckable(True)
            button.setFixedHeight(44)
            button.clicked.connect(
                lambda checked, i=idx, tid=tab_id: self._on_button_clicked(i, tid)
            )
            button.setStyleSheet(self._button_style())
            self.button_group.addButton(button)
            container_layout.addWidget(button)
            self.buttons[tab_id] = (button, idx)

        container_layout.addStretch()

        if self.tabs:
            first_button = self.buttons[self.tabs[0][0]][0]
            first_button.setChecked(True)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self._container)

    def _button_style(self) -> str:
        """Get button style without background (highlight provides it)."""
        return f"""
            QPushButton {{
                background-color: transparent;
                color: {TEXT_MUTED};
                border: none;
                padding: {LAYOUT["padding"]}px {LAYOUT["spacing_lg"]}px;
                border-radius: {LAYOUT["radius"]}px;
                font-family: {FONTS["family_monospace"]};
                font-size: 13px;
                font-weight: 500;
                text-align: left;
            }}
            QPushButton:hover {{
                color: {TEXT_LABEL};
            }}
            QPushButton:checked {{
                color: {TEXT_MAIN};
                font-weight: 600;
            }}
        """
