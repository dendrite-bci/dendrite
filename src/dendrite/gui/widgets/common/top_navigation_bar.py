"""Top Navigation Bar Widget - Full-width navigation at top of main window."""

from PyQt6 import QtCore, QtWidgets

from dendrite.gui.styles.design_tokens import (
    BG_INPUT,
    BG_PANEL,
    BG_TAB_ACTIVE,
    FONT_SIZE,
    FONT_WEIGHT,
    SEPARATOR_SUBTLE,
    TEXT_LABEL,
    TEXT_MAIN,
)
from dendrite.gui.styles.widget_styles import LAYOUT


class TopNavigationBar(QtWidgets.QWidget):
    """Full-width top bar with minimal text navigation and display switcher."""

    section_changed = QtCore.pyqtSignal(str)
    display_changed = QtCore.pyqtSignal(str)

    def __init__(self, tabs: list[tuple[str, str]], parent=None):
        """Initialize with tab definitions.

        Args:
            tabs: List of (tab_id, tab_label) tuples
        """
        super().__init__(parent)
        self._tabs = tabs
        self._nav_group = QtWidgets.QButtonGroup(self)
        self._display_group = QtWidgets.QButtonGroup(self)
        self._tab_ids: list[str] = []
        self._display_ids: list[str] = []
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet(f"background-color: {BG_PANEL};")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QHBoxLayout(content)
        content_layout.setContentsMargins(LAYOUT["margin"], 11, LAYOUT["margin"], 11)
        content_layout.setSpacing(LAYOUT["spacing_sm"])
        layout.addWidget(content)

        # Left-aligned main navigation
        self._nav_group.setExclusive(True)
        for i, (tab_id, label) in enumerate(self._tabs):
            btn = self._create_minimal_button(label)
            self._nav_group.addButton(btn, i)
            self._tab_ids.append(tab_id)
            content_layout.addWidget(btn)

        self._nav_group.idClicked.connect(self._on_nav_clicked)
        if self._tabs:
            self._nav_group.button(0).setChecked(True)

        content_layout.addStretch()

        # Right-aligned display switcher
        self._display_group.setExclusive(True)
        for i, (display_id, label) in enumerate([("log", "Log"), ("telemetry", "Telemetry")]):
            btn = self._create_minimal_button(label)
            self._display_group.addButton(btn, i)
            self._display_ids.append(display_id)
            content_layout.addWidget(btn)

        self._display_group.idClicked.connect(self._on_display_clicked)
        self._display_group.button(1).setChecked(True)

        # Bottom separator
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        separator.setStyleSheet(f"border: none; border-top: 1px solid {SEPARATOR_SUBTLE};")
        separator.setFixedHeight(1)
        layout.addWidget(separator)

    def _create_minimal_button(self, label: str) -> QtWidgets.QPushButton:
        """Create a minimal text button."""
        btn = QtWidgets.QPushButton(label)
        btn.setCheckable(True)
        btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(self._minimal_button_style())
        return btn

    def _minimal_button_style(self) -> str:
        """Return navigation button style with hover background."""
        return f"""
            QPushButton {{
                background: transparent;
                border: none;
                border-radius: 4px;
                color: {TEXT_LABEL};
                font-size: {FONT_SIZE["lg"]}px;
                padding: {LAYOUT["padding_md"]}px {LAYOUT["spacing_lg"]}px;
            }}
            QPushButton:hover {{
                background: {BG_INPUT};
                color: {TEXT_MAIN};
            }}
            QPushButton:checked {{
                background: {BG_TAB_ACTIVE};
                color: {TEXT_MAIN};
                font-weight: {FONT_WEIGHT["medium"]};
            }}
            QPushButton:checked:hover {{
                background: {BG_TAB_ACTIVE};
            }}
        """

    def _on_nav_clicked(self, button_id: int):
        """Handle main navigation selection."""
        self.section_changed.emit(self._tab_ids[button_id])

    def _on_display_clicked(self, button_id: int):
        """Handle display switcher selection."""
        self.display_changed.emit(self._display_ids[button_id])

    def set_current_tab(self, tab_id: str):
        """Set the current tab by ID."""
        if tab_id in self._tab_ids:
            index = self._tab_ids.index(tab_id)
            self._nav_group.button(index).setChecked(True)

    def get_current_index(self) -> int:
        """Get current tab index."""
        return self._nav_group.checkedId()
