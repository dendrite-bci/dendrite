"""Reusable section widgets for ML Workbench control panels."""

from PyQt6 import QtCore, QtWidgets

# Panel and widget size constants
PANEL_WIDTH = 560
INDICATOR_WIDTH = 3

from dendrite.gui.styles.design_tokens import (
    FONT_SIZE,
    STATUS_ERROR,
    STATUS_SUCCESS,
    STATUS_WARNING_ALT,
    TEXT_MUTED,
)
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles


def create_section(title: str) -> tuple[QtWidgets.QWidget, QtWidgets.QVBoxLayout]:
    """Create a card-style section with muted uppercase header.

    Args:
        title: Section title (will be uppercased)

    Returns:
        Tuple of (section_widget, content_layout)
    """
    section = QtWidgets.QWidget()
    section.setObjectName("section_card")
    section.setStyleSheet(WidgetStyles.section_card())
    layout = QtWidgets.QVBoxLayout(section)
    layout.setContentsMargins(
        LAYOUT["spacing_xl"], LAYOUT["spacing_lg"], LAYOUT["spacing_xl"], LAYOUT["spacing_xl"]
    )
    layout.setSpacing(LAYOUT["spacing_sm"])

    header = QtWidgets.QLabel(title.upper())
    header.setStyleSheet(f"""
        color: {TEXT_MUTED};
        font-size: {FONT_SIZE["sm"]}px;
        font-weight: 600;
        letter-spacing: 1px;
        background-color: transparent;
    """)
    layout.addWidget(header)

    return section, layout


def create_scrollable_panel(
    width: int = PANEL_WIDTH,
) -> tuple[QtWidgets.QWidget, QtWidgets.QVBoxLayout, QtWidgets.QVBoxLayout]:
    """Create a scrollable panel for control sections.

    Args:
        width: Fixed width for the panel (default 560px)

    Returns:
        Tuple of (panel_widget, content_layout, panel_layout).
        Add scrollable sections to content_layout.
        Add fixed footer widgets (action buttons, status) to panel_layout.
    """
    panel = QtWidgets.QWidget()
    panel.setFixedWidth(width)

    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setStyleSheet(WidgetStyles.scrollarea)
    scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

    content = QtWidgets.QWidget()
    content_layout = QtWidgets.QVBoxLayout(content)
    content_layout.setContentsMargins(0, 0, LAYOUT["spacing_md"], 0)
    content_layout.setSpacing(LAYOUT["spacing_md"])

    scroll.setWidget(content)

    panel_layout = QtWidgets.QVBoxLayout(panel)
    panel_layout.setContentsMargins(0, 0, 0, 0)
    panel_layout.addWidget(scroll, stretch=1)

    return panel, content_layout, panel_layout


def create_form_layout() -> QtWidgets.QFormLayout:
    """Create a form layout with standard ML Workbench settings.

    Returns:
        QFormLayout with AllNonFixedFieldsGrow policy and zero margins
    """
    form = QtWidgets.QFormLayout()
    form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    form.setContentsMargins(0, 0, 0, 0)
    return form


class CollapsibleSection(QtWidgets.QWidget):
    """A collapsible section with card styling and muted header."""

    def __init__(self, title: str, start_expanded: bool = False, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self._title = title
        self._setup_ui(start_expanded)

    def _setup_ui(self, start_expanded: bool):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._toggle_btn = QtWidgets.QPushButton()
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(start_expanded)
        self._toggle_btn.setStyleSheet(WidgetStyles.collapsible_header())
        self._update_button_text(start_expanded)
        self._toggle_btn.toggled.connect(self._on_toggle)
        layout.addWidget(self._toggle_btn)

        self._content_container = QtWidgets.QWidget()
        self._content_container.setObjectName("collapsible_content")
        self._content_container.setStyleSheet(WidgetStyles.collapsible_content())
        self._content_layout = QtWidgets.QVBoxLayout(self._content_container)
        self._content_layout.setContentsMargins(
            LAYOUT["spacing_xl"], 0, LAYOUT["spacing_xl"], LAYOUT["spacing_xl"]
        )
        self._content_layout.setSpacing(LAYOUT["spacing_sm"])
        self._content_container.setVisible(start_expanded)
        layout.addWidget(self._content_container)

    def _update_button_text(self, expanded: bool):
        arrow = "▼" if expanded else "▶"
        self._toggle_btn.setText(f"{arrow}  {self._title.upper()}")

    def _on_toggle(self, checked: bool):
        self._content_container.setVisible(checked)
        self._update_button_text(checked)
        self.updateGeometry()
        if self.parent():
            self.parent().updateGeometry()

    def content_layout(self) -> QtWidgets.QVBoxLayout:
        """Get the layout for adding content widgets."""
        return self._content_layout


class StatusContainer(QtWidgets.QWidget):
    """Status display with colored left indicator border."""

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(LAYOUT["spacing_md"])

        self._indicator = QtWidgets.QFrame()
        self._indicator.setFixedWidth(INDICATOR_WIDTH)
        self._indicator.setStyleSheet(f"background-color: {TEXT_MUTED};")
        layout.addWidget(self._indicator)

        content = QtWidgets.QVBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(LAYOUT["spacing_xs"])

        self._progress_bar = QtWidgets.QProgressBar()
        self._progress_bar.setMaximumHeight(12)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.hide()
        content.addWidget(self._progress_bar)

        self._status_label = QtWidgets.QLabel("Ready")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: {FONT_SIZE['sm']}px; background-color: transparent;"
        )
        content.addWidget(self._status_label)

        layout.addLayout(content, stretch=1)

    def set_status(self, text: str, severity: str = "normal"):
        """Set status text and severity color.

        Args:
            text: Status message
            severity: One of 'normal', 'success', 'warning', 'error'
        """
        self._status_label.setText(text)
        colors = {
            "normal": TEXT_MUTED,
            "success": STATUS_SUCCESS,
            "warning": STATUS_WARNING_ALT,
            "error": STATUS_ERROR,
        }
        color = colors.get(severity, TEXT_MUTED)
        self._indicator.setStyleSheet(f"background-color: {color};")
        self._status_label.setStyleSheet(
            f"color: {color}; font-size: {FONT_SIZE['sm']}px; background-color: transparent;"
        )

    def show_progress(self, show: bool = True, value: int = 0, maximum: int = 0):
        """Show or hide progress bar.

        Args:
            show: Whether to show the progress bar
            value: Current progress value (0 for indeterminate)
            maximum: Maximum value (0 for indeterminate)
        """
        if show:
            self._progress_bar.setRange(0, maximum)
            self._progress_bar.setValue(value)
            self._progress_bar.show()
        else:
            self._progress_bar.hide()

    def text(self) -> str:
        """Get current status text."""
        return self._status_label.text()
