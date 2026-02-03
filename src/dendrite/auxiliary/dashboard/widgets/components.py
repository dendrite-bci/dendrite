#!/usr/bin/env python
"""
UI Components and Styling

Contains UI styling constants, helper functions for creating UI components,
and reusable UI elements for the visualization dashboard.
"""

import time

from PyQt6 import QtCore, QtWidgets

from dendrite.gui.styles.design_tokens import (
    ACCENT,
    BG_ELEVATED,
    BG_PANEL,
    BORDER,
    STATUS_ERROR,
    STATUS_OK,
    TEXT_DISABLED,
    TEXT_LABEL,
    TEXT_MAIN,
)
from dendrite.gui.styles.widget_styles import WidgetStyles

# Tab widget styling
TAB_WIDGET_STYLESHEET = WidgetStyles.tabwidget

# Dashboard-specific tab widget - pill-style design matching main window navigation
TAB_WIDGET_DASHBOARD_STYLESHEET = f"""
    QTabWidget {{
        background-color: transparent;
        border: none;
    }}
    QTabWidget::pane {{
        background-color: transparent;
        border: none;
        border-radius: 8px;
        padding: 8px;
    }}
    QTabWidget::pane:!has-tabs {{
        border: none;
        padding: 0px;
    }}
    QTabBar {{
        background-color: {BG_PANEL};
        border-radius: 12px;
        padding: 4px;
    }}
    QTabBar::tab {{
        background-color: transparent;
        color: {TEXT_LABEL};
        padding: 8px 20px;
        margin-right: 2px;
        border-radius: 8px;
        font-weight: 500;
        font-size: 11px;
        min-width: 95px;
        border: none;
    }}
    QTabBar::tab:hover {{
        background-color: rgba(255, 255, 255, 0.05);
        color: {TEXT_MAIN};
    }}
    QTabBar::tab:selected {{
        background-color: {BG_ELEVATED};
        color: {TEXT_MAIN};
        font-weight: 500;
    }}
    QTabBar::tab:last {{
        margin-right: 0px;
    }}
"""

# Reconnect button - transparent blend style (subtle hover without blue)
RECONNECT_BUTTON_STYLESHEET = WidgetStyles.button(transparent=True, blend=True)

# Spinbox styling (already V2-compatible)
SPINBOX_STYLESHEET = WidgetStyles.spinbox

# Dashboard Control Sizing - standardized for compact inline controls
DASHBOARD_CONTROL_HEIGHT = 24
DASHBOARD_SPINBOX_WIDTH = 55

DASHBOARD_SPINBOX_STYLE = f"""
    {SPINBOX_STYLESHEET}
    QSpinBox {{
        max-height: {DASHBOARD_CONTROL_HEIGHT}px;
        min-height: {DASHBOARD_CONTROL_HEIGHT}px;
        padding: 2px 8px;
        max-width: {DASHBOARD_SPINBOX_WIDTH}px;
    }}
"""

# Shared combo style for dashboard controls
DASHBOARD_COMBO_STYLE = f"""
    {WidgetStyles.combobox}
    QComboBox {{
        min-height: {DASHBOARD_CONTROL_HEIGHT}px;
        max-height: {DASHBOARD_CONTROL_HEIGHT}px;
    }}
"""

# Shared checkbox style
DASHBOARD_CHECKBOX_STYLE = WidgetStyles.checkbox

# Preset buttons - small, transparent blend style
PRESET_BUTTON_STYLESHEET = WidgetStyles.button(size="small", transparent=True, blend=True)

# Collapsible section button - left-aligned with checked state, subtle hover, no border
COLLAPSIBLE_BUTTON_STYLESHEET = f"""
    {
    WidgetStyles.button(
        align="left", checked_bg=BG_ELEVATED, checked_text=TEXT_MAIN, transparent=True, blend=True
    )
}
    QPushButton {{
        border: none;
    }}
    QPushButton:hover {{
        border: none;
    }}
    QPushButton:checked {{
        border: none;
    }}
"""

# Label style definitions for create_styled_label()
LABEL_STYLES = {
    "title": f"font-weight: 600; color: {TEXT_LABEL}; font-size: 13px;",
    "subtitle": f"font-weight: 500; color: {TEXT_LABEL}; font-size: 11px;",
    "info": f"color: {TEXT_DISABLED}; font-size: 11px;",
    "note": f"font-style: italic; color: {TEXT_LABEL}; font-size: 10px;",
    "normal": f"color: {TEXT_LABEL}; font-size: 11px;",
}

# Shared label style for dashboard controls (must be after LABEL_STYLES)
DASHBOARD_LABEL_STYLE = LABEL_STYLES["normal"]

# Section title style (used in mode tabs)
SECTION_TITLE_STYLE = f"font-weight: 600; color: {TEXT_LABEL};"

# StatusBar style constants - pre-built for performance (no f-string generation per update)
STATUS_INDICATOR_STYLE = "font-size: 12px;"
STATUS_LABEL_STYLE = "font-size: 11px; font-weight: 500;"

# Pre-built status state styles (indicator + label for each state)
_STATUS_STYLES = {
    "disconnected": {
        "indicator": f"color: {STATUS_ERROR}; {STATUS_INDICATOR_STYLE}",
        "label": f"color: {STATUS_ERROR}; {STATUS_LABEL_STYLE}",
        "text": "Disconnected",
    },
    "stale": {
        "indicator": f"color: {ACCENT}; {STATUS_INDICATOR_STYLE}",
        "label": f"color: {ACCENT}; {STATUS_LABEL_STYLE}",
        "text": "Stale",
    },
    "connected": {
        "indicator": f"color: {STATUS_OK}; {STATUS_INDICATOR_STYLE}",
        "label": f"color: {STATUS_OK}; {STATUS_LABEL_STYLE}",
        "text": "Connected",
    },
}

# Separator style constant (exported for use by other modules)
SEPARATOR_STYLE = f"color: {TEXT_DISABLED}; font-size: 11px;"

# Dashboard slider style for channel count and similar controls
DASHBOARD_SLIDER_STYLE = f"""
    QSlider::groove:horizontal {{
        background: {BG_PANEL};
        height: 4px;
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {ACCENT};
        width: 12px;
        height: 12px;
        margin: -4px 0;
        border-radius: 6px;
    }}
    QSlider::sub-page:horizontal {{
        background: {ACCENT};
        border-radius: 2px;
    }}
"""

COLLAPSED_SECTION_HEIGHT = 30  # Fixed height for collapsed section header


class UIHelpers:
    """Helper functions for creating common UI elements"""

    @staticmethod
    def create_collapsible_section(
        title: str, parent_layout: QtWidgets.QLayout = None, start_expanded: bool = False
    ) -> tuple[QtWidgets.QWidget, QtWidgets.QLayout]:
        """Helper to create a collapsible QWidget section. Caller is responsible for adding to layout."""
        group = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(group)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        button = QtWidgets.QPushButton(f"{'▼' if start_expanded else '▶'} {title}")
        button.setCheckable(True)
        button.setChecked(start_expanded)
        button.setStyleSheet(COLLAPSIBLE_BUTTON_STYLESHEET)

        content_widget = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content_widget)
        content_layout.setContentsMargins(
            4, 2, 4, 0
        )  # No bottom padding - spacing handled by container
        content_widget.setVisible(start_expanded)
        if not start_expanded:
            group.setFixedHeight(COLLAPSED_SECTION_HEIGHT)

        layout.addWidget(button)
        layout.addWidget(content_widget, stretch=1)

        def toggle_section(checked):
            content_widget.setVisible(checked)
            button.setText(f"{'▼' if checked else '▶'} {title}")
            if checked:
                group.setMinimumHeight(0)
                group.setMaximumHeight(16777215)  # QWIDGETSIZE_MAX
            else:
                group.setFixedHeight(COLLAPSED_SECTION_HEIGHT)
            group.updateGeometry()
            if group.parent():
                group.parent().updateGeometry()

        button.toggled.connect(toggle_section)
        # Caller is responsible for adding group to layout with appropriate stretch
        return group, content_layout

    @staticmethod
    def create_channel_slider_controls(
        default_count: int = 16, max_count: int = 64
    ) -> tuple[QtWidgets.QWidget, dict]:
        """Create a simple channel count slider control for EEG channel selection."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(8)

        channels_label = QtWidgets.QLabel("Channels:")
        channels_label.setStyleSheet(LABEL_STYLES["subtitle"])
        layout.addWidget(channels_label)

        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setMinimum(1)
        slider.setMaximum(max_count)
        slider.setValue(default_count)
        slider.setFixedWidth(120)
        slider.setStyleSheet(DASHBOARD_SLIDER_STYLE)
        layout.addWidget(slider)

        count_label = QtWidgets.QLabel(f"{default_count}/{max_count}")
        count_label.setStyleSheet(LABEL_STYLES["normal"])
        layout.addWidget(count_label)

        layout.addStretch()

        return widget, {"slider": slider, "count_label": count_label}

    @staticmethod
    def create_channel_pagination_controls() -> tuple[QtWidgets.QWidget, dict]:
        """Create channel pagination with config button.

        Returns:
            (widget, controls_dict) where controls_dict contains:
            - prev_btn: Previous page button
            - next_btn: Next page button
            - page_label: Current page label
            - config_btn: Config popup button
            - summary_label: Channel count summary
        """
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(8)

        # Previous page button
        prev_btn = QtWidgets.QPushButton("\u25c0")  # ◀
        prev_btn.setFixedSize(24, 24)
        prev_btn.setStyleSheet(PRESET_BUTTON_STYLESHEET)
        prev_btn.setToolTip("Previous page")
        layout.addWidget(prev_btn)

        # Page label
        page_label = QtWidgets.QLabel("Page 1 of 1")
        page_label.setStyleSheet(LABEL_STYLES["normal"])
        page_label.setMinimumWidth(80)
        page_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(page_label)

        # Next page button
        next_btn = QtWidgets.QPushButton("\u25b6")  # ▶
        next_btn.setFixedSize(24, 24)
        next_btn.setStyleSheet(PRESET_BUTTON_STYLESHEET)
        next_btn.setToolTip("Next page")
        layout.addWidget(next_btn)

        layout.addSpacing(8)

        # Config button
        config_btn = QtWidgets.QPushButton("\u2699")  # ⚙
        config_btn.setFixedSize(24, 24)
        config_btn.setStyleSheet(PRESET_BUTTON_STYLESHEET)
        config_btn.setToolTip("Channel region presets")
        layout.addWidget(config_btn)

        layout.addSpacing(8)

        # Channel summary label
        summary_label = QtWidgets.QLabel("Channels 1-32 of 64")
        summary_label.setStyleSheet(LABEL_STYLES["info"])
        layout.addWidget(summary_label)

        layout.addStretch()

        return widget, {
            "prev_btn": prev_btn,
            "next_btn": next_btn,
            "page_label": page_label,
            "config_btn": config_btn,
            "summary_label": summary_label,
        }


class StatusBar(QtWidgets.QWidget):
    """
    Persistent status bar showing connection state and stream info.
    Designed to sit at the top of the dashboard for constant visibility.
    """

    # Signal emitted when reconnect button is clicked
    reconnect_clicked = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

        # State tracking
        self._connected = False
        self._stale = False
        self._last_data_time = 0.0
        self._stream_info = ""

        # Stale detection timer (checks every 1 second)
        self._stale_timer = QtCore.QTimer(self)
        self._stale_timer.timeout.connect(self._check_stale)
        self._stale_timer.start(1000)

    def _create_separator(self) -> QtWidgets.QLabel:
        """Create a styled vertical separator label."""
        sep = QtWidgets.QLabel("|")
        sep.setStyleSheet(SEPARATOR_STYLE)
        return sep

    def _setup_ui(self):
        """Setup the status bar layout."""
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(16)

        # Connection status indicator (icon + text)
        self._status_indicator = QtWidgets.QLabel("\u25cf")  # Filled circle
        self._status_indicator.setStyleSheet(_STATUS_STYLES["disconnected"]["indicator"])
        layout.addWidget(self._status_indicator)

        self._status_label = QtWidgets.QLabel("Disconnected")
        self._status_label.setStyleSheet(_STATUS_STYLES["disconnected"]["label"])
        layout.addWidget(self._status_label)

        layout.addWidget(self._create_separator())

        # Stream info
        self._stream_label = QtWidgets.QLabel("No streams")
        self._stream_label.setStyleSheet(f"color: {TEXT_LABEL}; font-size: 11px;")
        layout.addWidget(self._stream_label)

        layout.addStretch()

        # Minimal reconnect button (right-aligned)
        self._reconnect_btn = QtWidgets.QPushButton("Reset")
        self._reconnect_btn.setToolTip("Reconnect & Reset Dashboard")
        self._reconnect_btn.setStyleSheet(RECONNECT_BUTTON_STYLESHEET)
        self._reconnect_btn.setMinimumHeight(DASHBOARD_CONTROL_HEIGHT)
        self._reconnect_btn.clicked.connect(self.reconnect_clicked.emit)
        layout.addWidget(self._reconnect_btn)

        # Style the container
        self.setStyleSheet(f"""
            StatusBar {{
                background-color: {BG_ELEVATED};
                border-bottom: 1px solid {BORDER};
            }}
        """)
        self.setMinimumHeight(28)

    def set_connected(self, connected: bool):
        """Update connection status."""
        self._connected = connected
        self._stale = False
        self._update_status_display()

    def set_stream_info(self, info: str):
        """Update stream information display."""
        self._stream_info = info
        self._stream_label.setText(info if info else "No streams")

    def record_data_received(self):
        """Record that data was received (for stale detection)."""
        self._last_data_time = time.time()
        if self._stale:
            self._stale = False
            self._update_status_display()

    def _check_stale(self):
        """Check for stale data (no data for >2 seconds)."""
        if self._connected and self._last_data_time > 0:
            elapsed = time.time() - self._last_data_time
            if elapsed > 2.0 and not self._stale:
                self._stale = True
                self._update_status_display()

    def _update_status_display(self):
        """Update the connection status indicator using pre-built styles."""
        if not self._connected:
            state = "disconnected"
        elif self._stale:
            state = "stale"
        else:
            state = "connected"

        styles = _STATUS_STYLES[state]
        self._status_indicator.setStyleSheet(styles["indicator"])
        self._status_label.setText(styles["text"])
        self._status_label.setStyleSheet(styles["label"])


# Region presets for EEG channel filtering
EEG_REGION_PRESETS: list[tuple[str, str]] = [
    ("Central", "C"),
    ("Frontal", "F"),
    ("Parietal", "P"),
    ("Occipital", "O"),
]

CHANNELS_PER_PAGE = 32


class ChannelConfigPopup(QtWidgets.QFrame):
    """Popup for channel region preset selection."""

    preset_selected = QtCore.pyqtSignal(str)  # Emits prefix or empty string for "All"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.WindowType.Popup | QtCore.Qt.WindowType.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet(f"""
            ChannelConfigPopup {{
                background-color: {BG_ELEVATED};
                border: 1px solid {BORDER};
                border-radius: 6px;
                padding: 4px;
            }}
        """)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        # "All" button to reset to full channel list
        all_btn = QtWidgets.QPushButton("All")
        all_btn.setStyleSheet(PRESET_BUTTON_STYLESHEET)
        all_btn.setToolTip("Show all channels")
        all_btn.clicked.connect(lambda: self._emit_and_close(""))
        layout.addWidget(all_btn)

        # Region preset buttons
        for name, prefix in EEG_REGION_PRESETS:
            btn = QtWidgets.QPushButton(name)
            btn.setStyleSheet(PRESET_BUTTON_STYLESHEET)
            btn.setToolTip(f"Show {name} channels ({prefix}*)")
            btn.clicked.connect(lambda checked, p=prefix: self._emit_and_close(p))
            layout.addWidget(btn)

    def _emit_and_close(self, prefix: str):
        """Emit selected preset and close popup."""
        self.preset_selected.emit(prefix)
        self.close()

    def show_near(self, widget: QtWidgets.QWidget):
        """Show popup positioned near the given widget."""
        pos = widget.mapToGlobal(widget.rect().bottomLeft())
        self.move(pos)
        self.show()
