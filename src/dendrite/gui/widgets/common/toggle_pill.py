"""
Toggle Pill Widget

Modern minimal toggle switch with optional text label (ON/OFF).
Provides a clean interactive toggle with smooth animations.
"""

from PyQt6 import QtCore, QtGui, QtWidgets

from dendrite.gui.styles.design_tokens import ACCENT, TEXT_DISABLED, TEXT_MAIN


class _ToggleSwitch(QtWidgets.QWidget):
    """Internal widget that renders the sliding toggle switch."""

    def __init__(self, width: int = 36, height: int = 18, parent=None):
        super().__init__(parent)
        self.on_color = ACCENT
        self.off_color = TEXT_DISABLED
        self.current_position = 0.0  # 0.0 = left (OFF), 1.0 = right (ON)
        self.current_bg_color = QtGui.QColor(self.off_color)

        # Position animation
        self.position_animation = QtCore.QVariantAnimation()
        self.position_animation.setDuration(250)
        self.position_animation.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)
        self.position_animation.valueChanged.connect(self._on_position_changed)

        # Color animation
        self.color_animation = QtCore.QVariantAnimation()
        self.color_animation.setDuration(250)
        self.color_animation.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)
        self.color_animation.valueChanged.connect(self._on_color_changed)

        self.setFixedSize(width, height)

    def animate_to(self, checked: bool) -> None:
        """Animate to the target state."""
        target_position = 1.0 if checked else 0.0
        target_color = QtGui.QColor(self.on_color if checked else self.off_color)

        self.position_animation.setStartValue(self.current_position)
        self.position_animation.setEndValue(target_position)
        self.position_animation.start()

        self.color_animation.setStartValue(self.current_bg_color)
        self.color_animation.setEndValue(target_color)
        self.color_animation.start()

    def _on_position_changed(self, value: float) -> None:
        self.current_position = value
        self.update()

    def _on_color_changed(self, value: QtGui.QColor) -> None:
        self.current_bg_color = value
        self.update()

    def paintEvent(self, event) -> None:
        """Draw simple toggle switch."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        radius = height // 2

        # Draw background track
        bg_rect = QtCore.QRectF(0, 0, width, height)
        painter.setBrush(self.current_bg_color)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawRoundedRect(bg_rect, radius, radius)

        # Draw sliding thumb
        thumb_diameter = height - 4
        thumb_radius = thumb_diameter / 2
        thumb_travel = width - height
        thumb_x = radius + (thumb_travel * self.current_position)
        thumb_y = height / 2

        painter.setBrush(QtGui.QColor("#FFFFFF"))
        painter.drawEllipse(QtCore.QPointF(thumb_x, thumb_y), thumb_radius, thumb_radius)

        painter.end()


class TogglePillWidget(QtWidgets.QWidget):
    """
    Modern minimal toggle switch with optional text label.

    Visual Design:
        - ON state: Blue background, white thumb on right
        - OFF state: Gray background, white thumb on left
        - Smooth 250ms animation for thumb position and background color
        - Default size: 36x18px toggle + optional ON/OFF text label

    Usage:
        # Standard with label
        toggle = TogglePillWidget(initial_state=True)
        toggle.toggled.connect(self.on_toggle_changed)

        # Compact (no label) for inline form fields
        toggle = TogglePillWidget(initial_state=False, show_label=False, width=28, height=14)

        # Checkbox-compatible API
        if toggle.isChecked():
            do_something()
        toggle.setChecked(False)
    """

    # Signal emitted when toggle state changes (emits bool: True=ON, False=OFF)
    toggled = QtCore.pyqtSignal(bool)

    def __init__(
        self,
        initial_state: bool = False,
        on_color: str = ACCENT,
        off_color: str = TEXT_DISABLED,
        width: int = 36,
        height: int = 18,
        show_label: bool = True,
        parent=None,
    ):
        """
        Initialize toggle pill.

        Args:
            initial_state: Initial checked state (True=ON, False=OFF)
            on_color: Hex color for ON state (default: ACCENT blue)
            off_color: Hex color for OFF state (default: TEXT_DISABLED gray)
            width: Toggle width in pixels (default: 36)
            height: Toggle height in pixels (default: 18)
            show_label: Show ON/OFF text label (default: True). Set False for compact inline use.
            parent: Parent widget
        """
        super().__init__(parent)
        self._checked = initial_state
        self._show_label = show_label

        # Create horizontal layout
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6 if show_label else 0)

        # Create toggle switch
        self._switch = _ToggleSwitch(width, height, self)
        self._switch.on_color = on_color
        self._switch.off_color = off_color
        self._switch.current_position = 1.0 if initial_state else 0.0
        self._switch.current_bg_color = QtGui.QColor(on_color if initial_state else off_color)

        # Add switch to layout
        layout.addWidget(self._switch)

        # Create text label (only if show_label is True)
        if show_label:
            self._label = QtWidgets.QLabel(self)
            self._label.setMinimumWidth(26)  # Width for "OFF"
            self._update_label()
            layout.addWidget(self._label)
        else:
            self._label = None

        layout.addStretch()

        # Enable mouse interaction
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._switch.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    def isChecked(self) -> bool:
        """Get current checked state (checkbox-compatible API)."""
        return self._checked

    def setChecked(self, checked: bool) -> None:
        """
        Set checked state programmatically (checkbox-compatible API).

        Args:
            checked: New checked state
        """
        if self._checked != checked:
            self._checked = checked
            self._switch.animate_to(self._checked)
            self._update_label()
            self.toggled.emit(self._checked)

    def toggle(self) -> None:
        """Toggle the current state."""
        self.setChecked(not self._checked)

    def mousePressEvent(self, event) -> None:
        """Handle mouse click to toggle state."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.toggle()
        super().mousePressEvent(event)

    def _update_label(self) -> None:
        """Update text label to show current state."""
        if self._label is None:
            return

        text = "ON" if self._checked else "OFF"
        color = self._switch.on_color if self._checked else TEXT_MAIN

        self._label.setText(text)
        self._label.setStyleSheet(f"color: {color}; font-size: 11px;")
