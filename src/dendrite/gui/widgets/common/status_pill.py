"""
Status Pill Widget

Modern pill-shaped status indicator with animated sliding dot.
Provides a toggle-switch-style visual indicator for status (active/inactive).
"""

from PyQt6 import QtCore, QtGui, QtWidgets

from dendrite.gui.styles.design_tokens import BG_PANEL, TEXT_DISABLED


class StatusPillWidget(QtWidgets.QWidget):
    """
    Compact pill-shaped status indicator with animated sliding dot.

    Visual Design:
        - Compact size (default: 32x16px, configurable)
        - Gray background with subtle border for definition
        - Only the sliding dot changes color based on status
        - Dot moves from left (inactive) to right (active)
        - Smooth animation between states (250ms)
        - Clickable with cursor feedback

    Usage:
        pill = StatusPillWidget()  # Default 32x16px
        pill = StatusPillWidget(48, 24)  # Custom size
        pill.set_status(STATUS_OK, active=True)   # Gray pill, green dot on right
        pill.set_status(TEXT_DISABLED, active=False)  # Gray pill, gray dot on left
        pill.clicked.connect(on_click_handler)  # Handle clicks
    """

    clicked = QtCore.pyqtSignal()

    def __init__(self, width: int = 32, height: int = 16, parent=None):
        """
        Initialize status pill indicator.

        Args:
            width: Pill width in pixels (default: 32)
            height: Pill height in pixels (default: 16)
            parent: Parent widget
        """
        super().__init__(parent)
        self.status_color = TEXT_DISABLED
        self.target_position = 0.0  # 0.0 = left (inactive), 1.0 = right (active)
        self.current_position = 0.0  # Start at left (inactive)

        # Manual animation timer (~60fps)
        self._anim_timer = QtCore.QTimer()
        self._anim_timer.setInterval(16)
        self._anim_timer.timeout.connect(self._animate_step)

        # Configurable size (default: compact for status indicators)
        self.setFixedSize(width, height)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    def set_status(self, status_color: str, active: bool = False):
        """
        Update the pill status color and position.

        Args:
            status_color: Hex color string for the active state (e.g., STATUS_OK, STATUS_ERROR)
            active: Whether the indicator should be active (True) or inactive (False)
        """
        self.status_color = status_color
        self.target_position = 1.0 if active else 0.0

        if self.current_position != self.target_position:
            self._anim_timer.start()

        self.update()

    def animate_reject(self, from_active: bool = False):
        """
        Animate rejection feedback.

        Args:
            from_active: If True, animate OFF then back ON (rejecting turn-off).
                         If False, animate ON then back OFF (rejecting turn-on).
        """
        self._reject_pending = True
        self._reject_return_position = 1.0 if from_active else 0.0
        # Go to opposite position first
        self.target_position = 0.0 if from_active else 1.0
        self._anim_timer.start()

    def _animate_step(self):
        """Animate one step towards target."""
        diff = self.target_position - self.current_position
        if abs(diff) < 0.05:
            self.current_position = self.target_position
            self._anim_timer.stop()

            # If reject pending, flip back to return position
            if getattr(self, "_reject_pending", False):
                self._reject_pending = False
                self.target_position = getattr(self, "_reject_return_position", 0.0)
                self._anim_timer.start()
        else:
            self.current_position += diff * 0.25  # Smooth easing
        self.update()

    def paintEvent(self, event):
        """Custom paint event to draw the pill indicator."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        radius = height // 2
        dot_radius = (height - 6) // 2  # Inner dot radius with comfortable padding

        # Draw background pill with border for better contrast
        bg_rect = QtCore.QRectF(0, 0, width, height)
        painter.setBrush(QtGui.QColor(BG_PANEL))

        # Add subtle semi-transparent border for definition (reduces jagged edges)
        border_color = QtGui.QColor(255, 255, 255, 20)  # ~8% opacity
        pen = QtGui.QPen(border_color)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawRoundedRect(bg_rect, radius, radius)

        # Calculate dot position (interpolated)
        dot_travel = width - height  # Distance the dot can travel
        dot_x = radius + (dot_travel * self.current_position)
        dot_y = height / 2

        # Draw inner sliding dot - always use status_color
        # Position (left/right) indicates active/inactive state
        dot_color = QtGui.QColor(self.status_color)

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(dot_color)
        painter.drawEllipse(QtCore.QPointF(dot_x, dot_y), dot_radius, dot_radius)

        painter.end()

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        """Emit clicked signal on mouse press."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)
