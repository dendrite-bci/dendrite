"""
Latency Sparkline Widget

Minimal sparkline chart for visualizing latency over time.
"""

from collections import deque

from PyQt6 import QtCore, QtGui, QtWidgets

from dendrite.gui.styles.design_tokens import STATUS_DANGER, STATUS_SUCCESS, STATUS_WARNING_ALT


class LatencySparkline(QtWidgets.QWidget):
    """Compact sparkline for latency visualization."""

    def __init__(
        self,
        max_samples: int = 30,
        width: int = 80,
        height: int = 20,
        low_threshold: float = 10,
        high_threshold: float = 30,
        parent=None,
    ):
        """
        Initialize sparkline.

        Args:
            max_samples: Number of samples to display (ring buffer size)
            width: Widget width in pixels
            height: Widget height in pixels
            low_threshold: Latency below this is green (ms)
            high_threshold: Latency above this is red (ms)
        """
        super().__init__(parent)
        self.samples = deque(maxlen=max_samples)
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self._force_color = None  # Override color when set
        self.setFixedSize(width, height)

    def add_sample(self, value_ms: float) -> None:
        """Add a latency sample and trigger repaint."""
        self.samples.append(value_ms)
        self.update()

    def clear(self) -> None:
        """Clear all samples."""
        self.samples.clear()
        self._force_color = None
        self.update()

    def set_force_color(self, color: str) -> None:
        """Force sparkline to specific color (e.g., for dropped state)."""
        self._force_color = color
        self.update()

    def clear_force_color(self) -> None:
        """Clear forced color, return to normal threshold-based coloring."""
        self._force_color = None

    def _get_color(self, value: float) -> QtGui.QColor:
        """Get color based on latency value."""
        if value >= self.high_threshold:
            return QtGui.QColor(STATUS_DANGER)
        elif value >= self.low_threshold:
            return QtGui.QColor(STATUS_WARNING_ALT)
        return QtGui.QColor(STATUS_SUCCESS)

    def paintEvent(self, event) -> None:
        """Draw the sparkline."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()

        if len(self.samples) < 2:
            painter.end()
            return

        # Adaptive scale based on actual data range
        samples = list(self.samples)
        data_min = min(samples)
        data_max = max(samples)
        data_range = max(data_max - data_min, 2.0)  # At least 2ms range
        scale_pad = data_range * 0.15

        min_val = max(0, data_min - scale_pad)
        max_val = data_max + scale_pad

        pixel_pad = 2
        draw_width = width - 2 * pixel_pad
        draw_height = height - 2 * pixel_pad

        # Build path
        path = QtGui.QPainterPath()
        x_step = draw_width / (len(samples) - 1) if len(samples) > 1 else draw_width

        for i, val in enumerate(samples):
            x = pixel_pad + i * x_step
            y = pixel_pad + draw_height - (val - min_val) / (max_val - min_val) * draw_height
            y = max(pixel_pad, min(y, height - pixel_pad))  # Clamp

            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        # Draw line with color based on latest value (or forced color)
        if self._force_color:
            color = QtGui.QColor(self._force_color)
        else:
            color = self._get_color(samples[-1])
        pen = QtGui.QPen(color, 1.5)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawPath(path)

        painter.end()
