"""Indeterminate progress bar with Fusion-style streaming animation."""

from PyQt6 import QtCore, QtGui, QtWidgets

from dendrite.gui.styles.design_tokens import ACCENT


class IndeterminateProgressBar(QtWidgets.QWidget):
    """Custom indeterminate progress bar with streaming highlight animation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(6)

        # Animation state (-0.3 to 1.3 for smooth entry/exit)
        self._position = -0.3

        # Animation timer
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._animate)
        self._timer.setInterval(30)

    def start(self):
        """Start the animation."""
        self._timer.start()

    def stop(self):
        """Stop the animation."""
        self._timer.stop()

    def _animate(self):
        """Update animation position."""
        self._position += 0.012
        if self._position >= 1.3:
            self._position = -0.3
        self.update()

    def paintEvent(self, event):
        """Draw the progress bar with streaming highlight."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        radius = rect.height() / 2

        # Draw full bar background
        base_color = QtGui.QColor(ACCENT)
        painter.setBrush(base_color)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawRoundedRect(rect, radius, radius)

        # Draw traveling highlight (enters/exits smoothly off edges)
        highlight_width = rect.width() * 0.5
        highlight_x = rect.width() * self._position - highlight_width * 0.25

        gradient = QtGui.QLinearGradient(highlight_x, 0, highlight_x + highlight_width, 0)
        gradient.setColorAt(0.0, QtGui.QColor(255, 255, 255, 0))
        gradient.setColorAt(0.5, QtGui.QColor(255, 255, 255, 180))
        gradient.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))

        painter.setBrush(gradient)
        painter.drawRoundedRect(rect, radius, radius)

        painter.end()
