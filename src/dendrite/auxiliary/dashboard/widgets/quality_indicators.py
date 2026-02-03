#!/usr/bin/env python
"""
Quality Indicator Widgets

Compact horizontal strip of colored dots showing per-channel signal quality.
"""

from PyQt6 import QtCore, QtGui, QtWidgets

from dendrite.auxiliary.dashboard.backend.signal_quality import (
    ChannelQualityResult,
    QualityLevel,
)
from dendrite.gui.styles.design_tokens import (
    STATUS_ERROR,
    STATUS_OK,
    STATUS_WARN,
    TEXT_MUTED,
)

# Quality level -> QColor mapping
_QUALITY_COLORS: dict[QualityLevel, QtGui.QColor] = {
    QualityLevel.GOOD: QtGui.QColor(STATUS_OK),
    QualityLevel.WARNING: QtGui.QColor(STATUS_WARN),
    QualityLevel.BAD: QtGui.QColor(STATUS_ERROR),
    QualityLevel.UNKNOWN: QtGui.QColor(TEXT_MUTED),
}

DOT_RADIUS = 4
DOT_SPACING = 3


class QualityStripWidget(QtWidgets.QWidget):
    """Compact row of colored dots representing per-channel signal quality."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._results: list[ChannelQualityResult] = []
        self.setFixedHeight(20)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

    def update_quality(self, results: list[ChannelQualityResult]):
        """Update quality data and trigger repaint."""
        self._results = results
        self._update_tooltip()
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent):
        if not self._results:
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        diameter = DOT_RADIUS * 2
        step = diameter + DOT_SPACING
        y_center = self.height() / 2

        x = DOT_SPACING
        for result in self._results:
            color = _QUALITY_COLORS.get(result.level, _QUALITY_COLORS[QualityLevel.UNKNOWN])
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QBrush(color))
            painter.drawEllipse(
                QtCore.QPointF(x + DOT_RADIUS, y_center),
                DOT_RADIUS,
                DOT_RADIUS,
            )
            x += step

        painter.end()

    def _update_tooltip(self):
        """Build tooltip with per-channel metric details."""
        if not self._results:
            self.setToolTip("")
            return

        lines = []
        for r in self._results:
            lines.append(
                f"{r.channel_label}: {r.level.value}  "
                f"(var={r.variance_ratio:.2f}, noise={r.line_noise_ratio:.3f}, "
                f"z={r.variance_zscore:.1f})"
            )
        self.setToolTip("\n".join(lines))
