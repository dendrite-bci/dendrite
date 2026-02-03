#!/usr/bin/env python
"""
Plot Utilities

Shared utilities for plot managers to avoid code duplication.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtGui import QPainter
from pyqtgraph.graphicsItems.LegendItem import ItemSample
from pyqtgraph.graphicsItems.ScatterPlotItem import drawSymbol

from dendrite.gui.styles.design_tokens import BORDER, TEXT_LABEL, TEXT_MUTED

# Plot color palettes (tamed neons + orange overlap fix)

# Metric line colors (performance + async)
METRIC_ACCURACY = "#66B2FF"  # Clear blue        (unchanged)
METRIC_CONFIDENCE = "#E8B44C"  # Warm gold          (was #C896E0 lavender)
METRIC_CHANCE_LEVEL = "#8A8A8A"  # Neutral gray       (was #FFA500 orange)

# Signal trace colors
EEG_SIGNAL = "#40E8C0"  # Tamed teal        (was #00ffcc neon)
PSD_MEAN_SIGNAL = "#40E8C0"  # Same teal as EEG for visual consistency
PSD_CHANNEL_OVERLAY = (64, 232, 192, 60)  # Teal with low alpha for per-channel overlay
NEUROFEEDBACK_SIGNAL = "#33A1C9"  # Ocean blue        (unchanged)

# Async scatter colors (all unchanged - already vibrant, not neon)
ASYNC_PREDICTION_FILL = "#FF6B6B"
ASYNC_PREDICTION_EDGE = "#8B0000"
ASYNC_TRUE_EVENT_FILL = "#FFFFFF"
ASYNC_TRUE_EVENT_EDGE = "#808080"

# ERP event type base colors (8 hues, varied per channel via HSV)
ERP_EVENT_COLORS = [
    "#FF6B6B",
    "#4ECDC4",
    "#45B7D1",
    "#96CEB4",
    "#FFEAA7",
    "#DDA0DD",
    "#98D8C8",
    "#F7DC6F",
]

# Event marker colors (4 neons tamed, rest unchanged)
EVENT_COLORS = [
    "#40E8C0",
    "#ff9500",
    "#ff4d88",
    "#a864ff",
    "#ffcc00",
    "#40E890",
    "#00a2ff",
    "#ff5050",
    "#5cb8ff",
    "#ffa500",
    "#40E8A8",
    "#ff6ec7",
    "#c56cf0",
    "#ffb142",
    "#218c74",
    "#ff9ff3",
    "#cd84f1",
    "#ffcccc",
    "#50D0F0",
    "#30C8A0",
]

# Modality signal colors (3 neons tamed, rest unchanged)
MODALITY_COLORS = [
    "#ff4d4d",
    "#40E840",
    "#ffcc00",
    "#4e7fff",
    "#d175ff",
    "#ff75d1",
    "#88E880",
    "#78E8E0",
]


class DotSample(ItemSample):
    """Legend sample that shows a colored dot instead of a line."""

    def paint(self, p: QPainter, *args) -> None:
        if not self.item.isVisible():
            return
        opts = self.item.opts
        pen = opts.get("pen")
        if pen is not None:
            color = pg.mkPen(pen).color()
        else:
            color = pg.mkColor("#ffffff")

        p.translate(10, 10)
        drawSymbol(p, "o", 8, pg.mkPen(color), pg.mkBrush(color))


def style_legend(legend: pg.LegendItem) -> None:
    """Apply consistent styling to a plot legend."""
    legend.setLabelTextColor(TEXT_LABEL)
    legend.setLabelTextSize("9pt")
    legend.setBrush(pg.mkBrush(26, 26, 26, 180))
    legend.mouseDragEvent = lambda ev: ev.ignore()


def style_plot_axes(plot_item: pg.PlotItem) -> None:
    """Apply muted axis styling to both left and bottom axes."""
    for axis_name in ("left", "bottom"):
        axis = plot_item.getAxis(axis_name)
        axis.setPen(pg.mkPen(BORDER))
        axis.setTextPen(TEXT_MUTED)
        axis.setStyle(tickLength=0)


def calculate_y_range(data: np.ndarray, padding: float = 0.1) -> tuple[float, float]:
    """
    Calculate Y-axis range with padding for plot data.

    Args:
        data: Array of data values
        padding: Fraction of range to add as padding (default 0.1 = 10%)

    Returns:
        Tuple of (min_value, max_value) for Y-axis range
    """
    ch_min, ch_max = np.min(data), np.max(data)
    rng = ch_max - ch_min

    if rng < 1e-9:
        # Flat signal - use center with fixed range
        center = ch_min
        return center - 1.0, center + 1.0
    else:
        pad = rng * padding
        return ch_min - pad, ch_max + pad


def should_update_y_range(
    new_min: float, new_max: float, last_min: float, last_max: float, threshold: float = 0.1
) -> bool:
    """
    Check if Y-range changed significantly enough to warrant an update.

    Args:
        new_min: New calculated minimum
        new_max: New calculated maximum
        last_min: Previous minimum
        last_max: Previous maximum
        threshold: Fractional change threshold (default 0.1 = 10%)

    Returns:
        True if range should be updated, False otherwise
    """
    last_range = last_max - last_min
    new_range = new_max - new_min

    # Check if range size changed significantly
    range_change = abs(new_range - last_range) / (last_range + 1e-9)

    # Check if center shifted significantly
    center_shift = abs((new_min + new_max) - (last_min + last_max)) / (last_range + 1e-9)

    return range_change > threshold or center_shift > threshold


def update_plot_y_range_if_needed(
    data: np.ndarray,
    plot_idx: int,
    last_y_ranges: list[tuple[float, float]],
    plot_items: list,
    range_padding: float = 0.1,
    range_change_threshold: float = 0.1,
) -> None:
    """
    Calculate and update Y-range for a plot if the change is significant.

    Args:
        data: Array of data values for the plot
        plot_idx: Index of the plot in the lists
        last_y_ranges: List of (min, max) tuples for each plot
        plot_items: List of cached plot item references
        range_padding: Fraction of range to add as padding
        range_change_threshold: Fractional change required to trigger update
    """
    new_min, new_max = calculate_y_range(data, range_padding)

    if plot_idx < len(last_y_ranges):
        last_min, last_max = last_y_ranges[plot_idx]

        if should_update_y_range(new_min, new_max, last_min, last_max, range_change_threshold):
            if plot_idx < len(plot_items):
                plot_items[plot_idx].setYRange(new_min, new_max, padding=0)
                last_y_ranges[plot_idx] = (new_min, new_max)


def setup_signal_plot(
    plot_widget: pg.GraphicsLayoutWidget,
    row: int,
    col: int,
    title: str,
    x_max: float,
    y_range: tuple[float, float] = (-100, 100),
    color: str = EEG_SIGNAL,
    max_height: int | None = None,
) -> tuple[pg.PlotItem, pg.PlotDataItem]:
    """
    Standard signal plot setup for EEG, EMG, and other time-series displays.

    Args:
        plot_widget: GraphicsLayoutWidget to add plot to
        row: Grid row position
        col: Grid column position
        title: Plot title (channel label)
        x_max: Maximum X value (time axis limit)
        y_range: Initial Y-axis range
        color: Curve pen color
        max_height: Optional maximum plot height in pixels

    Returns:
        Tuple of (plot_item, curve)
    """
    p = plot_widget.addPlot(row=row, col=col)
    p.setMouseEnabled(x=False, y=True)
    p.setLimits(xMin=0, xMax=x_max)
    p.hideAxis("bottom")
    p.setTitle(title, color=TEXT_LABEL, size="8pt")

    y_min, y_max = y_range
    p.setYRange(y_min, y_max, padding=0)

    style_plot_axes(p)

    left_axis = p.getAxis("left")
    left_axis.setWidth(30)
    left_axis.setTicks([[(y_min, str(int(y_min))), (0, "0"), (y_max, str(int(y_max)))], []])

    p.showGrid(x=False, y=False)
    p.setMenuEnabled(False)

    if max_height:
        p.setMaximumHeight(max_height)

    curve = p.plot(pen=pg.mkPen(color, width=1))
    return p, curve
