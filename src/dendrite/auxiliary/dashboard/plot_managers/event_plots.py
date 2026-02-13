#!/usr/bin/env python
"""
Event Plot Manager

Handles initialization and updating of event visualization plots.
"""

import logging
import time

import pyqtgraph as pg
from PyQt6 import QtGui

from dendrite.auxiliary.dashboard.plot_managers.plot_utils import EVENT_COLORS
from dendrite.gui.styles.design_tokens import BORDER

EVENT_DISPLAY_DURATION_S = 30.0
MAX_RENDERED_EVENTS = 50


class EventPlotManager:
    """Manages event visualization plots"""

    def __init__(self, plot_widget: pg.GraphicsLayoutWidget, data_manager):
        self.plot_widget = plot_widget
        self.data_manager = data_manager
        self.stim_plot = None
        self.x_axis_line = None

        # Event visualization elements (reusable for performance)
        self.event_lines = []
        self.event_scatters = []
        self.event_point_texts = []

        # Event colors
        self.event_colors = EVENT_COLORS

        # Cache for pen/brush/color objects to avoid recreation every frame
        self._pen_cache: dict[str, pg.QtGui.QPen] = {}
        self._brush_cache: dict[str, pg.QtGui.QBrush] = {}
        self._color_cache: dict[str, QtGui.QColor] = {}

        self.event_display_duration = EVENT_DISPLAY_DURATION_S

    def _get_cached_pen(self, color: str, width: int = 2) -> pg.QtGui.QPen:
        """Get or create cached pen for color."""
        key = f"{color}_{width}"
        if key not in self._pen_cache:
            self._pen_cache[key] = pg.mkPen(color, width=width)
        return self._pen_cache[key]

    def _get_cached_brush(self, color: str) -> pg.QtGui.QBrush:
        """Get or create cached brush for color."""
        if color not in self._brush_cache:
            self._brush_cache[color] = pg.mkBrush(color)
        return self._brush_cache[color]

    def _get_cached_color(self, color: str) -> QtGui.QColor:
        """Get or create cached QColor for color string."""
        if color not in self._color_cache:
            self._color_cache[color] = QtGui.QColor(color)
        return self._color_cache[color]

    def initialize_plots(self):
        """Sets up the event plot structure with pre-allocated item pool."""
        self.plot_widget.clear()
        self.stim_plot = self.plot_widget.addPlot(title="Events", row=0, col=0)
        self.stim_plot.setFixedHeight(100)
        self.stim_plot.disableAutoRange()
        self.stim_plot.showAxis("left", False)
        self.stim_plot.showAxis("bottom", False)

        self.x_axis_line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(BORDER, width=1))
        self.stim_plot.addItem(self.x_axis_line)

        # Pre-allocate fixed pool to avoid addItem() during rendering
        self.event_lines = []
        self.event_scatters = []
        self.event_point_texts = []
        for _ in range(MAX_RENDERED_EVENTS):
            line = pg.PlotCurveItem(pen=pg.mkPen(width=2))
            line.setVisible(False)
            self.stim_plot.addItem(line)
            self.event_lines.append(line)

            scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush("w"), pen=None)
            scatter.setVisible(False)
            self.stim_plot.addItem(scatter)
            self.event_scatters.append(scatter)

            point_text = pg.TextItem(color=(255, 255, 255))
            point_text.setFont(QtGui.QFont("Arial", 8))
            point_text.setVisible(False)
            self.stim_plot.addItem(point_text)
            self.event_point_texts.append(point_text)

    def update_plots(self):
        """Efficient event visualization using pre-allocated plot item pool."""
        if not self.stim_plot or not self.data_manager.initialized:
            return

        # Remove old events efficiently using deque
        current_time = time.time()
        oldest_time = current_time - self.event_display_duration

        while (
            self.data_manager.event_history and self.data_manager.event_history[0][0] < oldest_time
        ):
            self.data_manager.event_history.popleft()

        time_window = self.data_manager.buffer_size / self.data_manager.sample_rate

        # Calculate x positions and filter to visible events
        x_positions = []
        filtered_events = []
        for event in self.data_manager.event_history:
            time_diff = current_time - event[0]
            if time_diff > time_window:
                continue
            x_positions.append(time_window - time_diff)
            filtered_events.append(event)

        # Cap to most recent events to keep rendering fast
        events_to_display = filtered_events[-MAX_RENDERED_EVENTS:]
        x_positions = x_positions[-MAX_RENDERED_EVENTS:]

        # Fixed Y range for raster-style display (all events at same height)
        self.stim_plot.setYRange(0, 1.5, padding=0)

        rendered = 0
        for event, x_pos in zip(events_to_display, x_positions, strict=False):
            _, event_value, _ = event
            display_value = int(event_value)
            if display_value < 0:
                continue
            color_idx = display_value % len(self.event_colors)
            color = self.event_colors[color_idx]

            line = self.event_lines[rendered]
            line.setData([x_pos, x_pos], [0, 1])
            line.setPen(self._get_cached_pen(color, width=2))
            line.setVisible(True)

            scatter = self.event_scatters[rendered]
            scatter.setData([x_pos], [1])
            scatter.setBrush(self._get_cached_brush(color))
            scatter.setVisible(True)

            point_text = self.event_point_texts[rendered]
            point_text.setText(str(display_value))
            point_text.setPos(x_pos, 1.15)
            point_text.setAnchor((0.5, 1))
            point_text.setColor(self._get_cached_color(color))
            point_text.setVisible(True)

            rendered += 1

        # Hide unused pool items
        for i in range(rendered, MAX_RENDERED_EVENTS):
            self.event_lines[i].setVisible(False)
            self.event_scatters[i].setVisible(False)
            self.event_point_texts[i].setVisible(False)

        self.stim_plot.setXRange(0, time_window, padding=0)

    def clear_plots(self):
        """Clear all event plots and reset plot items."""
        if self.plot_widget:
            self.plot_widget.clear()

        self.event_lines = []
        self.event_scatters = []
        self.event_point_texts = []
        self.stim_plot = None
        self.x_axis_line = None

        logging.info("Event plots cleared.")
