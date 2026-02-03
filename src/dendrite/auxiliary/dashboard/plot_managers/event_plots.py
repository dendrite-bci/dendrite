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

        # Cache for pen/brush objects to avoid recreation every frame
        self._pen_cache: dict[str, pg.QtGui.QPen] = {}
        self._brush_cache: dict[str, pg.QtGui.QBrush] = {}

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

    def initialize_plots(self):
        """Sets up the event plot structure."""
        self.plot_widget.clear()
        self.stim_plot = self.plot_widget.addPlot(title="Events", row=0, col=0)
        self.stim_plot.setFixedHeight(100)
        self.stim_plot.disableAutoRange()
        self.stim_plot.showAxis("left", False)
        self.stim_plot.showAxis("bottom", False)

        self.x_axis_line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(BORDER, width=1))
        self.stim_plot.addItem(self.x_axis_line)

    def update_plots(self):
        """Efficient event visualization using reusable plot items."""
        if not self.stim_plot or not self.data_manager.initialized:
            return

        # Remove old events efficiently using deque
        current_time = time.time()
        oldest_time = current_time - self.event_display_duration

        while (
            self.data_manager.event_history and self.data_manager.event_history[0][0] < oldest_time
        ):
            self.data_manager.event_history.popleft()

        events_to_display = list(self.data_manager.event_history)
        time_window = self.data_manager.buffer_size / self.data_manager.sample_rate

        # Calculate x positions and filter out events
        x_positions = []
        filtered_events = []
        for event in events_to_display:
            time_diff = current_time - event[0]
            # Check if event is outside visible range
            if time_diff > time_window:
                continue  # Skip events that would be beyond the left edge

            x_pos = time_window - time_diff
            x_positions.append(x_pos)
            filtered_events.append(event)

        # Use filtered events only
        events_to_display = filtered_events

        # Fixed Y range for raster-style display (all events at same height)
        self.stim_plot.setYRange(0, 1.5, padding=0)

        required_items = len(events_to_display)

        # Ensure enough plot items exist
        while len(self.event_lines) < required_items:
            line = pg.PlotCurveItem(pen=pg.mkPen(width=2))
            self.stim_plot.addItem(line)
            self.event_lines.append(line)
        while len(self.event_scatters) < required_items:
            scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush("w"), pen=None)
            self.stim_plot.addItem(scatter)
            self.event_scatters.append(scatter)
        while len(self.event_point_texts) < required_items:
            point_text = pg.TextItem(color=(255, 255, 255))
            point_text.setFont(QtGui.QFont("Arial", 8))
            self.stim_plot.addItem(point_text)
            self.event_point_texts.append(point_text)

        # Update existing items
        for i, (event, x_pos) in enumerate(zip(events_to_display, x_positions, strict=False)):
            timestamp, event_value, _ = event
            display_value = int(event_value)
            if display_value <= 0:
                continue
            color_idx = (display_value - 1) % len(self.event_colors)
            color = self.event_colors[color_idx]

            # Raster-style: all events at same height (y=1)
            y_pos = 1

            line = self.event_lines[i]
            line.setData([x_pos, x_pos], [0, y_pos])
            line.setPen(self._get_cached_pen(color, width=2))
            line.setVisible(True)

            scatter = self.event_scatters[i]
            scatter.setData([x_pos], [y_pos])
            scatter.setBrush(self._get_cached_brush(color))
            scatter.setVisible(True)

            # Update point text with event code above marker
            point_text = self.event_point_texts[i]
            point_text.setText(str(display_value))
            point_text.setPos(x_pos, y_pos + 0.15)
            point_text.setAnchor((0.5, 1))
            point_text.setColor(QtGui.QColor(color))
            point_text.setVisible(True)

        # Hide extra items
        for i in range(required_items, len(self.event_lines)):
            self.event_lines[i].setVisible(False)
        for i in range(required_items, len(self.event_scatters)):
            self.event_scatters[i].setVisible(False)
        for i in range(required_items, len(self.event_point_texts)):
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
