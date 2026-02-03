#!/usr/bin/env python
"""
Unified Asynchronous Mode Plot Manager

Handles real-time visualization for asynchronous mode with a focus on
minimal, performance-oriented display. Combines prediction tracking,
accuracy trends, and confidence monitoring in a unified interface.
"""

import logging
import time
from collections import deque
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore

from dendrite.auxiliary.dashboard.plot_managers.plot_utils import (
    ASYNC_PREDICTION_EDGE,
    ASYNC_PREDICTION_FILL,
    ASYNC_TRUE_EVENT_EDGE,
    ASYNC_TRUE_EVENT_FILL,
    METRIC_CHANCE_LEVEL,
    METRIC_CONFIDENCE,
    DotSample,
    style_legend,
    style_plot_axes,
)
from dendrite.gui.styles.design_tokens import TEXT_LABEL

# Configuration constants
ASYNC_WINDOW_DURATION = 30.0  # seconds - 30s window for better temporal context
ASYNC_MAX_POINTS = 300  # 30s @ 10Hz = 300 points (no downsampling needed)
CONFIDENCE_THRESHOLD_LINE = 0.6  # Visual threshold line position for confidence plot


class AsyncPlotManager:
    """
    Unified plot manager for asynchronous mode visualization.

    Provides a minimal, focused interface showing:
    - Accuracy trend over time (primary focus)
    - Chance level baseline
    - Prediction events as markers
    - Confidence trend (separate subplot)

    Design Philosophy:
    - Single data storage per mode (no duplication)
    - Event-driven updates (no internal throttling)
    - Consistent time-based X-axis across all subplots
    - Minimal visual clutter - focus on actionable metrics
    """

    def __init__(self, plot_widget: pg.GraphicsLayoutWidget):
        """
        Initialize the async plot manager.

        Args:
            plot_widget: PyQtGraph graphics layout widget for rendering
        """
        self.plot_widget = plot_widget

        # Data storage per mode
        self._mode_data: dict[str, dict[str, Any]] = {}

        # Plot objects per mode
        self._plots: dict[str, dict[str, Any]] = {}

        # Performance tracking
        self._last_update_times: dict[str, float] = {}
        self._processed_timestamps: dict[str, set] = {}  # For duplicate detection

    def initialize_for_mode(self, mode_name: str) -> None:
        """
        Initialize data structures and plots for a specific mode.

        Args:
            mode_name: Unique identifier for the mode instance
        """
        if mode_name in self._mode_data:
            return  # Already initialized

        # Create unified data storage
        self._mode_data[mode_name] = {
            "timestamps": deque(maxlen=ASYNC_MAX_POINTS),
            "predictions": deque(maxlen=ASYNC_MAX_POINTS),
            "confidences": deque(maxlen=ASYNC_MAX_POINTS),
            "true_labels": deque(maxlen=ASYNC_MAX_POINTS),  # Ground truth for visualization
        }

        self._processed_timestamps[mode_name] = set()
        self._last_update_times[mode_name] = 0.0

        # Create plot layout
        self._create_plots(mode_name)

        logging.info(f"AsyncPlotManager initialized for mode: {mode_name}")

    def _create_plots(self, mode_name: str) -> None:
        """
        Create the unified plot layout for async mode.

        Layout:
        - Row 0: Accuracy trend + chance level + prediction markers
        - Row 1: Confidence trend + threshold line

        Args:
            mode_name: Mode identifier for plot labeling
        """
        plots = {}

        prediction_plot = self.plot_widget.addPlot(row=0, col=0)
        prediction_plot.setTitle(
            f"{mode_name} - Predictions (30s window)", color=TEXT_LABEL, size="9pt"
        )
        legend = prediction_plot.addLegend(offset=(10, 10), sampleType=DotSample)
        style_legend(legend)
        prediction_plot.setLabel("left", "Predicted Class", color=TEXT_LABEL)
        prediction_plot.setLabel("bottom", "Time (s)", color=TEXT_LABEL)
        style_plot_axes(prediction_plot)
        prediction_plot.setClipToView(True)

        # Enable horizontal grid lines to separate classes visually
        prediction_plot.showGrid(y=True, alpha=0.3)

        # Prediction scatter plot - each point at its class Y-position
        prediction_scatter = prediction_plot.plot(
            pen=None,
            symbol="o",
            symbolBrush=ASYNC_PREDICTION_FILL,
            symbolPen=pg.mkPen(ASYNC_PREDICTION_EDGE, width=1),
            symbolSize=10,
            name="Predictions",
        )

        # True event scatter plot - ground truth markers
        true_event_scatter = prediction_plot.plot(
            pen=None,
            symbol="o",
            symbolBrush=ASYNC_TRUE_EVENT_FILL,
            symbolPen=pg.mkPen(ASYNC_TRUE_EVENT_EDGE, width=1.5),
            symbolSize=8,
            name="True Events",
        )

        plots["predictions"] = {
            "plot": prediction_plot,
            "prediction_scatter": prediction_scatter,
            "true_event_scatter": true_event_scatter,
        }

        confidence_plot = self.plot_widget.addPlot(row=1, col=0)
        confidence_plot.setTitle(
            f"{mode_name} - Confidence (30s window)", color=TEXT_LABEL, size="9pt"
        )
        legend = confidence_plot.addLegend(offset=(10, 10), sampleType=DotSample)
        style_legend(legend)
        confidence_plot.setLabel("left", "Confidence", color=TEXT_LABEL)
        confidence_plot.setLabel("bottom", "Time (s)", color=TEXT_LABEL)
        style_plot_axes(confidence_plot)
        confidence_plot.setYRange(0, 1.05, padding=0)
        confidence_plot.setClipToView(True)

        # Confidence curve
        confidence_curve = confidence_plot.plot(
            pen=pg.mkPen(METRIC_CONFIDENCE, width=1.5), name="Confidence"
        )

        # Threshold reference line (fixed position at 0.6 for visual reference)
        threshold_line = pg.InfiniteLine(
            pos=CONFIDENCE_THRESHOLD_LINE,
            angle=0,
            pen=pg.mkPen(METRIC_CHANCE_LEVEL, width=1, style=QtCore.Qt.PenStyle.DashLine),
        )
        confidence_plot.addItem(threshold_line)

        plots["confidence"] = {
            "plot": confidence_plot,
            "confidence_curve": confidence_curve,
            "threshold_line": threshold_line,
        }

        # Link X-axes for synchronized panning/zooming
        confidence_plot.setXLink(prediction_plot)

        self._plots[mode_name] = plots

    def handle_data(self, mode_name: str, item: dict[str, Any]) -> None:
        """
        Handle incoming async mode data and store for visualization.

        Expected data format (from asynchronous_mode.py):
        {
            'type': 'performance',
            'mode_name': str,
            'prediction': int,
            'confidence': float,
            'accuracy': float,
            'timestamp': float
        }

        Args:
            mode_name: Mode identifier
            item: Data packet from async mode
        """
        # Initialize if needed
        if mode_name not in self._mode_data:
            self.initialize_for_mode(mode_name)

        # Extract data payload
        payload = item.get("data", {})
        if not payload:
            logging.warning(f"AsyncPlot [{mode_name}]: No 'data' field in item")
            return

        # Extract fields
        timestamp = payload.get("timestamp", time.time())
        prediction = payload.get("prediction", 0)
        confidence = payload.get("confidence", 0.5)
        true_label = payload.get(
            "true_label", None
        )  # Ground truth label (-1 for background, 0+ for events)

        # Duplicate detection using timestamp
        timestamp_key = f"{timestamp:.6f}"
        if timestamp_key in self._processed_timestamps[mode_name]:
            return  # Already processed

        self._processed_timestamps[mode_name].add(timestamp_key)

        # Clear set when it exceeds limit (old timestamps won't be seen again)
        if len(self._processed_timestamps[mode_name]) > ASYNC_MAX_POINTS * 2:
            self._processed_timestamps[mode_name].clear()

        # Store data
        data = self._mode_data[mode_name]
        data["timestamps"].append(timestamp)
        data["predictions"].append(prediction)
        data["confidences"].append(confidence)
        data["true_labels"].append(true_label)

    def update_plots(self, mode_name: str) -> None:
        """
        Update visualizations with latest data.

        This method is called by the dashboard's update loop. No internal
        throttling is performed - the caller controls update frequency.

        Args:
            mode_name: Mode identifier to update
        """
        if mode_name not in self._mode_data:
            return  # Not initialized

        data = self._mode_data[mode_name]

        # Check if we have data to plot
        if not data["timestamps"] or not data["predictions"]:
            return

        # Check if new data arrived since last update
        current_latest = data["timestamps"][-1]
        last_update = self._last_update_times.get(mode_name, 0.0)

        if current_latest <= last_update:
            return  # No new data

        self._last_update_times[mode_name] = current_latest

        # Convert to numpy arrays
        timestamps = np.array(data["timestamps"])
        predictions = np.array(data["predictions"])
        confidences = np.array(data["confidences"])

        # Create relative time axis (seconds from start)
        time_axis = timestamps - timestamps[0]
        current_time = time_axis[-1]

        # Apply sliding window
        window_start = max(0, current_time - ASYNC_WINDOW_DURATION)
        window_mask = time_axis >= window_start

        if not np.any(window_mask):
            return

        # Extract windowed data
        windowed_time = time_axis[window_mask]
        windowed_predictions = predictions[window_mask]
        windowed_confidences = confidences[window_mask]

        # Extract true labels (convert to array, handling None values)
        true_labels = np.array(data["true_labels"], dtype=object)
        windowed_true_labels = true_labels[window_mask]

        plots = self._plots[mode_name]
        x_range = [window_start, current_time]

        pred_plot = plots["predictions"]

        # Filter true events: only show event periods (>=0), not background (-1)
        event_mask = np.array([label is not None and label >= 0 for label in windowed_true_labels])

        # Collect all unique classes from both predictions and true labels
        all_classes: set[int] = set()
        if len(windowed_predictions) > 0:
            all_classes.update(windowed_predictions.astype(int))
        if np.any(event_mask):
            true_classes = [
                int(label) for label, m in zip(windowed_true_labels, event_mask, strict=False) if m
            ]
            all_classes.update(true_classes)

        # Create mapping: class_value -> sequential_index for compact Y-axis
        unique_classes = sorted(all_classes) if all_classes else [0]
        class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}

        # Plot all predictions (no confidence filtering)
        if len(windowed_time) > 0:
            # Map predictions to sequential Y positions
            mapped_predictions = np.array([class_to_idx[int(p)] for p in windowed_predictions])
            pred_plot["prediction_scatter"].setData(windowed_time, mapped_predictions)
        else:
            pred_plot["prediction_scatter"].setData([], [])

        # Ground truth should always be visible regardless of prediction confidence
        if np.any(event_mask):
            true_event_times = windowed_time[event_mask]
            # Map true events to sequential Y positions
            mapped_true_events = np.array(
                [
                    class_to_idx[int(label)]
                    for label, m in zip(windowed_true_labels, event_mask, strict=False)
                    if m
                ]
            )
            pred_plot["true_event_scatter"].setData(true_event_times, mapped_true_events)
        else:
            pred_plot["true_event_scatter"].setData([], [])

        # Set Y-range based on number of unique classes (not max class value)
        n_classes = len(unique_classes)
        pred_plot["plot"].setYRange(-0.5, n_classes - 0.5, padding=0)

        # Y-ticks: show actual class labels at mapped positions
        y_ticks = [(idx, str(cls)) for cls, idx in class_to_idx.items()]
        pred_plot["plot"].getAxis("left").setTicks([y_ticks])

        pred_plot["plot"].setXRange(*x_range, padding=0)

        conf_plot = plots["confidence"]
        conf_plot["confidence_curve"].setData(windowed_time, windowed_confidences)
        conf_plot["plot"].setXRange(*x_range, padding=0)

    def clear_plots(self, mode_name: str | None = None) -> None:
        """
        Clear plots and data for a specific mode or all modes.

        Args:
            mode_name: Specific mode to clear, or None to clear all
        """
        if mode_name:
            # Clear specific mode
            if mode_name in self._plots:
                del self._plots[mode_name]
            if mode_name in self._mode_data:
                del self._mode_data[mode_name]
            if mode_name in self._processed_timestamps:
                del self._processed_timestamps[mode_name]
            logging.info(f"AsyncPlotManager cleared for mode: {mode_name}")
        else:
            if self.plot_widget:
                self.plot_widget.clear()
            self._plots.clear()
            self._mode_data.clear()
            self._processed_timestamps.clear()
            self._last_update_times.clear()
            logging.info("AsyncPlotManager cleared all modes")
