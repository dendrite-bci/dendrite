#!/usr/bin/env python
"""
Performance Plot Manager

Handles initialization and updating of performance metrics plots for synchronous and training modes.
"""

import logging
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from dendrite.auxiliary.dashboard.plot_managers.plot_utils import (
    METRIC_ACCURACY,
    METRIC_CHANCE_LEVEL,
    METRIC_CONFIDENCE,
    DotSample,
    style_legend,
    style_plot_axes,
)
from dendrite.gui.styles.design_tokens import TEXT_LABEL

EMA_ALPHA = 0.3  # Smoothing factor (lower = smoother)


class PerformancePlotManager:
    """Manages performance metrics plots"""

    def __init__(self, plot_widget: pg.GraphicsLayoutWidget):
        self.plot_widget = plot_widget
        self.plots_by_metric_set: dict[str, dict[str, Any]] = {}
        self._smoothing_enabled: dict[str, bool] = {}  # Per-mode smoothing state

        # Essential metrics only - always shown, no toggle needed
        self.plot_specs = {
            "accuracy": {"pen": pg.mkPen(METRIC_ACCURACY, width=1.5), "name": "Accuracy"},
            "confidence": {"pen": pg.mkPen(METRIC_CONFIDENCE, width=1.5), "name": "Confidence"},
            "chance_level": {
                "pen": pg.mkPen(METRIC_CHANCE_LEVEL, width=1.5, style=QtCore.Qt.PenStyle.DashLine),
                "name": "Chance Level",
            },
        }

    def create_controls_widget(self, mode_name: str) -> QtWidgets.QWidget:
        """Create smoothing toggle control for performance curves."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        smooth_checkbox = QtWidgets.QCheckBox("Smooth")
        smooth_checkbox.setChecked(self._smoothing_enabled.get(mode_name, False))
        smooth_checkbox.toggled.connect(lambda checked: self._set_smoothing(mode_name, checked))
        layout.addWidget(smooth_checkbox)

        return widget

    def _set_smoothing(self, mode_name: str, enabled: bool) -> None:
        """Toggle smoothing for a mode."""
        self._smoothing_enabled[mode_name] = enabled

    def _apply_ema(self, values: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
        """Apply exponential moving average to smooth values."""
        if len(values) == 0:
            return values
        smoothed = np.zeros_like(values)
        smoothed[0] = values[0]
        for i in range(1, len(values)):
            smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
        return smoothed

    def initialize_plots(self, mode_name: str):
        """Initialize performance plots for a specific mode."""
        if mode_name not in self.plots_by_metric_set:
            self.plots_by_metric_set[mode_name] = {}

        logging.info(f"Performance plot manager initialized for mode: {mode_name}")

    def update_plots(self, mode_name: str, perf_data: dict[str, Any]):
        """Update performance plots for a mode."""
        if mode_name not in self.plots_by_metric_set:
            self.initialize_plots(mode_name)

        mode_plots = self.plots_by_metric_set[mode_name]
        current_row = 0

        # Sort metric set keys for consistent ordering
        sorted_metric_sets = sorted(perf_data.keys())
        processed_metric_sets = set()

        for metric_set_key in sorted_metric_sets:
            processed_metric_sets.add(metric_set_key)
            rec = perf_data[metric_set_key]

            if not rec.get("metrics") or not rec.get("indices"):
                # Clear curves if no data
                if metric_set_key in mode_plots:
                    self._clear_metric_set_curves(mode_plots[metric_set_key])
                continue

            indices = np.array(rec["indices"])

            # Create plot if it doesn't exist
            if metric_set_key not in mode_plots:
                plot_item = self.plot_widget.addPlot(row=current_row, col=0)
                title = self._get_plot_title(metric_set_key, mode_name)
                plot_item.setTitle(title, color=TEXT_LABEL, size="9pt")
                legend = plot_item.addLegend(offset=(10, 10), sampleType=DotSample)
                style_legend(legend)
                plot_item.setLabel("left", "Value", color=TEXT_LABEL)
                plot_item.setLabel("bottom", "Samples", color=TEXT_LABEL)
                style_plot_axes(plot_item)

                mode_plots[metric_set_key] = {"plot": plot_item, "curves": {}}

            plot_config = mode_plots[metric_set_key]
            plot_item = plot_config["plot"]
            plot_item.setVisible(True)

            # Update curves for each metric
            active_curves = set()
            for metric_name, metric_data in rec["metrics"].items():
                if not metric_data:
                    continue

                # Only show metrics we have specs for (essential metrics only)
                spec = self.plot_specs.get(metric_name)
                if not spec:
                    continue

                curve_id = metric_name

                # Create curve if it doesn't exist
                if curve_id not in plot_config["curves"]:
                    curve = plot_item.plot(pen=spec["pen"], name=spec["name"], antialias=True)
                    plot_config["curves"][curve_id] = curve

                # Update curve data
                curve = plot_config["curves"][curve_id]
                try:
                    vals = np.array(metric_data)
                    if self._smoothing_enabled.get(mode_name, False):
                        vals = self._apply_ema(vals)
                    data_len = min(len(vals), len(indices))
                    if data_len > 0:
                        curve.setData(indices[:data_len], vals[:data_len])
                        curve.setVisible(True)
                        active_curves.add(curve_id)
                    else:
                        curve.clear()
                        curve.setVisible(False)
                except Exception as e:
                    logging.warning(
                        f"Failed to update plot for {mode_name}/{metric_set_key}/{metric_name}: {e}"
                    )
                    curve.setVisible(False)

            # Hide inactive curves
            for curve_id in list(plot_config["curves"].keys()):
                if curve_id not in active_curves:
                    curve = plot_config["curves"][curve_id]
                    curve.clear()
                    curve.setVisible(False)

            # Set plot ranges
            if len(indices) > 0:
                plot_item.setXRange(indices.min(), indices.max(), padding=0.02)
                plot_item.enableAutoRange(axis="y")

            current_row += 1

        # Clean up plots for removed metric sets
        for old_metric_set in list(mode_plots.keys()):
            if old_metric_set not in processed_metric_sets:
                self._remove_metric_set_plot(mode_plots, old_metric_set)

    def _get_plot_title(self, metric_set_key: str, mode_name: str) -> str:
        """Generate appropriate plot title."""
        if metric_set_key == "synchronous_metrics":
            return f"Synchronous Mode Metrics - {mode_name}"
        elif "async" in metric_set_key:
            return f"Asynchronous Classification Metrics - {mode_name}"
        else:
            title_part = metric_set_key.replace("_", " ").title()
            return f"{title_part} - {mode_name}"

    def _clear_metric_set_curves(self, plot_config: dict[str, Any]):
        """Clear all curves in a metric set plot."""
        if "plot" in plot_config and plot_config["plot"] is not None:
            for curve_id in list(plot_config.get("curves", {}).keys()):
                curve = plot_config["curves"].pop(curve_id)
                plot_config["plot"].removeItem(curve)

    def _remove_metric_set_plot(self, mode_plots: dict[str, Any], metric_set_key: str):
        """Remove a metric set plot completely."""
        plot_config = mode_plots.pop(metric_set_key)
        if "plot" in plot_config and plot_config["plot"] is not None:
            # Clear all curves first
            for curve_id in list(plot_config.get("curves", {}).keys()):
                curve = plot_config["curves"].pop(curve_id)
                plot_config["plot"].removeItem(curve)

            if plot_config["plot"].legend is not None:
                plot_config["plot"].legend.clear()
                plot_config["plot"].removeItem(plot_config["plot"].legend)
                plot_config["plot"].legend = None

            # Remove plot from widget
            self.plot_widget.removeItem(plot_config["plot"])
            logging.info(f"Removed plot for metric set: {metric_set_key}")

    def clear_plots(self, mode_name: str | None = None):
        """Clear performance plots for a specific mode or all modes."""
        if mode_name:
            if mode_name in self.plots_by_metric_set:
                mode_plots = self.plots_by_metric_set[mode_name]
                for metric_set_key in list(mode_plots.keys()):
                    self._remove_metric_set_plot(mode_plots, metric_set_key)
                del self.plots_by_metric_set[mode_name]
                logging.info(f"Cleared performance plots for mode: {mode_name}")
        else:
            # Clear all plots
            if self.plot_widget:
                self.plot_widget.clear()
            self.plots_by_metric_set.clear()
            logging.info("All performance plots cleared.")
