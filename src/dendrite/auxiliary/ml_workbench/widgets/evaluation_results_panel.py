"""Evaluation results visualization panel.

Self-contained widget that owns all evaluation visualization: prediction timeline
scatter plot, confidence curve, ground truth markers, rolling window, and stats.
"""

from collections import deque
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from dendrite.auxiliary.ml_workbench.utils import create_plot_container, create_styled_plot
from dendrite.auxiliary.ml_workbench.widgets.stats_panel import StatsPanel
from dendrite.gui.styles.design_tokens import (
    STATUS_ERROR,
    STATUS_SUCCESS,
    STATUS_WARNING_ALT,
    TEXT_MUTED,
)
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles

# Maximum points to keep in plot arrays (sliding window for performance)
MAX_PLOT_POINTS = 500


class EvaluationResultsPanel(QtWidgets.QWidget):
    """Panel displaying evaluation visualization with live-updating plots.

    Owns the prediction timeline scatter, confidence curve, ground truth markers,
    rolling window display, and stats panel. The parent tab feeds it data via
    public methods; all rendering logic is self-contained.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._class_names: dict[int, str] = {0: "Class 0", 1: "Class 1"}
        # Sliding window deques for O(1) append/trim
        self._pred_x: deque = deque(maxlen=MAX_PLOT_POINTS)
        self._pred_y: deque = deque(maxlen=MAX_PLOT_POINTS)
        self._pred_brushes: deque = deque(maxlen=MAX_PLOT_POINTS)
        self._conf_x: deque = deque(maxlen=MAX_PLOT_POINTS)
        self._conf_values: deque = deque(maxlen=MAX_PLOT_POINTS)
        self._truth_x: list[float] = []
        self._truth_y: list[int] = []
        self._latest_metrics: dict[str, Any] = {}
        self._prediction_count: int = 0
        self._setup_ui()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(LAYOUT["spacing_md"])

        results_container = create_plot_container("resultsContainer")
        container_layout = QtWidgets.QVBoxLayout(results_container)
        container_layout.setContentsMargins(
            LAYOUT["spacing_xl"], LAYOUT["spacing_xl"],
            LAYOUT["spacing_xl"], LAYOUT["spacing_xl"],
        )
        container_layout.setSpacing(LAYOUT["spacing_md"])

        self._stats_panel = StatsPanel()
        container_layout.addWidget(self._stats_panel)

        # Plots in splitter
        plots_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        # Prediction timeline plot
        self._pred_plot = create_styled_plot("Predictions vs Ground Truth")
        self._pred_plot.setLabel("bottom", "Time (s)")
        self._pred_plot.addLegend(offset=(10, 10))
        # Ground truth (larger white circles)
        self._truth_scatter = pg.ScatterPlotItem(
            size=14,
            pen=pg.mkPen(TEXT_MUTED, width=2),
            brush=pg.mkBrush(255, 255, 255, 150),
            symbol="o",
            name="Truth",
        )
        self._pred_plot.addItem(self._truth_scatter)
        # Predictions (smaller, colored by outcome)
        self._pred_scatter = pg.ScatterPlotItem(size=8, pen=None, name="Pred")
        self._pred_plot.addItem(self._pred_scatter)
        plots_splitter.addWidget(self._pred_plot)

        # Confidence plot (smaller, below prediction plot)
        self._conf_plot = create_styled_plot("Confidence", y_range=(0, 1))
        self._conf_plot.setLabel("left", "Conf")
        self._conf_plot.setLabel("bottom", "Time (s)")
        self._conf_curve = self._conf_plot.plot(pen=pg.mkPen(STATUS_WARNING_ALT, width=2))
        self._conf_plot.setXLink(self._pred_plot)
        plots_splitter.addWidget(self._conf_plot)

        plots_splitter.setSizes([200, 100])
        container_layout.addWidget(plots_splitter)

        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.setStyleSheet(WidgetStyles.button(transparent=True))
        clear_btn.clicked.connect(self.clear)
        container_layout.addWidget(clear_btn)

        layout.addWidget(results_container, stretch=1)

    # -- Public API --

    def set_class_names(self, class_names: dict[int, str]) -> None:
        """Update Y-axis tick labels from class name mapping."""
        self._class_names = class_names if class_names else {0: "Class 0", 1: "Class 1"}
        self._update_y_axis_labels()

    def set_ground_truth(self, truth_x: list[float], truth_y: list[int]) -> None:
        """Set ground truth marker positions (in seconds)."""
        self._truth_x = truth_x
        self._truth_y = truth_y
        self._truth_scatter.setData(self._truth_x, self._truth_y)

    def add_predictions(
        self,
        predictions: list[dict],
        sample_labels: np.ndarray | None,
        sampling_rate: float,
    ) -> None:
        """Process new predictions and update plots.

        Args:
            predictions: List of prediction dicts with sample_idx, prediction, confidence
            sample_labels: Precomputed sample->label array for O(1) ground truth lookup
            sampling_rate: Data sampling rate for time conversion
        """
        for pred in predictions:
            sample_idx = pred["sample_idx"]
            conf = pred.get("confidence", 0.5)
            label = int(sample_labels[sample_idx]) if sample_labels is not None else -1

            brush = self._get_point_color(pred["prediction"], label)

            time_sec = sample_idx / sampling_rate
            self._pred_x.append(time_sec)
            self._pred_y.append(int(pred["prediction"]))
            self._pred_brushes.append(brush)
            self._conf_x.append(time_sec)
            self._conf_values.append(conf)

            self._latest_metrics = {
                "accuracy": pred.get("accuracy", 0.0),
                "per_class_accuracy_named": pred.get("per_class_accuracy_named", {}),
            }

        self._prediction_count += len(predictions)

        # Update plots once per batch
        self._pred_scatter.setData(
            list(self._pred_x), list(self._pred_y), brush=list(self._pred_brushes),
        )
        self._conf_curve.setData(list(self._conf_x), list(self._conf_values))

        # Rolling window centered on latest prediction (20 seconds)
        if self._pred_x:
            current_time = self._pred_x[-1]
            view_window = 20
            x_min = max(0, current_time - view_window / 2)
            x_max = current_time + view_window / 2
            self._pred_plot.setXRange(x_min, x_max, padding=0)

    def get_latest_metrics(self) -> dict[str, Any]:
        """Return the latest metrics passed through predictions."""
        return self._latest_metrics

    def get_confidence_values(self) -> deque:
        """Return the confidence value deque for external metric computation."""
        return self._conf_values

    @property
    def prediction_count(self) -> int:
        """Number of predictions added so far."""
        return self._prediction_count

    def update_stats(self, metrics: dict) -> None:
        """Update the stats panel with computed metrics."""
        self._stats_panel.update_stats(metrics)

    def update_progress(self, count: int, total: int, duration_str: str) -> None:
        """Update the progress display."""
        self._stats_panel.update_progress(count, total, duration_str)

    def reset_plot_range(self) -> None:
        """Reset X-range to show full data with all events visible."""
        if not self._truth_x and not self._pred_x:
            return
        all_x = list(self._truth_x) + list(self._pred_x)
        x_min = min(all_x)
        x_max = max(all_x)
        padding = (x_max - x_min) * 0.02 if x_max > x_min else 1.0
        self._pred_plot.setXRange(x_min - padding, x_max + padding, padding=0)

    def clear(self) -> None:
        """Clear all plot state and stats."""
        self._pred_x.clear()
        self._pred_y.clear()
        self._pred_brushes.clear()
        self._conf_x.clear()
        self._conf_values.clear()
        self._pred_scatter.setData([], [])
        self._conf_curve.setData([], [])
        self._truth_x, self._truth_y = [], []
        self._truth_scatter.setData([], [])
        self._stats_panel.reset()
        self._latest_metrics = {}
        self._prediction_count = 0

    # -- Private helpers --

    def _update_y_axis_labels(self) -> None:
        """Update Y-axis ticks with class names."""
        ticks = []
        for label in sorted(self._class_names.keys()):
            name = self._class_names.get(label, f"Class {label}")
            ticks.append((label, name))

        if not ticks:
            ticks = [(0, "Class 0"), (1, "Class 1")]

        y_axis = self._pred_plot.getAxis("left")
        y_axis.setTicks([ticks])
        self._update_plot_y_range()

    def _update_plot_y_range(self) -> None:
        """Update prediction plot Y-range based on number of classes."""
        n_classes = max(len(self._class_names), 2)
        self._pred_plot.setYRange(-0.5, n_classes - 0.5, padding=0)

    def _get_point_color(self, pred: int, label: int) -> Any:
        """Get brush color for a prediction based on outcome."""
        if label == -1:
            return pg.mkBrush(TEXT_MUTED)
        elif pred == label:
            return pg.mkBrush(STATUS_SUCCESS)
        else:
            return pg.mkBrush(STATUS_ERROR)
