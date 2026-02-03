"""Stats panel widget for evaluation metrics display."""

import numpy as np
from PyQt6 import QtWidgets

from dendrite.gui.styles.design_tokens import (
    STATUS_ERROR,
    STATUS_SUCCESS,
    STATUS_WARNING_ALT,
    TEXT_MAIN,
    TEXT_MUTED,
)
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles


class StatsPanel(QtWidgets.QWidget):
    """Stats panel for trial-level evaluation metrics (minimal telemetry style)."""

    # Thresholds: (good, warning, higher_is_better)
    THRESHOLDS = {
        "accuracy": (0.8, 0.6, True),
        "ttd": (200, 400, False),
        "far": (1.0, 3.0, False),
        "conf": (0.7, 0.5, True),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._metrics = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(LAYOUT["spacing_sm"])

        # Progress row: "0 / 0" + progress bar
        progress_layout = QtWidgets.QHBoxLayout()
        self.progress_label = QtWidgets.QLabel("0 / 0")
        self.progress_label.setStyleSheet(
            WidgetStyles.inline_label(color=TEXT_MAIN, size=14, weight=500)
        )
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setMaximumHeight(12)
        self.progress_bar.setTextVisible(False)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar, stretch=1)
        layout.addLayout(progress_layout)

        layout.addSpacing(4)

        # Section header: "METRICS"
        header = QtWidgets.QLabel("METRICS")
        header.setStyleSheet(WidgetStyles.section_header())
        layout.addWidget(header)

        # Compact metric rows (HTML labels for inline colored values)
        self._detection_label = QtWidgets.QLabel()
        self._detection_label.setStyleSheet(WidgetStyles.inline_label(color=TEXT_MAIN, size=13))
        layout.addWidget(self._detection_label)

        self._timing_label = QtWidgets.QLabel()
        self._timing_label.setStyleSheet(WidgetStyles.inline_label(color=TEXT_MAIN, size=13))
        layout.addWidget(self._timing_label)

        self._summary_label = QtWidgets.QLabel()
        self._summary_label.setStyleSheet(WidgetStyles.inline_label(color=TEXT_MAIN, size=13))
        layout.addWidget(self._summary_label)

        self._update_display()

    def update_progress(self, current: int, total: int, duration_str: str = ""):
        """Update progress display."""
        if duration_str:
            self.progress_label.setText(f"{current} / {total}  ({duration_str})")
        else:
            self.progress_label.setText(f"{current} / {total}")
        if total > 0:
            self.progress_bar.setValue(int(100 * current / total))

    def update_stats(self, metrics: dict):
        """Update statistics display with compact colored metrics."""
        self._metrics.update(metrics)
        self._update_display()

    def _update_display(self):
        """Rebuild the metric display lines from current metrics."""
        m = self._metrics

        # Row 1: Detection - Per-class accuracy or overall accuracy
        detection_parts = []
        if m.get("per_class_accuracy_named"):
            for name, acc in m["per_class_accuracy_named"].items():
                detection_parts.append(self._format_metric(name, acc, "accuracy", "%", scale=100))
        elif "balanced_accuracy" in m:
            detection_parts.append(
                self._format_metric("Acc", m["balanced_accuracy"], "accuracy", "%", scale=100)
            )
        self._detection_label.setText("  •  ".join(detection_parts) if detection_parts else "--")

        # Row 2: Timing - TTD, FAR, ITR
        timing_parts = []
        if "ttd" in m and not np.isnan(m["ttd"]):
            timing_parts.append(self._format_metric("TTD", m["ttd"], "ttd", "ms"))
        if "far" in m:
            timing_parts.append(self._format_metric("FAR", m["far"], "far", "/m"))
        if "itr" in m:
            timing_parts.append(
                f'<span style="color: {TEXT_MUTED};">ITR:</span> <span style="color: {STATUS_WARNING_ALT};">{m["itr"]:.1f} b/m</span>'
            )
        self._timing_label.setText("  •  ".join(timing_parts) if timing_parts else "--")

        # Row 3: Summary - trials, confidence
        summary_parts = []
        if "n_trials" in m:
            n = int(m["n_trials"])
            summary_parts.append(f'<span style="color: {TEXT_MAIN};">{n} trials</span>')
        if "conf" in m:
            summary_parts.append(self._format_metric("Conf", m["conf"], "conf", "%", scale=100))
        self._summary_label.setText("  •  ".join(summary_parts) if summary_parts else "--")

    def _format_metric(
        self, label: str, value: float, metric_key: str, unit: str, scale: float = 1
    ) -> str:
        """Format a metric with threshold-based coloring."""
        color = self._get_metric_color(value, metric_key)
        display_value = value * scale
        return f'<span style="color: {TEXT_MUTED};">{label}:</span> <span style="color: {color};">{display_value:.0f}{unit}</span>'

    def _get_metric_color(self, value: float, metric_key: str) -> str:
        """Get color based on thresholds."""
        if metric_key not in self.THRESHOLDS:
            return TEXT_MAIN

        good, warning, higher_is_better = self.THRESHOLDS[metric_key]
        if higher_is_better:
            return (
                STATUS_SUCCESS
                if value >= good
                else STATUS_WARNING_ALT
                if value >= warning
                else STATUS_ERROR
            )
        return (
            STATUS_SUCCESS
            if value <= good
            else STATUS_WARNING_ALT
            if value <= warning
            else STATUS_ERROR
        )

    def reset(self):
        """Reset all displays."""
        self.progress_label.setText("0 / 0")
        self.progress_bar.setValue(0)
        self._metrics = {}
        self._update_display()
