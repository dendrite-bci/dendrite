"""Evaluation tab - Async BCI metrics and live simulation."""

from collections import deque
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from dendrite.auxiliary.ml_workbench.backend import SimulationWorker
from dendrite.auxiliary.ml_workbench.utils import (
    create_plot_container,
    create_styled_plot,
    format_duration,
    setup_worker_thread,
)
from dendrite.auxiliary.ml_workbench.widgets import (
    StatsPanel,
    create_form_layout,
    create_scrollable_panel,
    create_section,
)
from dendrite.gui.styles.design_tokens import (
    STATUS_ERROR,
    STATUS_SUCCESS,
    STATUS_WARNING_ALT,
    TEXT_DISABLED,
    TEXT_MUTED,
)
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles
from dendrite.gui.widgets.common import TogglePillWidget
from dendrite.utils.logger_central import get_logger

logger = get_logger(__name__)


# Maximum points to keep in plot arrays (sliding window for performance)
MAX_PLOT_POINTS = 500


class EvaluationTab(QtWidgets.QWidget):
    """Evaluation tab for async BCI metrics and live simulation."""

    evaluation_started = QtCore.pyqtSignal()
    evaluation_finished = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._study_data: dict[str, Any] | None = None
        self._app_state = None
        self._decoder = None
        self._decoder_name: str | None = None
        self._worker = None
        self._thread = None
        self._sim_predictions: list[dict] = []
        # Use deques for O(1) sliding window (no list copying)
        self._pred_x: deque = deque(maxlen=MAX_PLOT_POINTS)
        self._pred_y: deque = deque(maxlen=MAX_PLOT_POINTS)
        self._pred_brushes: deque = deque(maxlen=MAX_PLOT_POINTS)
        self._conf_x: deque = deque(maxlen=MAX_PLOT_POINTS)
        self._conf_values: deque = deque(maxlen=MAX_PLOT_POINTS)
        self._truth_x: list[float] = []
        self._truth_y: list[int] = []
        self._total_samples: int = 0
        self._sampling_rate: float = 500.0
        self._window_length: int = 0
        # Latest metrics from mode (passed through predictions)
        self._latest_metrics: dict[str, Any] = {}
        # Event data for trial-level tracking
        self._event_times: np.ndarray | None = None
        self._event_labels: np.ndarray | None = None
        # Precomputed sample -> label mapping for O(1) lookup
        self._sample_labels: np.ndarray | None = None
        # Holdout validation data from training
        self._validation_data = None  # (continuous, event_times, event_labels)
        # Class label -> name mapping (extracted from dataset events dict)
        self._class_names: dict[int, str] = {}
        # Timer-based polling for GUI updates (no signals during eval loop)
        self._poll_timer = QtCore.QTimer()
        self._poll_timer.timeout.connect(self._poll_results)
        self._last_poll_idx = 0
        self._setup_ui()

    def set_app_state(self, state):
        self._app_state = state
        if state:
            state.study_changed.connect(self._on_study_changed)
            state.model_trained.connect(self._on_model_trained)
            state.validation_data_ready.connect(self._on_validation_data_ready)

    def _setup_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(
            LAYOUT["spacing_lg"], LAYOUT["spacing_lg"], LAYOUT["spacing_lg"], LAYOUT["spacing_lg"]
        )
        layout.setSpacing(LAYOUT["spacing_lg"])

        layout.addWidget(self._create_config_panel())
        layout.addWidget(self._create_results_panel(), stretch=1)

    def _create_config_panel(self) -> QtWidgets.QWidget:
        panel, layout, footer_layout = create_scrollable_panel(560)

        # Dataset section
        dataset_section, dataset_layout = create_section("Dataset")
        self._dataset_label = QtWidgets.QLabel("No dataset loaded")
        self._dataset_label.setWordWrap(True)
        dataset_layout.addWidget(self._dataset_label)
        layout.addWidget(dataset_section)

        # Model section
        model_section, model_layout = create_section("Model")
        self._model_label = QtWidgets.QLabel("No model trained")
        self._model_label.setWordWrap(True)
        self._model_label.setStyleSheet(WidgetStyles.inline_label(color=TEXT_DISABLED))
        model_layout.addWidget(self._model_label)
        layout.addWidget(model_section)

        # Evaluation section
        eval_section, eval_layout_v = create_section("Evaluation")

        eval_form = create_form_layout()

        self._step_ms_spin = QtWidgets.QSpinBox()
        self._step_ms_spin.setRange(10, 1000)
        self._step_ms_spin.setSingleStep(10)
        self._step_ms_spin.setValue(50)
        self._step_ms_spin.setSuffix(" ms")
        eval_form.addRow("Step:", self._step_ms_spin)
        eval_layout_v.addLayout(eval_form)

        # Gating toggle row (inside Evaluation section)
        gating_row = QtWidgets.QHBoxLayout()
        gating_row.setSpacing(LAYOUT["spacing_sm"])
        gating_row.addWidget(QtWidgets.QLabel("Decision Gating"))
        self._gating_toggle = TogglePillWidget(initial_state=False, show_label=False)
        self._gating_toggle.toggled.connect(self._on_gating_toggled)
        gating_row.addWidget(self._gating_toggle)
        gating_row.addStretch()
        eval_layout_v.addLayout(gating_row)

        # Gating controls container (hidden by default)
        self._gating_container = QtWidgets.QWidget()
        gating_form = create_form_layout()
        self._gating_container.setLayout(gating_form)

        self._conf_threshold_spin = QtWidgets.QDoubleSpinBox()
        self._conf_threshold_spin.setRange(0.5, 0.99)
        self._conf_threshold_spin.setSingleStep(0.05)
        self._conf_threshold_spin.setDecimals(2)
        self._conf_threshold_spin.setValue(0.6)
        gating_form.addRow("Confidence:", self._conf_threshold_spin)

        self._dwell_time_spin = QtWidgets.QDoubleSpinBox()
        self._dwell_time_spin.setRange(0.1, 1.0)
        self._dwell_time_spin.setSingleStep(0.1)
        self._dwell_time_spin.setDecimals(1)
        self._dwell_time_spin.setValue(0.3)
        self._dwell_time_spin.setSuffix(" s")
        gating_form.addRow("Dwell time:", self._dwell_time_spin)

        self._gating_container.hide()
        eval_layout_v.addWidget(self._gating_container)

        layout.addWidget(eval_section)
        layout.addStretch()

        # Footer: action buttons pinned below scroll area
        btn_layout = QtWidgets.QHBoxLayout()

        self._eval_btn = QtWidgets.QPushButton("Evaluate")
        self._eval_btn.setStyleSheet(WidgetStyles.button(padding="8px 12px"))
        self._eval_btn.setToolTip("Run evaluation with live visualization")
        self._eval_btn.clicked.connect(self._on_evaluate)
        self._eval_btn.setEnabled(False)
        btn_layout.addWidget(self._eval_btn)

        self._quick_btn = QtWidgets.QPushButton("Quick Eval")
        self._quick_btn.setStyleSheet(WidgetStyles.button(padding="8px 12px"))
        self._quick_btn.setToolTip("Fast evaluation without visualization")
        self._quick_btn.clicked.connect(self._on_quick_eval)
        self._quick_btn.setEnabled(False)
        btn_layout.addWidget(self._quick_btn)

        self._stop_btn = QtWidgets.QPushButton("Stop")
        self._stop_btn.setStyleSheet(WidgetStyles.button(severity="error", padding="8px 12px"))
        self._stop_btn.clicked.connect(self._on_stop_evaluation)
        self._stop_btn.hide()
        btn_layout.addWidget(self._stop_btn)
        footer_layout.addLayout(btn_layout)

        self._status_label = QtWidgets.QLabel("Ready")
        self._status_label.setWordWrap(True)
        footer_layout.addWidget(self._status_label)

        return panel

    def _create_results_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(LAYOUT["spacing_md"])

        # Plot container (no border for minimal aesthetic)
        results_container = create_plot_container("resultsContainer")
        sim_layout = QtWidgets.QVBoxLayout(results_container)
        sim_layout.setContentsMargins(
            LAYOUT["spacing_xl"], LAYOUT["spacing_xl"], LAYOUT["spacing_xl"], LAYOUT["spacing_xl"]
        )
        sim_layout.setSpacing(LAYOUT["spacing_md"])

        # Stats panel with grouped metrics
        self._stats_panel = StatsPanel()
        sim_layout.addWidget(self._stats_panel)

        # Plots in splitter
        plots_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        # Prediction timeline plot
        self._pred_plot = create_styled_plot("Predictions vs Ground Truth")
        # No left axis label - tick labels show class names directly
        self._pred_plot.setLabel("bottom", "Time (s)")
        self._pred_plot.addLegend(offset=(10, 10))
        # Y-axis width auto-sizes based on tick label length (set in _update_y_axis_labels)
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
        # Link X-axis for synchronized scrolling
        self._conf_plot.setXLink(self._pred_plot)
        plots_splitter.addWidget(self._conf_plot)

        # Set initial splitter sizes (2:1 ratio)
        plots_splitter.setSizes([200, 100])
        sim_layout.addWidget(plots_splitter)

        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.setStyleSheet(WidgetStyles.button(transparent=True))
        clear_btn.clicked.connect(self._clear_simulation)
        sim_layout.addWidget(clear_btn)

        layout.addWidget(results_container, stretch=1)
        return panel

    def _on_study_changed(self, data):
        """Handle study selection change."""
        # Clear stale holdout data from previous study
        self._validation_data = None

        if data is None or not isinstance(data, dict) or "config" not in data:
            self._study_data = None
            self._dataset_label.setText("No study selected")
            self._class_names = {0: "Class 0", 1: "Class 1"}
            self._update_y_axis_labels()
            self._update_button_states()
            return

        self._study_data = data
        config = data["config"]
        selected_subj = data.get("selected_subject")

        # Extract class names from events dict (e.g., {'left_hand': 1, 'right_hand': 2})
        # Create 0-indexed mapping to match trainer's label normalization
        if hasattr(config, "events") and config.events:
            sorted_labels = sorted(config.events.values())
            self._class_names = {}
            for orig_label in sorted_labels:
                name = next(k for k, v in config.events.items() if v == orig_label)
                self._class_names[sorted_labels.index(orig_label)] = name
        else:
            self._class_names = {0: "Class 0", 1: "Class 1"}
        self._update_y_axis_labels()

        subj_str = (
            f"Subject {selected_subj}" if selected_subj else f"{len(config.subjects)} subjects"
        )

        self._dataset_label.setText(f"<b>{config.name}</b><br>{subj_str}")
        self._update_button_states()

    def _update_y_axis_labels(self):
        """Update Y-axis ticks with class names from dataset."""
        ticks = []
        for label in sorted(self._class_names.keys()):  # All classes
            name = self._class_names.get(label, f"Class {label}")
            ticks.append((label, name))

        if not ticks:
            ticks = [(0, "Class 0"), (1, "Class 1")]

        y_axis = self._pred_plot.getAxis("left")
        y_axis.setTicks([ticks])

        # Update Y-range based on number of classes
        self._update_plot_y_range()

    def _update_plot_y_range(self):
        """Update prediction plot Y-range based on number of classes."""
        n_classes = max(len(self._class_names), 2)
        # Use padding=0 to prevent labels from being pushed off-screen
        self._pred_plot.setYRange(-0.5, n_classes - 0.5, padding=0)

    def _on_model_trained(self, name: str, model):
        """Handle model trained."""
        self._decoder = model
        self._decoder_name = name

        # Build info string with decoder properties
        info_parts = [f"<b>{name}</b>"]

        if hasattr(model, "input_shapes") and model.input_shapes:
            for modality, shape in model.input_shapes.items():
                info_parts.append(f"{modality}: {shape[0]}ch × {shape[1]}samples")

        if hasattr(model, "num_classes"):
            info_parts.append(f"Classes: {model.num_classes}")

        if hasattr(model, "config") and model.config:
            if model.config.sample_rate:
                info_parts.append(f"Rate: {int(model.config.sample_rate)}Hz")

        self._model_label.setText("<br>".join(info_parts))
        self._model_label.setStyleSheet("")
        self._update_button_states()

    def _on_validation_data_ready(self, val_data):
        """Handle holdout validation data from training."""
        self._validation_data = val_data
        if val_data:
            continuous, times, labels = val_data
            # Calculate duration from continuous data shape and sample rate
            sample_rate = getattr(self._decoder, "sample_rate", 250.0) if self._decoder else 250.0
            duration = continuous.shape[1] / sample_rate
            self._status_label.setText(
                f"Holdout data ready: {len(labels)} events ({format_duration(duration)})"
            )
        else:
            self._status_label.setText("Ready (no holdout data)")

    def _update_button_states(self):
        has_data = self._study_data is not None
        has_holdout = self._validation_data is not None
        decoder_valid = self._decoder is not None and hasattr(self._decoder, "predict")

        can_eval = decoder_valid and has_holdout
        self._eval_btn.setEnabled(can_eval)
        self._quick_btn.setEnabled(can_eval)

        # Show helpful message when eval is disabled due to missing holdout
        if decoder_valid and has_data and not has_holdout:
            self._status_label.setText("Train with holdout % to enable evaluation")

        if self._decoder is not None and not decoder_valid:
            self._model_label.setText(
                f"<b>{self._decoder_name}</b><br><span style='color:orange'>Missing predict</span>"
            )

    def _on_gating_toggled(self, enabled: bool):
        """Show or hide gating controls when toggle changes."""
        self._gating_container.setVisible(enabled)

    def _get_gating_status_text(self) -> str:
        """Get gating status text for status label."""
        if not self._gating_toggle.isChecked():
            return ""
        parts = [
            f"conf>{self._conf_threshold_spin.value():.2f}",
            f"dwell={self._dwell_time_spin.value():.1f}s",
        ]
        return f" [{', '.join(parts)}]"

    def _prepare_evaluation_data(self, modality: str = "eeg"):
        """Prepare data for evaluation.

        Validates decoder, loads continuous data, and computes step parameters.

        Returns:
            tuple: (continuous_data, event_times, event_labels, window_size, step_samples)
                   or None if validation fails (error shown in status)
        """
        if not self._decoder:
            return None

        if not self._study_data and not self._validation_data:
            self._status_label.setText("No data available for evaluation")
            return None

        # Get window size from decoder input shapes
        if hasattr(self._decoder, "input_shapes") and self._decoder.input_shapes:
            if modality in self._decoder.input_shapes:
                shape = self._decoder.input_shapes[modality]
                window_size = shape[1] if len(shape) > 1 else shape[0]
            else:
                self._status_label.setText(f"Decoder has no input shape for {modality}")
                return None
        else:
            self._status_label.setText("Decoder missing input_shapes")
            return None

        if self._validation_data is None:
            self._status_label.setText("No holdout data - train with holdout % first")
            return None

        continuous_data, event_times, event_labels = self._validation_data
        config = self._study_data["config"] if self._study_data else None

        n_channels, n_samples = continuous_data.shape

        # Get sample rate from config, decoder, or keep default
        if config and config.sample_rate > 0:
            self._sampling_rate = config.sample_rate
        elif hasattr(self._decoder, "sample_rate") and self._decoder.sample_rate > 0:
            self._sampling_rate = self._decoder.sample_rate

        self._window_length = window_size

        # Compute step in samples
        step_ms = self._step_ms_spin.value()
        step_samples = max(1, int((step_ms / 1000) * self._sampling_rate))

        self._total_samples = max(1, (n_samples - window_size) // step_samples)

        return continuous_data, event_times, event_labels, window_size, step_samples

    def _run_evaluation(self, real_time: bool = True):
        """Run evaluation with optional live visualization.

        Args:
            real_time: If True, add delays for live visualization
        """
        modality = "eeg"
        result = self._prepare_evaluation_data(modality)
        if result is None:
            return

        continuous_data, event_times, event_labels, window_size, step_samples = result
        n_channels, n_samples = continuous_data.shape

        # Store event info for trial-level metrics
        self._event_times = event_times
        self._event_labels = event_labels

        if self._validation_data is not None:
            duration = n_samples / self._sampling_rate
            self._status_label.setText(
                f"Using holdout data: {len(event_labels)} events ({format_duration(duration)})"
            )

        self._clear_simulation()

        # Build label remapping to 0-indexed (same as trainer does)
        unique_labels = sorted(set(event_labels))
        label_remap = {old: new for new, old in enumerate(unique_labels)}

        # Precompute sample -> label mapping for O(1) lookup in batch handler
        self._sample_labels = np.full(n_samples, -1, dtype=np.int64)
        for event_time, label in zip(event_times, event_labels, strict=False):
            if label >= 0:
                start = int(event_time)
                end = min(start + window_size, n_samples)
                self._sample_labels[start:end] = label_remap.get(int(label), int(label))

        # Set ground truth markers at event positions (convert to seconds)
        self._truth_x = []
        self._truth_y = []
        for event_time, label in zip(event_times, event_labels, strict=False):
            if label >= 0:
                self._truth_x.append(event_time / self._sampling_rate)
                self._truth_y.append(label_remap.get(int(label), int(label)))
        self._truth_scatter.setData(self._truth_x, self._truth_y)

        self._eval_btn.setEnabled(False)
        self._quick_btn.setEnabled(False)

        X_continuous = continuous_data.T
        gating_config = (
            {
                "confidence_threshold": self._conf_threshold_spin.value(),
                "use_confidence_gating": True,
                "dwell_time_sec": self._dwell_time_spin.value(),
                "use_dwell_time_gating": True,
                "background_class": 0,
            }
            if self._gating_toggle.isChecked()
            else None
        )

        self._thread = QtCore.QThread()
        self._worker = SimulationWorker(
            decoder=self._decoder,
            X=X_continuous,
            event_times=event_times,
            event_labels=event_labels,
            epoch_length=window_size,
            step_size=step_samples,
            modality=modality,
            sample_rate=self._sampling_rate,
            gating_config=gating_config,
            real_time=real_time,
            class_names=self._class_names,
        )
        on_finished = self._on_evaluation_finished if real_time else self._on_quick_finished
        setup_worker_thread(
            self._worker,
            self._thread,
            on_finished=on_finished,
            on_error=lambda e: self._status_label.setText(f"Error: {e}"),
        )

        self._last_poll_idx = 0
        self._poll_timer.start(100)

        self._stop_btn.show()
        gating_info = self._get_gating_status_text()
        step_ms = self._step_ms_spin.value()
        prefix = "Running" if real_time else "Quick"
        self._status_label.setText(
            f"{prefix}: {n_channels}ch × {window_size}samples, step={step_ms:.0f}ms{gating_info}"
        )
        self._thread.start()

    def _on_evaluate(self):
        """Run sliding window evaluation with live visualization."""
        self._run_evaluation(real_time=True)

    def _on_quick_eval(self):
        """Run fast batch evaluation without live visualization."""
        self._run_evaluation(real_time=False)

    def _poll_results(self):
        """Poll worker for new results and update GUI (timer callback)."""
        if not self._worker:
            return

        new_results = self._worker.get_results_since(self._last_poll_idx)
        if not new_results:
            return

        # Process all new results
        for pred in new_results:
            sample_idx = pred["sample_idx"]
            self._sim_predictions.append(pred)

            conf = pred.get("confidence", 0.5)

            # O(1) label lookup from precomputed array
            label = int(self._sample_labels[sample_idx]) if self._sample_labels is not None else -1

            # Determine color based on prediction outcome
            brush = self._get_point_color(pred["prediction"], label)

            # Accumulate plot data (deques auto-trim) - convert to seconds
            time_sec = sample_idx / self._sampling_rate
            self._pred_x.append(time_sec)
            self._pred_y.append(int(pred["prediction"]))
            self._pred_brushes.append(brush)
            self._conf_x.append(time_sec)
            self._conf_values.append(conf)

            # Store latest metrics from mode (passed through prediction)
            self._latest_metrics = {
                "accuracy": pred.get("accuracy", 0.0),
                "per_class_accuracy_named": pred.get("per_class_accuracy_named", {}),
            }

        self._last_poll_idx += len(new_results)

        # Update plots ONCE per poll
        self._pred_scatter.setData(
            list(self._pred_x), list(self._pred_y), brush=list(self._pred_brushes)
        )
        self._conf_curve.setData(list(self._conf_x), list(self._conf_values))

        # Rolling window centered on latest prediction (20 seconds)
        if self._pred_x:
            current_time = self._pred_x[-1]  # Already in seconds
            view_window = 20  # 20 seconds
            x_min = max(0, current_time - view_window / 2)
            x_max = current_time + view_window / 2
            self._pred_plot.setXRange(x_min, x_max, padding=0)

        # Update stats and progress with duration
        count = len(self._sim_predictions)
        total_duration = self._total_samples * self._step_ms_spin.value() / 1000
        self._stats_panel.update_progress(
            count, self._total_samples, format_duration(total_duration)
        )
        self._update_stats_display()

    def _on_quick_finished(self):
        """Handle quick evaluation completion."""
        self._poll_timer.stop()
        self._poll_results()  # Final poll to get remaining results

        # Reset X-range to show full data with all events visible
        self._reset_plot_range()

        self._update_button_states()
        self._stop_btn.hide()

        count = len(self._sim_predictions)
        self._status_label.setText(f"Quick eval complete ({count} predictions)")
        self._update_stats_display(final=True)

    def _on_stop_evaluation(self):
        """Stop the running evaluation."""
        self._poll_timer.stop()
        if self._worker:
            self._worker.stop()
        self._stop_btn.hide()
        self._status_label.setText("Stopping evaluation...")

    def _get_point_color(self, pred: int, label: int) -> Any:
        """Get brush color for a prediction based on outcome."""
        if label == -1:
            return pg.mkBrush(TEXT_MUTED)  # Background (no event)
        elif pred == label:
            return pg.mkBrush(STATUS_SUCCESS)  # Correct prediction
        else:
            return pg.mkBrush(STATUS_ERROR)  # Incorrect prediction

    def _reset_plot_range(self):
        """Reset X-range to show full data with all events visible."""
        if not self._truth_x and not self._pred_x:
            return

        # Combine truth and prediction X values to get full range (in seconds)
        all_x = list(self._truth_x) + list(self._pred_x)
        x_min = min(all_x)
        x_max = max(all_x)
        padding = (x_max - x_min) * 0.02 if x_max > x_min else 1.0  # 1 second default
        self._pred_plot.setXRange(x_min - padding, x_max + padding, padding=0)

    def _on_evaluation_finished(self):
        """Handle evaluation completion."""
        self._poll_timer.stop()
        self._poll_results()  # Final poll to get remaining results

        # Reset X-range to show full data with all events visible
        self._reset_plot_range()

        # Re-enable buttons
        self._update_button_states()
        self._stop_btn.hide()

        count = len(self._sim_predictions)
        self._status_label.setText(f"Evaluation complete ({count} predictions)")

        # Final metrics update with ITR
        self._update_stats_display(final=True)

    def _update_stats_display(self, final: bool = False):
        """Update stats panel from mode's metrics (passed through predictions).

        Args:
            final: If True, get full metrics from worker's MetricsManager
        """
        count = len(self._sim_predictions)
        if count == 0:
            return

        # Use latest metrics passed from mode via predictions
        metrics = {
            "accuracy": self._latest_metrics.get("accuracy", 0),
            "per_class_accuracy_named": self._latest_metrics.get("per_class_accuracy_named", {}),
        }

        # For final stats, get full metrics from worker's MetricsManager
        if final and self._worker and self._worker.metrics_manager:
            async_metrics = self._worker.metrics_manager.get_current_metrics()
            metrics["n_trials"] = async_metrics.get("n_trials", 0)
            metrics["ttd"] = async_metrics.get("mean_ttd_ms", float("nan"))
            metrics["far"] = async_metrics.get("far_per_min", 0)
            metrics["per_class_accuracy_named"] = async_metrics.get("per_class_accuracy_named", {})

            # Calculate ITR
            if self._worker.metrics_manager.async_metrics:
                n_events = async_metrics.get("n_trials", 0)
                if n_events > 0 and self._total_samples > 0:
                    total_duration_sec = self._total_samples * self._step_ms_spin.value() / 1000
                    avg_selection_time = total_duration_sec / n_events
                    metrics["itr"] = self._worker.metrics_manager.async_metrics.calculate_itr(
                        num_classes=len(self._class_names) if self._class_names else 2,
                        mean_selection_time_sec=avg_selection_time,
                    )

        # Mean confidence (use deque directly)
        if self._conf_values:
            metrics["conf"] = np.mean(list(self._conf_values))

        total_duration = self._total_samples * self._step_ms_spin.value() / 1000
        self._stats_panel.update_progress(
            count, self._total_samples, format_duration(total_duration)
        )
        self._stats_panel.update_stats(metrics)

    def _clear_simulation(self):
        self._sim_predictions = []
        self._pred_x.clear()
        self._pred_y.clear()
        self._pred_brushes.clear()
        self._conf_x.clear()
        self._conf_values.clear()
        self._sample_labels = None
        self._pred_scatter.setData([], [])
        self._conf_curve.setData([], [])
        self._truth_x, self._truth_y = [], []
        self._truth_scatter.setData([], [])
        self._stats_panel.reset()
        self._latest_metrics = {}
