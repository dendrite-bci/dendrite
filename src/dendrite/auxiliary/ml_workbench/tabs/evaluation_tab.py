"""Evaluation tab - Async BCI metrics and live simulation."""

from functools import partial
from typing import Any

import numpy as np
from PyQt6 import QtCore, QtWidgets

from dendrite.auxiliary.ml_workbench.backend import SimulationWorker
from dendrite.auxiliary.ml_workbench.utils import format_duration, setup_worker_thread
from dendrite.auxiliary.ml_workbench.widgets import (
    EvaluationResultsPanel,
    create_form_layout,
    create_scrollable_panel,
    create_section,
)
from dendrite.gui.styles.design_tokens import TEXT_DISABLED
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles
from dendrite.gui.widgets.common import TogglePillWidget
from dendrite.utils.logger_central import get_logger

logger = get_logger(__name__)


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
        self._total_samples: int = 0
        self._sampling_rate: float = 500.0
        self._sample_labels: np.ndarray | None = None
        self._validation_data = None
        self._class_names: dict[int, str] = {}
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

    # -- UI Setup --

    def _setup_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(
            LAYOUT["spacing_lg"], LAYOUT["spacing_lg"], LAYOUT["spacing_lg"], LAYOUT["spacing_lg"]
        )
        layout.setSpacing(LAYOUT["spacing_lg"])

        layout.addWidget(self._create_config_panel())
        self._results_panel = EvaluationResultsPanel()
        layout.addWidget(self._results_panel, stretch=1)

    def _create_config_panel(self) -> QtWidgets.QWidget:
        panel, layout, footer_layout = create_scrollable_panel(560)

        layout.addWidget(self._create_dataset_section())
        layout.addWidget(self._create_model_section())
        layout.addWidget(self._create_evaluation_section())
        layout.addStretch()
        self._create_footer(footer_layout)

        return panel

    def _create_dataset_section(self) -> QtWidgets.QWidget:
        section, section_layout = create_section("Dataset")
        self._dataset_label = QtWidgets.QLabel("No dataset loaded")
        self._dataset_label.setWordWrap(True)
        section_layout.addWidget(self._dataset_label)
        return section

    def _create_model_section(self) -> QtWidgets.QWidget:
        section, section_layout = create_section("Model")
        self._model_label = QtWidgets.QLabel("No model trained")
        self._model_label.setWordWrap(True)
        self._model_label.setStyleSheet(WidgetStyles.inline_label(color=TEXT_DISABLED))
        section_layout.addWidget(self._model_label)
        return section

    def _create_evaluation_section(self) -> QtWidgets.QWidget:
        section, section_layout = create_section("Evaluation")

        eval_form = create_form_layout()
        self._step_ms_spin = QtWidgets.QSpinBox()
        self._step_ms_spin.setRange(10, 1000)
        self._step_ms_spin.setSingleStep(10)
        self._step_ms_spin.setValue(50)
        self._step_ms_spin.setSuffix(" ms")
        eval_form.addRow("Step:", self._step_ms_spin)
        section_layout.addLayout(eval_form)

        # Gating toggle
        gating_row = QtWidgets.QHBoxLayout()
        gating_row.setSpacing(LAYOUT["spacing_sm"])
        gating_row.addWidget(QtWidgets.QLabel("Decision Gating"))
        self._gating_toggle = TogglePillWidget(initial_state=False, show_label=False)
        self._gating_toggle.toggled.connect(self._on_gating_toggled)
        gating_row.addWidget(self._gating_toggle)
        gating_row.addStretch()
        section_layout.addLayout(gating_row)

        # Gating controls (hidden by default)
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
        section_layout.addWidget(self._gating_container)

        return section

    def _create_footer(self, footer_layout: QtWidgets.QVBoxLayout):
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

    # -- App state handlers --

    def _on_study_changed(self, data):
        """Handle study selection change."""
        self._validation_data = None

        if data is None or not isinstance(data, dict) or "config" not in data:
            self._study_data = None
            self._dataset_label.setText("No study selected")
            self._class_names = {0: "Class 0", 1: "Class 1"}
            self._results_panel.set_class_names(self._class_names)
            self._update_button_states()
            return

        self._study_data = data
        config = data["config"]
        selected_subj = data.get("selected_subject")

        if hasattr(config, "events") and config.events:
            sorted_labels = sorted(config.events.values())
            self._class_names = {}
            for orig_label in sorted_labels:
                name = next(k for k, v in config.events.items() if v == orig_label)
                self._class_names[sorted_labels.index(orig_label)] = name
        else:
            self._class_names = {0: "Class 0", 1: "Class 1"}
        self._results_panel.set_class_names(self._class_names)

        subj_str = (
            f"Subject {selected_subj}" if selected_subj else f"{len(config.subjects)} subjects"
        )

        self._dataset_label.setText(f"<b>{config.name}</b><br>{subj_str}")
        self._update_button_states()

    def _on_model_trained(self, name: str, model):
        """Handle model trained."""
        self._decoder = model
        self._decoder_name = name

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
            continuous, _, labels = val_data
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

    # -- Data preparation --

    def _prepare_evaluation_data(self, modality: str = "eeg"):
        """Prepare data for evaluation.

        Returns:
            tuple: (continuous_data, event_times, event_labels, window_size, step_samples)
                   or None if validation fails
        """
        if not self._decoder:
            return None

        if not self._study_data and not self._validation_data:
            self._status_label.setText("No data available for evaluation")
            return None

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

        n_samples = continuous_data.shape[1]

        if config and config.sample_rate > 0:
            self._sampling_rate = config.sample_rate
        elif hasattr(self._decoder, "sample_rate") and self._decoder.sample_rate > 0:
            self._sampling_rate = self._decoder.sample_rate

        step_ms = self._step_ms_spin.value()
        step_samples = max(1, int((step_ms / 1000) * self._sampling_rate))

        self._total_samples = max(1, (n_samples - window_size) // step_samples)

        return continuous_data, event_times, event_labels, window_size, step_samples

    # -- Evaluation lifecycle --

    def _run_evaluation(self, real_time: bool = True):
        """Run evaluation with optional live visualization."""
        result = self._prepare_evaluation_data("eeg")
        if result is None:
            return

        continuous_data, event_times, event_labels, window_size, step_samples = result
        n_channels, n_samples = continuous_data.shape

        if self._validation_data is not None:
            duration = n_samples / self._sampling_rate
            self._status_label.setText(
                f"Using holdout data: {len(event_labels)} events ({format_duration(duration)})"
            )

        self._results_panel.clear()
        self._setup_ground_truth(event_times, event_labels, window_size, n_samples)

        self._eval_btn.setEnabled(False)
        self._quick_btn.setEnabled(False)

        self._start_simulation_worker(
            continuous_data, event_times, event_labels,
            window_size, step_samples, real_time,
        )

        self._stop_btn.show()
        gating_info = self._get_gating_status_text()
        step_ms = self._step_ms_spin.value()
        prefix = "Running" if real_time else "Quick"
        self._status_label.setText(
            f"{prefix}: {n_channels}ch × {window_size}samples, step={step_ms:.0f}ms{gating_info}"
        )

    def _setup_ground_truth(
        self, event_times, event_labels, window_size: int, n_samples: int,
    ):
        """Build label remapping and precompute sample-level ground truth."""
        unique_labels = sorted(set(event_labels))
        label_remap = {old: new for new, old in enumerate(unique_labels)}

        self._sample_labels = np.full(n_samples, -1, dtype=np.int64)
        for event_time, label in zip(event_times, event_labels, strict=False):
            if label >= 0:
                start = int(event_time)
                end = min(start + window_size, n_samples)
                self._sample_labels[start:end] = label_remap.get(int(label), int(label))

        truth_x = []
        truth_y = []
        for event_time, label in zip(event_times, event_labels, strict=False):
            if label >= 0:
                truth_x.append(event_time / self._sampling_rate)
                truth_y.append(label_remap.get(int(label), int(label)))
        self._results_panel.set_ground_truth(truth_x, truth_y)

    def _get_gating_config(self) -> dict | None:
        """Build gating config from UI controls, or None if disabled."""
        if not self._gating_toggle.isChecked():
            return None
        return {
            "confidence_threshold": self._conf_threshold_spin.value(),
            "use_confidence_gating": True,
            "dwell_time_sec": self._dwell_time_spin.value(),
            "use_dwell_time_gating": True,
            "background_class": 0,
        }

    def _start_simulation_worker(
        self, continuous_data, event_times, event_labels,
        window_size: int, step_samples: int, real_time: bool,
    ):
        """Create and start the simulation worker thread."""
        self._thread = QtCore.QThread()
        self._worker = SimulationWorker(
            decoder=self._decoder,
            X=continuous_data.T,
            event_times=event_times,
            event_labels=event_labels,
            epoch_length=window_size,
            step_size=step_samples,
            modality="eeg",
            sample_rate=self._sampling_rate,
            gating_config=self._get_gating_config(),
            real_time=real_time,
            class_names=self._class_names,
        )
        on_finished = partial(
            self._on_evaluation_complete, "Evaluation" if real_time else "Quick eval"
        )
        setup_worker_thread(
            self._worker,
            self._thread,
            on_finished=on_finished,
            on_error=lambda e: self._status_label.setText(f"Error: {e}"),
        )

        self._last_poll_idx = 0
        self._poll_timer.start(100)
        self._thread.start()

    def _on_evaluate(self):
        """Run sliding window evaluation with live visualization."""
        self._run_evaluation(real_time=True)

    def _on_quick_eval(self):
        """Run fast batch evaluation without live visualization."""
        self._run_evaluation(real_time=False)

    def _on_stop_evaluation(self):
        """Stop the running evaluation."""
        self._poll_timer.stop()
        if self._worker:
            self._worker.stop()
        self._stop_btn.hide()
        self._status_label.setText("Stopping evaluation...")

    # -- Polling and completion --

    def _poll_results(self):
        """Poll worker for new results and update GUI."""
        if not self._worker:
            return

        new_results = self._worker.get_results_since(self._last_poll_idx)
        if not new_results:
            return

        self._last_poll_idx += len(new_results)

        self._results_panel.add_predictions(new_results, self._sample_labels, self._sampling_rate)
        self._update_stats_display()

    def _on_evaluation_complete(self, prefix: str):
        """Handle evaluation completion."""
        self._poll_timer.stop()
        self._poll_results()

        self._results_panel.reset_plot_range()
        self._update_button_states()
        self._stop_btn.hide()

        count = self._results_panel.prediction_count
        self._status_label.setText(f"{prefix} complete ({count} predictions)")
        self._update_stats_display(final=True)

    def _update_stats_display(self, final: bool = False):
        """Build metrics dict from worker state and delegate to panel.

        Args:
            final: If True, get full metrics from worker's MetricsManager
        """
        count = self._results_panel.prediction_count
        if count == 0:
            return

        latest = self._results_panel.get_latest_metrics()
        metrics = {
            "accuracy": latest.get("accuracy", 0),
            "per_class_accuracy_named": latest.get("per_class_accuracy_named", {}),
        }

        if final and self._worker and self._worker.metrics_manager:
            async_metrics = self._worker.metrics_manager.get_current_metrics()
            metrics["n_trials"] = async_metrics.get("n_trials", 0)
            metrics["ttd"] = async_metrics.get("mean_ttd_ms", float("nan"))
            metrics["far"] = async_metrics.get("far_per_min", 0)
            metrics["per_class_accuracy_named"] = async_metrics.get("per_class_accuracy_named", {})

            if self._worker.metrics_manager.async_metrics:
                n_events = async_metrics.get("n_trials", 0)
                if n_events > 0 and self._total_samples > 0:
                    total_duration_sec = self._total_samples * self._step_ms_spin.value() / 1000
                    avg_selection_time = total_duration_sec / n_events
                    metrics["itr"] = self._worker.metrics_manager.async_metrics.calculate_itr(
                        num_classes=len(self._class_names) if self._class_names else 2,
                        mean_selection_time_sec=avg_selection_time,
                    )

        conf_values = self._results_panel.get_confidence_values()
        if conf_values:
            metrics["conf"] = np.mean(list(conf_values))

        total_duration = self._total_samples * self._step_ms_spin.value() / 1000
        self._results_panel.update_progress(
            count, self._total_samples, format_duration(total_duration)
        )
        self._results_panel.update_stats(metrics)
