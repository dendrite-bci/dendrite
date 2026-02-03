"""Training tab - Configure and run model training."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from dendrite.auxiliary.ml_workbench import TrainResult
from dendrite.auxiliary.ml_workbench.backend.workers import (
    DataLoaderWorker,
    DirectTrainingWorker,
    OptunaTrainingWorker,
)
from dendrite.auxiliary.ml_workbench.dialogs import TrainingConfigDialog
from dendrite.auxiliary.ml_workbench.utils import create_styled_plot, setup_worker_thread
from dendrite.auxiliary.ml_workbench.widgets import (
    CollapsibleSection,
    create_form_layout,
    create_scrollable_panel,
    create_section,
)
from dendrite.data.storage.database import Database, DecoderRepository, StudyRepository
from dendrite.gui.styles.design_tokens import (
    ACCENT,
    FONT_SIZE,
    STATUS_SUCCESS,
    STATUS_WARNING_ALT,
    TEXT_MAIN,
    TEXT_MUTED,
)
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles
from dendrite.gui.widgets.common.toggle_pill import TogglePillWidget
from dendrite.ml.decoders import get_available_decoders
from dendrite.ml.decoders.decoder_schemas import DecoderConfig
from dendrite.ml.search import DEFAULT_SEARCH_SPACE, OptunaConfig
from dendrite.utils.logger_central import get_logger

logger = get_logger(__name__)


class TrainingTab(QtWidgets.QWidget):
    """Training tab for configuring and running model training."""

    training_finished = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._study_data: dict[str, Any] | None = None
        self._app_state = None
        self._worker = None
        self._thread = None
        self._data_loader_worker = None
        self._data_loader_thread = None
        self._last_result: TrainResult | None = None
        # Training config (just n_trials for search intensity)
        self._training_config = {"n_trials": 20}
        # Optuna config (always enabled now)
        self._optuna_config: OptunaConfig | None = None
        self._search_space: dict[str, Any] | None = None
        self._optuna_results: dict[str, Any] | None = None  # Stored after search
        self._final_training_config: DecoderConfig | None = None  # Best config from search
        self._total_epochs: int = 0  # Cached epoch count for split label
        self._is_training = False
        self._setup_ui()

    def set_app_state(self, state):
        """Set the app state for cross-tab communication."""
        self._app_state = state
        if state:
            state.study_changed.connect(self._on_study_changed)

    def _create_metric_card(
        self, title: str, primary: bool = False
    ) -> tuple[QtWidgets.QFrame, QtWidgets.QLabel]:
        """Create a styled metric card widget and return (card, value_label)."""
        card = QtWidgets.QFrame()
        card.setStyleSheet(WidgetStyles.metric_card())

        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(
            LAYOUT["spacing_lg"], LAYOUT["spacing_md"], LAYOUT["spacing_lg"], LAYOUT["spacing_md"]
        )
        layout.setSpacing(LAYOUT["spacing_xs"])

        title_lbl = QtWidgets.QLabel(title.upper())
        title_lbl.setStyleSheet(WidgetStyles.metric_card_title())
        layout.addWidget(title_lbl)

        value_lbl = QtWidgets.QLabel("--")
        size = FONT_SIZE["xl"] if primary else FONT_SIZE["lg"]
        weight = "bold" if primary else "600"
        value_lbl.setStyleSheet(WidgetStyles.metric_card_value(size=size, weight=weight))
        layout.addWidget(value_lbl)

        return card, value_lbl

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

        # Decoder section
        decoder_section, decoder_layout = create_section("Decoder")
        form = create_form_layout()
        self._model_combo = QtWidgets.QComboBox()
        self._model_combo.setStyleSheet(WidgetStyles.combobox())
        self._model_combo.addItems(get_available_decoders())
        form.addRow("Type:", self._model_combo)
        decoder_layout.addLayout(form)
        layout.addWidget(decoder_section)

        # Training section
        training_section, training_layout = create_section("Training")

        # Holdout slider with event count labels
        holdout_container = QtWidgets.QVBoxLayout()
        holdout_container.setContentsMargins(0, 0, 0, 0)

        slider_row = QtWidgets.QHBoxLayout()
        slider_row.addWidget(QtWidgets.QLabel("Eval Holdout:"))
        self._holdout_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._holdout_slider.setStyleSheet(WidgetStyles.slider())
        self._holdout_slider.setRange(0, 50)
        self._holdout_slider.setValue(30)
        self._holdout_slider.setToolTip("Hold out last X% of events for evaluation (0 = no split)")
        self._holdout_slider.valueChanged.connect(self._on_holdout_changed)
        slider_row.addWidget(self._holdout_slider, stretch=1)

        self._holdout_pct_label = QtWidgets.QLabel("30%")
        self._holdout_pct_label.setMinimumWidth(40)
        slider_row.addWidget(self._holdout_pct_label)
        holdout_container.addLayout(slider_row)

        self._split_label = QtWidgets.QLabel("Train: -- | Eval: --")
        self._split_label.setStyleSheet(
            WidgetStyles.inline_label(color=TEXT_MUTED, size=FONT_SIZE["sm"])
        )
        holdout_container.addWidget(self._split_label)

        training_layout.addLayout(holdout_container)

        # Optuna toggle with compact config button
        optuna_row = QtWidgets.QHBoxLayout()
        optuna_row.setSpacing(LAYOUT["spacing_sm"])
        optuna_row.addWidget(QtWidgets.QLabel("Optuna Search"))
        self._optuna_check = TogglePillWidget(initial_state=False)
        self._optuna_check.toggled.connect(self._on_optuna_toggled)
        optuna_row.addWidget(self._optuna_check)

        self._config_btn = QtWidgets.QPushButton("⚙")
        self._config_btn.setFixedSize(LAYOUT["icon_size"], LAYOUT["icon_size"])
        self._config_btn.setStyleSheet(WidgetStyles.button(variant="icon"))
        self._config_btn.setToolTip("Configure Optuna Search")
        self._config_btn.clicked.connect(self._on_configure_clicked)
        self._config_btn.setEnabled(False)
        optuna_row.addWidget(self._config_btn)

        optuna_row.addStretch()
        training_layout.addLayout(optuna_row)

        layout.addWidget(training_section)

        # Save Decoder section (collapsible, collapsed by default)
        self._save_section = CollapsibleSection("Save Decoder", start_expanded=False)
        save_layout = self._save_section.content_layout()

        save_form = create_form_layout()

        self._model_name_input = QtWidgets.QLineEdit()
        self._model_name_input.setStyleSheet(WidgetStyles.input())
        self._model_name_input.setPlaceholderText("e.g., decoder_subject1_v2")
        save_form.addRow("Name:", self._model_name_input)

        self._model_desc_input = QtWidgets.QLineEdit()
        self._model_desc_input.setStyleSheet(WidgetStyles.input())
        self._model_desc_input.setPlaceholderText("Optional description")
        save_form.addRow("Desc:", self._model_desc_input)

        save_layout.addLayout(save_form)

        self._save_btn = QtWidgets.QPushButton("Save to Database")
        self._save_btn.setStyleSheet(WidgetStyles.button(variant="text", padding="8px 12px"))
        self._save_btn.clicked.connect(self._on_save_clicked)
        self._save_btn.setEnabled(False)
        save_layout.addWidget(self._save_btn)

        self._save_status_label = QtWidgets.QLabel("")
        self._save_status_label.setWordWrap(True)
        save_layout.addWidget(self._save_status_label)

        layout.addWidget(self._save_section)
        layout.addStretch()

        # Footer: action buttons pinned below scroll area
        btn_row = QtWidgets.QHBoxLayout()

        self._train_btn = QtWidgets.QPushButton("Train")
        self._train_btn.setStyleSheet(WidgetStyles.button(padding="8px 12px"))
        self._train_btn.clicked.connect(self._on_train_btn_clicked)
        self._train_btn.setEnabled(False)
        btn_row.addWidget(self._train_btn)
        footer_layout.addLayout(btn_row)

        self._progress_bar = QtWidgets.QProgressBar()
        self._progress_bar.setRange(0, 0)
        self._progress_bar.hide()
        footer_layout.addWidget(self._progress_bar)

        self._status_label = QtWidgets.QLabel("Ready")
        self._status_label.setWordWrap(True)
        footer_layout.addWidget(self._status_label)

        return panel

    def _on_optuna_toggled(self, enabled: bool):
        """Handle Optuna toggle state change."""
        self._config_btn.setEnabled(enabled)

    def _on_holdout_changed(self, value: int):
        """Update labels when holdout slider changes."""
        self._holdout_pct_label.setText(f"{value}%")
        self._update_split_label()

    def _update_split_label(self):
        """Update the train/eval split label based on cached epoch count."""
        if self._total_epochs == 0:
            self._split_label.setText("Train: -- | Eval: --")
            return

        holdout_pct = self._holdout_slider.value()
        eval_count = int(self._total_epochs * holdout_pct / 100)
        train_count = self._total_epochs - eval_count

        self._split_label.setText(f"Train: {train_count} | Eval: {eval_count}")

    def _on_configure_clicked(self):
        """Open training configuration dialog."""
        dialog = TrainingConfigDialog(self._training_config, parent=self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._training_config = dialog.get_config()
            self._optuna_config = dialog.get_optuna_config()
            self._search_space = dialog.get_search_space()

    def _create_results_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(LAYOUT["spacing_md"])

        # Plot container (no border for minimal aesthetic)
        plots_container = QtWidgets.QFrame()
        plots_container.setObjectName("plotsContainer")
        plots_container.setStyleSheet(WidgetStyles.container(bg="main", radius=0))
        plots_container_layout = QtWidgets.QVBoxLayout(plots_container)
        plots_container_layout.setContentsMargins(
            LAYOUT["spacing_xl"], LAYOUT["spacing_xl"], LAYOUT["spacing_xl"], LAYOUT["spacing_xl"]
        )
        plots_container_layout.setSpacing(LAYOUT["spacing_md"])

        # Row 1: Training curves (Loss + Accuracy) - always 50/50 split
        curves_layout = QtWidgets.QHBoxLayout()
        curves_layout.setSpacing(LAYOUT["spacing_lg"])

        # Loss plot with legend
        self._loss_plot = create_styled_plot("Loss")
        self._loss_plot.addLegend(offset=(-10, 10))
        self._train_loss_curve = self._loss_plot.plot(
            pen=pg.mkPen(ACCENT, width=2.5), name="Train", antialias=True
        )
        self._val_loss_curve = self._loss_plot.plot(
            pen=pg.mkPen(STATUS_WARNING_ALT, width=2.5, style=QtCore.Qt.PenStyle.DashLine),
            name="Val",
            antialias=True,
        )
        curves_layout.addWidget(self._loss_plot)

        # Accuracy plot with legend
        self._acc_plot = create_styled_plot("Accuracy", y_range=(0, 1))
        self._acc_plot.addLegend(offset=(-10, 10))
        self._train_acc_curve = self._acc_plot.plot(
            pen=pg.mkPen(ACCENT, width=2.5), name="Train", antialias=True
        )
        self._val_acc_curve = self._acc_plot.plot(
            pen=pg.mkPen(STATUS_WARNING_ALT, width=2.5, style=QtCore.Qt.PenStyle.DashLine),
            name="Val",
            antialias=True,
        )
        # Chance level line (1/num_classes) - updated when training completes
        self._chance_line = pg.InfiniteLine(
            pos=0.5,
            angle=0,
            pen=pg.mkPen(color=TEXT_MUTED, width=1.5, style=QtCore.Qt.PenStyle.DotLine),
            label="chance",
            labelOpts={"position": 0.95, "color": TEXT_MUTED},
        )
        self._acc_plot.addItem(self._chance_line)
        curves_layout.addWidget(self._acc_plot)

        plots_container_layout.addLayout(curves_layout)

        # Row 2: Optuna search progress (full width, hidden by default)
        self._optuna_plot = create_styled_plot("Search Progress", height=160, y_range=(0, 1))
        self._optuna_plot.setLabel("bottom", "Trial")
        self._optuna_plot.setLabel("left", "Validation Accuracy")
        # Trial scatter points (regular trials)
        self._optuna_trials_scatter = pg.ScatterPlotItem(
            size=8, pen=pg.mkPen(ACCENT, width=1), brush=pg.mkBrush(ACCENT), symbol="o"
        )
        self._optuna_plot.addItem(self._optuna_trials_scatter)
        # Best trial scatter points (highlighted)
        self._optuna_best_scatter = pg.ScatterPlotItem(
            size=12,
            pen=pg.mkPen(STATUS_SUCCESS, width=2),
            brush=pg.mkBrush(STATUS_SUCCESS),
            symbol="star",
        )
        self._optuna_plot.addItem(self._optuna_best_scatter)
        # Best accuracy line
        self._optuna_best_line = pg.InfiniteLine(
            pos=0,
            angle=0,
            pen=pg.mkPen(color=STATUS_SUCCESS, width=1.5, style=QtCore.Qt.PenStyle.DashLine),
        )
        self._optuna_plot.addItem(self._optuna_best_line)
        # Text item for best accuracy
        self._optuna_best_text = pg.TextItem("", anchor=(1, 0), color=STATUS_SUCCESS)
        self._optuna_best_text.setFont(QtGui.QFont("sans-serif", 10, QtGui.QFont.Weight.Bold))
        self._optuna_plot.addItem(self._optuna_best_text)
        # Track trial data for plotting
        self._optuna_trial_data = []  # [(trial_num, accuracy, is_best), ...]
        self._optuna_plot.hide()  # Hidden by default, shown during Optuna search
        plots_container_layout.addWidget(self._optuna_plot)

        # Row 3: Analysis plots (Confusion matrix + Per-class metrics)
        analysis_layout = QtWidgets.QHBoxLayout()
        analysis_layout.setSpacing(LAYOUT["spacing_lg"])

        # Confusion matrix heatmap
        self._cm_plot = create_styled_plot("Confusion Matrix", height=180, show_grid=False)
        self._cm_plot.setAspectLocked(True)
        self._cm_image = pg.ImageItem()
        self._cm_plot.addItem(self._cm_image)
        # Text annotations for cell values (created dynamically)
        self._cm_labels = []
        analysis_layout.addWidget(self._cm_plot)

        # Per-class metrics bar chart
        self._class_plot = create_styled_plot("Per-Class Metrics", height=180, y_range=(0, 1))
        self._class_plot.addLegend(offset=(-10, 10))
        analysis_layout.addWidget(self._class_plot)

        plots_container_layout.addLayout(analysis_layout)

        # Row 4: Metrics summary cards
        metrics_layout = QtWidgets.QHBoxLayout()
        metrics_layout.setSpacing(LAYOUT["spacing_md"])

        acc_card, self._acc_label = self._create_metric_card("Accuracy", primary=True)
        metrics_layout.addWidget(acc_card)

        val_acc_card, self._val_acc_label = self._create_metric_card("Val Acc")
        metrics_layout.addWidget(val_acc_card)

        f1_card, self._f1_label = self._create_metric_card("F1 Score")
        metrics_layout.addWidget(f1_card)

        kappa_card, self._kappa_label = self._create_metric_card("Kappa")
        metrics_layout.addWidget(kappa_card)

        metrics_layout.addStretch()
        plots_container_layout.addLayout(metrics_layout)

        # Optuna summary (shown after Optuna search) - styled as card
        self._optuna_summary_frame = QtWidgets.QFrame()
        self._optuna_summary_frame.setStyleSheet(WidgetStyles.metric_card())
        summary_layout = QtWidgets.QVBoxLayout(self._optuna_summary_frame)
        summary_layout.setContentsMargins(
            LAYOUT["spacing_lg"], LAYOUT["spacing_md"], LAYOUT["spacing_lg"], LAYOUT["spacing_md"]
        )
        summary_layout.setSpacing(LAYOUT["spacing_xs"])

        optuna_title = QtWidgets.QLabel("SEARCH RESULTS")
        optuna_title.setStyleSheet(WidgetStyles.metric_card_title())
        summary_layout.addWidget(optuna_title)

        self._optuna_summary_label = QtWidgets.QLabel("")
        self._optuna_summary_label.setWordWrap(True)
        self._optuna_summary_label.setStyleSheet(WidgetStyles.inline_label(color=TEXT_MAIN))
        summary_layout.addWidget(self._optuna_summary_label)

        self._optuna_summary_frame.hide()
        plots_container_layout.addWidget(self._optuna_summary_frame)

        layout.addWidget(plots_container, stretch=1)

        return panel

    def _on_study_changed(self, data):
        """Handle study selection change from app state."""
        if data is None or not isinstance(data, dict) or "config" not in data:
            self._study_data = None
            self._dataset_label.setText("No study selected")
            self._train_btn.setEnabled(False)
            self._total_epochs = 0
            self._update_split_label()
            return

        self._study_data = data
        config = data["config"]
        capability = data.get("capability", "unknown")
        selected_subj = data.get("selected_subject")

        subj_str = (
            f"Subject {selected_subj}" if selected_subj else f"{len(config.subjects)} subjects"
        )

        self._dataset_label.setText(
            f"<b>{config.name}</b><br>{subj_str}<br>Capability: {capability}"
        )
        self._train_btn.setEnabled(True)

        # Reset split label (actual counts shown after data loads for training)
        self._total_epochs = 0
        self._update_split_label()

    def _get_base_config(self) -> DecoderConfig:
        """Build base config for Optuna search."""
        # Minimal config - search will optimize hyperparameters
        return DecoderConfig(
            model_type=self._model_combo.currentText(),
            modality="eeg",
        )

    def _on_train_btn_clicked(self):
        """Route train button click based on training state."""
        if self._is_training:
            self._on_stop_clicked()
        else:
            self._on_train_clicked()

    def _on_train_clicked(self):
        """Start training - first load data asynchronously, then train."""
        if not self._study_data:
            self._status_label.setText("No data loaded")
            return

        # Prepare for loading
        config = self._study_data["config"]
        loader = self._study_data["loader"]
        selected_subject = self._study_data.get("selected_subject")
        capability = self._study_data.get("capability", "epochs")

        subjects = [selected_subject] if selected_subject is not None else config.subjects

        self._is_training = True
        self._train_btn.setText("Stop")
        self._train_btn.setStyleSheet(WidgetStyles.button(severity="error", padding="8px 12px"))
        self._progress_bar.show()
        self._status_label.setText("Loading data...")

        # Clear metric labels and plots
        self._acc_label.setText("--")
        self._val_acc_label.setText("--")
        self._f1_label.setText("--")
        self._kappa_label.setText("--")
        self._train_loss_curve.setData([], [])
        self._val_loss_curve.setData([], [])
        self._train_acc_curve.setData([], [])
        self._val_acc_curve.setData([], [])
        self._update_confusion_matrix(None)
        self._class_plot.clear()
        self._update_cv_visualization(None, 0)
        self._optuna_summary_frame.hide()
        # Clear and hide Optuna progress plot
        self._optuna_plot.hide()
        self._optuna_trial_data = []
        self._optuna_trials_scatter.setData([], [])
        self._optuna_best_scatter.setData([], [])
        self._optuna_best_line.setValue(0)
        self._optuna_best_text.setText("")

        # Launch data loader in background thread
        holdout_pct = self._holdout_slider.value()

        self._data_loader_thread = QtCore.QThread()
        self._data_loader_worker = DataLoaderWorker(
            loader=loader,
            subjects=subjects,
            capability=capability,
            holdout_pct=holdout_pct,
        )
        setup_worker_thread(
            self._data_loader_worker,
            self._data_loader_thread,
            on_finished=self._on_data_loaded,
            on_error=self._on_training_error,
            on_progress=self._on_progress,
        )
        self._data_loader_worker.validation_ready.connect(self._on_validation_data_ready)
        self._data_loader_thread.start()

    def _on_data_loaded(self, X: np.ndarray | None, y: np.ndarray | None):
        """Handle data loading completion - start training (with or without Optuna)."""
        if X is None or y is None:
            self._is_training = False
            self._train_btn.setText("Train")
            self._train_btn.setStyleSheet(WidgetStyles.button(padding="8px 12px"))
            self._progress_bar.hide()
            self._status_label.setText("Failed to load data")
            return

        base_config = self._get_base_config()
        self._thread = QtCore.QThread()

        self._optuna_results = None
        self._final_training_config = None

        if self._optuna_check.isChecked():
            # Optuna search enabled - show the progress plot
            self._optuna_plot.show()

            # Initialize Optuna config if not set (user didn't open configure dialog)
            if self._optuna_config is None:
                self._optuna_config = OptunaConfig(
                    n_trials=self._training_config.get("n_trials", 20),
                    direction="maximize",
                )
                self._search_space = DEFAULT_SEARCH_SPACE.copy()

            n_trials = self._optuna_config.n_trials
            self._status_label.setText(
                f"Loaded {len(X)} epochs. Starting search ({n_trials} trials)..."
            )
            self._worker = OptunaTrainingWorker(
                X, y, base_config, self._optuna_config, self._search_space
            )
            self._worker.optuna_finished.connect(self._on_optuna_finished)
            self._worker.trial_completed.connect(self._on_trial_completed)
        else:
            # Direct training without Optuna
            self._status_label.setText(f"Loaded {len(X)} epochs. Starting training...")
            self._worker = DirectTrainingWorker(X, y, base_config)
            self._worker.epoch_completed.connect(self._on_epoch_completed)
            # Initialize live epoch tracking
            self._live_train_losses = []
            self._live_train_accs = []
            self._live_val_losses = []
            self._live_val_accs = []

        setup_worker_thread(
            self._worker,
            self._thread,
            on_finished=self._on_training_finished,
            on_error=self._on_training_error,
            on_progress=self._on_progress,
        )
        self._thread.start()

    def _on_stop_clicked(self):
        if self._data_loader_worker:
            self._data_loader_worker.stop()
        if self._worker:
            self._worker.stop()
        self._status_label.setText("Stopping...")

    def _on_progress(self, msg: str):
        self._status_label.setText(msg)

    def _on_validation_data_ready(self, val_data):
        """Forward validation data to AppState for Evaluation tab."""
        if self._app_state and val_data is not None:
            self._app_state.set_validation_data(val_data)

    def _on_trial_completed(self, trial_num: int, accuracy: float, is_best: bool):
        """Update Optuna progress plot with new trial result."""
        self._optuna_trial_data.append((trial_num, accuracy, is_best))

        # Extract all trials and best trials
        all_trials = [(t[0], t[1]) for t in self._optuna_trial_data]
        best_trials = [(t[0], t[1]) for t in self._optuna_trial_data if t[2]]

        # Update scatter plots
        if all_trials:
            x_all, y_all = zip(*all_trials, strict=False)
            self._optuna_trials_scatter.setData(x_all, y_all)

        if best_trials:
            x_best, y_best = zip(*best_trials, strict=False)
            self._optuna_best_scatter.setData(x_best, y_best)

            # Update best line and text
            current_best = max(y_best)
            self._optuna_best_line.setValue(current_best)
            self._optuna_best_text.setText(f"Best: {current_best * 100:.1f}%")
            self._optuna_best_text.setPos(trial_num, current_best + 0.05)

        # Update X range
        self._optuna_plot.setXRange(0, trial_num + 1, padding=0.05)

    def _on_epoch_completed(
        self,
        epoch: int,
        total: int,
        train_loss: float,
        train_acc: float,
        val_loss: float | None,
        val_acc: float | None,
    ):
        """Update loss/accuracy curves after each epoch (live updates)."""
        # Append to live tracking lists
        self._live_train_losses.append(train_loss)
        self._live_train_accs.append(train_acc)

        epochs = list(range(1, len(self._live_train_losses) + 1))

        # Update training curves
        self._train_loss_curve.setData(epochs, self._live_train_losses)
        self._train_acc_curve.setData(epochs, self._live_train_accs)

        # Update validation curves if available
        if val_loss is not None:
            self._live_val_losses.append(val_loss)
            self._val_loss_curve.setData(epochs, self._live_val_losses)
        if val_acc is not None:
            self._live_val_accs.append(val_acc)
            self._val_acc_curve.setData(epochs, self._live_val_accs)

    def _on_optuna_finished(self, results: dict[str, Any]):
        """Store Optuna results and final config for display and saving."""
        self._optuna_results = results

        # Build final DecoderConfig from best params
        if results and "best_params" in results:
            best_params = results["best_params"]
            base_config = self._get_base_config()
            config_dict = base_config.model_dump()
            config_dict.update(best_params)
            self._final_training_config = DecoderConfig(**config_dict)

    def _on_training_error(self, error: str):
        self._status_label.setText(f"Error: {error}")

    def _on_training_finished(self, result: TrainResult | None):
        self._is_training = False
        self._train_btn.setText("Train")
        self._train_btn.setStyleSheet(WidgetStyles.button(padding="8px 12px"))
        self._progress_bar.hide()

        if result is None:
            self._status_label.setText("Training stopped or failed")
            self._save_btn.setEnabled(False)
            return

        self._last_result = result
        self._status_label.setText("Training complete! You can now save the model.")

        # Enable save button
        self._save_btn.setEnabled(True)
        self._save_status_label.setText("")

        # Auto-populate model name if empty
        if not self._model_name_input.text():
            model_type = self._model_combo.currentText()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._model_name_input.setText(f"{model_type}_{timestamp}")

        # Update training curves
        history = result.train_history
        if history:
            epochs = list(range(1, len(history.get("loss", [])) + 1))
            if epochs:
                self._train_loss_curve.setData(epochs, history.get("loss", []))
                self._val_loss_curve.setData(epochs, history.get("val_loss", []))
                self._train_acc_curve.setData(epochs, history.get("accuracy", []))
                self._val_acc_curve.setData(epochs, history.get("val_accuracy", []))

        # Update chance level line based on number of classes
        if result.confusion_matrix is not None and result.confusion_matrix.size > 0:
            n_classes = result.confusion_matrix.shape[0]
            chance_level = 1.0 / n_classes
            self._chance_line.setValue(chance_level)

        # Extract class names from decoder config for plot labels
        self._class_names = None
        if hasattr(result.decoder, "config") and hasattr(result.decoder.config, "label_mapping"):
            label_mapping = result.decoder.config.label_mapping
            if label_mapping:
                # label_mapping is {class_name: class_idx}, invert to {idx: name}
                self._class_names = {v: k for k, v in label_mapping.items()}
        # Fallback to numeric labels
        if not self._class_names and result.confusion_matrix is not None:
            n_classes = result.confusion_matrix.shape[0]
            self._class_names = {i: str(i) for i in range(n_classes)}

        # Update confusion matrix heatmap
        self._update_confusion_matrix(result.confusion_matrix)

        # Update per-class metrics
        self._update_class_metrics(result.confusion_matrix)

        # Update CV fold visualization if available
        self._update_cv_visualization(
            result.cv_results, len(history.get("accuracy", [])) if history else 0
        )

        # Update metric labels
        self._update_metric_labels(result)

        # Update optuna summary if available
        self._update_optuna_summary()

        self.training_finished.emit(result)

    def _update_optuna_summary(self):
        """Update optuna summary display."""
        if not self._optuna_results:
            self._optuna_summary_frame.hide()
            return

        r = self._optuna_results
        best_params = r.get("best_params", {})
        best_value = r.get("best_value", 0)
        n_completed = r.get("n_completed", 0)
        n_pruned = r.get("n_pruned", 0)
        n_failed = r.get("n_failed", 0)

        # Format best params
        param_lines = []
        for k, v in best_params.items():
            if isinstance(v, float):
                param_lines.append(f"  {k}: {v:.4g}")
            else:
                param_lines.append(f"  {k}: {v}")
        params_str = "\n".join(param_lines) if param_lines else "  (none)"

        summary = (
            f"Trials: {n_completed} completed, {n_pruned} pruned, {n_failed} failed\n"
            f"Best accuracy: {best_value * 100:.1f}%\n\n"
            f"Best Parameters:\n{params_str}"
        )

        self._optuna_summary_label.setText(summary)
        self._optuna_summary_frame.show()

    def _update_metric_labels(self, result: TrainResult) -> None:
        """Update the metric summary labels with training results."""
        from sklearn.metrics import cohen_kappa_score, f1_score

        self._acc_label.setText(f"{result.accuracy * 100:.1f}%")
        self._val_acc_label.setText(f"{result.val_accuracy * 100:.1f}%")

        # Compute F1 and Kappa from confusion matrix
        cm = result.confusion_matrix
        if cm is not None and cm.size > 0:
            y_true, y_pred = [], []
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    y_true.extend([i] * cm[i, j])
                    y_pred.extend([j] * cm[i, j])

            if y_true:
                f1 = f1_score(y_true, y_pred, average="weighted")
                kappa = cohen_kappa_score(y_true, y_pred)
                self._f1_label.setText(f"{f1:.3f}")
                self._kappa_label.setText(f"{kappa:.3f}")

    def _create_confusion_matrix_colormap(self) -> np.ndarray:
        """Create colormap lookup table for confusion matrix.

        Returns:
            256x4 RGBA lookup table array
        """
        # Smooth colormap: dark background -> accent blue -> success green
        colors = np.array(
            [
                [21, 21, 21, 255],  # Low: dark (BG_MAIN)
                [0, 80, 160, 255],  # Medium-low: accent blue
                [0, 123, 255, 255],  # Medium: bright accent
                [0, 180, 140, 255],  # Medium-high: teal
                [0, 212, 170, 255],  # High: success green
            ],
            dtype=np.uint8,
        )

        lut = np.zeros((256, 4), dtype=np.uint8)
        n_colors = len(colors)
        for i in range(256):
            t = i / 255.0 * (n_colors - 1)
            idx = min(int(t), n_colors - 2)
            frac = t - idx
            lut[i] = colors[idx] * (1 - frac) + colors[idx + 1] * frac
        return lut

    def _update_confusion_matrix(self, cm: np.ndarray | None) -> None:
        """Update confusion matrix heatmap visualization."""
        # Clear previous labels
        for label in self._cm_labels:
            self._cm_plot.removeItem(label)
        self._cm_labels.clear()

        if cm is None or cm.size == 0:
            self._cm_image.clear()
            return

        n_classes = cm.shape[0]

        # Normalize for color intensity (0-1 range per row)
        cm_normalized = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        cm_normalized = cm_normalized / row_sums

        # Set image and colormap
        self._cm_image.setImage(cm_normalized.T)  # Transpose for pyqtgraph (x, y)
        self._cm_image.setLookupTable(self._create_confusion_matrix_colormap())

        # Set position and scale
        self._cm_image.setRect(0, 0, n_classes, n_classes)
        self._cm_plot.setXRange(-0.5, n_classes - 0.5, padding=0.1)
        self._cm_plot.setYRange(-0.5, n_classes - 0.5, padding=0.1)

        # Set axis labels using class names if available
        class_names = getattr(self, "_class_names", None) or {i: str(i) for i in range(n_classes)}
        ax = self._cm_plot.getAxis("bottom")
        ax.setTicks([[(i, class_names.get(i, str(i))) for i in range(n_classes)]])
        ax.setLabel("Predicted")
        ay = self._cm_plot.getAxis("left")
        ay.setTicks([[(i, class_names.get(i, str(i))) for i in range(n_classes)]])
        ay.setLabel("Actual")

        # Add text annotations with count and percentage values
        for i in range(n_classes):
            for j in range(n_classes):
                value = cm[i, j]
                normalized = cm_normalized[i, j]
                pct = normalized * 100
                # Dynamic text color: white on dark, dark on bright
                text_color = TEXT_MAIN if normalized < 0.6 else "#1a1a1a"
                # Show count and percentage, e.g., "42\n(71%)"
                label = f"{value}\n({pct:.0f}%)"
                text = pg.TextItem(label, anchor=(0.5, 0.5), color=text_color)
                text.setFont(QtGui.QFont("sans-serif", 9, QtGui.QFont.Weight.Bold))
                text.setPos(j + 0.5, i + 0.5)
                self._cm_plot.addItem(text)
                self._cm_labels.append(text)

    def _update_class_metrics(self, cm: np.ndarray | None) -> None:
        """Update per-class precision/recall/F1 bar chart."""
        self._class_plot.clear()
        self._class_plot.addLegend(offset=(-10, 10))

        if cm is None or cm.size == 0:
            return

        n_classes = cm.shape[0]

        # Calculate per-class metrics
        precision = np.zeros(n_classes)
        recall = np.zeros(n_classes)
        f1 = np.zeros(n_classes)

        for i in range(n_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp  # Column sum minus diagonal
            fn = cm[i, :].sum() - tp  # Row sum minus diagonal

            precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1[i] = (
                2 * precision[i] * recall[i] / (precision[i] + recall[i])
                if (precision[i] + recall[i]) > 0
                else 0
            )

        # Bar positions
        bar_width = 0.25
        x = np.arange(n_classes)

        # Create bar graph items
        precision_bars = pg.BarGraphItem(
            x=x - bar_width,
            height=precision,
            width=bar_width,
            brush=pg.mkBrush(STATUS_SUCCESS),
            name="Precision",
        )
        recall_bars = pg.BarGraphItem(
            x=x, height=recall, width=bar_width, brush=pg.mkBrush(ACCENT), name="Recall"
        )
        f1_bars = pg.BarGraphItem(
            x=x + bar_width,
            height=f1,
            width=bar_width,
            brush=pg.mkBrush(STATUS_WARNING_ALT),
            name="F1",
        )

        self._class_plot.addItem(precision_bars)
        self._class_plot.addItem(recall_bars)
        self._class_plot.addItem(f1_bars)

        # Set axis labels using class names if available
        class_names = getattr(self, "_class_names", None) or {
            i: f"Class {i}" for i in range(n_classes)
        }
        ax = self._class_plot.getAxis("bottom")
        ax.setTicks([[(i, class_names.get(i, f"Class {i}")) for i in range(n_classes)]])
        self._class_plot.setXRange(-0.5, n_classes - 0.5, padding=0.1)

    def _update_cv_visualization(self, cv_results: dict[str, Any] | None, n_epochs: int) -> None:
        """Update CV fold visualization on accuracy plot."""
        # Remove previous CV items if any
        if hasattr(self, "_cv_items"):
            for item in self._cv_items:
                self._acc_plot.removeItem(item)
        self._cv_items = []

        if cv_results is None:
            return

        fold_scores = cv_results.get("fold_scores", [])
        mean_acc = cv_results.get("mean_accuracy", 0)
        std_acc = cv_results.get("std_accuracy", 0)

        if not fold_scores:
            return

        # Position fold scores at the right side of the plot (after training epochs)
        x_pos = n_epochs + 1 if n_epochs > 0 else 1
        n_folds = len(fold_scores)

        # Add scatter points for each fold
        fold_scatter = pg.ScatterPlotItem(
            x=[x_pos] * n_folds,
            y=fold_scores,
            size=10,
            pen=pg.mkPen("w", width=1),
            brush=pg.mkBrush(STATUS_SUCCESS),
            symbol="o",
            name="CV Folds",
        )
        self._acc_plot.addItem(fold_scatter)
        self._cv_items.append(fold_scatter)

        # Add mean line
        mean_line = pg.InfiniteLine(
            pos=mean_acc,
            angle=0,
            pen=pg.mkPen(color=STATUS_SUCCESS, width=2, style=QtCore.Qt.PenStyle.DashLine),
            label=f"CV mean: {mean_acc * 100:.1f}%",
            labelOpts={"position": 0.05, "color": STATUS_SUCCESS},
        )
        self._acc_plot.addItem(mean_line)
        self._cv_items.append(mean_line)

        # Add error band (mean +/- std) as a fill region
        if std_acc > 0:
            x_vals = [0, n_epochs + 2] if n_epochs > 0 else [0, 2]
            upper = pg.PlotDataItem(x_vals, [mean_acc + std_acc] * 2)
            lower = pg.PlotDataItem(x_vals, [mean_acc - std_acc] * 2)
            fill = pg.FillBetweenItem(upper, lower, brush=pg.mkBrush(80, 180, 120, 40))
            self._acc_plot.addItem(fill)
            self._cv_items.append(fill)

    def _on_save_clicked(self):
        """Handle save button click."""
        if not self._last_result:
            self._save_status_label.setText("No trained model to save")
            self._save_status_label.setStyleSheet(f"color: {STATUS_WARNING_ALT};")
            return

        # Validate model name
        model_name = self._model_name_input.text().strip()
        if not model_name:
            self._save_status_label.setText("Please enter a model name")
            self._save_status_label.setStyleSheet(f"color: {STATUS_WARNING_ALT};")
            return

        # Sanitize name (remove special characters)
        model_name = re.sub(r"[^\w\-]", "_", model_name)

        description = self._model_desc_input.text().strip()

        self._save_status_label.setText("Saving...")
        self._save_status_label.setStyleSheet("")
        QtWidgets.QApplication.processEvents()

        saved_path = self._save_decoder_to_db(self._last_result, model_name, description)

        if saved_path:
            self._save_status_label.setText(f"Saved: {Path(saved_path).name}")
            self._save_status_label.setStyleSheet(f"color: {ACCENT};")
        else:
            self._save_status_label.setText("Save failed - check logs")
            self._save_status_label.setStyleSheet(f"color: {STATUS_WARNING_ALT};")

    def _extract_decoder_metadata(self, result: TrainResult) -> dict[str, Any]:
        """Extract metadata from training result for database storage."""
        metadata = {
            "cv_mean": None,
            "cv_std": None,
            "cv_folds": None,
            "channel_names_json": None,
            "class_labels_json": None,
            "modality": None,
            "training_dataset_name": None,
        }

        # CV results
        if result.cv_results:
            metadata["cv_mean"] = result.cv_results.get("mean_accuracy")
            metadata["cv_std"] = result.cv_results.get("std_accuracy")
            metadata["cv_folds"] = result.cv_results.get("n_folds")

        # Decoder config metadata
        decoder_config = result.decoder.config
        if decoder_config.channel_labels:
            metadata["channel_names_json"] = json.dumps(decoder_config.channel_labels)
        if decoder_config.label_mapping:
            metadata["class_labels_json"] = json.dumps(decoder_config.label_mapping)
        if decoder_config.modality:
            metadata["modality"] = decoder_config.modality

        # Training dataset
        if self._study_data and "config" in self._study_data:
            metadata["training_dataset_name"] = self._study_data["config"].name

        return metadata

    def _prepare_search_result_json(self) -> str | None:
        """Prepare search result JSON if Optuna search was done."""
        if not self._optuna_results:
            return None
        return json.dumps(
            {
                "n_trials": self._optuna_results.get("n_completed", 0),
                "best_value": self._optuna_results.get("best_value"),
                "best_params": self._optuna_results.get("best_params", {}),
                "n_pruned": self._optuna_results.get("n_pruned", 0),
                "n_failed": self._optuna_results.get("n_failed", 0),
            }
        )

    def _save_decoder_to_db(
        self, result: TrainResult, decoder_name: str, description: str = ""
    ) -> str | None:
        """Save trained decoder to disk and register in database.

        Args:
            result: Training result containing the decoder
            decoder_name: User-provided name for the decoder
            description: Optional description

        Returns:
            Path to saved decoder, or None if save failed
        """
        try:
            # Get study name first (needed for both saving and registration)
            study_name = None
            if self._app_state and hasattr(self._app_state, "study_name"):
                study_name = self._app_state.study_name
            study_name = study_name or "default_study"

            # Save decoder to disk under study's decoders directory
            saved_path = result.decoder.save(decoder_name, study_name=study_name)

            # Get config and metadata
            config = self._final_training_config or self._get_base_config()
            metadata = self._extract_decoder_metadata(result)

            db = Database()
            study_repo = StudyRepository(db)
            decoder_repo = DecoderRepository(db)

            study = study_repo.get_or_create(study_name)
            study_id = study["study_id"]

            # Register in database
            decoder_id = decoder_repo.add_decoder(
                study_id=study_id,
                decoder_name=decoder_name,
                decoder_path=saved_path,
                model_type=config.model_type,
                training_accuracy=result.accuracy,
                validation_accuracy=result.val_accuracy,
                cv_mean_accuracy=metadata["cv_mean"],
                cv_std_accuracy=metadata["cv_std"],
                cv_folds=metadata["cv_folds"],
                source="offline_trainer",
                description=description,
                channel_names=metadata["channel_names_json"],
                class_labels=metadata["class_labels_json"],
                training_dataset_name=metadata["training_dataset_name"],
                modality=metadata["modality"],
                training_config=json.dumps(config.model_dump(exclude_none=True)),
                search_result=self._prepare_search_result_json(),
            )

            if decoder_id:
                logger.info(f"Registered decoder in database with ID: {decoder_id}")
            else:
                logger.warning("Failed to register decoder in database (may already exist)")

            return saved_path

        except Exception as e:
            logger.error(f"Failed to save decoder: {e}")
            return None
