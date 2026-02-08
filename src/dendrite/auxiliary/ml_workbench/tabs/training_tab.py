"""Training tab - Configure and run model training."""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6 import QtCore, QtWidgets

from dendrite.auxiliary.ml_workbench import TrainResult
from dendrite.auxiliary.ml_workbench.backend.decoder_saver import save_decoder
from dendrite.auxiliary.ml_workbench.backend.workers import (
    DataLoaderWorker,
    DirectTrainingWorker,
    OptunaTrainingWorker,
)
from dendrite.auxiliary.ml_workbench.dialogs import TrainingConfigDialog
from dendrite.auxiliary.ml_workbench.utils import setup_worker_thread
from dendrite.auxiliary.ml_workbench.widgets import (
    CollapsibleSection,
    create_form_layout,
    create_scrollable_panel,
    create_section,
)
from dendrite.auxiliary.ml_workbench.widgets.training_results_panel import TrainingResultsPanel
from dendrite.gui.styles.design_tokens import (
    ACCENT,
    FONT_SIZE,
    STATUS_WARNING_ALT,
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
        self._training_config = {"n_trials": 20}
        self._optuna_config: OptunaConfig | None = None
        self._search_space: dict[str, Any] | None = None
        self._optuna_results: dict[str, Any] | None = None
        self._final_training_config: DecoderConfig | None = None
        self._total_epochs: int = 0
        self._is_training = False
        self._setup_ui()

    def set_app_state(self, state):
        """Set the app state for cross-tab communication."""
        self._app_state = state
        if state:
            state.study_changed.connect(self._on_study_changed)

    # -- UI Setup --

    def _setup_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(
            LAYOUT["spacing_lg"], LAYOUT["spacing_lg"], LAYOUT["spacing_lg"], LAYOUT["spacing_lg"]
        )
        layout.setSpacing(LAYOUT["spacing_lg"])

        layout.addWidget(self._create_config_panel())

        self._results_panel = TrainingResultsPanel()
        layout.addWidget(self._results_panel, stretch=1)

    def _create_config_panel(self) -> QtWidgets.QWidget:
        panel, layout, footer_layout = create_scrollable_panel(560)

        layout.addWidget(self._create_dataset_section())
        layout.addWidget(self._create_decoder_section())
        layout.addWidget(self._create_training_section())
        layout.addWidget(self._create_save_section())
        layout.addStretch()
        self._create_footer(footer_layout)

        return panel

    def _create_dataset_section(self) -> QtWidgets.QWidget:
        section, section_layout = create_section("Dataset")
        self._dataset_label = QtWidgets.QLabel("No dataset loaded")
        self._dataset_label.setWordWrap(True)
        section_layout.addWidget(self._dataset_label)
        return section

    def _create_decoder_section(self) -> QtWidgets.QWidget:
        section, section_layout = create_section("Decoder")
        form = create_form_layout()
        self._model_combo = QtWidgets.QComboBox()
        self._model_combo.setStyleSheet(WidgetStyles.combobox())
        self._model_combo.addItems(get_available_decoders())
        form.addRow("Type:", self._model_combo)
        section_layout.addLayout(form)
        return section

    def _create_training_section(self) -> QtWidgets.QWidget:
        section, section_layout = create_section("Training")

        # Holdout slider
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
        section_layout.addLayout(holdout_container)

        # Optuna toggle
        optuna_row = QtWidgets.QHBoxLayout()
        optuna_row.setSpacing(LAYOUT["spacing_sm"])
        optuna_row.addWidget(QtWidgets.QLabel("Optuna Search"))
        self._optuna_check = TogglePillWidget(initial_state=False)
        self._optuna_check.toggled.connect(self._on_optuna_toggled)
        optuna_row.addWidget(self._optuna_check)

        self._config_btn = QtWidgets.QPushButton("\u2699")
        self._config_btn.setFixedSize(LAYOUT["icon_size"], LAYOUT["icon_size"])
        self._config_btn.setStyleSheet(WidgetStyles.button(variant="icon"))
        self._config_btn.setToolTip("Configure Optuna Search")
        self._config_btn.clicked.connect(self._on_configure_clicked)
        self._config_btn.setEnabled(False)
        optuna_row.addWidget(self._config_btn)

        optuna_row.addStretch()
        section_layout.addLayout(optuna_row)

        return section

    def _create_save_section(self) -> QtWidgets.QWidget:
        section = CollapsibleSection("Save Decoder", start_expanded=False)
        save_layout = section.content_layout()

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

        return section

    def _create_footer(self, footer_layout: QtWidgets.QVBoxLayout):
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

    # -- Config Handlers --

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
        selected_subj = data.get("selected_subject")

        subj_str = (
            f"Subject {selected_subj}" if selected_subj else f"{len(config.subjects)} subjects"
        )

        self._dataset_label.setText(f"<b>{config.name}</b><br>{subj_str}")
        self._train_btn.setEnabled(True)

        self._total_epochs = 0
        self._update_split_label()

    def _get_base_config(self) -> DecoderConfig:
        """Build base config for Optuna search."""
        return DecoderConfig(
            model_type=self._model_combo.currentText(),
            modality="eeg",
        )

    # -- Training Lifecycle --

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

        config = self._study_data["config"]
        loader = self._study_data["loader"]
        selected_subject = self._study_data.get("selected_subject")

        subjects = [selected_subject] if selected_subject is not None else config.subjects

        self._is_training = True
        self._train_btn.setText("Stop")
        self._train_btn.setStyleSheet(WidgetStyles.button(severity="error", padding="8px 12px"))
        self._progress_bar.show()
        self._status_label.setText("Loading data...")

        self._results_panel.clear()

        holdout_pct = self._holdout_slider.value()

        self._data_loader_thread = QtCore.QThread()
        self._data_loader_worker = DataLoaderWorker(
            loader=loader,
            subjects=subjects,
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
            self._results_panel.show_optuna_plot()

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
            self._worker.trial_completed.connect(self._results_panel.update_trial)
        else:
            self._status_label.setText(f"Loaded {len(X)} epochs. Starting training...")
            self._worker = DirectTrainingWorker(X, y, base_config)
            self._worker.epoch_completed.connect(self._results_panel.update_epoch)
            self._results_panel.reset_live_tracking()

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

    def _on_optuna_finished(self, results: dict[str, Any]):
        """Store Optuna results and final config for display and saving."""
        self._optuna_results = results

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

        self._save_btn.setEnabled(True)
        self._save_status_label.setText("")

        if not self._model_name_input.text():
            model_type = self._model_combo.currentText()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._model_name_input.setText(f"{model_type}_{timestamp}")

        class_names = None
        if hasattr(result.decoder, "config") and hasattr(result.decoder.config, "label_mapping"):
            label_mapping = result.decoder.config.label_mapping
            if label_mapping:
                class_names = {v: k for k, v in label_mapping.items()}
        if not class_names and result.confusion_matrix is not None:
            n_classes = result.confusion_matrix.shape[0]
            class_names = {i: str(i) for i in range(n_classes)}

        self._results_panel.update_from_result(result, class_names)
        self._results_panel.update_optuna_summary(self._optuna_results)

        self.training_finished.emit(result)

    # -- Save --

    def _on_save_clicked(self):
        """Handle save button click."""
        if not self._last_result:
            self._save_status_label.setText("No trained model to save")
            self._save_status_label.setStyleSheet(f"color: {STATUS_WARNING_ALT};")
            return

        model_name = self._model_name_input.text().strip()
        if not model_name:
            self._save_status_label.setText("Please enter a model name")
            self._save_status_label.setStyleSheet(f"color: {STATUS_WARNING_ALT};")
            return

        model_name = re.sub(r"[^\w\-]", "_", model_name)
        description = self._model_desc_input.text().strip()

        self._save_status_label.setText("Saving...")
        self._save_status_label.setStyleSheet("")
        QtWidgets.QApplication.processEvents()

        study_name = None
        if self._app_state and hasattr(self._app_state, "study_name"):
            study_name = self._app_state.study_name
        study_name = study_name or "default_study"

        training_dataset_name = None
        if self._study_data and "config" in self._study_data:
            training_dataset_name = self._study_data["config"].name

        saved_path = save_decoder(
            result=self._last_result,
            decoder_name=model_name,
            study_name=study_name,
            description=description,
            optuna_results=self._optuna_results,
            final_config=self._final_training_config,
            training_dataset_name=training_dataset_name,
        )

        if saved_path:
            self._save_status_label.setText(f"Saved: {Path(saved_path).name}")
            self._save_status_label.setStyleSheet(f"color: {ACCENT};")
        else:
            self._save_status_label.setText("Save failed - check logs")
            self._save_status_label.setStyleSheet(f"color: {STATUS_WARNING_ALT};")
