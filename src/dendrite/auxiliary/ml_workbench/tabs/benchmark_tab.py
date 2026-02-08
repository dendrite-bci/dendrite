"""Benchmark tab for model comparison and evaluation."""

from typing import Any

from PyQt6 import QtCore, QtWidgets

from dendrite.auxiliary.ml_workbench.backend import (
    BenchmarkRow,
    BenchmarkWorker,
    export_benchmark_results,
)
from dendrite.auxiliary.ml_workbench.dialogs import TrainingConfigDialog
from dendrite.auxiliary.ml_workbench.utils import is_model_compatible
from dendrite.auxiliary.ml_workbench.widgets import (
    CollapsibleSection,
    create_form_layout,
    create_scrollable_panel,
    create_section,
)
from dendrite.gui.styles.design_tokens import (
    STATUS_ERROR,
    STATUS_SUCCESS,
    STATUS_WARN,
    TEXT_DISABLED,
)
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles
from dendrite.gui.widgets.common.toggle_pill import TogglePillWidget
from dendrite.ml.decoders import get_available_decoders
from dendrite.ml.search import OptunaConfig
from dendrite.utils.logger_central import get_logger

logger = get_logger(__name__)


class BenchmarkTab(QtWidgets.QWidget):
    """Tab for running benchmarks on loaded studies."""

    def __init__(self, app_state=None, parent=None):
        super().__init__(parent)
        self._app_state = app_state
        self._study_data: dict[str, Any] | None = None
        self._running = False
        self._training_config = {
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
        }
        self._optuna_config = OptunaConfig(n_trials=10)
        self._n_folds = 5
        self._setup_ui()

        if self._app_state:
            self._app_state.study_changed.connect(self._on_study_changed)

    def set_app_state(self, app_state):
        self._app_state = app_state
        self._app_state.study_changed.connect(self._on_study_changed)
        self._update_data_status()

    # -- UI Setup --

    def _setup_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(
            LAYOUT["spacing_lg"], LAYOUT["spacing_lg"], LAYOUT["spacing_lg"], LAYOUT["spacing_lg"]
        )
        layout.setSpacing(LAYOUT["spacing_lg"])

        left_panel, left_layout, left_footer = create_scrollable_panel(560)
        left_layout.addWidget(self._create_data_source_section())
        left_layout.addWidget(self._create_decoders_section())
        left_layout.addWidget(self._create_eval_strategy_section())
        left_layout.addWidget(self._create_training_config_section())
        left_layout.addStretch()
        self._create_footer(left_footer)

        layout.addWidget(left_panel)
        layout.addWidget(self._create_results_panel(), 1)
        self._update_data_status()
        self._update_config_summary()

    def _create_data_source_section(self) -> QtWidgets.QWidget:
        section, section_layout = create_section("Data Source")
        self._data_label = QtWidgets.QLabel("No study loaded")
        self._data_label.setStyleSheet(WidgetStyles.inline_label(color=STATUS_WARN))
        self._data_label.setWordWrap(True)
        section_layout.addWidget(self._data_label)
        return section

    def _create_decoders_section(self) -> QtWidgets.QWidget:
        section, section_layout = create_section("Decoders")

        self._model_checkboxes: dict[str, QtWidgets.QCheckBox] = {}

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(200)
        scroll.setStyleSheet(WidgetStyles.scrollarea)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll_widget = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(
            LAYOUT["spacing_xs"], LAYOUT["spacing_xs"], LAYOUT["spacing_xs"], LAYOUT["spacing_xs"]
        )
        scroll_layout.setSpacing(LAYOUT["spacing_xs"])

        for decoder_name in get_available_decoders():
            cb = QtWidgets.QCheckBox(decoder_name)
            if decoder_name in ["EEGNet", "CSP+LDA"]:
                cb.setChecked(True)
            self._model_checkboxes[decoder_name] = cb
            scroll_layout.addWidget(cb)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        section_layout.addWidget(scroll)

        btn_row = QtWidgets.QHBoxLayout()
        select_all = QtWidgets.QPushButton("All")
        select_all.setStyleSheet(WidgetStyles.button(transparent=True))
        select_all.clicked.connect(lambda: self._select_models(True))
        select_none = QtWidgets.QPushButton("None")
        select_none.setStyleSheet(WidgetStyles.button(transparent=True))
        select_none.clicked.connect(lambda: self._select_models(False))
        btn_row.addWidget(select_all)
        btn_row.addWidget(select_none)
        btn_row.addStretch()
        section_layout.addLayout(btn_row)

        return section

    def _create_eval_strategy_section(self) -> QtWidgets.QWidget:
        section, section_layout = create_section("Evaluation Strategy")
        self._eval_group = section

        form = create_form_layout()
        form.setSpacing(LAYOUT["spacing_md"])

        self._eval_type_combo = QtWidgets.QComboBox()
        self._eval_type_combo.setStyleSheet(WidgetStyles.combobox())
        self._eval_type_combo.addItem("Within Session (k-fold CV)", "within_session")
        self._eval_type_combo.addItem("Cross Session", "cross_session")
        self._eval_type_combo.addItem("Cross Subject (LOSO)", "cross_subject")
        form.addRow("Type:", self._eval_type_combo)

        self._eval_info_label = QtWidgets.QLabel("")
        self._eval_info_label.setStyleSheet(WidgetStyles.inline_label(color=TEXT_DISABLED, size=11))
        self._eval_info_label.setWordWrap(True)
        form.addRow("", self._eval_info_label)

        section_layout.addLayout(form)
        return section

    def _create_training_config_section(self) -> QtWidgets.QWidget:
        section = CollapsibleSection("Training Configuration", start_expanded=False)
        config_layout = section.content_layout()

        self._config_summary = QtWidgets.QLabel("")
        self._config_summary.setStyleSheet(WidgetStyles.inline_label(color=TEXT_DISABLED))
        self._config_summary.setWordWrap(True)
        config_layout.addWidget(self._config_summary)

        optuna_row = QtWidgets.QHBoxLayout()
        optuna_row.addWidget(QtWidgets.QLabel("Optuna Search"))
        self._optuna_check = TogglePillWidget(initial_state=True)
        self._optuna_check.toggled.connect(self._on_optuna_toggled)
        optuna_row.addWidget(self._optuna_check)
        optuna_row.addStretch()
        config_layout.addLayout(optuna_row)

        self._optuna_label = QtWidgets.QLabel("")
        self._optuna_label.setStyleSheet(WidgetStyles.inline_label(color=STATUS_WARN))
        config_layout.addWidget(self._optuna_label)

        self._config_btn = QtWidgets.QPushButton("Configure Search...")
        self._config_btn.setStyleSheet(WidgetStyles.button(transparent=True))
        self._config_btn.clicked.connect(self._on_configure_clicked)
        config_layout.addWidget(self._config_btn)

        return section

    def _create_footer(self, footer_layout: QtWidgets.QVBoxLayout):
        self._run_btn = QtWidgets.QPushButton("Run Benchmark")
        self._run_btn.setMinimumHeight(40)
        self._run_btn.setStyleSheet(WidgetStyles.button(padding="8px 12px"))
        self._run_btn.clicked.connect(self._on_run)
        self._run_btn.setEnabled(False)
        footer_layout.addWidget(self._run_btn)

        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setMaximumHeight(16)
        self._progress.setStyleSheet(WidgetStyles.progress_bar())
        self._progress.setVisible(False)
        footer_layout.addWidget(self._progress)

        self._progress_label = QtWidgets.QLabel("Ready")
        self._progress_label.setWordWrap(True)
        self._progress_label.setStyleSheet(WidgetStyles.inline_label(color=TEXT_DISABLED))
        footer_layout.addWidget(self._progress_label)

    def _create_results_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(LAYOUT["spacing_md"])

        results_group = QtWidgets.QGroupBox("Results")
        results_layout = QtWidgets.QVBoxLayout(results_group)

        self._results_table = QtWidgets.QTableWidget()
        self._results_table.setColumnCount(4)
        self._results_table.setHorizontalHeaderLabels(["Model", "Accuracy", "Time", "Subjects"])
        self._results_table.horizontalHeader().setStretchLastSection(True)
        self._results_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self._results_table.setAlternatingRowColors(True)
        self._results_table.setStyleSheet(WidgetStyles.benchmark_table())
        results_layout.addWidget(self._results_table)

        export_row = QtWidgets.QHBoxLayout()
        export_row.addStretch()
        self._export_btn = QtWidgets.QPushButton("Export Results...")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export)
        export_row.addWidget(self._export_btn)
        results_layout.addLayout(export_row)

        layout.addWidget(results_group)
        return panel

    # -- Config & State --

    def _select_models(self, select: bool):
        for cb in self._model_checkboxes.values():
            cb.setChecked(select)

    def _on_configure_clicked(self):
        """Open training configuration dialog."""
        dialog = TrainingConfigDialog(self._training_config, parent=self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._training_config = dialog.get_config()
            self._optuna_config = dialog.get_optuna_config()
            self._update_config_summary()

    def _on_optuna_toggled(self, enabled: bool):
        """Handle Optuna toggle state change."""
        self._config_btn.setEnabled(enabled)
        self._update_config_summary()

    def _update_config_summary(self):
        """Update config summary label."""
        c = self._training_config
        self._config_summary.setText(
            f"Epochs: {c.get('epochs', 100)}, "
            f"Batch: {c.get('batch_size', 32)}, "
            f"LR: {c.get('learning_rate', 0.001):.4f}"
        )
        if self._optuna_check.isChecked():
            n_trials = self._optuna_config.n_trials if self._optuna_config else 10
            self._optuna_label.setText(f"Search: {n_trials} trials")
            self._optuna_label.setStyleSheet(WidgetStyles.inline_label(color=STATUS_WARN))
        else:
            self._optuna_label.setText("Standard training (no HPO)")
            self._optuna_label.setStyleSheet(WidgetStyles.inline_label(color=TEXT_DISABLED))

    def _on_study_changed(self, data):
        if data is None or not isinstance(data, dict) or "config" not in data:
            self._study_data = None
        else:
            self._study_data = data
        self._update_data_status()

    def _update_data_status(self):
        if self._study_data:
            config = self._study_data["config"]
            is_moabb = config.source_type == "moabb"
            selected_subject = self._study_data.get("selected_subject")
            n_subjects = 1 if selected_subject is not None else len(config.subjects)

            source_str = "[MOABB]" if is_moabb else ""
            self._data_label.setText(
                f"<b>{config.name}</b> {source_str}<br>Subjects: {n_subjects}"
            )
            self._data_label.setStyleSheet(WidgetStyles.inline_label(color=STATUS_SUCCESS))
            self._run_btn.setEnabled(True)
            self._on_modality_changed("EEG")

            self._eval_group.setVisible(True)
            self._update_eval_options(n_subjects, is_moabb)
        else:
            self._data_label.setText("No study loaded.\nLoad data in the Data tab first.")
            self._data_label.setStyleSheet(WidgetStyles.inline_label(color=STATUS_WARN))
            self._run_btn.setEnabled(False)
            self._eval_group.setVisible(False)

    def _update_eval_options(self, n_subjects: int, is_moabb: bool = True):
        """Update eval type options based on dataset capabilities."""
        cross_subject_ok = n_subjects >= 2
        cross_session_ok = is_moabb

        model = self._eval_type_combo.model()
        model.item(1).setEnabled(cross_session_ok)
        model.item(2).setEnabled(cross_subject_ok)

        info_parts = []
        if not cross_session_ok:
            info_parts.append("Cross Session: MOABB only")
        if not cross_subject_ok:
            info_parts.append("Cross Subject: needs 2+ subjects")
        self._eval_info_label.setText(" | ".join(info_parts))

        current_idx = self._eval_type_combo.currentIndex()
        if (current_idx == 1 and not cross_session_ok) or (
            current_idx == 2 and not cross_subject_ok
        ):
            self._eval_type_combo.setCurrentIndex(0)

    def _on_modality_changed(self, modality: str):
        for model_name, checkbox in self._model_checkboxes.items():
            compatible = is_model_compatible(model_name, modality)
            checkbox.setEnabled(compatible)
            if not compatible:
                checkbox.setChecked(False)

    def _get_selected_models(self) -> list[str]:
        return [name for name, cb in self._model_checkboxes.items() if cb.isChecked()]

    # -- Benchmark Run --

    def _on_run(self):
        if not self._study_data:
            QtWidgets.QMessageBox.warning(
                self, "No Data", "Please load a study in the Data tab first."
            )
            return

        selected_models = self._get_selected_models()
        if not selected_models:
            QtWidgets.QMessageBox.warning(self, "No Models", "Please select at least one model.")
            return

        self._results_table.setRowCount(0)
        self._export_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._progress_label.setStyleSheet(WidgetStyles.inline_label(color=TEXT_DISABLED))
        self._progress_label.setText("Starting benchmark...")
        self._running = True
        self._run_btn.setEnabled(False)

        config = self._study_data["config"]
        eval_type = self._eval_type_combo.currentData()

        if not self._validate_cross_session(config, eval_type):
            return

        self._worker = BenchmarkWorker(
            study_data=self._study_data,
            models=selected_models,
            epochs=self._training_config.get("epochs", 100),
            batch_size=self._training_config.get("batch_size", 32),
            learning_rate=self._training_config.get("learning_rate", 0.001),
            n_folds=self._n_folds,
            optuna_config=self._optuna_config if self._optuna_check.isChecked() else None,
            eval_type=eval_type,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.result_ready.connect(self._on_model_result)
        self._worker.finished.connect(self._on_benchmark_finished)
        self._worker.error.connect(self._on_benchmark_error)
        self._worker.start()

    def _validate_cross_session(self, config, eval_type: str) -> bool:
        """Validate cross_session at run time. Returns False if invalid."""
        if eval_type != "cross_session" or config.source_type != "moabb":
            return True

        loader = self._study_data.get("loader")
        if not loader or not hasattr(loader, "get_sessions"):
            return True

        try:
            first_subj = config.subjects[0] if config.subjects else 1
            n_sessions = len(loader.get_sessions(first_subj))
            if n_sessions < 2:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid Evaluation",
                    f"Cross Session requires 2+ sessions, but dataset has {n_sessions}.",
                )
                self._running = False
                self._progress.setVisible(False)
                self._run_btn.setEnabled(True)
                return False
        except Exception as e:
            logger.warning(f"Could not check sessions: {e}")

        return True

    def _on_progress(self, percent: int, message: str):
        self._progress.setValue(percent)
        self._progress_label.setText(message)

    def _on_model_result(self, model_name: str, metrics: dict):
        row = self._results_table.rowCount()
        self._results_table.insertRow(row)

        self._results_table.setItem(row, 0, QtWidgets.QTableWidgetItem(model_name))

        acc_text = f"{metrics.get('accuracy', 0) * 100:.1f}%"
        if metrics.get("accuracy_std", 0) > 0:
            acc_text += f" ±{metrics['accuracy_std'] * 100:.1f}"
        self._results_table.setItem(row, 1, QtWidgets.QTableWidgetItem(acc_text))

        time_val = metrics.get("eval_time", metrics.get("train_time", 0))
        self._results_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{time_val:.1f}s"))

        n_subjects = metrics.get("n_subjects", 1)
        subjects_item = QtWidgets.QTableWidgetItem(str(n_subjects))
        subjects_item.setData(
            QtCore.Qt.ItemDataRole.UserRole,
            {
                "best_params": metrics.get("best_params", {}),
                "per_subject": metrics.get("per_subject", []),
                "n_subjects": n_subjects,
            },
        )
        self._results_table.setItem(row, 3, subjects_item)

        best_params = metrics.get("best_params", {})
        if self._app_state and best_params:
            self._app_state.set_benchmark_result(model_name, best_params)

    def _on_benchmark_finished(self):
        self._running = False
        self._progress.setVisible(False)
        self._progress_label.setText("Benchmark complete!")
        self._run_btn.setEnabled(True)
        has_results = self._results_table.rowCount() > 0
        self._export_btn.setEnabled(has_results)

    def _on_benchmark_error(self, error_msg: str):
        self._running = False
        self._progress.setVisible(False)
        self._progress_label.setText(f"Error: {error_msg}")
        self._progress_label.setStyleSheet(WidgetStyles.inline_label(color=STATUS_ERROR))
        self._run_btn.setEnabled(True)
        QtWidgets.QMessageBox.critical(self, "Benchmark Error", error_msg)

    # -- Export --

    def _on_export(self):
        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "benchmark_results.csv",
            "CSV Files (*.csv);;JSON Files (*.json)",
        )
        if filepath:
            self._export_results(filepath)

    def _export_results(self, filepath: str):
        rows = []
        for row_idx in range(self._results_table.rowCount()):
            stored_data = (
                self._results_table.item(row_idx, 3).data(QtCore.Qt.ItemDataRole.UserRole) or {}
            )
            rows.append(
                BenchmarkRow(
                    model=self._results_table.item(row_idx, 0).text(),
                    accuracy=self._results_table.item(row_idx, 1).text(),
                    time=self._results_table.item(row_idx, 2).text(),
                    n_subjects=stored_data.get("n_subjects", 1),
                    best_params=stored_data.get("best_params", {}),
                    per_subject=stored_data.get("per_subject", []),
                )
            )

        per_subj_path = export_benchmark_results(filepath, rows)

        if per_subj_path:
            self._progress_label.setText(f"Results exported to {filepath} and {per_subj_path}")
        else:
            self._progress_label.setText(f"Results exported to {filepath}")
