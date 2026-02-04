"""Benchmark tab for model comparison and evaluation."""

import csv
import json
from typing import Any

from PyQt6 import QtCore, QtWidgets

from dendrite.auxiliary.ml_workbench.backend import BENCHMARK_SEED, BenchmarkWorker
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
        self._setup_ui()

        if self._app_state:
            self._app_state.study_changed.connect(self._on_study_changed)

    def set_app_state(self, app_state):
        self._app_state = app_state
        self._app_state.study_changed.connect(self._on_study_changed)
        self._update_data_status()

    def _setup_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(
            LAYOUT["spacing_lg"], LAYOUT["spacing_lg"], LAYOUT["spacing_lg"], LAYOUT["spacing_lg"]
        )
        layout.setSpacing(LAYOUT["spacing_lg"])

        # Left panel (scrollable)
        left_panel, left_layout, left_footer = create_scrollable_panel(560)

        # Data Source section
        data_section, data_layout = create_section("Data Source")
        self._data_label = QtWidgets.QLabel("No study loaded")
        self._data_label.setStyleSheet(WidgetStyles.inline_label(color=STATUS_WARN))
        self._data_label.setWordWrap(True)
        data_layout.addWidget(self._data_label)
        left_layout.addWidget(data_section)

        # Decoders section
        decoder_section, decoder_layout = create_section("Decoders")

        self._model_checkboxes: dict[str, QtWidgets.QCheckBox] = {}
        decoder_names = get_available_decoders()

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

        for decoder_name in decoder_names:
            cb = QtWidgets.QCheckBox(decoder_name)
            if decoder_name in ["EEGNet", "CSP+LDA"]:
                cb.setChecked(True)
            self._model_checkboxes[decoder_name] = cb
            scroll_layout.addWidget(cb)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        decoder_layout.addWidget(scroll)

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
        decoder_layout.addLayout(btn_row)
        left_layout.addWidget(decoder_section)

        # Evaluation Strategy section
        eval_section, eval_layout_v = create_section("Evaluation Strategy")

        eval_form = create_form_layout()
        eval_form.setSpacing(LAYOUT["spacing_md"])

        self._eval_type_combo = QtWidgets.QComboBox()
        self._eval_type_combo.setStyleSheet(WidgetStyles.combobox())
        self._eval_type_combo.addItem("Within Session (k-fold CV)", "within_session")
        self._eval_type_combo.addItem("Cross Session", "cross_session")
        self._eval_type_combo.addItem("Cross Subject (LOSO)", "cross_subject")
        eval_form.addRow("Type:", self._eval_type_combo)

        self._eval_info_label = QtWidgets.QLabel("")
        self._eval_info_label.setStyleSheet(WidgetStyles.inline_label(color=TEXT_DISABLED, size=11))
        self._eval_info_label.setWordWrap(True)
        eval_form.addRow("", self._eval_info_label)

        eval_layout_v.addLayout(eval_form)
        left_layout.addWidget(eval_section)
        self._eval_group = eval_section

        # Training Configuration section (collapsible, collapsed by default)
        self._config_section = CollapsibleSection("Training Configuration", start_expanded=False)
        config_layout = self._config_section.content_layout()

        self._config_summary = QtWidgets.QLabel("Epochs: 100, Batch: 32, LR: 0.001")
        self._config_summary.setStyleSheet(WidgetStyles.inline_label(color=TEXT_DISABLED))
        self._config_summary.setWordWrap(True)
        config_layout.addWidget(self._config_summary)

        # Optuna toggle
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

        left_layout.addWidget(self._config_section)
        left_layout.addStretch()

        # Store training config
        self._training_config = {
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
        }
        self._optuna_config = OptunaConfig(n_trials=10)
        self._n_folds = 5
        self._update_config_summary()

        # Footer: action buttons pinned below scroll area
        self._run_btn = QtWidgets.QPushButton("Run Benchmark")
        self._run_btn.setMinimumHeight(40)
        self._run_btn.setStyleSheet(WidgetStyles.button(padding="8px 12px"))
        self._run_btn.clicked.connect(self._on_run)
        self._run_btn.setEnabled(False)
        left_footer.addWidget(self._run_btn)

        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setMaximumHeight(16)
        self._progress.setStyleSheet(WidgetStyles.progress_bar())
        self._progress.setVisible(False)
        left_footer.addWidget(self._progress)

        self._progress_label = QtWidgets.QLabel("Ready")
        self._progress_label.setWordWrap(True)
        self._progress_label.setStyleSheet(WidgetStyles.inline_label(color=TEXT_DISABLED))
        left_footer.addWidget(self._progress_label)

        layout.addWidget(left_panel)

        # Right panel - Results
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(LAYOUT["spacing_md"])

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

        right_layout.addWidget(results_group)

        layout.addWidget(right_panel, 1)
        self._update_data_status()

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

            # Count available subjects
            n_subjects = 1 if selected_subject is not None else len(config.subjects)

            source_str = "[MOABB]" if is_moabb else ""
            self._data_label.setText(
                f"<b>{config.name}</b> {source_str}<br>" f"Subjects: {n_subjects}"
            )
            self._data_label.setStyleSheet(WidgetStyles.inline_label(color=STATUS_SUCCESS))
            self._run_btn.setEnabled(True)
            self._on_modality_changed("EEG")

            # Show evaluation options for all datasets (unified MOABB eval)
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
        # Internal datasets typically have 1 session per subject
        cross_session_ok = is_moabb  # Only MOABB datasets may have multiple sessions

        # Update combo items (disable unavailable options)
        model = self._eval_type_combo.model()
        # Cross Session - only for MOABB (internal typically single session)
        model.item(1).setEnabled(cross_session_ok)
        # Cross Subject - disabled if < 2 subjects
        model.item(2).setEnabled(cross_subject_ok)

        # Update info label
        info_parts = []
        if not cross_session_ok:
            info_parts.append("Cross Session: MOABB only")
        if not cross_subject_ok:
            info_parts.append("Cross Subject: needs 2+ subjects")
        self._eval_info_label.setText(" | ".join(info_parts))

        # Reset if selected option not valid
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
        self._progress_label.setStyleSheet(
            WidgetStyles.inline_label(color=TEXT_DISABLED)
        )  # Reset from error style
        self._progress_label.setText("Starting benchmark...")
        self._running = True
        self._run_btn.setEnabled(False)

        # Always use MOABB evaluation (unified path for all datasets)
        config = self._study_data["config"]
        eval_type = self._eval_type_combo.currentData()

        # Validate cross_session at run time (avoids data download on selection)
        if eval_type == "cross_session" and config.source_type == "moabb":
            loader = self._study_data.get("loader")
            if loader and hasattr(loader, "get_sessions"):
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
                        return
                except Exception as e:
                    logger.warning(f"Could not check sessions: {e}")

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

    def _on_progress(self, percent: int, message: str):
        self._progress.setValue(percent)
        self._progress_label.setText(message)

    def _on_model_result(self, model_name: str, metrics: dict):
        row = self._results_table.rowCount()
        self._results_table.insertRow(row)

        # Col 0: Model name
        self._results_table.setItem(row, 0, QtWidgets.QTableWidgetItem(model_name))

        # Col 1: Accuracy with std
        acc_text = f"{metrics.get('accuracy', 0) * 100:.1f}%"
        if metrics.get("accuracy_std", 0) > 0:
            acc_text += f" ±{metrics['accuracy_std'] * 100:.1f}"
        self._results_table.setItem(row, 1, QtWidgets.QTableWidgetItem(acc_text))

        # Col 2: Time (train_time for holdout eval, eval_time for MOABB eval)
        time_val = metrics.get("eval_time", metrics.get("train_time", 0))
        self._results_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{time_val:.1f}s"))

        # Col 3: Number of subjects (store metadata for export)
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

        # Store Optuna results if available
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
        all_per_subject = []

        for row in range(self._results_table.rowCount()):
            model_name = self._results_table.item(row, 0).text()
            # Metadata stored in Subjects column (col 3)
            stored_data = (
                self._results_table.item(row, 3).data(QtCore.Qt.ItemDataRole.UserRole) or {}
            )
            best_params = stored_data.get("best_params", {})
            per_subject = stored_data.get("per_subject", [])

            rows.append(
                {
                    "model": model_name,
                    "accuracy": self._results_table.item(row, 1).text(),
                    "time": self._results_table.item(row, 2).text(),
                    "n_subjects": stored_data.get("n_subjects", 1),
                    "best_params": best_params,
                }
            )

            # Collect per-subject data for detailed export
            for subj_data in per_subject:
                all_per_subject.append(
                    {
                        "model": model_name,
                        "subject": subj_data.get("subject", ""),
                        "accuracy": subj_data.get("accuracy", ""),
                        "kappa": subj_data.get("kappa", ""),
                        "f1": subj_data.get("f1", ""),
                        "balanced_accuracy": subj_data.get("balanced_accuracy", ""),
                    }
                )

        if filepath.endswith(".json"):
            # JSON includes full per-subject breakdown
            export_data = {
                "summary": rows,
                "per_subject": all_per_subject,
                "seed": BENCHMARK_SEED,
            }
            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2)
        else:
            # CSV exports summary (per-subject available in JSON)
            # Serialize best_params dict to JSON string for CSV
            csv_rows = []
            for row in rows:
                csv_row = row.copy()
                csv_row["best_params"] = json.dumps(row.get("best_params", {}))
                csv_rows.append(csv_row)

            with open(filepath, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["model", "accuracy", "time", "n_subjects", "best_params"]
                )
                writer.writeheader()
                writer.writerows(csv_rows)

            # Also export per-subject CSV if data available
            if all_per_subject:
                per_subj_path = filepath.replace(".csv", "_per_subject.csv")
                with open(per_subj_path, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "model",
                            "subject",
                            "accuracy",
                            "kappa",
                            "f1",
                            "balanced_accuracy",
                        ],
                    )
                    writer.writeheader()
                    writer.writerows(all_per_subject)
                self._progress_label.setText(f"Results exported to {filepath} and {per_subj_path}")
                return

        self._progress_label.setText(f"Results exported to {filepath}")
