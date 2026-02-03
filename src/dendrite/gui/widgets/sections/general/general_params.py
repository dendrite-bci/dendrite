"""
General Parameters Widget

A widget for general system parameters like study name, subject, session, and recording name.
Provides database-backed dropdowns for BIDS-compliant data organization.
"""

import subprocess
import sys
from contextlib import contextmanager

from pydantic import ValidationError
from PyQt6 import QtCore, QtWidgets

from dendrite.constants import (
    DEFAULT_RECORDING_NAME,
    DEFAULT_SESSION_ID,
    DEFAULT_STUDY_NAME,
    DEFAULT_SUBJECT_ID,
)
from dendrite.gui.config.study_schemas import StudyConfig
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles
from dendrite.gui.utils import load_icon
from dendrite.utils.logger_central import get_logger

logger = get_logger("GeneralParametersWidget")

# External apps configuration: (label, icon, module)
APP_CONFIGS = [
    ("Stream Manager", "icons/stream.svg", "dendrite.auxiliary.stream_manager.app"),
    ("Dashboard", "icons/dashboard.svg", "dendrite.auxiliary.dashboard.app"),
    ("DB Explorer", "icons/database.svg", "dendrite.auxiliary.database.app"),
    ("Trainer", "icons/brain.svg", "dendrite.auxiliary.ml_workbench.app"),
]


class GeneralParametersWidget(QtWidgets.QWidget):
    """Widget for general system parameters and utility functions."""

    load_config_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self._db = None  # Lazy-loaded database
        self._repo = None  # Lazy-loaded recording repository
        self._study_repo = None  # Lazy-loaded study repository

        self.setup_ui()
        self.refresh_from_db()  # Initial population
        self._refresh_run_number()  # Initialize run number display

    def setup_ui(self):
        """Set up the user interface."""
        general_layout = QtWidgets.QVBoxLayout(self)
        general_layout.setContentsMargins(0, 0, 0, 0)
        general_layout.setSpacing(WidgetStyles.layout["spacing"])

        # Header row: "STUDY" label + config icon button
        header_row = QtWidgets.QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)

        study_header = QtWidgets.QLabel("STUDY")
        study_header.setStyleSheet(
            WidgetStyles.label("muted_header", uppercase=True, letter_spacing=1)
        )
        header_row.addWidget(study_header)

        header_row.addStretch()

        self.load_config_button = QtWidgets.QPushButton()
        self.load_config_button.setToolTip("Load Configuration")
        self.load_config_button.setStyleSheet(
            WidgetStyles.button(variant="text", padding="4px 8px")
        )
        self.load_config_button.setFixedSize(28, 28)
        config_icon = load_icon("icons/database.svg")
        if not config_icon.isNull():
            self.load_config_button.setIcon(config_icon)
            self.load_config_button.setIconSize(QtCore.QSize(16, 16))
        self.load_config_button.clicked.connect(self.load_config_requested.emit)
        header_row.addWidget(self.load_config_button)

        general_layout.addLayout(header_row)
        general_layout.addSpacing(LAYOUT["spacing_sm"])  # Gap after header

        # BIDS fields in grid layout for proper alignment
        label_style = WidgetStyles.label("small")

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(LAYOUT["spacing_sm"])
        grid.setContentsMargins(0, 0, 0, 0)

        # Row 0: Study, Task
        study_label = QtWidgets.QLabel("Study:")
        study_label.setStyleSheet(label_style)
        grid.addWidget(study_label, 0, 0)

        self.study_name_combo = QtWidgets.QComboBox()
        self.study_name_combo.setEditable(True)
        self.study_name_combo.setStyleSheet(WidgetStyles.combobox())
        self.study_name_combo.currentTextChanged.connect(self._on_study_changed)
        grid.addWidget(self.study_name_combo, 0, 1)

        task_label = QtWidgets.QLabel("Task:")
        task_label.setStyleSheet(label_style)
        grid.addWidget(task_label, 0, 2)

        self.file_name_input = QtWidgets.QLineEdit(DEFAULT_RECORDING_NAME)
        self.file_name_input.setStyleSheet(WidgetStyles.input())
        self.file_name_input.textChanged.connect(self._on_task_changed)
        grid.addWidget(self.file_name_input, 0, 3, 1, 3)  # span cols 3-5

        # Row 1: Sub, Ses, Run
        sub_label = QtWidgets.QLabel("Sub:")
        sub_label.setStyleSheet(label_style)
        grid.addWidget(sub_label, 1, 0)

        self.subject_id_combo = QtWidgets.QComboBox()
        self.subject_id_combo.setEditable(True)
        self.subject_id_combo.setStyleSheet(WidgetStyles.combobox())
        self.subject_id_combo.currentTextChanged.connect(self._on_subject_changed)
        grid.addWidget(self.subject_id_combo, 1, 1)

        ses_label = QtWidgets.QLabel("Ses:")
        ses_label.setStyleSheet(label_style)
        grid.addWidget(ses_label, 1, 2)

        self.session_id_combo = QtWidgets.QComboBox()
        self.session_id_combo.setEditable(True)
        self.session_id_combo.setStyleSheet(WidgetStyles.combobox())
        self.session_id_combo.currentTextChanged.connect(self._on_session_changed)
        grid.addWidget(self.session_id_combo, 1, 3)

        run_label_prefix = QtWidgets.QLabel("Run:")
        run_label_prefix.setStyleSheet(label_style)
        grid.addWidget(run_label_prefix, 1, 4)

        self.run_label = QtWidgets.QLabel("01")
        self.run_label.setStyleSheet(label_style)
        self.run_label.setToolTip("Auto-calculated run number")
        grid.addWidget(self.run_label, 1, 5)

        # Column stretches for proportional sizing
        grid.setColumnStretch(1, 1)  # Study combo
        grid.setColumnStretch(3, 1)  # Task input / Ses combo

        general_layout.addLayout(grid)

        # External Apps section
        self.setup_external_apps_section(general_layout)

    def setup_external_apps_section(self, parent_layout):
        """Set up the compact external apps section."""
        parent_layout.addSpacing(LAYOUT["spacing_xl"])

        header = QtWidgets.QLabel("APPS")
        header.setStyleSheet(WidgetStyles.label("muted_header", uppercase=True, letter_spacing=1))
        parent_layout.addWidget(header)
        parent_layout.addSpacing(LAYOUT["spacing_sm"])

        buttons_layout = QtWidgets.QGridLayout()
        buttons_layout.setSpacing(LAYOUT["spacing_sm"])
        buttons_layout.setContentsMargins(0, 0, 0, 0)

        for i, (label, icon_name, module) in enumerate(APP_CONFIGS):
            btn = self._create_app_button(label, icon_name, module)
            buttons_layout.addWidget(btn, i // 2, i % 2)

        parent_layout.addLayout(buttons_layout)
        parent_layout.addSpacing(LAYOUT["spacing_xl"])

    def _create_app_button(self, label: str, icon_name: str, module: str) -> QtWidgets.QPushButton:
        """Create a styled app launcher button."""
        btn = QtWidgets.QPushButton(label)
        btn.setStyleSheet(
            WidgetStyles.button(variant="primary", align="center", padding="8px 10px")
        )
        btn.setToolTip(f"Launch {label}")
        icon = load_icon(icon_name)
        if not icon.isNull():
            btn.setIcon(icon)
            btn.setIconSize(QtCore.QSize(16, 16))
        btn.clicked.connect(lambda checked, m=module, l=label: self.launch_external_app(m, l))
        return btn

    def _init_db(self):
        """Lazy-initialize database helpers."""
        if self._db is None:
            # Local import for lazy loading: database module imported only when first needed
            from dendrite.data.storage.database import (
                Database,
                RecordingRepository,
                StudyRepository,
            )

            self._db = Database()
            self._db.init_db()
            self._repo = RecordingRepository(self._db)
            self._study_repo = StudyRepository(self._db)

    @contextmanager
    def _block_combo_signals(self):
        """Context manager to block signals on all BIDS combo boxes."""
        self.study_name_combo.blockSignals(True)
        self.subject_id_combo.blockSignals(True)
        self.session_id_combo.blockSignals(True)
        try:
            yield
        finally:
            self.study_name_combo.blockSignals(False)
            self.subject_id_combo.blockSignals(False)
            self.session_id_combo.blockSignals(False)

    def refresh_from_db(self):
        """Refresh all dropdowns from database."""
        try:
            self._init_db()
            current_study = self.study_name_combo.currentText()
            current_subject = self.subject_id_combo.currentText()
            current_session = self.session_id_combo.currentText()

            with self._block_combo_signals():
                # Refresh studies
                self.study_name_combo.clear()
                studies = self._study_repo.get_all_studies()
                self.study_name_combo.addItems([s["study_name"] for s in studies])
                self.study_name_combo.setCurrentText(current_study or DEFAULT_STUDY_NAME)

                # Refresh subjects for current study
                self._refresh_subjects()
                if current_subject:
                    self.subject_id_combo.setCurrentText(current_subject)
                elif self.subject_id_combo.count() == 0:
                    self.subject_id_combo.setCurrentText(DEFAULT_SUBJECT_ID)

                # Refresh sessions
                self._refresh_sessions()
                if current_session:
                    self.session_id_combo.setCurrentText(current_session)
                elif self.session_id_combo.count() == 0:
                    self.session_id_combo.setCurrentText(DEFAULT_SESSION_ID)

        except Exception as e:
            logger.warning(f"Could not refresh from database: {e}")
            self.study_name_combo.setCurrentText(DEFAULT_STUDY_NAME)
            self.subject_id_combo.setCurrentText(DEFAULT_SUBJECT_ID)
            self.session_id_combo.setCurrentText(DEFAULT_SESSION_ID)

    def _refresh_subjects(self):
        """Refresh subject list for current study."""
        try:
            self._init_db()
            study = self.study_name_combo.currentText().strip()
            self.subject_id_combo.clear()
            if study:
                subjects = self._repo.get_existing_subjects(study)
                self.subject_id_combo.addItems(subjects)
        except Exception as e:
            logger.debug(f"Could not refresh subjects: {e}")

    def _refresh_sessions(self):
        """Refresh session list for current study+subject, suggest next."""
        try:
            self._init_db()
            study = self.study_name_combo.currentText().strip()
            subject = self.subject_id_combo.currentText().strip()
            self.session_id_combo.clear()
            if study and subject:
                sessions = self._repo.get_existing_sessions(study, subject)
                self.session_id_combo.addItems(sessions)
                # Suggest next session
                next_session = self._repo.get_next_session_id(study, subject)
                if next_session not in sessions:
                    self.session_id_combo.setCurrentText(next_session)
        except Exception as e:
            logger.debug(f"Could not refresh sessions: {e}")

    def _on_study_changed(self, text: str):
        """Handle study selection change - refresh subjects, sessions, run."""
        with self._block_combo_signals():
            self._refresh_subjects()
            self._refresh_sessions()
            self._refresh_run_number()

    def _on_subject_changed(self, text: str):
        """Handle subject selection change - refresh sessions, run."""
        with self._block_combo_signals():
            self._refresh_sessions()
            self._refresh_run_number()

    def _on_session_changed(self, text: str):
        """Handle session selection change - refresh run number."""
        self._refresh_run_number()

    def _on_task_changed(self, text: str):
        """Handle task name change - refresh run number."""
        self._refresh_run_number()

    def _refresh_run_number(self):
        """Refresh the run number display based on current BIDS fields."""
        try:
            self._init_db()
            subject = self.subject_id_combo.currentText().strip()
            session = self.session_id_combo.currentText().strip()
            task = self.file_name_input.text().strip()
            if subject and session and task:
                run_num = self._repo.get_next_run_number(subject, session, task)
                self.run_label.setText(f"{run_num:02d}")
            else:
                self.run_label.setText("01")
        except Exception as e:
            logger.debug(f"Could not refresh run number: {e}")
            self.run_label.setText("01")

    def get_general_config(self) -> dict:
        """Get the current general configuration."""
        return {
            "study_name": self.study_name_combo.currentText().strip() or DEFAULT_STUDY_NAME,
            "subject_id": self.subject_id_combo.currentText().strip(),
            "session_id": self.session_id_combo.currentText().strip(),
            "recording_name": self.file_name_input.text().strip() or DEFAULT_RECORDING_NAME,
        }

    def set_general_config(self, config: dict):
        """Set the general configuration from a dictionary."""
        with self._block_combo_signals():
            if "study_name" in config:
                self.study_name_combo.setCurrentText(config["study_name"])
                self._refresh_subjects()
            if "subject_id" in config:
                self.subject_id_combo.setCurrentText(config["subject_id"])
                self._refresh_sessions()
            if "session_id" in config:
                self.session_id_combo.setCurrentText(config["session_id"])
            if "recording_name" in config:
                self.file_name_input.setText(config["recording_name"])

    def validate_required_fields(self) -> tuple[bool, str]:
        """Validate BIDS fields using Pydantic schema. Returns (valid, error_message)."""
        try:
            StudyConfig(
                study_name=self.study_name_combo.currentText(),
                subject_id=self.subject_id_combo.currentText(),
                session_id=self.session_id_combo.currentText(),
                recording_name=self.file_name_input.text(),
            )
            return True, ""
        except ValidationError as e:
            return False, e.errors()[0]["msg"]

    def launch_external_app(self, module_name: str, app_display_name: str):
        """Launch an external application as a Python module."""
        try:
            logger.info(f"Launching {app_display_name}: python -m {module_name}")

            # Launch the application as a module in a new console/process
            args = [sys.executable, "-m", module_name]
            if sys.platform.startswith("win"):
                subprocess.Popen(args, creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                subprocess.Popen(args)

            logger.info(f"Launched {app_display_name}")

        except Exception as e:
            logger.error(f"Failed to launch {app_display_name}: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "Launch Error", f"Failed to launch {app_display_name}:\n{e!s}"
            )
