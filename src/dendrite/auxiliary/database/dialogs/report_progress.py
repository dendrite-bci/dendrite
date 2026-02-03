"""
Report Progress Dialog

Dialog showing report generation progress with subprocess monitoring.
"""

import logging
import re
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

from PyQt6 import QtCore, QtWidgets

from dendrite.constants import TEMP_REPORTS_DIR
from dendrite.gui.styles.design_tokens import (
    ACCENT,
    BG_INPUT,
    BG_MAIN,
    BORDER,
    RADIUS,
    SPACING,
    TEXT_MAIN,
)
from dendrite.gui.styles.widget_styles import WidgetStyles

logger = logging.getLogger(__name__)


class ReportProgressDialog(QtWidgets.QDialog):
    """Dialog showing report generation progress."""

    def __init__(self, file_path: str, parent=None, record: dict | None = None):
        super().__init__(parent)
        self.file_path = file_path
        self.record = record
        self.process = None

        # Generate predictable output path
        basename = Path(self.file_path).stem
        timestamp = re.search(r"(\d{8}_\d{6})", basename)
        ts = timestamp.group(1) if timestamp else basename
        self.report_path = TEMP_REPORTS_DIR / f"session_report_{ts}.html"
        self.report_path.parent.mkdir(parents=True, exist_ok=True)

        self.setWindowTitle("Generating Session Report")
        self.setFixedSize(400, 200)
        self.setModal(True)
        self.setup_ui()
        self.apply_styles()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(SPACING["md"])

        info_label = QtWidgets.QLabel(
            f"Generating session report for:\n{Path(self.file_path).name}"
        )
        info_label.setStyleSheet(WidgetStyles.label())
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Progress bar
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)  # Indeterminate
        layout.addWidget(self.progress)

        self.status_label = QtWidgets.QLabel("Initializing...")
        self.status_label.setStyleSheet(WidgetStyles.label(size="small"))
        layout.addWidget(self.status_label)

        # Cancel/Close button
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.setStyleSheet(WidgetStyles.button(variant="secondary"))
        self.cancel_btn.clicked.connect(self.cancel_report)
        layout.addWidget(self.cancel_btn)

    def apply_styles(self):
        self.setStyleSheet(f"""
            QDialog {{
                background: {BG_MAIN};
                color: {TEXT_MAIN};
            }}
            QProgressBar {{
                border: 1px solid {BORDER};
                border-radius: {RADIUS["md"]}px;
                text-align: center;
                background: {BG_INPUT};
            }}
            QProgressBar::chunk {{
                background: {ACCENT};
                border-radius: {RADIUS["md"]}px;
            }}
        """)

    def start_report(self):
        """Start the report generation process."""
        try:
            # Use unified session report for both types
            module_name = "dendrite.utils.reports.session_report"
            self.status_label.setText("Generating session report...")

            # Build command with explicit output path
            cmd = [
                sys.executable,
                "-m",
                module_name,
                "--file",
                self.file_path,
                "--output",
                str(self.report_path),
            ]

            # Add study name and metrics path if available from database record
            if self.record:
                # Study name
                study_name = self.record.get("study_name")
                if study_name:
                    cmd.extend(["--study", study_name])
                    logger.info(f"Using database study name for report: '{study_name}'")

                # Metrics file path - pass explicitly from database
                metrics_path = self.record.get("metrics_file_path")
                if metrics_path and Path(metrics_path).exists():
                    cmd.extend(["--metrics", metrics_path])
                    logger.info(f"Using database metrics file: '{metrics_path}'")

            logger.info(f"Report command: {' '.join(cmd)}")

            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self.monitor_process)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()

        except Exception as e:
            self.status_label.setText(f"Error: {e!s}")

    def monitor_process(self):
        """Monitor the subprocess in a separate thread."""
        try:
            stdout, stderr = self.process.communicate()
            QtCore.QMetaObject.invokeMethod(
                self,
                "process_finished",
                QtCore.Qt.ConnectionType.QueuedConnection,
                QtCore.Q_ARG(int, self.process.returncode),
                QtCore.Q_ARG(str, stdout),
                QtCore.Q_ARG(str, stderr),
            )
        except Exception as e:
            QtCore.QMetaObject.invokeMethod(
                self,
                "process_error",
                QtCore.Qt.ConnectionType.QueuedConnection,
                QtCore.Q_ARG(str, str(e)),
            )

    @QtCore.pyqtSlot(int, str, str)
    def process_finished(self, return_code: int, stdout: str, stderr: str):
        """Handle process completion."""
        if return_code == 0:
            self.status_label.setText("Report generated successfully!")
            self.progress.setRange(0, 1)
            self.progress.setValue(1)

            self.cancel_btn.setText("Close")

            # Try to open the report
            self.try_open_report()
        else:
            self.status_label.setText(f"Report generation failed (code {return_code})")
            if stderr:
                logger.error(f"Report error: {stderr}")

    @QtCore.pyqtSlot(str)
    def process_error(self, error: str):
        """Handle process error."""
        self.status_label.setText(f"Process error: {error}")

    def try_open_report(self):
        """Try to open the generated report."""
        if self.report_path and self.report_path.exists():
            reply = QtWidgets.QMessageBox.question(
                self,
                "Open Report",
                "Report generated successfully! Would you like to open it?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            )
            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                webbrowser.open(self.report_path.as_uri())
        else:
            QtWidgets.QMessageBox.information(
                self,
                "Report Generated",
                f"Report generated successfully!\nCheck the {TEMP_REPORTS_DIR} directory for the output.",
            )

    def cancel_report(self):
        """Cancel the report generation."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
        self.reject()
