"""
File Validation Utilities

H5 file type detection, corruption checking, and error dialogs.
"""

import logging
import os

logger = logging.getLogger(__name__)


def detect_h5_file_type(file_path: str) -> str:
    """
    Detect if an H5 file is a metrics file or a regular EEG file.

    Uses get_h5_info() for consistent H5 handling.

    Args:
        file_path: Path to H5 file

    Returns:
        'metrics': File created by MetricsSaver with classifier groups
        'eeg_data': Regular EEG file with EEG/EMG/Event datasets
        'corrupted': File is corrupted or truncated
        'unknown': Cannot determine file type
    """
    from dendrite.data.io import get_h5_info

    try:
        # Quick file size check - if file is very small, it's likely corrupted
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            if file_size < 1024:  # Less than 1KB is likely corrupted
                return "corrupted"

        # Use centralized H5 info function
        h5_info = get_h5_info(file_path)
        datasets = set(h5_info["datasets"].keys())
        groups = set(h5_info["groups"].keys())

        # Check for APM metrics file structure
        has_script_metadata = "script_metadata" in groups or "script_metadata" in datasets
        has_classifier_groups = bool(groups - {"script_metadata"})  # Has groups besides metadata

        # Check for regular EEG file structure
        has_eeg_data = "EEG" in datasets
        has_event_data = "Event" in datasets

        if has_script_metadata and has_classifier_groups:
            return "metrics"
        elif has_eeg_data or has_event_data:
            return "eeg_data"
        else:
            return "unknown"

    except Exception as e:
        # Check for specific H5 corruption errors
        error_msg = str(e).lower()
        if any(
            keyword in error_msg for keyword in ["truncated", "eof", "corrupted", "unable to open"]
        ):
            return "corrupted"
        else:
            logger.error(f"Error detecting H5 file type for {file_path}: {e}")
            return "unknown"


def show_file_error(parent, error_type: str, file_path: str | None = None):
    """
    Centralized error dialog for file-related issues.

    Args:
        parent: Parent widget for dialog
        error_type: Type of error ('corrupted', 'not_found', 'no_eeg_file', 'no_metrics_file')
        file_path: Optional file path to display in error message
    """
    from PyQt6 import QtWidgets

    if error_type == "corrupted":
        QtWidgets.QMessageBox.warning(
            parent, "Corrupted File", f"The file appears to be corrupted or truncated:\n{file_path}"
        )
    elif error_type == "not_found":
        QtWidgets.QMessageBox.warning(parent, "File Not Found", f"File not found:\n{file_path}")
    elif error_type == "no_eeg_file":
        QtWidgets.QMessageBox.warning(
            parent, "File Not Found", "Raw EEG file not found for this recording."
        )
    elif error_type == "no_metrics_file":
        QtWidgets.QMessageBox.warning(
            parent, "File Not Found", "Metrics file not found for this recording."
        )
