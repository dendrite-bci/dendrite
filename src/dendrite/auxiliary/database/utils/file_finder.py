"""
File Finding Utilities

Centralized logic for finding associated files (raw EEG, metrics, etc.).
"""

import glob
import os


def find_raw_eeg_file(record: dict) -> str | None:
    """
    Find the raw EEG file associated with a recording.

    Centralized method to avoid duplication across dialogs.

    Args:
        record: Database record dictionary

    Returns:
        Path to raw EEG file if found, None otherwise
    """
    from .file_validation import detect_h5_file_type

    base_path = record.get("hdf5_file_path", "")
    if not base_path:
        return None

    # If the current file is already a raw EEG file, use it
    file_type = detect_h5_file_type(base_path)
    if file_type == "eeg_data":
        return base_path

    # Otherwise, try to find the corresponding raw EEG file
    base_dir = os.path.dirname(base_path)
    filename = os.path.basename(base_path)

    if filename.startswith("metrics_"):
        unique_id = filename.replace("metrics_", "").replace(".h5", "")
        raw_file_pattern = f"eeg_data_*{unique_id}*.h5"

        search_dirs = [
            base_dir,
            os.path.join(base_dir, "..", "raw"),
            os.path.join(base_dir, "..", "..", "raw"),
        ]

        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                matches = glob.glob(os.path.join(search_dir, raw_file_pattern))
                if matches:
                    return matches[0]

    return None


def find_metrics_file(record: dict) -> str | None:
    """Return the metrics file path from the database record."""
    path = record.get("metrics_file_path")
    return path if path and os.path.exists(path) else None
