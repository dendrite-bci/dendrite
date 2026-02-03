"""
Format Helper Utilities

Formatting functions for file sizes, timestamps, and H5 filename parsing.
"""

import logging
import os
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def format_file_size(size_mb: float) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_mb: File size in megabytes

    Returns:
        Formatted string (e.g., "1.5 GB", "250 MB", "512 KB")
    """
    if size_mb >= 1024:
        return f"{size_mb / 1024:.2f} GB"
    elif size_mb >= 1:
        return f"{size_mb:.1f} MB"
    else:
        return f"{size_mb * 1024:.0f} KB"


def parse_h5_filename(file_path: str) -> dict[str, Any]:
    """
    Extract basic metadata from H5 file.

    Simple extraction - study name should be selected via ComboBox (get_existing_studies).
    File naming standards will be improved in the future.

    Args:
        file_path: Path to H5 file

    Returns:
        Dictionary with keys:
            - study_name: Empty (user selects from ComboBox)
            - experiment_name: Cleaned filename
            - session_timestamp: File modification time
            - file_type: 'eeg_data', 'metrics', or 'unknown'
    """
    filename = os.path.basename(file_path)
    result = {
        "study_name": "",  # User selects from analysis/ folder dropdown
        "experiment_name": "",
        "session_timestamp": "",
        "file_type": "unknown",
    }

    # Determine file type from prefix
    if filename.startswith("eeg_data_"):
        result["file_type"] = "eeg_data"
    elif filename.startswith("metrics_"):
        result["file_type"] = "metrics"

    # Use cleaned filename as experiment name
    name_part = filename.replace("eeg_data_", "").replace("metrics_", "").replace(".h5", "")
    name_part = name_part.replace("_preprocessed", "").replace("_raw", "")
    result["experiment_name"] = name_part if name_part else filename.replace(".h5", "")

    # Use file modification time as timestamp
    try:
        mtime = os.path.getmtime(file_path)
        result["session_timestamp"] = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
    except OSError:
        result["session_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return result
