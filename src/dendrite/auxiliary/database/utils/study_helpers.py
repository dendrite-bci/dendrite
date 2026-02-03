"""
Study Management Utilities

Helper functions for managing study directories in the analysis folder.
"""

import os
from pathlib import Path


def get_existing_studies() -> list[str]:
    """
    Get list of existing study names from analysis folder.

    Returns:
        Sorted list of study directory names (excluding special directories)
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[4]
    analysis_dir = project_root / "analysis"

    if not analysis_dir.exists():
        return []

    exclude = {"__pycache__", "v1_legacy", ".git", ".ipynb_checkpoints"}

    studies = []
    try:
        for item in os.listdir(analysis_dir):
            item_path = analysis_dir / item
            if item_path.is_dir() and item not in exclude and not item.startswith("."):
                studies.append(item)
    except OSError:
        pass

    return sorted(studies)
