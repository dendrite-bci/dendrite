"""
Dendrite Constants
"""

from pathlib import Path

# Project paths (src/dendrite/constants.py -> parents[2] -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# --- Data Storage ---
STUDIES_DIR = DATA_DIR / "studies"
DATABASE_PATH = DATA_DIR / "dendrite.db"
DEFAULT_STUDY_NAME = "default_study"


def get_study_paths(study_name: str) -> dict[str, Path]:
    """Get all data paths for a study."""
    base = STUDIES_DIR / study_name
    return {
        "config": base / "config",
        "raw": base / "raw",
        "metrics": base / "metrics",
        "decoders": base / "decoders",
        "logs": base / "logs",
    }

# --- LSL / Streams ---
SAMPLE_PULL_TIMEOUT = 0.1  # Per-sample pull timeout

# --- Processing ---
QUEUE_SIZE_LARGE = 1000  # Multiprocessing queue capacity (mode outputs, save)

# --- Process Timeouts (seconds) ---
TIMEOUT_DATA_ACQUISITION = 5
TIMEOUT_DATA_SAVER = 5
TIMEOUT_MODE_PROCESS = 2
TIMEOUT_METRICS_SAVER = 2
