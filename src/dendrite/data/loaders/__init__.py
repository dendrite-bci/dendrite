"""Dataset loading, epoching, and preprocessing utilities."""

from pathlib import Path

from ._training_data import (
    load_epochs,
    load_moabb_for_training,
    load_study_history,
    merge_recordings,
)
from ._types import EpochedData, RawData
from .fif_loader import FIFLoader
from .moabb_discovery import (
    discover_moabb_datasets,
    get_moabb_dataset_info,
)
from .moabb_loader import MoabbConfig, MOABBLoader
from .raw_h5_loader import RawH5Loader


def load_file(file_path: str, *, modality: str | None = None) -> RawData:
    """Load data from file, auto-detecting format by extension."""
    ext = Path(file_path).suffix.lower()
    if ext in FIFLoader.EXTENSIONS:
        return FIFLoader(file_path).load()
    elif ext in RawH5Loader.EXTENSIONS:
        return RawH5Loader(file_path).load(modality=modality)
    raise ValueError(f"Unsupported format: {ext}")


def is_supported_format(file_path: str) -> bool:
    """Check if file format is supported."""
    ext = Path(file_path).suffix.lower()
    return ext in FIFLoader.EXTENSIONS or ext in RawH5Loader.EXTENSIONS
