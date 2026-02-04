"""Dataset import utilities for offline ML."""

from pathlib import Path

from ._types import LoadedData
from .base_loader import BaseLoader
from .config import DatasetConfig
from .fif_loader import FIFLoader
from .h5_loader import H5Loader
from .internal_moabb_wrapper import InternalDatasetWrapper
from .moabb_discovery import (
    discover_moabb_datasets,
    get_moabb_dataset_info,
    load_moabb_dataset_details,
)
from .moabb_loader import MOAABLoader


def load_file(file_path: str) -> LoadedData:
    """Load data from file, auto-detecting format by extension."""
    ext = Path(file_path).suffix.lower()
    if ext in FIFLoader.EXTENSIONS:
        return FIFLoader.load_file(file_path)
    elif ext in H5Loader.EXTENSIONS:
        return H5Loader.load_file(file_path)
    raise ValueError(f"Unsupported format: {ext}")


def get_file_filter() -> str:
    """Get file dialog filter string for supported formats."""
    return (
        "All Supported (*.fif *.h5 *.hdf5);;"
        "MNE-Python (*.fif);;"
        "HDF5 (*.h5 *.hdf5);;"
        "All Files (*)"
    )


def is_supported_format(file_path: str) -> bool:
    """Check if file format is supported."""
    ext = Path(file_path).suffix.lower()
    return ext in FIFLoader.EXTENSIONS or ext in H5Loader.EXTENSIONS
