"""Dataset management for offline ML."""

from .cache import DataCache
from .config import DatasetConfig
from .custom_loader import CustomFIFLoader
from .internal_moabb_wrapper import InternalDatasetWrapper
from .loader import DataLoader
from .moabb_discovery import (
    discover_moabb_datasets,
    get_moabb_dataset_info,
    list_moabb_paradigms,
    load_moabb_dataset_details,
)
from .moabb_loader import MOAABLoader
from .study_item import StudyItem

__all__ = [
    "DatasetConfig",
    "DataLoader",
    "DataCache",
    "StudyItem",
    # MOABB integration
    "MOAABLoader",
    "discover_moabb_datasets",
    "get_moabb_dataset_info",
    "list_moabb_paradigms",
    "load_moabb_dataset_details",
    "InternalDatasetWrapper",
    # Custom/Internal loaders
    "CustomFIFLoader",
]
