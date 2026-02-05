"""Dynamic MOABB dataset discovery.

Generates DatasetConfig for all MOABB datasets at runtime instead of
hardcoding presets. This provides access to all 36+ MOABB datasets.
"""

import logging
from typing import Any

from .config import DatasetConfig

logger = logging.getLogger(__name__)

# Module-level cache for discovered datasets
_dataset_cache: list[DatasetConfig] | None = None

# Map MOABB paradigm strings to our paradigm class names
_PARADIGM_MAP = {
    "imagery": "MotorImagery",
    "p300": "P300",
    "ssvep": "SSVEP",
    "cvep": "CVEP",
    "rstate": "RestingState",
}


def discover_moabb_datasets(
    paradigm_filter: str | None = None,
    cache: bool = True,
) -> list[DatasetConfig]:
    """Discover all available MOABB datasets dynamically.

    Args:
        paradigm_filter: Optional filter by paradigm ('imagery', 'p300', etc.)
        cache: Whether to cache results (default True)

    Returns:
        List of DatasetConfig objects for each MOABB dataset
    """
    global _dataset_cache

    try:
        from moabb.datasets import utils as dataset_utils
    except ImportError:
        logger.warning("MOABB not installed, no datasets available")
        return []

    if cache and _dataset_cache is not None:
        if paradigm_filter:
            return [c for c in _dataset_cache if c.moabb_paradigm == _PARADIGM_MAP.get(paradigm_filter)]
        return _dataset_cache

    configs = []
    for ds_class in dataset_utils.dataset_list:
        try:
            # Get paradigm from class attribute - no instantiation needed
            paradigm = getattr(ds_class, "paradigm", "imagery")

            # Skip if filtering and doesn't match
            if paradigm_filter and paradigm != paradigm_filter:
                continue

            # Map to our paradigm name
            paradigm_name = _PARADIGM_MAP.get(paradigm, "MotorImagery")

            # Try to get metadata from class attributes (no instantiation)
            # If not available, will be loaded on click via load_moabb_dataset_details()
            subjects = []
            if hasattr(ds_class, "subject_list"):
                try:
                    subjects = list(ds_class.subject_list)
                except (TypeError, AttributeError):
                    pass

            events = {}
            if hasattr(ds_class, "event_id"):
                try:
                    event_id = ds_class.event_id
                    if isinstance(event_id, dict):
                        events = dict(event_id)
                    elif isinstance(event_id, (list, tuple)):
                        events = {str(e): i for i, e in enumerate(event_id)}
                except (TypeError, AttributeError):
                    pass

            interval = getattr(ds_class, "interval", None)
            epoch_tmin = interval[0] if interval else 0.0
            epoch_tmax = interval[1] if interval else 4.0

            n_subjects = len(subjects) if subjects else "?"
            description = f"MOABB {paradigm} dataset with {n_subjects} subjects"

            config = DatasetConfig(
                name=ds_class.__name__,
                description=description,
                source_type="moabb",
                moabb_dataset=ds_class.__name__,
                moabb_paradigm=paradigm_name,
                events=events,
                subjects=subjects,
                epoch_tmin=epoch_tmin,
                epoch_tmax=epoch_tmax,
                sample_rate=0,
            )
            configs.append(config)

        except Exception as e:
            logger.debug(f"Could not load MOABB dataset {ds_class.__name__}: {e}")
            continue

    if cache:
        _dataset_cache = configs

    logger.info(f"Discovered {len(configs)} MOABB datasets")
    return configs


def get_moabb_dataset_info(name: str) -> dict[str, Any] | None:
    """Get info about a specific MOABB dataset.

    Args:
        name: Dataset class name (e.g., 'BNCI2014_001')

    Returns:
        Dict with dataset info or None if not found
    """
    for config in discover_moabb_datasets():
        if config.name == name or config.moabb_dataset == name:
            return {
                "name": config.name,
                "paradigm": config.moabb_paradigm,
                "n_subjects": len(config.subjects),
                "events": config.events,
                "config": config,
            }
    return None


def load_moabb_dataset_details(config: DatasetConfig) -> None:
    """Load subjects/events/interval for a BIDS dataset on-demand."""
    if config.subjects:
        return  # Already loaded

    try:
        from moabb.datasets import utils as dataset_utils

        for ds_class in dataset_utils.dataset_list:
            if ds_class.__name__ == config.moabb_dataset:
                ds = ds_class()
                config.subjects = list(ds.subject_list) if hasattr(ds, "subject_list") else []
                if hasattr(ds, "event_id") and ds.event_id:
                    if isinstance(ds.event_id, dict):
                        config.events = dict(ds.event_id)
                    elif isinstance(ds.event_id, (list, tuple)):
                        config.events = {str(e): i for i, e in enumerate(ds.event_id)}
                if hasattr(ds, "interval") and ds.interval:
                    config.epoch_tmin = ds.interval[0]
                    config.epoch_tmax = ds.interval[1]
                break
    except Exception as e:
        logger.warning(f"Could not load details for {config.moabb_dataset}: {e}")
