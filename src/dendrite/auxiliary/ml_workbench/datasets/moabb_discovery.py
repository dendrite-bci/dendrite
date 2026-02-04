"""Dynamic MOABB dataset discovery.

Generates DatasetConfig for all MOABB datasets at runtime instead of
hardcoding presets. This provides access to all 36+ MOABB datasets.
"""

import logging
from typing import Any

from .config import DatasetConfig

logger = logging.getLogger(__name__)


# MOABB doesn't expose a "check if cached" API - accessing data triggers downloads.
# To display sampling rates without downloading, we read MNE's cache directly:
#
# 1. _build_srate_cache() scans ~/mne_data/ once at startup (in background thread)
# 2. For each directory, reads ONE file header using MNE (preload=False = fast)
# 3. _get_cached_sampling_rate() does instant dict lookup when user clicks dataset
#
# This moves all file I/O to startup (~1.3s) so UI clicks are instant (<0.01ms).
# Returns 0 Hz for uncached datasets - user must download data first.

_SRATE_CACHE = {}
_SRATE_CACHE_BUILT = False


def _build_srate_cache() -> None:
    """Pre-scan all cache directories once to build sampling rate cache."""
    global _SRATE_CACHE_BUILT
    if _SRATE_CACHE_BUILT:
        return

    from pathlib import Path

    import mne

    mne_data = Path(mne.get_config("MNE_DATA", str(Path.home() / "mne_data")))
    if not mne_data.exists():
        _SRATE_CACHE_BUILT = True
        return

    for cache_dir in mne_data.iterdir():
        if not cache_dir.is_dir() or cache_dir.name.startswith("."):
            continue

        dir_key = cache_dir.name.lower()

        # Try MNE readers directly
        f = next(cache_dir.rglob("*.edf"), None)
        if f:
            try:
                _SRATE_CACHE[dir_key] = mne.io.read_raw_edf(f, preload=False, verbose=False).info[
                    "sfreq"
                ]
                continue
            except (OSError, ValueError, RuntimeError, KeyError) as e:
                logger.debug(f"Could not read EDF {f.name}: {e}")

        f = next(cache_dir.rglob("*.fif"), None)
        if f:
            try:
                _SRATE_CACHE[dir_key] = mne.io.read_info(f, verbose=False)["sfreq"]
                continue
            except (OSError, ValueError, RuntimeError, KeyError) as e:
                logger.debug(f"Could not read FIF {f.name}: {e}")

        f = next(cache_dir.rglob("*.gdf"), None)
        if f:
            try:
                _SRATE_CACHE[dir_key] = mne.io.read_raw_gdf(f, preload=False, verbose=False).info[
                    "sfreq"
                ]
                continue
            except (OSError, ValueError, RuntimeError, KeyError) as e:
                logger.debug(f"Could not read GDF {f.name}: {e}")

        # MAT files
        f = next(cache_dir.rglob("*.mat"), None)
        if f:
            try:
                import scipy.io as sio

                mat = sio.loadmat(str(f))
                for v in mat.values():
                    if hasattr(v, "dtype") and v.dtype == object and v.size:
                        elem = v.flat[0]
                        if hasattr(elem, "dtype") and elem.dtype.names and "fs" in elem.dtype.names:
                            _SRATE_CACHE[dir_key] = float(elem["fs"].flat[0])
                            break
            except (OSError, ValueError, KeyError) as e:
                logger.debug(f"Could not read MAT {f.name}: {e}")

    _SRATE_CACHE_BUILT = True


def _get_cached_sampling_rate(ds_code: str) -> float:
    """Get sampling rate from pre-built cache."""
    _build_srate_cache()

    # MOABB dataset codes don't always match MNE cache directory names
    # e.g., PhysionetMI -> MNE-eegbci-data, Cho2017 -> MNE-gigadb-data
    CODE_FIXES = {"physionet": "eegbci", "alexandre": "alex", "cho2017": "gigadb"}

    # Match first 4 chars of normalized code (or mapped name) against cache dir names
    # e.g., "BNCI2014_001" -> "bnci" matches "mne-bnci-data"
    code = ds_code.lower().replace("-", "").replace("_", "")
    search = CODE_FIXES.get(code[:9], code[:4])

    for dir_key, srate in _SRATE_CACHE.items():
        if search in dir_key:
            return srate
    return 0


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
    try:
        from moabb.datasets import utils as dataset_utils
    except ImportError:
        logger.warning("MOABB not installed, no datasets available")
        return []

    if cache and hasattr(discover_moabb_datasets, "_cache"):
        configs = discover_moabb_datasets._cache
        if paradigm_filter:
            return [c for c in configs if c.moabb_paradigm == _PARADIGM_MAP.get(paradigm_filter)]
        return configs

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
        discover_moabb_datasets._cache = configs

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


def list_moabb_paradigms() -> list[str]:
    """List available MOABB paradigm types."""
    return list(_PARADIGM_MAP.keys())
