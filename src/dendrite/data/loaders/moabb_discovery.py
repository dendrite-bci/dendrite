"""Dynamic MOABB dataset discovery.

Scans all MOABB datasets at runtime, returns plain dicts.
Instantiates dataset objects (no download) for accurate metadata.
"""

import logging
from typing import Any

from .moabb_loader import MoabbConfig

logger = logging.getLogger(__name__)

_dataset_cache: list[dict[str, Any]] | None = None
_paradigm_filters_cache: dict[str, list[float] | None] = {}

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
) -> list[dict[str, Any]]:
    """Discover all available MOABB datasets. Returns list of info dicts."""
    global _dataset_cache

    try:
        from moabb.datasets import utils as dataset_utils
    except ImportError:
        logger.warning("MOABB not installed, no datasets available")
        return []

    if cache and _dataset_cache is not None:
        if paradigm_filter:
            return [d for d in _dataset_cache if d["raw_paradigm"] == paradigm_filter]
        return _dataset_cache

    results: list[dict[str, Any]] = []
    for ds_class in dataset_utils.dataset_list:
        try:
            # Instantiate for accurate metadata (no download triggered)
            ds = ds_class()

            paradigm = getattr(ds, "paradigm", "imagery")
            if paradigm_filter and paradigm != paradigm_filter:
                continue

            paradigm_name = _PARADIGM_MAP.get(paradigm, paradigm.title())
            subjects = list(getattr(ds, "subject_list", []) or [])

            events: dict[str, int] = {}
            event_id = getattr(ds, "event_id", None)
            if isinstance(event_id, dict):
                events = dict(event_id)

            interval = getattr(ds, "interval", None)

            # Get paradigm preprocessing info (cached per paradigm type)
            paradigm_filters = _paradigm_filters_cache.get(paradigm_name)
            if paradigm_filters is None:
                try:
                    from .moabb_loader import _get_moabb_paradigm
                    p = _get_moabb_paradigm(paradigm_name)
                    filters = getattr(p, "filters", [])
                    paradigm_filters = filters[0] if filters else None
                except Exception:
                    paradigm_filters = None
                _paradigm_filters_cache[paradigm_name] = paradigm_filters

            results.append({
                "code": ds_class.__name__,
                "name": ds_class.__name__,
                "paradigm": paradigm_name,
                "raw_paradigm": paradigm,
                "n_subjects": len(subjects),
                "subjects": subjects,
                "events": events,
                "interval": [interval[0], interval[1]] if interval else [0.0, 4.0],
                "paradigm_bandpass": paradigm_filters,
            })
        except Exception as e:
            logger.debug(f"Could not load MOABB dataset {ds_class.__name__}: {e}")

    if cache:
        _dataset_cache = results

    logger.info(f"Discovered {len(results)} MOABB datasets")
    return results


def get_moabb_dataset_info(name: str) -> dict[str, Any] | None:
    """Get info + MoabbConfig for a specific MOABB dataset."""
    for ds in discover_moabb_datasets():
        if ds["code"] == name:
            return {
                **ds,
                "config": MoabbConfig(
                    dataset=ds["code"],
                    paradigm=ds["paradigm"],
                    events=ds["events"],
                ),
            }
    return None
