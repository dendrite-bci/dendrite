"""Wrapper to use internal FIF datasets with MOABB evaluation.

This allows internal datasets to be evaluated using MOABB's standardized
evaluation methods (WithinSession, CrossSession, CrossSubject).
"""

import logging
import re

from moabb.datasets.base import BaseDataset

from .config import DatasetConfig
from .loader import DataLoader


def _to_camel_kebab(name: str) -> str:
    """Convert name to MOABB-compatible code that abbreviates class name.

    MOABB requires the class name to be an abbreviation of the code.
    For InternalDatasetWrapper, we prefix with 'Internal-Dataset-Wrapper-'
    so the class name letters appear in order in the code.
    """
    # Remove special characters except spaces and dashes
    cleaned = re.sub(r"[^a-zA-Z0-9\s\-]", "", name)
    # Split on spaces, underscores, dashes
    parts = re.split(r"[\s_\-]+", cleaned)
    # Capitalize each part
    camel = "".join(part.capitalize() for part in parts if part)
    # Prefix ensures InternalDatasetWrapper is abbreviation of code
    return f"Internal-Dataset-Wrapper-{camel or 'Data'}"


class InternalDatasetWrapper(BaseDataset):
    """Wrap internal FIF data as MOABB-compatible dataset.

    This enables using MOABB's evaluation framework on internal datasets,
    providing standardized, reproducible evaluation methodology.

    Example:
        loader = DataLoader(config)
        dataset = InternalDatasetWrapper(loader, config)
        evaluation = WithinSessionEvaluation(paradigm, [dataset])
        results = evaluation.process(pipelines)
    """

    def __init__(self, loader: DataLoader, config: DatasetConfig):
        """Initialize wrapper with existing loader and config.

        Args:
            loader: DataLoader instance for loading FIF files
            config: Dataset configuration
        """
        self._loader = loader
        self._config = config

        # Map config events to MOABB format
        events = config.events or {"Target": 1, "NonTarget": 0}

        # Determine paradigm from config
        paradigm = self._detect_paradigm(config)

        # Suppress MOABB's cosmetic abbreviation warning during init
        # MOABB requires class name to be abbreviation of code, but our wrapper
        # name doesn't follow this pattern. The warning is harmless.
        moabb_logger = logging.getLogger("moabb.datasets.base")
        original_level = moabb_logger.level
        moabb_logger.setLevel(logging.ERROR)

        try:
            super().__init__(
                subjects=config.subjects,
                sessions_per_subject=1,  # Internal data typically single session
                events=events,
                code=_to_camel_kebab(config.name),  # Convert to valid Camel-KebabCase
                interval=[config.epoch_tmin, config.epoch_tmax],
                paradigm=paradigm,
            )
        finally:
            moabb_logger.setLevel(original_level)

    def _detect_paradigm(self, config: DatasetConfig) -> str:
        """Detect MOABB paradigm from config."""
        events = config.events or {}
        event_names = [k.lower() for k in events.keys()]

        # Check for P300 indicators
        if any(n in event_names for n in ["target", "nontarget", "non_target"]):
            return "p300"

        # Check for motor imagery indicators
        if any(n in event_names for n in ["left", "right", "hand", "feet"]):
            return "imagery"

        # Check for ErrP indicators
        if any(n in event_names for n in ["error", "correct", "errp"]):
            return "imagery"  # ErrP uses imagery paradigm structure

        return "imagery"  # Default

    def _get_single_subject_data(self, subject):
        """Return data structured as MOABB expects.

        Returns:
            Dict with structure: {'session_id': {'run_id': mne.io.Raw}}

        Note:
            Scales EEG data from V to μV (1e6) to match MOABB's internal scaling.
            This ensures models trained on MOABB data can be compared to internal data.
        """
        try:
            raw = self._loader.load_raw(subject)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not load data for subject {subject}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Error loading data for subject {subject}: {e}") from e

        # Scale EEG from V to μV to match MOABB paradigm preprocessing
        # MOABB paradigms apply 1e6 scaling internally (see moabb_loader.py:170)
        raw = raw.copy()
        raw.apply_function(lambda x: x * 1e6, picks="eeg")

        return {"0": {"0": raw}}

    def data_path(self, subject, path=None, force_update=False, update_path=None, verbose=None):
        """Return paths to subject's data files.

        Required by MOABB but we don't need download functionality
        since data is already local.
        """
        try:
            fif_path = self._loader.get_fif_path(subject)
            return [str(fif_path)]
        except FileNotFoundError:
            return []
