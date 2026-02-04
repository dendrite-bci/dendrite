"""FIF file data loader.

Loads data from a FIF file with event mapping from JSON.
"""

import json
import logging
from pathlib import Path
from typing import Any

import mne
import numpy as np

from ._event_utils import apply_preprocessing, create_epochs, encode_labels, filter_events_by_codes
from .base_loader import BaseLoader
from .config import DatasetConfig

logger = logging.getLogger(__name__)


class FIFLoader(BaseLoader):
    """Loader for FIF file datasets.

    Handles custom datasets with a FIF file and event mapping from JSON.
    """

    def __init__(
        self,
        config: DatasetConfig,
        file_path: str,
        event_mapping: dict[str, int] | None = None,
        preproc_overrides: dict | None = None,
    ):
        """Initialize loader with configuration.

        Args:
            config: Dataset configuration
            file_path: Path to the FIF file
            event_mapping: Event name->code mapping
            preproc_overrides: Optional preprocessing overrides
        """
        super().__init__(config)
        self._file_path = Path(file_path)
        self._event_mapping = event_mapping or {}
        self._preproc_overrides = preproc_overrides or {}
        self._raw_cache: dict[tuple, mne.io.Raw] = {}

    @classmethod
    def from_dataset_info(cls, config: DatasetConfig, dataset_info: dict) -> "FIFLoader":
        """Create loader from database dataset_info dict.

        Args:
            config: Dataset configuration
            dataset_info: Dict from datasets table with file_path, events_json, etc.

        Returns:
            FIFLoader configured for the dataset
        """
        file_path = dataset_info.get("file_path")
        if not file_path:
            raise ValueError("No file_path in dataset_info")

        # Parse event mapping from JSON
        event_mapping = None
        events_json = dataset_info.get("events_json")
        if events_json:
            event_mapping = json.loads(events_json)

        # Collect preprocessing overrides
        preproc_overrides = {
            "lowcut": dataset_info.get("preproc_lowcut"),
            "highcut": dataset_info.get("preproc_highcut"),
            "rereference": dataset_info.get("preproc_rereference", False),
            "target_sample_rate": dataset_info.get("target_sample_rate"),
            "modality": dataset_info.get("modality"),
            "sampling_rate": dataset_info.get("sampling_rate", 250.0),
            "epoch_tmin": dataset_info.get("epoch_tmin"),
            "epoch_tmax": dataset_info.get("epoch_tmax"),
        }

        return cls(config, file_path, event_mapping=event_mapping, preproc_overrides=preproc_overrides)

    # Path resolution methods

    def get_fif_path(
        self,
        subject_id: int,
        session: str | None = None,
        run: str | None = None,
    ) -> Path:
        """Return the single file path (ignores arguments).

        Args:
            subject_id: Ignored (single file)
            session: Ignored (single file)
            run: Ignored (single file)

        Returns:
            Path to FIF file
        """
        return self._file_path

    def get_subject_list(self) -> list[int]:
        """Single file is treated as subject 1.

        Returns:
            List containing only subject 1
        """
        return [1]

    def get_sample_rate(self) -> float:
        """Get effective sample rate (target if resampled, else original).

        Returns:
            Sample rate in Hz
        """
        target = self._preproc_overrides.get("target_sample_rate")
        if target:
            return float(target)
        sampling_rate = self._preproc_overrides.get("sampling_rate")
        if sampling_rate:
            return float(sampling_rate)
        return self.config.sample_rate

    # Helper methods

    def _get_preproc_params(self) -> tuple[float | None, float | None, bool, float | None]:
        """Get preprocessing parameters (with overrides applied).

        Returns:
            (lowcut, highcut, rereference, target_sample_rate)
        """
        lowcut = self._preproc_overrides.get("lowcut") or self.config.preproc_lowcut
        highcut = self._preproc_overrides.get("highcut") or self.config.preproc_highcut
        rereference = self._preproc_overrides.get("rereference") or self.config.preproc_rereference
        target_sample_rate = self._preproc_overrides.get("target_sample_rate")
        return lowcut, highcut, rereference, target_sample_rate

    def _get_channel_picks(self) -> str:
        """Get channel pick string for data extraction.

        Returns:
            Channel type string (e.g., 'eeg')
        """
        modality = self._preproc_overrides.get("modality")
        if modality:
            return modality.lower()
        return self.config.channels if self.config.channels else "eeg"

    def _get_epoch_window(self) -> tuple[float, float]:
        """Get epoch window (tmin, tmax) from config or overrides.

        Returns:
            (tmin, tmax) tuple
        """
        tmin = self._preproc_overrides.get("epoch_tmin") or self.config.epoch_tmin or -0.2
        tmax = self._preproc_overrides.get("epoch_tmax") or self.config.epoch_tmax or 0.8
        return tmin, tmax

    # Data loading methods

    def load_raw(
        self,
        subject_id: int,
        preprocess: bool = True,
        session: str | None = None,
        run: str | None = None,
    ) -> mne.io.Raw:
        """Load raw data with optional preprocessing.

        Args:
            subject_id: Ignored (single file = single subject)
            preprocess: Apply bandpass filtering if configured
            session: Ignored (single file)
            run: Ignored (single file)

        Returns:
            Raw MNE object
        """
        cache_key = (subject_id, preprocess)
        if cache_key in self._raw_cache:
            return self._raw_cache[cache_key]

        raw = mne.io.read_raw_fif(self._file_path, preload=True, verbose=False)

        if preprocess:
            lowcut, highcut, rereference, target_sample_rate = self._get_preproc_params()
            if lowcut or highcut or target_sample_rate:
                raw = apply_preprocessing(raw, lowcut, highcut, rereference, target_sample_rate)

        self._raw_cache[cache_key] = raw
        return raw

    def load_epochs(
        self,
        subject_id: int,
        block: int | None = None,
        session: str | None = None,
        run: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load epoched data from FIF file.

        Args:
            subject_id: Ignored (single file = single subject)
            block: Ignored (single file uses event_mapping directly)
            session: Ignored (single file)
            run: Ignored (single file)

        Returns:
            X: (n_epochs, n_channels, n_times) - epoched data
            y: (n_epochs,) - integer labels
        """
        logger.info(f"Loading custom FIF: {self._file_path}")

        raw = self.load_raw(subject_id, preprocess=True)
        mne_events, mne_event_id = mne.events_from_annotations(raw)

        selected_codes = set(self._event_mapping.values())
        filtered_events = filter_events_by_codes(mne_events, selected_codes)

        if len(filtered_events) == 0:
            raise ValueError(
                f"No matching events. File has: {list(mne_event_id.values())}, wanted: {selected_codes}"
            )

        tmin, tmax = self._get_epoch_window()
        picks = self._get_channel_picks()
        epochs = create_epochs(raw, filtered_events, selected_codes, tmin, tmax, picks=picks)

        X = epochs.get_data()
        y = encode_labels(epochs.events[:, 2], self._event_mapping)

        logger.info(f"Loaded {X.shape[0]} epochs, shape: {X.shape}, classes: {np.unique(y)}")
        return X, y

    def load_continuous(
        self,
        subject_id: int,
        block: int | None = None,
        session: str | None = None,
        run: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
        """Load continuous data for sliding window evaluation.

        Args:
            subject_id: Ignored (single file = single subject)
            block: Ignored (single file uses event_mapping directly)
            session: Ignored (single file)
            run: Ignored (single file)

        Returns:
            data: (n_channels, n_samples) - continuous EEG
            event_times: (n_events,) - sample indices of events
            event_labels: (n_events,) - integer labels
            event_mapping: Dict mapping event names to integer labels
        """
        raw = self.load_raw(subject_id, preprocess=True)
        mne_events, _ = mne.events_from_annotations(raw, verbose=False)

        selected_codes = set(self._event_mapping.values())
        filtered = filter_events_by_codes(mne_events, selected_codes)

        picks = self._get_channel_picks()
        raw_picked = raw.copy().pick(picks)
        data = raw_picked.get_data()

        event_times = filtered[:, 0] if len(filtered) > 0 else np.array([])
        event_labels = (
            encode_labels(filtered[:, 2], self._event_mapping) if len(filtered) > 0 else np.array([])
        )

        if len(event_times) == 0:
            logger.warning(
                "No events found. Check event configuration matches annotations in data."
            )

        return data, event_times, event_labels, self._event_mapping

    def load_data_split(
        self,
        subject_id: int = 1,
        block: int = 1,
        val_ratio: float = 0.3,
    ) -> tuple[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]],
        dict[str, Any],
    ]:
        """Load data split into train epochs and validation continuous.

        Args:
            subject_id: Ignored (single file = single subject)
            block: Ignored (single file uses event_mapping directly)
            val_ratio: Fraction of events for validation (default 0.3)

        Returns:
            train_data: (X_train, y_train) - epochs for training
            val_data: (continuous, event_times, event_labels, event_mapping) - for async eval
            split_info: Metadata about split method
        """
        raw = self.load_raw(subject_id, preprocess=True)
        mne_events, _ = mne.events_from_annotations(raw, verbose=False)

        selected_codes = set(self._event_mapping.values())
        filtered = filter_events_by_codes(mne_events, selected_codes)
        # Sort by time
        filtered = filtered[filtered[:, 0].argsort()]

        # Split temporally
        n_val = int(len(filtered) * val_ratio)
        if n_val == 0 or n_val >= len(filtered):
            raise ValueError(f"Not enough events to split: {len(filtered)} total")

        train_events = filtered[:-n_val]
        val_events = filtered[-n_val:]

        # Training epochs
        tmin, tmax = self._get_epoch_window()
        picks = self._get_channel_picks()
        epochs = create_epochs(raw, train_events, selected_codes, tmin, tmax, picks=picks)
        X_train = epochs.get_data()
        y_train = encode_labels(epochs.events[:, 2], self._event_mapping)

        # Validation continuous
        split_sample = val_events[0, 0]
        buffer = int(tmax * self.get_sample_rate()) + 100
        val_start = max(0, split_sample - buffer)

        raw_picked = raw.copy().pick(picks)
        val_continuous = raw_picked.get_data()[:, val_start:]
        val_times = val_events[:, 0] - val_start
        val_labels = encode_labels(val_events[:, 2], self._event_mapping)

        n_train = len(train_events)
        split_info: dict[str, Any] = {
            "method": "temporal",
            "val_ratio": val_ratio,
            "n_train": n_train,
            "n_val": n_val,
        }

        logger.info(f"Split data: {n_train} train, {n_val} val events")
        return (X_train, y_train), (val_continuous, val_times, val_labels, self._event_mapping), split_info

    def get_n_times(self, subject_id: int = 1) -> int:
        """Get number of time samples per epoch.

        Args:
            subject_id: Ignored (single file)

        Returns:
            Number of time samples
        """
        # Load epochs to get actual shape
        X, _ = self.load_epochs(subject_id)
        return X.shape[2]
