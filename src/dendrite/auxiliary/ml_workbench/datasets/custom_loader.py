"""Loader for custom FIF/SET datasets.

Loads data from user-added FIF files stored in the datasets table.
"""

import json
import logging

import mne
import numpy as np

from ._event_utils import apply_preprocessing, create_epochs, encode_labels, filter_events_by_codes
from .config import DatasetConfig

logger = logging.getLogger(__name__)


class CustomFIFLoader:
    """Load epoched data from custom FIF/SET files.

    Example:
        loader = CustomFIFLoader(config, dataset_info)
        X, y = loader.load_epochs(subject_id=1)  # subject_id ignored for single-file
    """

    def __init__(self, config: DatasetConfig, dataset_info: dict):
        """Initialize loader.

        Args:
            config: DatasetConfig with epoch settings
            dataset_info: Dict from datasets table with file_path, events_json, etc.
        """
        self.config = config
        self.dataset_info = dataset_info
        self._epoch_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self._raw_cache: dict[tuple, mne.io.Raw] = {}

    def get_subject_list(self) -> list[int]:
        """Custom FIF has single 'subject'."""
        return [1]

    def get_fif_path(self, subject_id: int = 1) -> str:
        """Get file path for MOABB compatibility."""
        file_path = self.dataset_info.get("file_path")
        if not file_path:
            raise FileNotFoundError("No file_path in dataset_info")
        return file_path

    def load_raw(self, subject_id: int = 1, preprocess: bool = True) -> mne.io.Raw:
        """Load raw data with optional preprocessing.

        Args:
            subject_id: Ignored (single file = single subject)
            preprocess: Apply bandpass filtering if configured

        Returns:
            Raw MNE object
        """
        cache_key = (subject_id, preprocess)
        if cache_key in self._raw_cache:
            return self._raw_cache[cache_key]

        file_path = self.dataset_info.get("file_path")
        if not file_path:
            raise ValueError("No file_path in dataset_info")

        raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)

        if preprocess:
            lowcut = self.dataset_info.get("preproc_lowcut") or self.config.preproc_lowcut
            highcut = self.dataset_info.get("preproc_highcut") or self.config.preproc_highcut
            rereference = (
                self.dataset_info.get("preproc_rereference", False)
                or self.config.preproc_rereference
            )
            target_sample_rate = self.dataset_info.get("target_sample_rate")
            if lowcut or highcut or target_sample_rate:
                raw = apply_preprocessing(raw, lowcut, highcut, rereference, target_sample_rate)

        self._raw_cache[cache_key] = raw
        return raw

    def _get_event_mapping(self) -> dict[str, int]:
        """Get event mapping from dataset_info."""
        events_json = self.dataset_info.get("events_json")
        if not events_json:
            raise ValueError("No events configured")
        return json.loads(events_json)

    def _get_epoch_window(self) -> tuple[float, float]:
        """Get epoch window (tmin, tmax) from config."""
        tmin = self.dataset_info.get("epoch_tmin", self.config.epoch_tmin or -0.2)
        tmax = self.dataset_info.get("epoch_tmax", self.config.epoch_tmax or 0.8)
        return tmin, tmax

    def _get_channel_picks(self) -> str:
        """Get channel pick string for data extraction.

        Matches create_epochs behavior for consistent channel selection.
        """
        # Check dataset_info for modality (eeg, emg, etc.)
        modality = self.dataset_info.get("modality")
        if modality:
            return modality.lower()
        # Fall back to config channels, then default 'eeg'
        return self.config.channels if self.config.channels else "eeg"

    def load_epochs(
        self,
        subject_id: int = 1,
        block: int | None = None,
        events: list[str] | None = None,
        preprocess: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load epoched data from FIF file."""
        cache_key = (subject_id, preprocess)
        if cache_key in self._epoch_cache:
            return self._epoch_cache[cache_key]

        logger.info(f"Loading custom FIF: {self.dataset_info.get('file_path')}")

        raw = self.load_raw(subject_id, preprocess)
        event_mapping = self._get_event_mapping()
        mne_events, mne_event_id = mne.events_from_annotations(raw)

        selected_codes = set(event_mapping.values())
        filtered_events = filter_events_by_codes(mne_events, selected_codes)

        if len(filtered_events) == 0:
            raise ValueError(
                f"No matching events. File has: {list(mne_event_id.values())}, wanted: {selected_codes}"
            )

        tmin, tmax = self._get_epoch_window()
        picks = self._get_channel_picks()
        epochs = create_epochs(raw, filtered_events, selected_codes, tmin, tmax, picks=picks)

        X = epochs.get_data()
        y = encode_labels(epochs.events[:, 2], event_mapping)

        logger.info(f"Loaded {X.shape[0]} epochs, shape: {X.shape}, classes: {np.unique(y)}")

        self._epoch_cache[cache_key] = (X, y)
        return X, y

    def load_continuous(
        self,
        subject_id: int = 1,
        block: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load continuous data for sliding window evaluation.

        Returns:
            data: (n_channels, n_samples) - continuous EEG
            event_times: (n_events,) - sample indices of events
            event_labels: (n_events,) - integer labels
        """
        raw = self.load_raw(subject_id, preprocess=True)
        event_mapping = self._get_event_mapping()
        mne_events, _ = mne.events_from_annotations(raw, verbose=False)

        selected_codes = set(event_mapping.values())
        filtered = filter_events_by_codes(mne_events, selected_codes)

        # Pick channels matching load_epochs behavior
        picks = self._get_channel_picks()
        raw_picked = raw.copy().pick(picks)
        data = raw_picked.get_data()
        event_times = filtered[:, 0] if len(filtered) > 0 else np.array([])
        event_labels = (
            encode_labels(filtered[:, 2], event_mapping) if len(filtered) > 0 else np.array([])
        )

        if len(event_times) == 0:
            logger.warning(
                f"No events found for subject {subject_id}. "
                f"Check event configuration matches annotations in data."
            )

        return data, event_times, event_labels

    def load_data_split(
        self,
        subject_id: int = 1,
        block: int = 1,
        val_ratio: float = 0.3,
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Load data split into train epochs and validation continuous.

        Returns:
            train_data: (X_train, y_train) - epochs for training
            val_data: (continuous, event_times, event_labels) - for async eval
        """
        raw = self.load_raw(subject_id, preprocess=True)
        event_mapping = self._get_event_mapping()
        mne_events, _ = mne.events_from_annotations(raw, verbose=False)

        selected_codes = set(event_mapping.values())
        filtered = filter_events_by_codes(mne_events, selected_codes)
        # Sort by time
        filtered = filtered[filtered[:, 0].argsort()]

        # Split temporally
        n_val = int(len(filtered) * val_ratio)
        if n_val == 0 or n_val >= len(filtered):
            raise ValueError(f"Not enough events to split: {len(filtered)} total")

        train_events = filtered[:-n_val]
        val_events = filtered[-n_val:]

        # Training epochs using create_epochs (picks must match validation)
        tmin, tmax = self._get_epoch_window()
        picks = self._get_channel_picks()
        epochs = create_epochs(raw, train_events, selected_codes, tmin, tmax, picks=picks)
        X_train = epochs.get_data()
        y_train = encode_labels(epochs.events[:, 2], event_mapping)

        # Validation continuous
        split_sample = val_events[0, 0]
        buffer = int(tmax * self.get_sample_rate()) + 100
        val_start = max(0, split_sample - buffer)

        # Pick channels matching load_epochs behavior
        picks = self._get_channel_picks()
        raw_picked = raw.copy().pick(picks)
        val_continuous = raw_picked.get_data()[:, val_start:]
        val_times = val_events[:, 0] - val_start
        val_labels = encode_labels(val_events[:, 2], event_mapping)

        logger.info(f"Split data: {len(train_events)} train, {len(val_events)} val events")
        return (X_train, y_train), (val_continuous, val_times, val_labels)

    def get_n_channels(self, subject_id: int = 1) -> int:
        X, _ = self.load_epochs(subject_id)
        return X.shape[1]

    def get_n_times(self, subject_id: int = 1) -> int:
        X, _ = self.load_epochs(subject_id)
        return X.shape[2]

    def get_sample_rate(self) -> float:
        """Get effective sample rate (target if resampled, else original)."""
        target = self.dataset_info.get("target_sample_rate")
        if target:
            return float(target)
        return self.dataset_info.get("sampling_rate", 250.0)
