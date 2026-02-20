"""Base classes for data loaders.

Defines the abstract interface and shared file-based loader logic.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import mne
import numpy as np

from ._event_utils import apply_preprocessing, create_epochs, encode_labels, filter_events_by_codes
from .config import DatasetConfig

logger = logging.getLogger(__name__)


def build_label_info(events: dict[str, int]) -> tuple[dict[int, str], dict[str, int]]:
    """Build (event_mapping, label_mapping) from a {name: code} dict.

    Args:
        events: Mapping of event names to integer codes, e.g. {'left': 1, 'right': 2}.

    Returns:
        event_mapping: {code: name} reverse lookup.
        label_mapping: {name: class_index} sorted by code value to match encode_labels() ordering.
    """
    if not events:
        return {}, {}
    event_mapping = {code: name for name, code in events.items()}
    label_mapping = {
        name: idx for idx, (name, _) in enumerate(sorted(events.items(), key=lambda x: x[1]))
    }
    return event_mapping, label_mapping


class BaseLoader(ABC):
    """Abstract base class for EEG data loaders.

    Defines the common interface for loading raw data, epochs,
    and continuous data from different data sources.
    """

    def __init__(self, config: DatasetConfig):
        """Initialize loader with configuration."""
        self.config = config

    @abstractmethod
    def get_subject_list(self) -> list[int]:
        """Get list of available subject IDs."""
        ...

    @abstractmethod
    def get_sample_rate(self) -> float:
        """Get sample rate in Hz."""
        ...

    @abstractmethod
    def load_raw(
        self,
        subject_id: int,
        preprocess: bool = True,
        session: str | None = None,
        run: str | None = None,
    ) -> mne.io.Raw:
        """Load raw MNE object with optional preprocessing."""
        ...

    @abstractmethod
    def load_epochs(
        self,
        subject_id: int,
        block: int | None = None,
        session: str | None = None,
        run: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load epoched data (n_epochs, n_channels, n_times) and labels."""
        ...

    @abstractmethod
    def load_continuous(
        self,
        subject_id: int,
        block: int | None = None,
        session: str | None = None,
        run: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
        """Load continuous data (n_channels, n_samples), event times, labels, and mapping."""
        ...

    @abstractmethod
    def load_data_split(
        self,
        subject_id: int,
        block: int = 1,
        val_ratio: float = 0.3,
    ) -> tuple[
        tuple[np.ndarray, np.ndarray],  # train: (X, y)
        tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]],  # val: (cont, times, labels, mapping)
        dict[str, Any],  # split_info
    ]:
        """Load train epochs and validation continuous data with split metadata."""
        ...

    @abstractmethod
    def get_label_info(self) -> tuple[dict[int, str], dict[str, int]]:
        """Return (event_mapping, label_mapping) for decoder configuration.

        Returns:
            event_mapping: {event_code: event_name} e.g. {1: 'left', 2: 'right'}
            label_mapping: {event_name: class_index} e.g. {'left': 0, 'right': 1}
        """
        ...

    def _get_channel_picks(self) -> str:
        """Get channel pick string (e.g., 'eeg') from config."""
        return self.config.channels if self.config.channels else "eeg"

    def _get_epoch_window(self) -> tuple[float, float]:
        """Get epoch window (tmin, tmax) from config."""
        tmin = self.config.epoch_tmin or -0.2
        tmax = self.config.epoch_tmax or 0.8
        return tmin, tmax

    def get_channel_names(self, subject_id: int) -> list[str]:
        """Get channel names from a subject's raw file."""
        raw = self.load_raw(subject_id, preprocess=False)
        picks = self._get_channel_picks()
        raw_picked = raw.copy().pick(picks)
        return list(raw_picked.ch_names)

    def get_n_channels(self, subject_id: int = 1) -> int:
        """Get number of channels for a subject."""
        return len(self.get_channel_names(subject_id))

    def get_n_times(self, subject_id: int = 1) -> int:
        """Get number of time samples per epoch."""
        return self.config.window_samples


class FileDatasetLoader(BaseLoader):
    """Base loader for single-file datasets (FIF, H5, etc.).

    Provides shared training interface (load_epochs, load_continuous,
    load_data_split) on top of an MNE Raw object. Subclasses implement
    load_raw() to produce the Raw from their specific format.
    """

    EXTENSIONS: set[str] = set()

    def __init__(
        self,
        config: DatasetConfig,
        file_path: str,
        event_mapping: dict[str, int] | None = None,
        preproc_overrides: dict | None = None,
    ):
        """Initialize loader with file path and optional event mapping."""
        super().__init__(config)
        self._file_path = Path(file_path)
        self._event_mapping = event_mapping or {}
        self._preproc_overrides = preproc_overrides or {}
        self._raw_cache: dict[tuple, mne.io.Raw] = {}

    @classmethod
    def from_dataset_info(cls, config: DatasetConfig, dataset_info: dict) -> "FileDatasetLoader":
        """Create loader from database dataset_info dict."""
        file_path = dataset_info.get("file_path")
        if not file_path:
            raise ValueError("No file_path in dataset_info")

        event_mapping = None
        events_json = dataset_info.get("events_json")
        if events_json:
            event_mapping = json.loads(events_json)

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

    def get_subject_list(self) -> list[int]:
        """Single file is treated as subject 1."""
        return [1]

    def get_sample_rate(self) -> float:
        """Get effective sample rate (target if resampled, else original)."""
        target = self._preproc_overrides.get("target_sample_rate")
        if target:
            return float(target)
        sampling_rate = self._preproc_overrides.get("sampling_rate")
        if sampling_rate:
            return float(sampling_rate)
        return self.config.sample_rate

    def _get_preproc_params(self) -> tuple[float | None, float | None, bool, float | None]:
        """Get preprocessing parameters with overrides applied."""
        lowcut = self._preproc_overrides.get("lowcut") or self.config.preproc_lowcut
        highcut = self._preproc_overrides.get("highcut") or self.config.preproc_highcut
        rereference = self._preproc_overrides.get("rereference") or self.config.preproc_rereference
        target_sample_rate = self._preproc_overrides.get("target_sample_rate")
        return lowcut, highcut, rereference, target_sample_rate

    def _get_channel_picks(self) -> str:
        """Get channel pick string (e.g., 'eeg') from config or overrides."""
        modality = self._preproc_overrides.get("modality")
        if modality:
            return modality.lower()
        return self.config.channels if self.config.channels else "eeg"

    def _get_epoch_window(self) -> tuple[float, float]:
        """Get epoch window (tmin, tmax) from config or overrides."""
        tmin = self._preproc_overrides.get("epoch_tmin") or self.config.epoch_tmin or -0.2
        tmax = self._preproc_overrides.get("epoch_tmax") or self.config.epoch_tmax or 0.8
        return tmin, tmax

    def load_epochs(
        self,
        subject_id: int,
        block: int | None = None,
        session: str | None = None,
        run: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load epoched data (n_epochs, n_channels, n_times) and labels."""
        logger.info(f"Loading epochs from: {self._file_path}")

        raw = self.load_raw(subject_id, preprocess=True)
        mne_events, mne_event_id = mne.events_from_annotations(raw, event_id=self._event_mapping)

        if not self._event_mapping:
            raise ValueError(
                "event_mapping is empty. Provide an event_mapping dict to specify which events to load. "
                f"File contains events: {list(mne_event_id.keys())}"
            )

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
        """Load continuous data (n_channels, n_samples), event times, labels, and mapping."""
        raw = self.load_raw(subject_id, preprocess=True)
        mne_events, _ = mne.events_from_annotations(raw, event_id=self._event_mapping, verbose=False)

        selected_codes = set(self._event_mapping.values())
        filtered = filter_events_by_codes(mne_events, selected_codes)

        picks = self._get_channel_picks()
        raw_picked = raw.copy().pick(picks)
        data = raw_picked.get_data()

        if len(filtered) > 0:
            event_times = filtered[:, 0]
            event_labels = encode_labels(filtered[:, 2], self._event_mapping)
        else:
            event_times = np.array([])
            event_labels = np.array([])

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
        """Load train epochs and validation continuous data with split metadata."""
        raw = self.load_raw(subject_id, preprocess=True)
        mne_events, _ = mne.events_from_annotations(raw, event_id=self._event_mapping, verbose=False)

        selected_codes = set(self._event_mapping.values())
        filtered = filter_events_by_codes(mne_events, selected_codes)
        filtered = filtered[filtered[:, 0].argsort()]

        n_val = int(len(filtered) * val_ratio)
        if n_val == 0 or n_val >= len(filtered):
            raise ValueError(f"Not enough events to split: {len(filtered)} total")

        train_events = filtered[:-n_val]
        val_events = filtered[-n_val:]

        tmin, tmax = self._get_epoch_window()
        picks = self._get_channel_picks()
        epochs = create_epochs(raw, train_events, selected_codes, tmin, tmax, picks=picks)
        X_train = epochs.get_data()
        y_train = encode_labels(epochs.events[:, 2], self._event_mapping)

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

    def get_label_info(self) -> tuple[dict[int, str], dict[str, int]]:
        """Return (event_mapping, label_mapping) for decoder configuration."""
        return build_label_info(self._event_mapping)

    def get_n_times(self, subject_id: int = 1) -> int:
        """Get number of time samples per epoch."""
        X, _ = self.load_epochs(subject_id)
        return X.shape[2]
