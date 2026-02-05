"""Abstract base class for data loaders.

Defines the interface that all loaders must implement.
"""

from abc import ABC, abstractmethod
from typing import Any

import mne
import numpy as np

from .config import DatasetConfig


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
