"""Abstract base class for data loaders.

Defines the interface that all loaders must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
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
        """Initialize loader with configuration.

        Args:
            config: Dataset configuration
        """
        self.config = config

    @abstractmethod
    def get_fif_path(
        self,
        subject_id: int,
        session: str | None = None,
        run: str | None = None,
    ) -> Path:
        """Get path to FIF file for a subject.

        Args:
            subject_id: Subject ID
            session: Session identifier (e.g., "ses-01")
            run: Run identifier (e.g., "run-01")

        Returns:
            Path to FIF file
        """
        ...

    @abstractmethod
    def get_subject_list(self) -> list[int]:
        """Get list of available subjects.

        Returns:
            List of subject IDs
        """
        ...

    @abstractmethod
    def get_sample_rate(self) -> float:
        """Get sample rate in Hz.

        Returns:
            Sample rate
        """
        ...

    @abstractmethod
    def load_raw(
        self,
        subject_id: int,
        preprocess: bool = True,
        session: str | None = None,
        run: str | None = None,
    ) -> mne.io.Raw:
        """Load raw MNE object with optional preprocessing.

        Args:
            subject_id: Subject ID
            preprocess: Apply preprocessing (bandpass, rereference)
            session: Session identifier
            run: Run identifier

        Returns:
            MNE Raw object
        """
        ...

    @abstractmethod
    def load_epochs(
        self,
        subject_id: int,
        block: int | None = None,
        session: str | None = None,
        run: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load epoched data for training.

        Args:
            subject_id: Subject ID
            block: Block number for event filtering
            session: Session identifier
            run: Run identifier

        Returns:
            X: (n_epochs, n_channels, n_times) - epoched data
            y: (n_epochs,) - integer labels
        """
        ...

    @abstractmethod
    def load_continuous(
        self,
        subject_id: int,
        block: int | None = None,
        session: str | None = None,
        run: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
        """Load continuous data for sliding window evaluation.

        Args:
            subject_id: Subject ID
            block: Block number for event filtering
            session: Session identifier
            run: Run identifier

        Returns:
            data: (n_channels, n_samples) - continuous EEG
            event_times: (n_events,) - sample indices of events
            event_labels: (n_events,) - integer labels
            event_mapping: Dict mapping event names to integer labels
        """
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
        """Load data split into train epochs and validation continuous.

        Args:
            subject_id: Subject ID
            block: Block number for event filtering
            val_ratio: Fraction of data for validation

        Returns:
            train_data: (X_train, y_train) - epochs for training
            val_data: (continuous, event_times, event_labels, event_mapping) - for async eval
            split_info: Metadata about split method (e.g., {"method": "temporal", "val_ratio": 0.3})
        """
        ...

    # Concrete shared methods

    def _get_channel_picks(self) -> str:
        """Get channel pick string for data extraction.

        Returns:
            Channel type string (e.g., 'eeg')
        """
        return self.config.channels if self.config.channels else "eeg"

    def _get_epoch_window(self) -> tuple[float, float]:
        """Get epoch window (tmin, tmax) from config.

        Returns:
            (tmin, tmax) tuple
        """
        tmin = self.config.epoch_tmin or -0.2
        tmax = self.config.epoch_tmax or 0.8
        return tmin, tmax

    def get_channel_names(self, subject_id: int) -> list[str]:
        """Get channel names from a subject's raw file.

        Args:
            subject_id: Subject ID

        Returns:
            List of channel names
        """
        raw = self.load_raw(subject_id, preprocess=False)
        picks = self._get_channel_picks()
        raw_picked = raw.copy().pick(picks)
        return list(raw_picked.ch_names)

    def get_n_channels(self, subject_id: int = 1) -> int:
        """Get number of channels for a subject.

        Args:
            subject_id: Subject ID

        Returns:
            Number of channels
        """
        return len(self.get_channel_names(subject_id))

    def get_n_times(self, subject_id: int = 1) -> int:
        """Get number of time samples per epoch.

        Args:
            subject_id: Subject ID

        Returns:
            Number of time samples
        """
        return self.config.window_samples
