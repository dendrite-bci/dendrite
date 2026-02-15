"""FIF file data loader.

Loads data from a FIF file with event mapping from JSON.
"""

import logging

import mne
import numpy as np

from ._event_utils import apply_preprocessing
from .base_loader import FileDatasetLoader
from ._types import LoadedData

logger = logging.getLogger(__name__)


class FIFLoader(FileDatasetLoader):
    """Loader for FIF file datasets.

    Handles custom datasets with a FIF file and event mapping from JSON.
    """

    EXTENSIONS = {".fif"}

    @staticmethod
    def load_file(file_path: str) -> LoadedData:
        """Load data from FIF file for streaming."""
        logger.info(f"Loading FIF file via MNE: {file_path}")
        raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)

        data = raw.get_data().T  # (samples, channels)
        channel_names = list(raw.ch_names)
        channel_types = raw.get_channel_types()
        sample_rate = raw.info["sfreq"]

        events = []
        event_id = None
        try:
            events_array, event_id = mne.events_from_annotations(raw)
            events = [(int(e[0]), int(e[2])) for e in events_array]
        except (ValueError, KeyError, RuntimeError):
            pass

        logger.info(f"Loaded: {data.shape[0]} samples, {data.shape[1]} channels @ {sample_rate} Hz")
        return LoadedData(data, channel_names, channel_types, sample_rate, events, event_id)

    def load_raw(
        self,
        subject_id: int,
        preprocess: bool = True,
        session: str | None = None,
        run: str | None = None,
    ) -> mne.io.Raw:
        """Load raw FIF data with optional preprocessing."""
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
