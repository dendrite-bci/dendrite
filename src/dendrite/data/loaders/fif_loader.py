"""FIF file data loader."""

import json
import logging

import mne
import numpy as np

from ._types import RawData

logger = logging.getLogger(__name__)


class FIFLoader:
    """Loader for raw FIF files."""

    EXTENSIONS = {".fif"}

    def __init__(self, file_path: str):
        self._file_path = file_path

    def load(self) -> RawData:
        """Load continuous data from a raw FIF file."""
        logger.info(f"Loading FIF file: {self._file_path}")
        raw = mne.io.read_raw_fif(self._file_path, preload=True, verbose=False)

        data: np.ndarray = np.asarray(raw.get_data())  # (channels, samples)
        channel_names = list(raw.ch_names)
        channel_types = raw.get_channel_types()
        sample_rate = raw.info["sfreq"]

        # 1. Check embedded metadata (FIF derivatives from make_derivative)
        events: list[tuple[int, int]] = []
        event_id: dict[str, int] | None = None
        desc = raw.info.get("description")
        if desc:
            try:
                meta = json.loads(desc)
                if "event_id" in meta:
                    event_id = {k: int(v) for k, v in meta["event_id"].items()}
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        # 2. Extract events — use embedded mapping with annotations, else auto-derive
        try:
            if event_id:
                events_array, event_id = mne.events_from_annotations(
                    raw,
                    event_id=event_id,  # type: ignore[arg-type]
                )
            else:
                events_array, event_id = mne.events_from_annotations(raw)
            events = [(int(e[0]), int(e[2])) for e in events_array]
        except (ValueError, KeyError, RuntimeError):
            pass

        logger.info(f"Loaded: {data.shape[0]} samples, {data.shape[1]} channels @ {sample_rate} Hz")
        return RawData(data, channel_names, channel_types, sample_rate, events, event_id)
