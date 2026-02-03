"""Data loading for offline ML datasets.

Ported from analysis/errp_pretrain_study/data_loader.py with generalization
to support any dataset via DatasetConfig.
"""

import logging
import re

import mne
import numpy as np

logger = logging.getLogger(__name__)
from pathlib import Path

from dendrite.processing.preprocessing.offline_adapter import apply_online_preprocessing_offline

from .cache import get_global_cache
from .config import DatasetConfig


class DataLoader:
    """Unified data loader for offline ML datasets.

    Loads epoched data for training and continuous data for async evaluation.
    Uses DatasetConfig to determine paths, events, and preprocessing.
    """

    def __init__(self, config: DatasetConfig, use_cache: bool = True):
        """Initialize loader with dataset config.

        Args:
            config: Dataset configuration
            use_cache: Whether to use global cache for loaded data
        """
        self.config = config
        self.use_cache = use_cache
        self._cache = get_global_cache() if use_cache else None

    def get_sessions(self, subject_id: int) -> list[str]:
        """Discover available sessions for a subject (BIDS structure)."""
        subject_dir = Path(self.config.data_root) / f"sub-{subject_id:02d}"
        if not subject_dir.exists():
            return []
        sessions = sorted(
            [d.name for d in subject_dir.iterdir() if d.is_dir() and d.name.startswith("ses-")]
        )
        return sessions

    def get_runs(self, subject_id: int, session: str) -> list[str]:
        """Discover available runs for a session (BIDS structure)."""
        eeg_dir = Path(self.config.data_root) / f"sub-{subject_id:02d}" / session / "eeg"
        if not eeg_dir.exists():
            return []

        runs = set()
        for f in eeg_dir.glob("*_eeg.fif"):
            match = re.search(r"run-(\d+)", f.name)
            if match:
                runs.add(f"run-{match.group(1)}")
        return sorted(runs)

    def _cache_key(self, *parts: str) -> str:
        """Generate cache key from parts."""
        return f"{self.config.name}_{'-'.join(str(p) for p in parts)}"

    def get_fif_path(
        self,
        subject_id: int,
        session: str | None = None,
        run: str | None = None,
    ) -> Path:
        """Get path to FIF file for a subject.

        Supports both BIDS structure (session/run) and legacy structure.

        Args:
            subject_id: Subject ID
            session: BIDS session (e.g., "ses-01") or None for auto-detection
            run: BIDS run (e.g., "run-01") or None for first available

        Returns:
            Path to FIF file
        """
        data_root = Path(self.config.data_root)

        # Try BIDS structure if session specified
        if session:
            eeg_dir = data_root / f"sub-{subject_id:02d}" / session / "eeg"
            if eeg_dir.exists():
                pattern = f"*{run}*_eeg.fif" if run else "*_eeg.fif"
                fif_files = sorted(eeg_dir.glob(pattern))
                if fif_files:
                    return fif_files[0]

        # Auto-detect BIDS if no session specified
        sessions = self.get_sessions(subject_id)
        if sessions:
            session = sessions[0]
            eeg_dir = data_root / f"sub-{subject_id:02d}" / session / "eeg"
            if eeg_dir.exists():
                pattern = f"*{run}*_eeg.fif" if run else "*_eeg.fif"
                fif_files = sorted(eeg_dir.glob(pattern))
                if fif_files:
                    return fif_files[0]

        # Fallback to legacy structure: sub-XX/raw_fif/*_raw.fif
        subject_dir = data_root / f"sub-{subject_id:02d}" / "raw_fif"
        fif_files = list(subject_dir.glob("*_raw.fif"))
        if fif_files:
            return fif_files[0]

        raise FileNotFoundError(f"No FIF files found for subject {subject_id} in {data_root}")

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
            session: BIDS session (e.g., "ses-01") or None for auto-detection
            run: BIDS run (e.g., "run-01") or None for first available

        Returns:
            MNE Raw object
        """
        cache_key = self._cache_key("raw", subject_id, preprocess, session or "", run or "")

        def _load() -> mne.io.Raw:
            fif_path = self.get_fif_path(subject_id, session=session, run=run)
            raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)

            if preprocess and (
                self.config.preproc_lowcut is not None or self.config.preproc_highcut is not None
            ):
                n_eeg_channels = len(mne.pick_types(raw.info, eeg=True))
                modality_preprocessing = {
                    "EEG": {
                        "num_channels": n_eeg_channels,
                        "sample_rate": raw.info["sfreq"],
                        "lowcut": self.config.preproc_lowcut,
                        "highcut": self.config.preproc_highcut,
                        "apply_rereferencing": self.config.preproc_rereference,
                        "downsample_factor": 1,
                    }
                }
                raw, _ = apply_online_preprocessing_offline(raw, modality_preprocessing)

            return raw

        if self._cache:
            return self._cache.get_or_load(cache_key, _load)
        return _load()

    def _get_event_names(self, block: int | None = None) -> list[str]:
        """Get list of event names to include for given block.

        Args:
            block: Block number (1, 3, etc.) or None for all blocks

        Returns:
            List of event names to include
        """
        # Check for block-specific event patterns (legacy ErrP support)
        if self.config.event_patterns and block is not None:
            if block in self.config.event_patterns:
                pattern = self.config.event_patterns[block]
                # Flatten all event names from pattern dict
                all_names = []
                for names in pattern.values():
                    all_names.extend(names)
                return all_names

        # Default: use all events from config.events
        return list(self.config.events.keys())

    def _get_event_mapping(self, block: int | None = None) -> dict[str, int]:
        """Get event name to label mapping for given block.

        Args:
            block: Block number or None for all blocks

        Returns:
            Dict mapping event names to integer labels
        """
        event_names = self._get_event_names(block)

        # Build mapping from config.events for relevant event names
        mapping = {}
        for name in event_names:
            if name in self.config.events:
                mapping[name] = self.config.events[name]

        return mapping

    def _filter_mne_events(
        self,
        event_id: dict[str, int],
        block: int | None = None,
    ) -> dict[str, int]:
        """Filter MNE events based on dataset config and block.

        Args:
            event_id: MNE event name to code mapping
            block: Optional block number

        Returns:
            Filtered event_id dict containing only relevant events
        """
        event_names = self._get_event_names(block)
        return {k: v for k, v in event_id.items() if k in event_names}

    def _load_events_with_mapping(
        self,
        raw: mne.io.Raw,
        block: int | None = None,
    ) -> tuple[np.ndarray, dict[str, int], dict[int, int]]:
        """Extract events from raw and build code-to-label mapping.

        Consolidates event extraction logic used by load_epochs(),
        load_continuous(), and load_data_split().

        Args:
            raw: MNE Raw object with annotations
            block: Block number for event filtering

        Returns:
            mne_events: MNE events array (n_events, 3)
            filtered_event_id: Filtered event name->code mapping
            code_to_label: MNE code->config label mapping
        """
        mne_events, event_id = mne.events_from_annotations(raw, verbose=False)
        event_mapping = self._get_event_mapping(block)
        filtered_event_id = self._filter_mne_events(event_id, block=block)

        code_to_label = {}
        for event_name, code in filtered_event_id.items():
            if event_name in event_mapping:
                code_to_label[code] = event_mapping[event_name]

        return mne_events, filtered_event_id, code_to_label

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
            block: Block number (uses config.blocks if set)
            session: BIDS session (e.g., "ses-01") or None for auto-detection
            run: BIDS run (e.g., "run-01") or None for first available

        Returns:
            X: (n_epochs, n_channels, n_times) - epoched EEG data
            y: (n_epochs,) - integer labels based on config.events mapping
        """
        cache_key = self._cache_key("epochs", subject_id, block, session or "", run or "")

        def _load() -> tuple[np.ndarray, np.ndarray]:
            raw = self.load_raw(subject_id, session=session, run=run)
            mne_events, filtered_event_id, code_to_label = self._load_events_with_mapping(
                raw, block
            )

            if not filtered_event_id:
                raise ValueError(
                    f"No events found for subject {subject_id}, block {block}. "
                    f"Available events in raw: check annotations"
                )

            picks = self.config.channels if self.config.channels else "eeg"
            epochs = mne.Epochs(
                raw,
                mne_events,
                event_id=filtered_event_id,
                tmin=self.config.epoch_tmin,
                tmax=self.config.epoch_tmax,
                picks=picks,
                baseline=None,
                preload=True,
                verbose=False,
            )

            X = epochs.get_data()  # (n_trials, n_channels, n_times)
            event_codes = epochs.events[:, 2]

            # Convert event codes to labels
            y = np.array(
                [code_to_label.get(code, 0) for code in event_codes],
                dtype=np.int64,
            )

            return X, y

        if self._cache:
            return self._cache.get_or_load(cache_key, _load)
        return _load()

    def load_continuous(
        self,
        subject_id: int,
        block: int | None = None,
        session: str | None = None,
        run: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load continuous data for sliding window evaluation.

        Args:
            subject_id: Subject ID
            block: Block number for event filtering
            session: BIDS session (e.g., "ses-01") or None for auto-detection
            run: BIDS run (e.g., "run-01") or None for first available

        Returns:
            data: (n_channels, n_samples) - continuous EEG
            event_times: (n_events,) - sample indices of events
            event_labels: (n_events,) - integer labels based on config.events
        """
        cache_key = self._cache_key("continuous", subject_id, block, session or "", run or "")

        def _load() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            raw = self.load_raw(subject_id, session=session, run=run)
            mne_events, _, code_to_label = self._load_events_with_mapping(raw, block)
            all_codes = set(code_to_label.keys())

            # Extract continuous data
            picks = self.config.channels if self.config.channels else "eeg"
            raw_picked = raw.copy().pick(picks)
            data = raw_picked.get_data()  # (n_channels, n_samples)

            # Get event times and labels
            event_times = []
            event_labels = []

            for event in mne_events:
                sample_idx, _, code = event
                if code in all_codes:
                    event_times.append(sample_idx)
                    event_labels.append(code_to_label[code])

            if not event_times:
                logger.warning(
                    f"No events found for subject {subject_id}, block {block}. "
                    f"Check event configuration matches annotations in data."
                )

            return data, np.array(event_times), np.array(event_labels, dtype=np.int64)

        if self._cache:
            return self._cache.get_or_load(cache_key, _load)
        return _load()

    def load_data_split(
        self,
        subject_id: int,
        block: int = 1,
        val_ratio: float = 0.3,
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Load subject data split into train epochs and validation continuous.

        Splits temporally: first (1-val_ratio) events for training,
        last val_ratio events for validation (async eval).

        Args:
            subject_id: Subject ID
            block: Block number for event filtering
            val_ratio: Fraction of events for validation (default 0.3)

        Returns:
            train_data: (X_train, y_train) - epochs for training
            val_data: (continuous, event_times, event_labels) - for async eval
        """
        raw = self.load_raw(subject_id)
        mne_events, filtered_event_id, code_to_label = self._load_events_with_mapping(raw, block)
        all_codes = set(code_to_label.keys())

        # Get all relevant events sorted by time
        relevant_events = [(e[0], e[2]) for e in mne_events if e[2] in all_codes]
        relevant_events.sort(key=lambda x: x[0])

        n_total = len(relevant_events)
        if n_total == 0:
            logger.warning(
                f"No events found for subject {subject_id}, block {block}. "
                f"Check event configuration matches annotations in data."
            )
            raise ValueError(
                f"No events found for subject {subject_id}, block {block}. "
                f"Check event configuration matches annotations."
            )

        n_val = int(n_total * val_ratio)
        n_train = n_total - n_val

        train_events = relevant_events[:n_train]
        val_events = relevant_events[n_train:]

        if not train_events or not val_events:
            raise ValueError(f"Not enough events to split: {n_total} total (need at least 2)")

        split_sample = val_events[0][0]

        picks = self.config.channels if self.config.channels else "eeg"
        train_event_array = np.array([[e[0], 0, e[1]] for e in train_events])
        epochs = mne.Epochs(
            raw,
            train_event_array,
            event_id=filtered_event_id,
            tmin=self.config.epoch_tmin,
            tmax=self.config.epoch_tmax,
            picks=picks,
            baseline=None,
            preload=True,
            verbose=False,
        )

        X_train = epochs.get_data()
        event_codes = epochs.events[:, 2]
        y_train = np.array(
            [code_to_label.get(code, 0) for code in event_codes],
            dtype=np.int64,
        )

        raw_picked = raw.copy().pick(picks)
        full_data = raw_picked.get_data()

        # Add buffer before first val event
        buffer_samples = int(self.config.epoch_tmax * self.config.sample_rate) + 100
        val_start = max(0, split_sample - buffer_samples)
        val_continuous = full_data[:, val_start:]

        # Adjust event times relative to val_continuous start
        val_event_times = np.array([e[0] - val_start for e in val_events])
        val_event_labels = np.array(
            [code_to_label.get(e[1], 0) for e in val_events],
            dtype=np.int64,
        )

        return (X_train, y_train), (val_continuous, val_event_times, val_event_labels)

    def get_channel_names(self, subject_id: int) -> list[str]:
        """Get channel names from a subject's raw file."""
        raw = self.load_raw(subject_id, preprocess=False)
        picks = self.config.channels if self.config.channels else "eeg"
        raw_picked = raw.copy().pick(picks)
        return list(raw_picked.ch_names)

    def get_subject_list(self) -> list[int]:
        """Get list of available subjects."""
        return self.config.subjects

    def get_n_channels(self, subject_id: int) -> int:
        """Get number of channels for a subject."""
        return len(self.get_channel_names(subject_id))

    def get_n_times(self, subject_id: int = 0) -> int:
        """Get number of time samples per epoch.

        Args:
            subject_id: Ignored (window size is config-based)
        """
        return self.config.window_samples

    def get_sample_rate(self) -> float:
        """Get sample rate in Hz."""
        return self.config.sample_rate
