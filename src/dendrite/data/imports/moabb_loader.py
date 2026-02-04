"""MOABB dataset loader for offline ML.

Loads data from MOABB benchmark datasets (BNCI2014_001, PhysionetMI, etc.)
for training and benchmarking BCI decoders.
"""

import logging
from typing import Any

import mne
import numpy as np

from .config import DatasetConfig

logger = logging.getLogger(__name__)


# Lazy imports to avoid loading MOABB until needed
def _get_moabb_dataset(dataset_name: str):
    """Get MOABB dataset class by name (dynamic lookup)."""
    from moabb import datasets

    # Dynamic lookup - works with all 36+ MOABB datasets
    if hasattr(datasets, dataset_name):
        return getattr(datasets, dataset_name)()

    raise ValueError(f"Unknown MOABB dataset: {dataset_name}")


def _get_moabb_paradigm(paradigm_name: str, **kwargs):
    """Get MOABB paradigm class by name."""
    from moabb import paradigms

    paradigm_map = {
        "MotorImagery": paradigms.MotorImagery,
        "LeftRightImagery": paradigms.LeftRightImagery,
        "FilterBankMotorImagery": paradigms.FilterBankMotorImagery,
        "P300": paradigms.P300,
        "SinglePass": paradigms.SinglePass,
    }

    if paradigm_name not in paradigm_map:
        available = list(paradigm_map.keys())
        raise ValueError(f"Unknown MOABB paradigm: {paradigm_name}. Available: {available}")

    return paradigm_map[paradigm_name](**kwargs)


class MOAABLoader:
    """Load data from MOABB benchmark datasets.

    Provides a unified interface for loading epoched EEG data from MOABB's
    collection of open BCI datasets. Data is automatically downloaded and
    cached by MOABB.

    Example:
        config = DatasetConfig(
            name="bnci_mi",
            source_type="moabb",
            moabb_dataset="BNCI2014_001",
            moabb_paradigm="MotorImagery",
            events={"left_hand": 0, "right_hand": 1},
        )
        loader = MOAABLoader(config)
        X, y = loader.load_epochs(subject_id=1)
    """

    def __init__(self, config: DatasetConfig):
        """Initialize loader with dataset config.

        Args:
            config: Dataset configuration with moabb_dataset and moabb_paradigm set
        """
        self.config = config
        self._validate_config()

        self._dataset = None
        self._paradigm = None
        self._cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def _validate_config(self):
        """Validate that config has required MOABB fields."""
        if self.config.source_type != "moabb":
            raise ValueError(
                f"MOAABLoader requires source_type='moabb', got '{self.config.source_type}'"
            )

        if not self.config.moabb_dataset:
            raise ValueError("moabb_dataset is required for MOABB source type")

        if not self.config.moabb_paradigm:
            raise ValueError("moabb_paradigm is required for MOABB source type")

    @property
    def dataset(self):
        """Lazy-load MOABB dataset."""
        if self._dataset is None:
            logger.info(f"Loading MOABB dataset: {self.config.moabb_dataset}")
            self._dataset = _get_moabb_dataset(self.config.moabb_dataset)
        return self._dataset

    @property
    def paradigm(self):
        """Lazy-load MOABB paradigm."""
        if self._paradigm is None:
            logger.info(f"Loading MOABB paradigm: {self.config.moabb_paradigm}")
            # Pass any paradigm-specific config
            paradigm_kwargs = {}
            if self.config.moabb_n_classes:
                paradigm_kwargs["n_classes"] = self.config.moabb_n_classes
            if self.config.moabb_events:
                paradigm_kwargs["events"] = self.config.moabb_events
            self._paradigm = _get_moabb_paradigm(self.config.moabb_paradigm, **paradigm_kwargs)
        return self._paradigm

    def get_subject_list(self) -> list[int]:
        """Get list of available subjects in the dataset."""
        return list(self.dataset.subject_list)

    def get_n_channels(self, subject_id: int) -> int:
        """Get number of channels for a subject."""
        X, _ = self.load_epochs(subject_id)
        return X.shape[1]

    def get_n_times(self, subject_id: int) -> int:
        """Get number of time samples per epoch."""
        X, _ = self.load_epochs(subject_id)
        return X.shape[2]

    def get_sample_rate(self) -> float:
        """Get sample rate from paradigm."""
        # Most MOABB paradigms have a resample parameter or use dataset's native rate
        # Default to config value if set, otherwise use 250 Hz (common MOABB default)
        if self.config.sample_rate:
            return self.config.sample_rate
        return 250.0

    def _get_channel_picks(self) -> str:
        """Get channel pick string for data extraction.

        Matches load_epochs behavior for consistent channel selection.
        MOABB paradigms typically select EEG channels only.
        """
        return self.config.channels if self.config.channels else "eeg"

    def _get_paradigm_filters(self) -> tuple[float, float] | None:
        """Get bandpass filter settings from paradigm.

        Returns:
            Tuple of (low_freq, high_freq) or None if no filtering defined
        """
        filters = getattr(self.paradigm, "filters", None)
        if filters and len(filters) > 0 and len(filters[0]) == 2:
            return tuple(filters[0])
        return None

    def _apply_paradigm_preprocessing(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Apply same preprocessing as paradigm (channel picks + bandpass + scaling).

        Ensures continuous data matches the filtered epochs from load_epochs().
        MOABB paradigms convert V to μV internally, so we apply the same scaling.
        """
        raw = raw.copy()
        raw.pick(self._get_channel_picks())

        fmin_fmax = self._get_paradigm_filters()
        if fmin_fmax:
            fmin, fmax = fmin_fmax
            raw.filter(fmin, fmax, verbose=False)
            logger.debug(f"Applied paradigm bandpass filter: {fmin}-{fmax} Hz")

        # MOABB paradigm converts V to μV (1e6 scaling) - apply same to match training
        raw.apply_function(lambda x: x * 1e6, picks="eeg")

        return raw

    def _extract_events_from_raw(
        self,
        raw: mne.io.BaseRaw,
    ) -> tuple[list[int], list[int], dict[str, int]]:
        """Extract event times, labels, and mapping from raw MNE object.

        If config.events is specified, filters to those events. Otherwise,
        auto-generates a mapping from all MNE event types (consistent with
        _encode_labels behavior for epoch loading).

        Args:
            raw: MNE Raw object with annotations

        Returns:
            event_times: List of sample indices
            event_labels: List of integer labels
            label_map: Dict mapping event names to integer labels
        """
        events_array, event_id = mne.events_from_annotations(raw, verbose=False)
        logger.debug(f"MNE event_id mapping: {event_id}")
        logger.debug(f"Config events filter: {self.config.events}")

        # Build reverse lookup: code -> event name
        code_to_name = {code: name for name, code in event_id.items()}

        # If config.events is empty, auto-generate mapping from MNE event_id
        # (matches _encode_labels behavior for consistency)
        if self.config.events:
            label_map = self.config.events
        else:
            # Auto-generate: assign sequential integers to sorted event names
            label_map = {name: i for i, name in enumerate(sorted(event_id.keys()))}
            logger.info(f"Auto-generated event mapping for continuous data: {label_map}")

        event_times = []
        event_labels = []
        for event in events_array:
            sample_idx, _, code = event
            event_name = code_to_name.get(code)
            if event_name and event_name in label_map:
                event_times.append(sample_idx)
                event_labels.append(label_map[event_name])

        return event_times, event_labels, label_map

    def load_epochs(
        self,
        subject_id: int,
        block: int | None = None,
        events: list[str] | None = None,
        session: str | None = None,
        run: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load epoched data for a subject.

        Args:
            subject_id: Subject ID (1-indexed typically)
            block: Ignored for MOABB (sessions handled internally)
            events: Ignored for MOABB (uses paradigm's event selection)
            session: Filter to specific session (e.g., '0train')
            run: Filter to specific run (e.g., '0')

        Returns:
            X: (n_epochs, n_channels, n_times) - epoched EEG data
            y: (n_epochs,) - integer labels (0, 1, 2, ...)
        """
        # Check cache (only for unfiltered requests)
        cache_key = (subject_id, session, run)
        if cache_key in self._cache:
            return self._cache[cache_key]

        logger.info(f"Loading MOABB data for subject {subject_id}")

        # Get data from MOABB
        X, y_str, metadata = self.paradigm.get_data(
            self.dataset,
            subjects=[subject_id],
        )

        # Convert string labels to integers, get validity mask
        y, valid_mask = self._encode_labels(y_str)

        # Build filter mask
        mask = valid_mask.copy()
        if session is not None:
            mask &= (metadata["session"] == session).values
        if run is not None:
            mask &= (metadata["run"] == run).values

        X = X[mask]
        y = y[mask]

        # Cache result
        self._cache[cache_key] = (X, y)

        logger.info(f"Loaded {X.shape[0]} epochs, shape: {X.shape}, classes: {np.unique(y)}")

        return X, y

    def _encode_labels(self, y_str: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert string labels to integers, return validity mask.

        Args:
            y_str: Array of string labels from MOABB (e.g., 'left_hand', 'right_hand')

        Returns:
            y_int: Integer labels (-1 for unmapped labels)
            valid_mask: Boolean mask of valid (mapped) labels
        """
        if not self.config.events:
            # No mapping specified - create automatic mapping (all labels valid)
            unique_labels = sorted(set(y_str))
            label_map = {label: i for i, label in enumerate(unique_labels)}
            logger.info(f"Auto-generated label mapping: {label_map}")
        else:
            label_map = self.config.events

        y_int = np.full(len(y_str), -1, dtype=np.int64)
        valid_mask = np.zeros(len(y_str), dtype=bool)

        for i, label in enumerate(y_str):
            str_label = str(label)  # Handle numpy string types
            if str_label in label_map:
                y_int[i] = label_map[str_label]
                valid_mask[i] = True

        n_filtered = len(y_str) - valid_mask.sum()
        if n_filtered > 0:
            logger.info(f"Filtered {n_filtered}/{len(y_str)} epochs with unmapped labels")

        return y_int, valid_mask

    def get_sessions(self, subject_id: int) -> list[str]:
        """Get available session names for a subject.

        Args:
            subject_id: Subject ID

        Returns:
            List of session names (e.g., ['0train', '1test'])
        """
        raw_data = self.dataset.get_data(subjects=[subject_id])
        return list(raw_data[subject_id].keys())

    def get_runs(self, subject_id: int, session: str) -> list[str]:
        """Get available run names for a session.

        Args:
            subject_id: Subject ID
            session: Session name

        Returns:
            List of run names (e.g., ['0', '1', '2'])
        """
        raw_data = self.dataset.get_data(subjects=[subject_id])
        return list(raw_data[subject_id][session].keys())

    def load_raw(
        self,
        subject_id: int,
        session: str | None = None,
        run: str | None = None,
    ) -> mne.io.BaseRaw:
        """Load raw continuous MNE object for a subject.

        Args:
            subject_id: Subject ID
            session: Session name (None = first available)
            run: Run name (None = first available)

        Returns:
            MNE Raw object with continuous EEG data
        """
        raw_data = self.dataset.get_data(subjects=[subject_id])
        subj_data = raw_data[subject_id]

        # Get first session if not specified
        if session is None:
            session = list(subj_data.keys())[0]

        # Get first run if not specified
        if run is None:
            run = list(subj_data[session].keys())[0]

        raw = subj_data[session][run]
        logger.info(
            f"Loaded raw data for subject {subject_id}, session {session}, run {run}: "
            f"{raw.get_data().shape}"
        )
        return raw

    def load_raw_concatenated(
        self,
        subject_id: int,
        session: str | None = None,
    ) -> mne.io.BaseRaw:
        """Load and concatenate all runs for a session.

        Args:
            subject_id: Subject ID
            session: Session name (None = first available)

        Returns:
            Concatenated MNE Raw object
        """
        raw_data = self.dataset.get_data(subjects=[subject_id])
        subj_data = raw_data[subject_id]

        if session is None:
            session = list(subj_data.keys())[0]

        runs = list(subj_data[session].values())
        if len(runs) == 1:
            return runs[0]

        # Concatenate all runs
        concatenated = mne.concatenate_raws(runs)
        logger.info(
            f"Concatenated {len(runs)} runs for subject {subject_id}, session {session}: "
            f"{concatenated.get_data().shape}"
        )
        return concatenated

    def load_continuous(
        self,
        subject_id: int,
        block: int | None = None,
        session: str | None = None,
        run: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
        """Load continuous data with events for async evaluation.

        Matches the DataLoader interface for continuous data access.

        Args:
            subject_id: Subject ID
            block: Ignored (for API compatibility with DataLoader)
            session: Session name (None = first available)
            run: Run name (None = all runs in session)

        Returns:
            data: (n_channels, n_samples) - continuous EEG
            event_times: (n_events,) - sample indices of events
            event_labels: (n_events,) - integer labels
            event_mapping: Dict mapping event names to integer labels
        """
        if run is not None:
            # Load specific run
            raw = self.load_raw(subject_id, session, run)
        else:
            # Load all runs in session
            raw = self.load_raw_concatenated(subject_id, session)

        # Apply same preprocessing as paradigm (channel picks + bandpass filter)
        raw = self._apply_paradigm_preprocessing(raw)
        data = raw.get_data()  # (n_channels, n_samples)

        # Extract and filter events
        event_times, event_labels, event_mapping = self._extract_events_from_raw(raw)

        if not event_times:
            logger.warning(
                f"No events found for subject {subject_id} after filtering. "
                f"Check config.events matches dataset event names."
            )
        else:
            logger.info(
                f"Loaded continuous data for subject {subject_id}: "
                f"data shape {data.shape}, {len(event_times)} events"
            )

        return data, np.array(event_times), np.array(event_labels), event_mapping

    def load_data_split(
        self,
        subject_id: int,
        block: int = 1,
        val_ratio: float = 0.3,
    ) -> tuple[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]],
        dict[str, Any],
    ]:
        """Split data using sessions/runs for train and eval.

        Uses natural dataset structure for clean train/eval separation:
        1. Multiple sessions (e.g., 'train'/'test') → first for train, second for eval
        2. Single session with multiple runs → first runs for train, last for eval
        3. Single run → temporal split (fallback)

        Args:
            subject_id: Subject ID
            block: Ignored (for API compatibility)
            val_ratio: Used for run-based split ratio (ignored if session split)

        Returns:
            train_data: (X_train, y_train) - epochs for training
            val_data: (continuous, event_times, event_labels, event_mapping) - for async eval
            split_info: Metadata about split method
        """
        sessions = self.get_sessions(subject_id)

        if len(sessions) >= 2:
            # Use first session for training, second for evaluation
            train_session = sessions[0]
            eval_session = sessions[1]
            logger.info(f"Using session split: train={train_session}, eval={eval_session}")

            # Get epochs from paradigm with session filtering via metadata
            X_all, y_str, metadata = self.paradigm.get_data(self.dataset, subjects=[subject_id])
            y_all, valid_mask = self._encode_labels(y_str)

            # Filter epochs by session AND valid labels
            train_mask = (metadata["session"] == train_session) & valid_mask
            X_train = X_all[train_mask]
            y_train = y_all[train_mask]

            # Load continuous data from eval session
            val_continuous, val_times, val_labels, val_mapping = self.load_continuous(
                subject_id, session=eval_session
            )

            split_info: dict[str, Any] = {
                "method": "session",
                "train": train_session,
                "eval": eval_session,
            }

        else:
            # Single session - split by runs
            session = sessions[0]
            runs = self.get_runs(subject_id, session)

            if len(runs) >= 2:
                # Use last run for evaluation
                n_eval_runs = max(1, int(len(runs) * val_ratio))
                train_runs = runs[:-n_eval_runs]
                eval_runs = runs[-n_eval_runs:]
                logger.info(f"Using run split: train={train_runs}, eval={eval_runs}")

                # Get epochs and filter by run
                X_all, y_str, metadata = self.paradigm.get_data(self.dataset, subjects=[subject_id])
                y_all, valid_mask = self._encode_labels(y_str)

                # Filter by run AND valid labels
                train_mask = metadata["run"].isin(train_runs) & valid_mask
                X_train = X_all[train_mask]
                y_train = y_all[train_mask]

                # Load continuous from eval runs
                val_continuous, val_times, val_labels, val_mapping = self._load_runs_continuous(
                    subject_id, session, eval_runs
                )

                split_info = {
                    "method": "run",
                    "train_runs": train_runs,
                    "eval_runs": eval_runs,
                }
            else:
                # Single run - temporal split of epochs
                logger.info("Single run - using temporal epoch split")
                X_all, y_all = self.load_epochs(subject_id)

                n_val = int(len(X_all) * val_ratio)
                if n_val == 0:
                    raise ValueError(f"Not enough epochs: {len(X_all)}")

                X_train = X_all[:-n_val]
                y_train = y_all[:-n_val]

                # For continuous, just use the full session
                val_continuous, val_times, val_labels, val_mapping = self.load_continuous(subject_id)
                # Take only last portion of events
                n_val_events = int(len(val_times) * val_ratio)
                val_times = val_times[-n_val_events:]
                val_labels = val_labels[-n_val_events:]

                split_info = {
                    "method": "temporal",
                    "val_ratio": val_ratio,
                }

        logger.info(
            f"Split MOABB data for subject {subject_id}: "
            f"{len(X_train)} train epochs, {len(val_labels)} eval events"
        )

        return (X_train, y_train), (val_continuous, val_times, val_labels, val_mapping), split_info

    def _load_runs_continuous(
        self,
        subject_id: int,
        session: str,
        runs: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
        """Load continuous data from specific runs within a session."""
        raw_data = self.dataset.get_data(subjects=[subject_id])
        subj_data = raw_data[subject_id][session]

        # Concatenate selected runs
        raw_list = [subj_data[run] for run in runs if run in subj_data]
        if len(raw_list) == 1:
            raw = raw_list[0]
        else:
            raw = mne.concatenate_raws(raw_list)

        # Apply preprocessing
        raw = self._apply_paradigm_preprocessing(raw)
        data = raw.get_data()

        # Extract and filter events
        event_times, event_labels, event_mapping = self._extract_events_from_raw(raw)

        return data, np.array(event_times), np.array(event_labels), event_mapping

    def get_channel_names(self, subject_id: int) -> list[str]:
        """Get channel names from preprocessed data (matching load_continuous)."""
        raw = self.load_raw(subject_id)
        raw = self._apply_paradigm_preprocessing(raw)
        return list(raw.ch_names)

    def get_channel_types(self, subject_id: int) -> list[str]:
        """Get channel types from preprocessed data (matching load_continuous)."""
        raw = self.load_raw(subject_id)
        raw = self._apply_paradigm_preprocessing(raw)
        return raw.get_channel_types()
