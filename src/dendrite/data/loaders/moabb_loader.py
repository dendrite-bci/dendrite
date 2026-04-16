"""MOABB dataset loader for offline ML.

Loads data from MOABB benchmark datasets (BNCI2014_001, PhysionetMI, etc.)
for training and benchmarking BCI decoders.
"""

import logging
from dataclasses import dataclass, field

import mne
import numpy as np

from ._types import EpochedData, RawData

logger = logging.getLogger(__name__)


def _get_moabb_dataset(dataset_name: str):
    """Get MOABB dataset class by name (dynamic lookup)."""
    from moabb import datasets

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
        raise ValueError(f"Unknown MOABB paradigm: {paradigm_name}. Available: {list(paradigm_map)}")

    return paradigm_map[paradigm_name](**kwargs)


@dataclass
class MoabbConfig:
    """Lightweight config for MOABBLoader."""

    dataset: str
    paradigm: str
    channels: str = "eeg"
    events: dict[str, int] = field(default_factory=dict)
    n_classes: int | None = None
    event_filter: list[str] | None = None


class MOABBLoader:
    """Load data from MOABB benchmark datasets as RawData.

    Example:
        loader = MOABBLoader(MoabbConfig(dataset="BNCI2014_001", paradigm="MotorImagery"))
        raw = loader.load_as_raw(subject_id=1)
    """

    def __init__(self, config: MoabbConfig):
        self.config = config
        self._dataset = None
        self._paradigm = None

    @property
    def dataset(self):
        """Lazy-load MOABB dataset."""
        if self._dataset is None:
            logger.info(f"Loading MOABB dataset: {self.config.dataset}")
            self._dataset = _get_moabb_dataset(self.config.dataset)
        return self._dataset

    @property
    def paradigm(self):
        """Lazy-load MOABB paradigm."""
        if self._paradigm is None:
            kwargs = {}
            if self.config.n_classes:
                kwargs["n_classes"] = self.config.n_classes
            if self.config.event_filter:
                kwargs["events"] = self.config.event_filter
            self._paradigm = _get_moabb_paradigm(self.config.paradigm, **kwargs)
        return self._paradigm

    def load_paradigm_epochs(
        self,
        subject_id: int,
    ) -> EpochedData:
        """Load pre-epoched data using MOABB's paradigm preprocessing.

        Uses paradigm.get_data() which applies the paradigm's own bandpass,
        epoch windowing, and event filtering — matches published benchmarks.
        """
        X, y_str, metadata = self.paradigm.get_data(
            self.dataset, subjects=[subject_id],
        )
        X = X.astype(np.float32)

        # Map string labels to integers (0..N-1)
        unique_labels = sorted(set(y_str))
        label_map = {label: i for i, label in enumerate(unique_labels)}
        y = np.array([label_map[s] for s in y_str], dtype=np.int64)

        # Get channel info from raw (paradigm doesn't expose it directly)
        raw = self._load_raw(subject_id)
        picks = self.config.channels or "eeg"
        raw = raw.copy().pick(picks)

        epoched = EpochedData(
            X=X, y=y,
            metadata={
                "paradigm": self.config.paradigm,
                "n_channels": X.shape[1],
                "n_times": X.shape[2],
                "subject": subject_id,
                "dataset_code": self.config.dataset,
            },
            source="moabb",
            source_id=self.config.dataset or "",
            sample_rate=float(raw.info["sfreq"]),
            channel_names=list(raw.ch_names),
            channel_types=raw.get_channel_types(),
        )
        epoched.encode_labels()
        return epoched

    def load_as_raw(
        self,
        subject_id: int,
        session: str | None = None,
        run: str | None = None,
    ) -> RawData:
        """Load continuous data as RawData — bridges MOABB into the standard pipeline."""
        raw = self._load_raw(subject_id, session, run)
        picks = self.config.channels or "eeg"
        raw = raw.copy().pick(picks)

        event_times, event_labels, event_mapping = self._extract_events_from_raw(raw)
        events = list(zip(event_times, event_labels, strict=True))

        return RawData(
            data=np.asarray(raw.get_data()),
            channel_names=list(raw.ch_names),
            channel_types=raw.get_channel_types(),
            sample_rate=float(raw.info["sfreq"]),
            events=events,
            event_id=event_mapping,
        )

    # -- Private helpers --

    def _load_raw(
        self,
        subject_id: int,
        session: str | None = None,
        run: str | None = None,
    ) -> mne.io.BaseRaw:
        """Load raw MNE object. Concatenates runs if run is None."""
        raw_data = self.dataset.get_data(subjects=[subject_id])
        subj_data = raw_data[subject_id]

        if session is None:
            session = list(subj_data.keys())[0]
        if run is None:
            runs = list(subj_data[session].values())
            if len(runs) == 1:
                return runs[0]
            return mne.concatenate_raws(runs)  # type: ignore[return-value]

        return subj_data[session][run]

    def _extract_events_from_raw(
        self, raw: mne.io.BaseRaw,
    ) -> tuple[list[int], list[int], dict[str, int]]:
        """Extract event times, labels, and mapping from MNE annotations."""
        events_array, event_id = mne.events_from_annotations(raw, verbose=False)
        code_to_name = {code: name for name, code in event_id.items()}

        label_map: dict[str, int] = (
            dict(self.config.events) if self.config.events else dict(event_id)
        )

        # Marker channel uses 0 as "no event" sentinel — shift codes to start at 1
        if 0 in label_map.values():
            offset = 1 - min(label_map.values())
            label_map = {name: code + offset for name, code in label_map.items()}
            logger.info(f"Shifted event codes by +{offset} to avoid marker 0 conflict")

        event_times = []
        event_labels = []
        for event in events_array:
            sample_idx, _, code = event
            event_name = code_to_name.get(code)
            if event_name and event_name in label_map:
                event_times.append(sample_idx)
                event_labels.append(label_map[event_name])

        return event_times, event_labels, label_map
