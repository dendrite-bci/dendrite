"""Shared types for data loaders."""

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def build_preprocessing_config(config: dict[str, Any]) -> Any | None:
    """Extract preprocessing fields from a config dict into a ModalityPreprocessing."""
    from dendrite.processing.preprocessing.preprocessing_schemas import ModalityPreprocessing

    fields: dict[str, Any] = {}
    if config.get("lowcut") is not None:
        fields["lowcut"] = config["lowcut"]
    if config.get("highcut") is not None:
        fields["highcut"] = config["highcut"]
    if config.get("apply_rereferencing"):
        fields["apply_rereferencing"] = True
    return ModalityPreprocessing(**fields) if fields else None


def _interpolate_bad_channels(
    data: np.ndarray, channel_names: list[str], bad_indices: list[int],
) -> None:
    """Correlation-based bad-channel interpolation, in-place.

    Offline counterpart of the live interpolation freeze: the correlation matrix
    comes from the recording data itself.
    """
    if not bad_indices or not channel_names:
        return
    from dendrite.processing.preprocessing.interpolation import (
        CorrelationInterpolationMatrix,
        InterpolationApplicator,
    )

    corr = np.nan_to_num(np.corrcoef(data), nan=0.0)
    result = CorrelationInterpolationMatrix.compute(
        channel_names, bad_indices, corr, bad_during_warmup=bad_indices,
    )
    if result is not None:
        InterpolationApplicator(result).apply(data)


@dataclass
class RawData:
    """Standardized output from file loaders.

    Supports method-chaining for the common load → filter → preprocess → epoch pipeline:
        loaded = load_file(path)
        loaded.filter_modality("eeg")
        loaded.preprocess(config)
        epoched = loaded.epoch(config)
    """

    data: np.ndarray  # (channels, samples)
    channel_names: list[str]
    channel_types: list[str]  # lowercase: 'eeg', 'eog', 'emg', etc.
    sample_rate: float
    events: list[tuple[int, int]]  # [(sample_idx, event_code), ...]
    event_id: dict[str, int] | None = None  # event_name → protocol code
    units: str = "V"

    @property
    def n_channels(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    @property
    def duration(self) -> float:
        return self.n_samples / self.sample_rate if self.sample_rate > 0 else 0.0

    def filter_modality(self, modality: str) -> None:
        """Filter channels in-place to keep only the given modality."""
        if not self.channel_types:
            return
        mod = modality.lower()
        indices = [i for i, t in enumerate(self.channel_types) if t == mod]
        if not indices:
            logger.warning(
                f"No '{modality}' channels found "
                f"(available: {set(self.channel_types)})"
            )
            return
        if len(indices) < len(self.channel_names):
            self.data = self.data[indices]
            self.channel_names = [self.channel_names[i] for i in indices]
            self.channel_types = [self.channel_types[i] for i in indices]

    def preprocess_with_eog_correction(
        self, eeg_config,
        bad_channels: dict[str, list[int]] | None = None,
    ) -> bool:
        """CAR + bandpass the EEG and regress out EOG, via the live code path.

        Offline counterpart of the online EOG correction.  Interpolates bad EEG
        channels, then runs EEG+EOG through an ``OnlinePreprocessor`` (EOG regression
        fit on the whole recording in the live CAR-referenced, band-limited domain),
        so training data follows the same adaptive trajectory online inference
        produces.  Collapses ``self`` to the processed EEG channels in-place.

        The correction is driven by the EEG ``apply_eog_correction`` flag and the raw
        EOG channels (``eog``/``veog``/``heog``) found in the recording — no ``eog``
        config is needed.  Returns True if correction was applied; False (no EOG / no
        EEG present) so the caller can fall back to the normal single-modality
        preprocess.
        """
        if not self.channel_types:
            return False
        types = [t.lower() for t in self.channel_types]
        eeg_idx = [i for i, t in enumerate(types) if t == "eeg"]
        eog_idx = [i for i, t in enumerate(types) if t in ("eog", "veog", "heog")]
        if not eeg_idx or not eog_idx:
            return False

        from dendrite.processing.preprocessing.offline_adapter import (
            apply_eeg_eog_correction_offline,
        )

        def _cfg(c) -> dict:
            return c.model_dump(exclude_none=True) if hasattr(c, "model_dump") else dict(c or {})

        eeg = self.data[eeg_idx].astype(np.float64)
        if bad_channels:
            _interpolate_bad_channels(
                eeg, [self.channel_names[i] for i in eeg_idx],
                bad_channels.get("eeg", []),
            )

        eeg_cfg = _cfg(eeg_config)
        processed = apply_eeg_eog_correction_offline(
            eeg,
            self.data[eog_idx].astype(np.float64),
            self.sample_rate, eeg_cfg,
        )
        self.data = processed
        self.channel_names = [self.channel_names[i] for i in eeg_idx]
        self.channel_types = [self.channel_types[i] for i in eeg_idx]
        ds = eeg_cfg.get("downsample_factor", 1)
        if ds and ds > 1:
            self.sample_rate = self.sample_rate / ds
        return True

    def pick_channels(self, names: list[str]) -> None:
        """Keep only the named channels, in-place."""
        name_set = set(names)
        indices = [i for i, n in enumerate(self.channel_names) if n in name_set]
        if not indices:
            return
        self.data = self.data[indices]
        self.channel_names = [self.channel_names[i] for i in indices]
        self.channel_types = [self.channel_types[i] for i in indices]

    def preprocess(
        self, config, *, modality: str = "eeg",
        bad_channels: dict[str, list[int]] | None = None,
    ) -> None:
        """Apply bandpass, CAR, resampling in-place. Config is ModalityPreprocessing or dict."""
        if config is None:
            return

        from dendrite.processing.preprocessing.offline_adapter import (
            apply_preprocessing_offline,
        )

        cfg = config.model_dump(exclude_none=True) if hasattr(config, "model_dump") else dict(config)

        # Bad channel interpolation (before bandpass/CAR)
        if bad_channels:
            _interpolate_bad_channels(
                self.data, self.channel_names, bad_channels.get(modality, []),
            )

        self.data = apply_preprocessing_offline(
            self.data, self.sample_rate, modality, cfg,
            chunk_size=int(self.sample_rate),
            bad_channels=bad_channels,
        )
        # Update sample rate if downsampled
        ds = cfg.get("downsample_factor", 1)
        if ds and ds > 1:
            self.sample_rate = self.sample_rate / ds

    def epoch(self, config: dict[str, Any]) -> "EpochedData":
        """Extract event-locked epochs. Returns EpochedData."""
        if not self.events:
            raise ValueError("No events found. Cannot create epochs.")

        event_mapping = config.get("event_mapping")
        label_mapping = config.get("label_mapping")
        channel_indices = config.get("channel_indices")
        tmin = config.get("epoch_tmin", 0.0)
        tmax = config.get("epoch_tmax", 2.0)
        s0 = int(tmin * self.sample_rate)
        s1 = int(tmax * self.sample_rate)
        if s1 <= s0:
            raise ValueError(f"Invalid epoch window: {tmin}s to {tmax}s")

        epochs, labels = [], []
        for idx, code in self.events:
            if event_mapping and code not in event_mapping:
                continue
            if event_mapping:
                name = event_mapping[code]
                if label_mapping and name not in label_mapping:
                    continue
                label = label_mapping[name] if label_mapping else code
            else:
                label = code

            lo, hi = idx + s0, idx + s1
            if lo < 0 or hi > self.data.shape[1]:
                continue
            ep = self.data[:, lo:hi]
            if channel_indices:
                ep = ep[channel_indices, :]
            epochs.append(ep)
            labels.append(label)

        if not epochs:
            raise ValueError("No valid epochs could be extracted")

        if config.get("include_background", False):
            bg_label = label_mapping["rest"] if label_mapping and "rest" in label_mapping else max(labels) + 1
            n_target = min(Counter(labels).values())
            bg_epochs = self._sample_background_epochs(s0, s1, n_target, channel_indices)
            if bg_epochs:
                epochs.extend(bg_epochs)
                labels.extend([bg_label] * len(bg_epochs))
                logger.info(f"Background: sampled {len(bg_epochs)} rest epochs (label={bg_label})")

        X = np.stack(epochs).astype(np.float32)
        y = np.array(labels, dtype=np.int64)

        if config.get("use_epoch_qc", True):
            X, y = self._apply_epoch_qc(X, y)

        return EpochedData(
            X=X, y=y,
            metadata={"event_id": self.event_id} if self.event_id else {},
            sample_rate=self.sample_rate,
            channel_names=self.channel_names,
            channel_types=self.channel_types,
        )

    # --- Internal helpers ---

    _MNE_TYPE_MAP = {"markers": "stim"}

    def to_mne_raw(self):
        """Convert to MNE RawArray with channel info and event annotations."""
        import mne

        types = [self._MNE_TYPE_MAP.get(t, t) for t in self.channel_types]
        info = mne.create_info(
            self.channel_names,
            self.sample_rate,
            ch_types=types,  # type: ignore[arg-type]
        )
        raw = mne.io.RawArray(self.data, info, verbose=False)
        if self.events:
            raw.set_annotations(mne.Annotations(
                onset=[idx / self.sample_rate for idx, _ in self.events],
                duration=[0.0] * len(self.events),
                description=[str(code) for _, code in self.events],
            ))
        return raw

    @staticmethod
    def _apply_epoch_qc(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from dendrite.data.quality import EpochQualityChecker

        checker = EpochQualityChecker()
        good = np.array([not checker.check({"epoch": X[i]})[0] for i in range(len(X))])
        n_bad = int(np.sum(~good))
        if n_bad > 0:
            logger.info(f"Epoch QC: rejected {n_bad}/{len(X)} ({checker.get_stats_summary()})")
            X, y = X[good], y[good]
            if len(X) == 0:
                raise ValueError("All epochs rejected by quality checker")
        return X, y

    def _sample_background_epochs(
        self,
        s0: int, s1: int,
        n_target: int,
        channel_indices: list[int] | None,
    ) -> list[np.ndarray]:
        """Sample random epochs from inter-trial gaps for background/rest class."""
        epoch_len = s1 - s0
        n_total = self.data.shape[1]
        sorted_events = sorted(self.events, key=lambda e: e[0])

        # Build gaps: before first event, between events, after last event
        boundaries = [(0, sorted_events[0][0] + s0)]
        for i in range(len(sorted_events) - 1):
            gap_start = sorted_events[i][0] + s1
            gap_end = sorted_events[i + 1][0] + s0
            boundaries.append((gap_start, gap_end))
        boundaries.append((sorted_events[-1][0] + s1, n_total))

        # Keep gaps that can fit at least one epoch
        valid_gaps = [(a, b) for a, b in boundaries if b - a >= epoch_len]
        if not valid_gaps:
            return []

        # Weight each gap by how many start positions it offers
        weights = np.array([b - a - epoch_len + 1 for a, b in valid_gaps], dtype=np.float64)
        weights /= weights.sum()

        rng = np.random.default_rng(42)
        result = []
        gap_indices = rng.choice(len(valid_gaps), size=n_target, p=weights)
        for gi in gap_indices:
            a, b = valid_gaps[gi]
            start = rng.integers(a, b - epoch_len + 1)
            ep = self.data[:, start : start + epoch_len]
            if channel_indices:
                ep = ep[channel_indices, :]
            result.append(ep)
        return result


@dataclass
class EpochedData:
    """In-memory cache of loaded (X, y) data for ML workflows."""

    X: np.ndarray  # (n_samples, n_channels, n_times)
    y: np.ndarray  # (n_samples,)
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "internal"
    source_id: str = ""
    sample_rate: float = 250.0
    channel_names: list[str] = field(default_factory=list)
    channel_types: list[str] = field(default_factory=list)

    def info(self) -> dict[str, Any]:
        """Return JSON-safe metadata dict (no numpy arrays)."""
        return {
            "source": self.source, "source_id": self.source_id,
            "sample_rate": self.sample_rate,
            "channel_names": self.channel_names, "channel_types": self.channel_types,
            "n_samples": int(self.X.shape[0]),
            "n_channels": int(self.X.shape[1]) if self.X.ndim >= 2 else 0,
            "n_times": int(self.X.shape[2]) if self.X.ndim >= 3 else 0,
            "shape": list(self.X.shape), "metadata": self.metadata,
        }

    def encode_labels(self) -> dict[str, int]:
        """Re-encode y to 0..N-1 class indices. Returns the label_map used."""
        raw_codes = sorted(set(self.y.tolist()))
        # Resolve integer event codes to human-readable names via event_id
        event_id = self.metadata.get("event_id", {})
        code_to_name = {code: name for name, code in event_id.items()} if event_id else {}
        class_names = [code_to_name.get(code, str(code)) for code in raw_codes]
        label_map = {code: i for i, code in enumerate(raw_codes)}
        self.y = np.array([label_map[v] for v in self.y], dtype=np.int64)
        self.metadata["class_names"] = class_names
        idx_counts = Counter(self.y.tolist())
        self.metadata["class_counts"] = {name: idx_counts.get(i, 0) for i, name in enumerate(class_names)}
        self.metadata["label_map"] = label_map
        return label_map

    def split_eval(self, ratio: float = 0.2) -> "EpochedData":
        """Split off a stratified eval portion. Modifies self in-place, returns eval."""
        from sklearn.model_selection import StratifiedShuffleSplit

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=42)
        train_idx, eval_idx = next(splitter.split(self.X, self.y))

        eval_data = EpochedData(
            X=self.X[eval_idx], y=self.y[eval_idx],
            metadata={**self.metadata, "auto_split": True},
            source=self.source, source_id=self.source_id,
            sample_rate=self.sample_rate,
            channel_names=list(self.channel_names),
            channel_types=list(self.channel_types),
        )
        self.X = self.X[train_idx]
        self.y = self.y[train_idx]
        return eval_data
