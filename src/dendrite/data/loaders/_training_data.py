"""Training data preparation — composes loaders + epoching for training workflows."""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from ._types import EpochedData, RawData, build_preprocessing_config

logger = logging.getLogger(__name__)


def load_moabb_for_training(
    config: dict[str, Any],
    broadcast_step: Callable[[str], None] | None = None,
) -> EpochedData:
    """Load a MOABB dataset for training.

    Two modes:
    - use_paradigm_epochs=True: MOABB's paradigm handles preprocessing + epoching
      (matches published benchmarks)
    - use_paradigm_epochs=False (default): load continuous → RawData → our pipeline
      (custom preprocessing, consistent with recordings)
    """
    from .moabb_loader import MoabbConfig, MOABBLoader

    step = broadcast_step or (lambda s: None)

    dataset_code = config["dataset_code"]
    subject = config.get("subject", 1)
    paradigm_name = config.get("paradigm", "MotorImagery")

    step(f"Downloading & loading {dataset_code} (subject {subject})...")
    loader = MOABBLoader(MoabbConfig(dataset=dataset_code, paradigm=paradigm_name))

    if config.get("use_paradigm_epochs", False):
        step("Using paradigm preprocessing...")
        return loader.load_paradigm_epochs(subject)

    step("Preprocessing & epoching...")
    loaded = loader.load_as_raw(subject)
    epoched = load_epochs(config, loaded)
    epoched.source = "moabb"
    epoched.source_id = dataset_code
    epoched.metadata.update({
        "paradigm": paradigm_name,
        "subject": subject,
        "dataset_code": dataset_code,
    })
    return epoched


def load_epochs(
    config: dict[str, Any],
    source: "str | RawData",
    *,
    swmr: bool = False,
) -> EpochedData:
    """Core epoch loading: load → filter modality → preprocess → epoch.

    Args:
        config: Training config (event_mapping, preprocessing, etc.)
        source: File path (str) or pre-loaded RawData
        swmr: Use SWMR mode for live session HDF5 files
    """
    modalities = config.get("modalities", [config.get("modality")])
    modality = modalities[0] if modalities else None

    if isinstance(source, RawData):
        loaded = source
    elif swmr:
        from .raw_h5_loader import RawH5Loader
        loaded = RawH5Loader(source, swmr=True).load(modality=modality)
    else:
        from . import load_file
        loaded = load_file(source, modality=modality)

    if not modality and loaded.channel_types:
        from collections import Counter
        data_types = [t for t in loaded.channel_types if t != "markers"]
        modality = Counter(data_types).most_common(1)[0][0] if data_types else None

    if modality:
        loaded.filter_modality(modality)

    # Preprocessing: prefer mode_preprocessing (online), fall back to workbench config
    mode_preproc = config.get("mode_preprocessing", {})
    if modality and modality in mode_preproc:
        loaded.preprocess(
            mode_preproc[modality],
            modality=modality,
            bad_channels=config.get("effective_bad"),
        )
    else:
        preprocessing = build_preprocessing_config(config)
        if preprocessing:
            loaded.preprocess(preprocessing)

    # Workbench: selected_events → rebuild event_mapping from recording's own event_id
    selected_events = config.get("selected_events")
    if selected_events and loaded.event_id:
        config = {**config, "event_mapping": {
            code: name for name, code in loaded.event_id.items()
            if name in selected_events
        }}

    # Ensure int keys for event_mapping
    if "event_mapping" in config:
        config = {**config, "event_mapping": {
            int(k): v for k, v in config["event_mapping"].items()
        }}

    # Sync channel metadata if channel_indices subset was requested
    channel_indices = config.get("channel_indices")
    if channel_indices:
        valid = [i for i in channel_indices if i < loaded.n_channels]
        loaded.channel_names = [loaded.channel_names[i] for i in valid]
        loaded.channel_types = [loaded.channel_types[i] for i in valid]

    epoched = loaded.epoch(config)
    epoched.encode_labels()
    return epoched


def merge_recordings(
    recording_ids: list[int],
    config: dict[str, Any],
    data_service: Any,
    broadcast_step: Callable[[str], None] | None = None,
) -> EpochedData:
    """Load multiple recordings and merge into a single EpochedData."""
    step = broadcast_step or (lambda s: None)
    datasets: list[EpochedData] = []
    subject_counts: dict[str, int] = {}

    for i, rid in enumerate(recording_ids):
        rec = data_service.recordings.get_by_id(rid)
        if not rec:
            raise ValueError(f"Recording {rid} not found")
        step(f"Loading recording {i + 1}/{len(recording_ids)}...")
        data = load_epochs(config, rec["hdf5_file_path"])
        data.metadata.update({
            "recording_id": rid,
            "recording_name": rec["recording_name"],
        })
        datasets.append(data)
        subj = rec["subject_id"]
        subject_counts[subj] = subject_counts.get(subj, 0) + 1

    if len(datasets) == 1:
        d = datasets[0]
        d.metadata["n_recordings"] = 1
        d.metadata["subject_breakdown"] = subject_counts
        return d

    # Validate spatial shapes match (channels + timepoints)
    ref_shape = datasets[0].X.shape[1:]
    for i, d in enumerate(datasets[1:], 1):
        if d.X.shape[1:] != ref_shape:
            raise ValueError(
                f"Shape mismatch: recording {recording_ids[0]} has shape "
                f"{ref_shape}, recording {recording_ids[i]} has shape "
                f"{d.X.shape[1:]}. All recordings must have matching "
                f"channels and epoch length."
            )

    # Check if label maps are compatible
    ref_labels = datasets[0].metadata.get("label_map", {})
    all_same = all(
        d.metadata.get("label_map", {}) == ref_labels for d in datasets[1:]
    )

    if all_same:
        X = np.concatenate([d.X for d in datasets], axis=0)
        y = np.concatenate([d.y for d in datasets], axis=0)
        class_names = datasets[0].metadata.get("class_names", [])
    else:
        # Unified encoding: decode each y to raw event codes, re-encode
        all_class_names = set()
        for d in datasets:
            all_class_names.update(d.metadata.get("class_names", []))
        class_names = sorted(all_class_names)
        ref_labels = {name: i for i, name in enumerate(class_names)}

        ys = []
        for d in datasets:
            inv = {v: k for k, v in d.metadata["label_map"].items()}
            raw = np.array([inv[int(label)] for label in d.y])
            ys.append(np.array(
                [ref_labels[code] for code in raw], dtype=np.int64,
            ))
        X = np.concatenate([d.X for d in datasets], axis=0)
        y = np.concatenate(ys, axis=0)

    merged_counts: dict[str, int] = {}
    for d in datasets:
        for k, v in d.metadata.get("class_counts", {}).items():
            merged_counts[str(k)] = merged_counts.get(str(k), 0) + v

    # Merge event_id mappings from all recordings
    merged_event_id: dict[str, int] = {}
    for d in datasets:
        merged_event_id.update(d.metadata.get("event_id", {}))

    return EpochedData(
        X=X, y=y,
        metadata={
            "paradigm": "Recording",
            "class_names": class_names,
            "class_counts": merged_counts,
            "label_map": ref_labels,
            "n_channels": int(X.shape[1]),
            "n_times": int(X.shape[2]),
            "n_recordings": len(datasets),
            "recording_ids": recording_ids,
            "subject_breakdown": subject_counts,
            "dataset_name": f"{len(datasets)} recordings (pooled)",
            "event_id": merged_event_id,
        },
        source="recording",
        source_id=",".join(str(rid) for rid in recording_ids),
        sample_rate=datasets[0].sample_rate,
        channel_names=datasets[0].channel_names,
        channel_types=datasets[0].channel_types,
    )


def load_study_history(
    request: dict[str, Any],
    study_name: str,
    data_service: Any,
    ref_shape: tuple | None,
    recording_ids: list[int] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load compatible historical epochs from other recordings in the study.

    Args:
        request: Training request config (event_mapping, preprocessing, etc.)
        study_name: Study to query recordings from
        data_service: Data service for DB queries
        ref_shape: Expected (n_channels, n_times) shape, or None to infer from first recording
        recording_ids: If provided, only load these specific recordings

    Returns (X, y) concatenated from all compatible recordings, or (None, None).
    Skips recordings that fail to load or have incompatible shapes/events.
    """
    if recording_ids:
        recordings = [data_service.recordings.get_by_id(rid) for rid in recording_ids]
        recordings = [r for r in recordings if r]
    else:
        study = data_service.studies.get_or_create(study_name)
        if not study:
            return None, None
        recordings = data_service.recordings.get_recordings_by_study(study["study_id"])

    current_file_id = request.get("file_identifier", "")

    datasets: list[tuple[np.ndarray, np.ndarray]] = []
    for rec in recordings:
        # Skip the current live session
        if current_file_id and current_file_id in rec.get("file_identifier", ""):
            continue

        rid = rec["recording_id"]
        file_path = rec.get("hdf5_file_path")
        if not file_path:
            continue
        try:
            epoched = load_epochs(request, file_path)
            if len(epoched.X) == 0:
                continue
            # Shape validation: infer from first recording if no reference
            if ref_shape is None:
                ref_shape = epoched.X.shape[1:]
            elif epoched.X.shape[1:] != ref_shape:
                logger.debug(f"Skipping recording {rid}: shape mismatch")
                continue
            datasets.append((epoched.X, epoched.y))
            logger.info(f"Loaded {len(epoched.y)} historical epochs from recording {rid}")
        except Exception as e:
            logger.debug(f"Skipping recording {rid}: {e}")
            continue

    if not datasets:
        return None, None

    X = np.concatenate([d[0] for d in datasets], axis=0)
    y = np.concatenate([d[1] for d in datasets], axis=0)
    return X, y


