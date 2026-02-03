"""
Offline Adapter for OnlinePreprocessor

Simple wrapper to apply OnlinePreprocessor to continuous MNE data.
"""

from typing import Any

import mne
import numpy as np

from dendrite.processing.preprocessing.preprocessor import OnlinePreprocessor
from dendrite.utils.logger_central import get_logger


def apply_online_preprocessing_offline(
    raw: mne.io.Raw, modality_preprocessing: dict[str, dict[str, Any]], chunk_size: int = 250
) -> tuple[mne.io.Raw, np.ndarray]:
    """
    Apply OnlinePreprocessor to continuous MNE Raw data.

    Args:
        raw: MNE Raw object
        modality_preprocessing: Preprocessing configs per modality (e.g., {'eeg': {...}})
        chunk_size: Samples per chunk for simulated streaming

    Returns:
        (preprocessed_raw, adjusted_events)
    """
    logger = get_logger()

    # Normalize modality keys to lowercase (matches OnlinePreprocessor behavior)
    modality_preprocessing = {k.lower(): v for k, v in modality_preprocessing.items()}

    # Extract events before preprocessing
    events, _ = mne.events_from_annotations(raw) if raw.annotations else (np.array([]), {})

    # Get EEG data
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    if len(eeg_picks) == 0:
        logger.warning("No EEG channels found, returning original data")
        return raw, events

    eeg_data = raw.get_data(picks=eeg_picks)  # (n_channels, n_samples)
    original_sfreq = raw.info["sfreq"]

    # Create OnlinePreprocessor (EOG removal handled separately in DataProcessor)
    preprocessor = OnlinePreprocessor(modality_preprocessing=modality_preprocessing)

    # Process in chunks
    processed_chunks = []
    n_samples = eeg_data.shape[1]

    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        chunk = eeg_data[:, start_idx:end_idx]

        # Process chunk (lowercase modality key)
        processed = preprocessor.process({"eeg": chunk})
        processed_chunks.append(processed["eeg"])

    # Concatenate processed chunks
    processed_data = np.concatenate(processed_chunks, axis=1)

    # Determine new sampling rate (after downsampling)
    downsample_factor = modality_preprocessing.get("eeg", {}).get("downsample_factor", 1)
    new_sfreq = original_sfreq / downsample_factor

    # Create new MNE Raw with processed data
    eeg_ch_names = [raw.ch_names[i] for i in eeg_picks]
    info = mne.create_info(ch_names=eeg_ch_names, sfreq=new_sfreq, ch_types="eeg")
    preprocessed_raw = mne.io.RawArray(processed_data, info, verbose=False)

    # Copy annotations from original raw (needed for event extraction)
    if raw.annotations and len(raw.annotations) > 0:
        if not np.isclose(original_sfreq, new_sfreq):
            # Adjust onset times if sampling rate changed
            scale_factor = new_sfreq / original_sfreq
            preprocessed_raw.set_annotations(
                mne.Annotations(
                    onset=raw.annotations.onset * scale_factor,
                    duration=raw.annotations.duration * scale_factor,
                    description=raw.annotations.description,
                )
            )
        else:
            preprocessed_raw.set_annotations(raw.annotations)

    # Adjust event timing if sampling rate changed
    if len(events) > 0 and not np.isclose(original_sfreq, new_sfreq):
        scale_factor = new_sfreq / original_sfreq
        adjusted_events = events.copy()
        adjusted_events[:, 0] = np.round(events[:, 0] * scale_factor).astype(int)
    else:
        adjusted_events = events

    logger.info(
        f"Offline preprocessing: {original_sfreq:.0f}Hz -> {new_sfreq:.0f}Hz, "
        f"{n_samples} -> {processed_data.shape[1]} samples"
    )

    return preprocessed_raw, adjusted_events
