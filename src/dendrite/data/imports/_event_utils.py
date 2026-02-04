"""Shared utilities for event handling across loaders."""

import logging

import mne
import numpy as np

logger = logging.getLogger(__name__)


def filter_events_by_codes(
    mne_events: np.ndarray,
    selected_codes: set[int],
) -> np.ndarray:
    """Filter MNE events array to only include specified event codes.

    Args:
        mne_events: MNE events array (n_events, 3)
        selected_codes: Set of event codes to keep

    Returns:
        Filtered events array
    """
    if not selected_codes:
        return mne_events
    mask = np.isin(mne_events[:, 2], list(selected_codes))
    return mne_events[mask]


def encode_labels(
    event_codes: np.ndarray,
    event_mapping: dict[str, int],
) -> np.ndarray:
    """Convert event codes to 0-indexed class labels.

    Args:
        event_codes: Array of event codes from epochs.events[:, 2]
        event_mapping: Dict mapping event names to codes, e.g. {'left': 1, 'right': 2}

    Returns:
        Array of integer labels (0, 1, 2, ...)
    """
    # Create code -> index mapping sorted by code value
    code_to_idx = {}
    for idx, (_name, code) in enumerate(sorted(event_mapping.items(), key=lambda x: x[1])):
        code_to_idx[code] = idx

    return np.array([code_to_idx.get(c, 0) for c in event_codes], dtype=np.int64)


def create_epochs(
    raw: mne.io.Raw,
    events: np.ndarray,
    event_codes: set[int],
    tmin: float,
    tmax: float,
    picks: str | None = None,
) -> mne.Epochs:
    """Create MNE Epochs from raw data and events.

    Args:
        raw: MNE Raw object
        events: MNE events array (n_events, 3)
        event_codes: Set of event codes to include
        tmin: Epoch start time relative to event
        tmax: Epoch end time relative to event
        picks: Channel selection (default: 'eeg')

    Returns:
        MNE Epochs object
    """
    # Create event_id dict for MNE (code as string -> code)
    event_id = {str(c): c for c in event_codes}

    return mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        picks=picks or "eeg",
        baseline=None,
        preload=True,
        verbose=False,
    )


def apply_preprocessing(
    raw: mne.io.Raw,
    lowcut: float | None = None,
    highcut: float | None = None,
    rereference: bool = False,
    target_sample_rate: float | None = None,
) -> mne.io.Raw:
    """Apply preprocessing to raw data.

    Args:
        raw: MNE Raw object (will be modified in place if preload=True)
        lowcut: High-pass filter cutoff (Hz)
        highcut: Low-pass filter cutoff (Hz)
        rereference: Apply common average reference
        target_sample_rate: Target sample rate for resampling (None = no resampling)

    Returns:
        Preprocessed MNE Raw object
    """
    from dendrite.processing.preprocessing.offline_adapter import apply_online_preprocessing_offline

    original_sfreq = raw.info["sfreq"]

    # Calculate downsample factor (must be integer divisor)
    if target_sample_rate and target_sample_rate < original_sfreq:
        if original_sfreq % target_sample_rate != 0:
            logger.warning(
                f"Target sample rate {target_sample_rate} Hz is not an integer divisor of "
                f"source rate {original_sfreq} Hz. Using no resampling."
            )
            downsample_factor = 1
        else:
            downsample_factor = int(original_sfreq // target_sample_rate)
    else:
        downsample_factor = 1

    # Skip if nothing to do
    if lowcut is None and highcut is None and downsample_factor == 1:
        return raw

    n_eeg_channels = len(mne.pick_types(raw.info, eeg=True))
    modality_preprocessing = {
        "EEG": {
            "num_channels": n_eeg_channels,
            "sample_rate": original_sfreq,
            "lowcut": lowcut,
            "highcut": highcut,
            "apply_rereferencing": rereference,
            "downsample_factor": downsample_factor,
        }
    }

    raw_processed, _ = apply_online_preprocessing_offline(raw, modality_preprocessing)
    return raw_processed
