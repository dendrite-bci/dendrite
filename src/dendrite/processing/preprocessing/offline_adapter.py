"""Offline adapter: apply OnlinePreprocessor to continuous numpy data in chunks."""

from typing import Any

import numpy as np

from dendrite.processing.preprocessing.preprocessor import OnlinePreprocessor
from dendrite.utils.logger_central import get_logger


def apply_preprocessing_offline(
    data: np.ndarray,
    sample_rate: float,
    modality: str,
    config: dict[str, Any],
    chunk_size: int = 250,
    bad_channels: dict[str, list[int]] | None = None,
) -> np.ndarray:
    """Apply OnlinePreprocessor to continuous data in chunks.

    Args:
        data: Raw data array (n_channels, n_samples).
        sample_rate: Sampling rate in Hz.
        modality: Modality name (e.g. "eeg").
        config: Preprocessing config dict for this modality.
        chunk_size: Samples per processing chunk.
        bad_channels: Per-modality bad channel indices.

    Returns:
        Processed data array (n_channels_out, n_samples_out).
    """
    logger = get_logger()

    config = {**config, "num_channels": data.shape[0], "sample_rate": sample_rate}
    preprocessor = OnlinePreprocessor({modality: config})

    processed_chunks = []
    n_samples = data.shape[1]

    for start in range(0, n_samples, chunk_size):
        chunk = data[:, start : start + chunk_size]
        processed = preprocessor.process({modality: chunk}, bad_channels=bad_channels)
        processed_chunks.append(processed[modality])

    processed_data = np.concatenate(processed_chunks, axis=1)

    downsample_factor = config.get("downsample_factor", 1)
    new_rate = sample_rate / downsample_factor

    logger.info(
        f"Offline preprocessing ({modality}): {sample_rate:.0f}Hz -> {new_rate:.0f}Hz, "
        f"{n_samples} -> {processed_data.shape[1]} samples"
    )

    return processed_data


def apply_eeg_eog_correction_offline(
    eeg: np.ndarray,
    eog: np.ndarray,
    sample_rate: float,
    eeg_config: dict[str, Any],
    chunk_size: int | None = None,
) -> np.ndarray:
    """Full-recording EEG preprocessing with EOG correction, via the live code path.

    Builds an ``OnlinePreprocessor`` and streams the recording through ``process()``
    in chunks, so the offline result goes through exactly the live code path: the
    adaptive EOG regression converges along the same trajectory it would online
    (refit timing can differ by at most one chunk boundary).  Returns the processed
    (CAR + band-passed + EOG-corrected) EEG.

    The correction is configured by the EEG ``apply_eog_correction`` flag alone and
    takes its reference from the raw EOG passed to ``process()`` — no ``eog`` config
    entry is needed.
    """
    chunk_size = chunk_size or int(sample_rate)
    # This helper is only called when correction is wanted — ensure it activates.
    eeg_cfg = {**eeg_config, "apply_eog_correction": True,
               "num_channels": eeg.shape[0], "sample_rate": sample_rate}
    pre = OnlinePreprocessor({"eeg": eeg_cfg})

    out = []
    for start in range(0, eeg.shape[1], chunk_size):
        chunk = pre.process({
            "eeg": eeg[:, start : start + chunk_size],
            "eog": eog[:, start : start + chunk_size],
        })
        out.append(chunk["eeg"])
    return np.concatenate(out, axis=1)
