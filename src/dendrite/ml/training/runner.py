"""Training runner — pure ML functions for decoder training.

Module-level functions suitable for subprocess execution (pickling).
"""

import logging
import time
from collections.abc import Callable
from queue import Full

import numpy as np

from dendrite.ml.decoders.decoder import Decoder
from dendrite.ml.decoders.decoder_schemas import DecoderConfig

logger = logging.getLogger(__name__)


def decoder_config_from_dict(
    config: dict, num_classes: int, input_shapes: dict,
) -> DecoderConfig:
    """Build DecoderConfig from a training config dict.

    Explicit num_classes/input_shapes override any values in config
    (they're computed from the actual data, not caller-provided).
    """
    return DecoderConfig(**{**config, "num_classes": num_classes, "input_shapes": input_shapes})


def train_decoder(
    X: np.ndarray, y: np.ndarray,
    decoder_config: DecoderConfig,
    modality: str = "eeg",
    epoch_callback: Callable[..., None] | None = None,
    stop_event=None,
) -> Decoder:
    """Create decoder, set input shapes, fit, return fitted decoder."""
    decoder = Decoder(decoder_config)
    decoder.input_shapes = {modality: list(X.shape[1:])}
    decoder.fit(X, y, epoch_callback=epoch_callback, stop_event=stop_event)
    return decoder


def run_training(
    X: np.ndarray, y: np.ndarray, config: dict, save_name: str,
    progress_queue=None, stop_event=None,
) -> dict:
    """Train a decoder (runs in subprocess). Returns {path, elapsed, metrics, eval_metrics}."""
    import os

    max_threads = config.get("max_threads", 2)
    os.environ["OMP_NUM_THREADS"] = str(max_threads)
    os.environ["MKL_NUM_THREADS"] = str(max_threads)

    import torch
    torch.set_num_threads(max_threads)

    num_classes = int(np.max(y) + 1)
    modality = config.get("modality", "eeg")
    input_shapes = {modality: list(X.shape[1:])}
    decoder_config = decoder_config_from_dict(config, num_classes, input_shapes)

    epoch_callback: Callable[..., None] | None = None
    if progress_queue is not None:
        def _epoch_callback(epoch, total, train_loss, train_acc, val_loss, val_acc):
            try:
                progress_queue.put_nowait({
                    "epoch": epoch, "total_epochs": total,
                    "train_loss": train_loss, "train_acc": train_acc,
                    "val_loss": val_loss, "val_acc": val_acc,
                })
            except (Full, OSError):
                pass
        epoch_callback = _epoch_callback

    start_time = time.time()
    decoder = train_decoder(X, y, decoder_config, modality, epoch_callback, stop_event)
    elapsed = time.time() - start_time

    if progress_queue is not None:
        progress_queue.put_nowait(None)  # sentinel

    # Eval metrics are computed by the caller on properly held-out data
    # (split before training). Computing them here would evaluate on training
    # data since decoder.fit(X, y) sees all of X.
    eval_metrics = {}

    # Store epoch offsets so the decoder is self-describing
    if config.get("epoch_tmin") is not None:
        decoder.config.epoch_tmin = config["epoch_tmin"]
    if config.get("epoch_tmax") is not None:
        decoder.config.epoch_tmax = config["epoch_tmax"]

    # Store preprocessing config so the decoder is self-describing
    mode_preproc = config.get("mode_preprocessing", {})
    if mode_preproc:
        from dendrite.processing.preprocessing.preprocessing_schemas import (
            ModalityPreprocessing,
            PreprocessingConfig,
        )
        mod_preproc = {}
        for mod, mp in mode_preproc.items():
            if isinstance(mp, dict):
                mod_preproc[mod] = ModalityPreprocessing(**mp)
            elif hasattr(mp, "model_dump"):
                mod_preproc[mod] = mp
        if mod_preproc:
            decoder.config.preprocessing_config = PreprocessingConfig(
                modality_preprocessing=mod_preproc,
            )

    # Store channel labels so the decoder is self-describing
    channel_labels = config.get("channel_labels")
    if channel_labels:
        decoder.channel_labels = channel_labels

    # Training data provenance
    rec_ids = config.get("recording_ids")
    rec_id = config.get("recording_id")
    if rec_ids:
        decoder.config.training_recording_ids = rec_ids
    elif rec_id:
        decoder.config.training_recording_ids = [rec_id]
    if config.get("file_identifier"):
        decoder.config.training_file_identifier = config["file_identifier"]

    path = decoder.save(save_name)
    return {
        "path": path, "elapsed": elapsed, "n_epochs": len(y),
        "training_metrics": decoder.get_training_metrics() or {},
        "eval_metrics": eval_metrics,
    }
