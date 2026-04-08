"""
ML utilities for Dendrite decoders.

This module provides modular ML components:
- trainer: TrainingLoop class for config-driven training loops
- losses: Custom loss functions (FocalLoss)
"""

from dendrite.ml.training.losses import FocalLoss
from dendrite.ml.training.trainer import TrainingLoop

__all__ = [
    "TrainingLoop",
    "FocalLoss",
]

# NOTE: runner.py functions (run_training, train_decoder, decoder_config_from_dict)
# are imported directly via dendrite.ml.training.runner to avoid circular imports
# (runner imports Decoder which imports TrainingLoop from this package).
