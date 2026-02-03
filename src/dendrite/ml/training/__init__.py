"""
ML utilities for Dendrite decoders.

This module provides modular ML components:
- trainer: TrainingLoop class for config-driven training loops
- losses: Custom loss functions (FocalLoss)
- inference: Uncertainty estimation (MC Dropout)
"""

from dendrite.ml.training.inference import mc_dropout_predict
from dendrite.ml.training.losses import FocalLoss
from dendrite.ml.training.trainer import TrainingLoop

__all__ = [
    "TrainingLoop",
    "FocalLoss",
    "mc_dropout_predict",
]
