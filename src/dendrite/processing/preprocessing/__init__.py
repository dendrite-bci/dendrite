"""Preprocessing components for brain-machine interfaces."""

from dendrite.processing.preprocessing.preprocessor import ModalityProcessor, OnlinePreprocessor
from dendrite.processing.preprocessing.scalers import ChannelScaler

__all__ = ["ChannelScaler", "ModalityProcessor", "OnlinePreprocessor"]
