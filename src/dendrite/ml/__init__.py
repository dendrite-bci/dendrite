"""
Dendrite ML - Machine Learning Module

Factory functions for models and decoders. For internal classes, import directly:
- `from dendrite.ml.models.base import ModelBase`
- `from dendrite.ml.decoders.decoder import Decoder`
- `from dendrite.ml.metrics import SynchronousMetrics`
"""

from .decoders import (
    Decoder,
    check_decoder_compatibility,
    create_decoder,
    get_available_decoders,
    get_decoder_capabilities,
    load_decoder,
)
from .models import (
    create_model,
    get_available_models,
)

__all__ = [
    "check_decoder_compatibility",
    "create_decoder",
    "create_model",
    "Decoder",
    "get_available_decoders",
    "get_available_models",
    "get_decoder_capabilities",
    "load_decoder",
]
