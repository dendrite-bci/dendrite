"""
Dendrite ML - Machine Learning Module

Factory functions for models and decoders.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_WAIT_POLICY", "passive")
os.environ.setdefault("GOMP_SPINCOUNT", "0")

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
