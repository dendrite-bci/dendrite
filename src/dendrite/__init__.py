"""
Dendrite - Open-source platform for real-time neural signal processing and brain-computer interfaces.
"""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version("dendrite")
except PackageNotFoundError:
    __version__ = "unknown"

# Project paths (src/dendrite/__init__.py -> parents[2] -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

from dendrite.ml.decoders import create_decoder, get_available_decoders, load_decoder
from dendrite.ml.models import create_model, get_available_models
from dendrite.processing.modes import create_mode, get_available_modes

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "create_mode",
    "get_available_modes",
    "create_decoder",
    "load_decoder",
    "get_available_decoders",
    "create_model",
    "get_available_models",
]
