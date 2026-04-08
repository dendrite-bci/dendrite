"""Shared test fixtures for dendrite v2."""

import os
from pathlib import Path

import numpy as np
import pytest

# --- Load .env from project root  ---

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_dotenv = _PROJECT_ROOT / ".env"
if _dotenv.exists():
    for _line in _dotenv.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

# --- External data roots (configure via .env or environment) ---

SWARM_STUDY_ROOT = Path(os.environ.get("SWARM_STUDY_ROOT", ""))
DENDRITE_DATA = _PROJECT_ROOT / "data" / "studies"


# --- Synthetic EEG data ---


@pytest.fixture
def eeg_data_2class():
    """Tiny 2-class EEG dataset: (20, 8, 64) float32, labels [0,1]."""
    X = np.random.RandomState(42).randn(20, 8, 64).astype(np.float32)
    y = np.array([0, 1] * 10)
    return X, y


@pytest.fixture
def eeg_data_3class():
    """Tiny 3-class EEG dataset: (30, 8, 64)."""
    X = np.random.RandomState(42).randn(30, 8, 64).astype(np.float32)
    y = np.array([0, 1, 2] * 10)
    return X, y


@pytest.fixture
def eeg_data_varied_channels():
    """32-channel EEG data for channel variability tests."""
    X = np.random.RandomState(42).randn(20, 32, 128).astype(np.float32)
    y = np.array([0, 1] * 10)
    return X, y


# --- Decoder configs ---


@pytest.fixture
def fast_decoder_config():
    """DecoderConfig dict tuned for speed: 2 epochs, no augmentation."""
    return {
        "model_type": "EEGNet",
        "num_classes": 2,
        "epochs": 2,
        "batch_size": 10,
        "validation_split": 0.2,
        "device": "cpu",
        "use_early_stopping": False,
        "use_augmentation": False,
        "use_swa": False,
        "use_lr_scheduler": False,
        "use_lr_warmup": False,
    }


@pytest.fixture
def event_mapping():
    """Standard 2-class event mapping."""
    return {7: "left_hand", 8: "right_hand"}



