"""Shared fixtures for processing modes tests."""

import logging

import numpy as np
import pytest

from dendrite.processing.modes.mode_utils import Buffer


@pytest.fixture
def logger():
    return logging.getLogger("test_modes")


@pytest.fixture
def make_buffer(logger):
    """Factory fixture for creating Buffer instances."""

    def _make(modalities=None, buffer_size=10):
        if modalities is None:
            modalities = ["eeg"]
        return Buffer(modalities=modalities, buffer_size=buffer_size, logger=logger)

    return _make


@pytest.fixture
def make_sample():
    """Factory fixture for creating data samples."""

    def _make(n_channels=4, value=1.0, marker=None, timestamp=None, modalities=None):
        if modalities is None:
            modalities = ["eeg"]
        sample = {}
        for mod in modalities:
            sample[mod] = np.full((n_channels, 1), value, dtype=np.float32)
        if marker is not None:
            sample["markers"] = marker
        sample["_receive_ns"] = timestamp or 0
        return sample

    return _make


def fill_buffer(buf, make_sample_fn, n_samples, n_channels=4, **kwargs):
    """Helper: fill a buffer with n_samples sequential-valued samples."""
    for i in range(n_samples):
        sample = make_sample_fn(n_channels=n_channels, value=float(i), **kwargs)
        buf.add_sample(sample)
