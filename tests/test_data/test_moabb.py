"""Tests for MOABB discovery — skip if moabb is not installed."""

import pytest

moabb = pytest.importorskip("moabb")

from dendrite.data.loaders import (
    discover_moabb_datasets,
    get_moabb_dataset_info,
)


class TestMOAABDiscovery:
    def test_discover_returns_dicts(self):
        """discover_moabb_datasets returns non-empty list of dicts."""
        datasets = discover_moabb_datasets(cache=False)
        assert len(datasets) > 0
        ds = datasets[0]
        assert "code" in ds
        assert "paradigm" in ds
        assert "subjects" in ds
        assert "events" in ds

    def test_get_dataset_info_known(self):
        """Known dataset returns dict with expected keys."""
        info = get_moabb_dataset_info("BNCI2014_001")
        assert info is not None
        assert "code" in info
        assert "paradigm" in info
        assert "n_subjects" in info
        assert "config" in info
