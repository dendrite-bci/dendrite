"""Tests for ConfigService — config aggregation and roundtrips."""

from unittest.mock import MagicMock

import pytest

from dendrite.web.services.config_service import ConfigService
from dendrite.web.services.mode_service import ModeService


def make_service() -> ConfigService:
    """Create ConfigService with mock stream service and real mode service."""
    stream_svc = MagicMock()
    stream_svc.has_streams.return_value = False
    stream_svc.get_streams.return_value = {}
    stream_svc.get_modalities_by_stream.return_value = {}

    mode_svc = ModeService()
    return ConfigService(stream_service=stream_svc, mode_service=mode_svc)


def test_build_configuration_defaults():
    svc = make_service()
    cfg = svc.build_configuration()

    assert cfg.study_name == "default_study"
    assert cfg.mode_instances == {}
    assert cfg.output["lsl"]["enabled"] is True
    assert cfg.stream_configs == []


def test_general_config_roundtrip():
    svc = make_service()
    svc.set_general_config({
        "study_name": "my_study",
        "subject_id": "sub01",
        "session_id": "ses01",
        "recording_name": "run1",
    })
    got = svc.get_general_config()
    assert got["study_name"] == "my_study"
    assert got["subject_id"] == "sub01"
    assert got["session_id"] == "ses01"
    assert got["recording_name"] == "run1"


def test_build_includes_mode_instances():
    svc = make_service()
    svc._mode_service.add_instance("P300", {
        "name": "P300",
        "mode": "synchronous",
        "channel_selection": {"eeg": [0, 1, 2, 3]},
        "event_mapping": {1: "left", 2: "right"},
    })
    cfg = svc.build_configuration()
    assert "P300" in cfg.mode_instances


def test_partial_general_update():
    """Setting only some general fields should not clear others."""
    svc = make_service()
    svc.set_general_config({"study_name": "first"})
    svc.set_general_config({"subject_id": "sub01"})
    got = svc.get_general_config()
    assert got["study_name"] == "first"
    assert got["subject_id"] == "sub01"


def test_set_general_config_invalid_chars_raises():
    """Invalid BIDS characters in study_name should raise ValueError."""
    svc = make_service()
    with pytest.raises(ValueError, match="invalid characters"):
        svc.set_general_config({"study_name": "bad<name>"})
