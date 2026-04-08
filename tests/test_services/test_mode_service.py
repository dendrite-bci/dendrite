"""Tests for ModeService — mode instance CRUD."""

import pytest

from dendrite.web.services.mode_service import ModeService


def _valid_sync_config(name: str = "P300") -> dict:
    """Return a minimal valid synchronous mode config."""
    return {
        "name": name,
        "mode": "synchronous",
        "channel_selection": {"eeg": [0, 1, 2, 3]},
        "event_mapping": {1: "left", 2: "right"},
    }


def _valid_async_config(name: str = "Async1") -> dict:
    """Return a minimal valid asynchronous mode config."""
    return {
        "name": name,
        "mode": "asynchronous",
        "channel_selection": {"eeg": [0, 1, 2, 3]},
        "decoder_source": "database",
    }


def make_service() -> ModeService:
    return ModeService()


def test_add_instance():
    svc = make_service()
    assert svc.add_instance("P300", _valid_sync_config())
    assert "P300" in svc.get_all_instance_names()


def test_add_duplicate_returns_false():
    svc = make_service()
    svc.add_instance("P300", _valid_sync_config())
    assert svc.add_instance("P300", _valid_async_config("P300")) is False


def test_remove_instance():
    svc = make_service()
    svc.add_instance("P300", _valid_sync_config())
    assert svc.remove_instance("P300")
    assert "P300" not in svc.get_all_instance_names()


def test_remove_nonexistent_returns_false():
    svc = make_service()
    assert svc.remove_instance("nope") is False


def test_rename_instance():
    svc = make_service()
    svc.add_instance("old", _valid_sync_config("old"))
    assert svc.rename_instance("old", "new")
    assert "old" not in svc.get_all_instance_names()
    assert "new" in svc.get_all_instance_names()
    assert svc.get_instance("new")["name"] == "new"


def test_generate_unique_name():
    svc = make_service()
    assert svc.generate_unique_name("P300") == "P300"
    svc.add_instance("P300", _valid_sync_config("P300"))
    assert svc.generate_unique_name("P300") == "P300_1"
    svc.add_instance("P300_1", _valid_sync_config("P300_1"))
    assert svc.generate_unique_name("P300") == "P300_2"


def test_get_instance_returns_deep_copy():
    svc = make_service()
    cfg = _valid_sync_config()
    svc.add_instance("test", cfg)
    got = svc.get_instance("test")
    got["channel_selection"]["eeg"].append(99)
    # Original should be unmodified
    assert 99 not in svc.get_instance("test")["channel_selection"]["eeg"]


def test_update_instance():
    svc = make_service()
    svc.add_instance("test", _valid_sync_config("test"))
    updated = _valid_sync_config("test")
    updated["event_mapping"] = {1: "up", 2: "down"}
    assert svc.update_instance("test", updated)
    assert svc.get_instance("test")["event_mapping"] == {1: "up", 2: "down"}


def test_clear_all():
    svc = make_service()
    svc.add_instance("a", _valid_sync_config("a"))
    svc.add_instance("b", _valid_sync_config("b"))
    svc.clear_all()
    assert len(svc.get_all_instance_names()) == 0


def test_add_invalid_config_raises():
    """Adding an invalid config raises ValueError."""
    svc = make_service()
    with pytest.raises(ValueError, match="event"):
        svc.add_instance("bad", {"name": "bad", "mode": "synchronous",
                                  "channel_selection": {"eeg": [0, 1]},
                                  "event_mapping": {1: "only_one"}})


def test_update_invalid_config_raises():
    """Updating with an invalid config raises ValueError."""
    svc = make_service()
    svc.add_instance("test", _valid_sync_config("test"))
    with pytest.raises(ValueError):
        svc.update_instance("test", {"name": "test", "mode": "synchronous",
                                      "channel_selection": {"eeg": [0, 1]},
                                      "event_mapping": {1: "only_one"}})


def test_validated_config_gets_defaults():
    """Validated config should have Pydantic defaults filled in."""
    svc = make_service()
    svc.add_instance("P300", _valid_sync_config())
    instance = svc.get_instance("P300")
    # Pydantic defaults from SynchronousInstanceConfig
    assert "decoder_config" in instance
    assert "training_interval" in instance
