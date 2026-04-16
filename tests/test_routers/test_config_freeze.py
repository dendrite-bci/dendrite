"""Tests for config freeze — mutation endpoints blocked during recording."""

import pytest

from dendrite.web.services.pipeline_service import PipelineService


@pytest.fixture
def simulate_recording(app):
    """Monkeypatch PipelineService.is_recording to return True."""
    original = PipelineService.is_recording
    PipelineService.is_recording = property(lambda self: True)
    yield
    PipelineService.is_recording = original


# --- Config mutation endpoints blocked during recording ---


@pytest.mark.parametrize("method,path,body", [
    # DAQ/Processor-level config is still fully blocked during recording
    ("PUT", "/api/config/general", {
        "study_name": "test", "subject_id": "01",
        "session_id": "01", "recording_name": "task",
    }),
    ("PUT", "/api/config/output", {"protocols": {}}),
    ("POST", "/api/config/load?file_path=test.json", None),
    ("POST", "/api/streams/configure", {"selected_uids": []}),
])
async def test_mutation_blocked_when_recording(client, simulate_recording, method, path, body):
    resp = await client.request(method, path, json=body)
    assert resp.status_code == 409
    assert "recording" in resp.json()["detail"].lower()


# Mode CRUD is allowed during recording when the target mode is not running
async def test_add_mode_allowed_during_recording(client, simulate_recording):
    """Adding a new mode config is allowed during recording."""
    resp = await client.post("/api/modes", json={
        "mode": "synchronous",
        "config": {
            "channel_selection": {"eeg": [0, 1, 2, 3]},
            "event_mapping": {"1": "left", "2": "right"},
        },
    })
    assert resp.status_code == 200


# --- GET endpoints still work during recording ---


@pytest.mark.parametrize("path", ["/api/config/general", "/api/modes", "/api/streams"])
async def test_get_allowed_when_recording(client, simulate_recording, path):
    resp = await client.get(path)
    assert resp.status_code == 200


# --- Mutations work when NOT recording ---


async def test_put_general_allowed_when_idle(client):
    resp = await client.put("/api/config/general", json={
        "study_name": "test", "subject_id": "01",
        "session_id": "01", "recording_name": "task",
    })
    assert resp.status_code == 200


async def test_add_mode_allowed_when_idle(client):
    resp = await client.post("/api/modes", json={
        "mode": "synchronous",
        "config": {
            "channel_selection": {"eeg": [0, 1, 2, 3]},
            "event_mapping": {"1": "left", "2": "right"},
        },
    })
    assert resp.status_code == 200


# --- Preflight endpoint ---


async def test_preflight_returns_checks(client):
    resp = await client.get("/api/pipeline/preflight")
    assert resp.status_code == 200
    data = resp.json()
    assert "ready" in data
    assert "checks" in data
    assert isinstance(data["checks"], list)
    # With no streams/modes configured, should not be ready
    assert data["ready"] is False


async def test_start_blocked_when_preflight_fails(client):
    """Start should return 422 when preflight checks fail."""
    resp = await client.post("/api/pipeline/start")
    assert resp.status_code == 422
    data = resp.json()
    assert "checks" in data["detail"]
