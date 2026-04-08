"""Tests for pipeline REST endpoints."""


async def test_debug_endpoint_returns_stats(client):
    resp = await client.get("/api/pipeline/debug")
    assert resp.status_code == 200
    data = resp.json()

    assert "recording" in data
    assert "elapsed_seconds" in data
    assert "bridge_stats" in data
    assert "drain_tasks_active" in data
    assert "queues" in data
    assert isinstance(data["bridge_stats"], dict)


async def test_debug_endpoint_when_not_recording(client):
    resp = await client.get("/api/pipeline/debug")
    data = resp.json()

    assert data["recording"] is False
    assert data["elapsed_seconds"] == 0.0
    assert data["drain_tasks_active"] == 0
    assert data["queues"] == {}


async def test_status_endpoint(client):
    resp = await client.get("/api/pipeline/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["recording"] is False


async def test_start_lock_prevents_concurrent_starts(client):
    """Second concurrent start returns 409 when lock is held."""
    from dendrite.web.routers.pipeline import _start_lock

    # Acquire the lock to simulate an in-progress start
    await _start_lock.acquire()
    try:
        resp = await client.post("/api/pipeline/start")
        assert resp.status_code == 409
        assert "already in progress" in resp.json()["detail"]
    finally:
        _start_lock.release()
