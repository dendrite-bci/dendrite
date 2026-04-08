"""Tests for ML workbench REST endpoints."""

import pytest

from dendrite.web.deps import get_ml_service


@pytest.fixture(autouse=True)
async def _ml_bridge(app):
    """Set bridge on ML service so training broadcasts work."""
    get_ml_service().set_bridge(app.state.queue_bridge)


# --- Models ---


async def test_list_models_returns_200(client):
    resp = await client.get("/api/ml/models")
    assert resp.status_code == 200
    models = resp.json()
    assert isinstance(models, list)
    assert len(models) > 0
    names = [m["model_type"] for m in models]
    assert "EEGNet" in names


async def test_get_model_schema_returns_200(client):
    resp = await client.get("/api/ml/models/EEGNet/schema")
    assert resp.status_code == 200
    schema = resp.json()
    assert "properties" in schema


async def test_get_model_schema_not_found(client):
    resp = await client.get("/api/ml/models/FakeModel/schema")
    assert resp.status_code == 404


# --- Jobs ---


async def test_list_jobs_returns_200(client):
    resp = await client.get("/api/ml/jobs")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


async def test_get_job_not_found(client):
    resp = await client.get("/api/ml/jobs/9999")
    assert resp.status_code == 404


async def test_delete_job_not_found(client):
    resp = await client.delete("/api/ml/jobs/9999")
    assert resp.status_code == 404


# --- Training ---


async def test_start_training_invalid_study(client):
    resp = await client.post("/api/ml/train", json={
        "study_id": 9999,
        "model_type": "EEGNet",
    })
    assert resp.status_code == 422


async def test_start_training_returns_job_id(client):
    # Create a study first
    study_resp = await client.post(
        "/api/data/studies",
        json={"study_name": "ml_test_study"},
    )
    study_id = study_resp.json()["study_id"]

    # Start training (will fail due to no data, but job should be created)
    resp = await client.post("/api/ml/train", json={
        "study_id": study_id,
        "model_type": "EEGNet",
        "epochs": 2,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
