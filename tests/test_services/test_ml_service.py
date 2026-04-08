"""Tests for MLService — model listing, job CRUD, and training."""

import json

import pytest

from dendrite.web.services.data_service import DataService
from dendrite.web.services.ml_service import MLService


@pytest.fixture
def data_svc(tmp_path):
    return DataService(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def ml_svc(data_svc):
    return MLService(data_svc)


# --- Model listing ---


def test_list_models_returns_entries(ml_svc):
    models = ml_svc.list_models()
    assert len(models) > 0
    names = [m["model_type"] for m in models]
    assert "EEGNet" in names
    assert "CSP+LDA" in names


def test_list_models_has_required_fields(ml_svc):
    models = ml_svc.list_models()
    for m in models:
        assert "model_type" in m
        assert "description" in m
        assert "modalities" in m


def test_get_model_config_schema_eegnet(ml_svc):
    schema = ml_svc.get_model_config_schema("EEGNet")
    assert schema is not None
    assert "properties" in schema


def test_get_model_config_schema_classical(ml_svc):
    """Classical classifiers (LDA, SVM) have config schemas. CSP is a feature extractor, not in MODEL_REGISTRY."""
    schema = ml_svc.get_model_config_schema("LDA")
    assert schema is not None
    assert schema["type"] == "object"


def test_get_model_config_schema_unknown(ml_svc):
    assert ml_svc.get_model_config_schema("NonExistentModel") is None


# --- Job CRUD ---


def test_create_and_get_job(ml_svc, data_svc):
    study = data_svc.studies.get_or_create("test_study")
    job_id = ml_svc._job_repo.create_job(
        study_id=study["study_id"],
        model_type="EEGNet",
        config_json=json.dumps({"model_type": "EEGNet"}),
    )
    job = ml_svc.get_job(job_id)
    assert job is not None
    assert job["model_type"] == "EEGNet"
    assert job["status"] == "pending"


def test_list_jobs_empty(ml_svc):
    assert ml_svc.list_jobs() == []


def test_list_jobs_by_study(ml_svc, data_svc):
    s1 = data_svc.studies.get_or_create("study_a")
    s2 = data_svc.studies.get_or_create("study_b")

    ml_svc._job_repo.create_job(s1["study_id"], "EEGNet", "{}")
    ml_svc._job_repo.create_job(s2["study_id"], "CSP+LDA", "{}")

    all_jobs = ml_svc.list_jobs()
    assert len(all_jobs) == 2

    study_a_jobs = ml_svc.list_jobs(study_id=s1["study_id"])
    assert len(study_a_jobs) == 1
    assert study_a_jobs[0]["model_type"] == "EEGNet"


def test_delete_job(ml_svc, data_svc):
    study = data_svc.studies.get_or_create("test")
    job_id = ml_svc._job_repo.create_job(study["study_id"], "EEGNet", "{}")
    assert ml_svc.delete_job(job_id)
    assert ml_svc.get_job(job_id) is None


def test_delete_nonexistent_job(ml_svc):
    assert not ml_svc.delete_job(9999)


# --- Training (async) ---


@pytest.mark.slow
async def test_start_training_with_synthetic_data(ml_svc, data_svc, tmp_path):
    """End-to-end training test with small synthetic FIF dataset."""
    import mne
    import numpy as np

    # Create synthetic raw FIF with event annotations
    n_channels = 8
    sfreq = 250.0
    duration = 20.0  # seconds
    n_total = int(duration * sfreq)
    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)],
        sfreq=sfreq,
        ch_types="eeg",
    )
    raw_data = np.random.randn(n_channels, n_total).astype(np.float32) * 1e-6
    raw = mne.io.RawArray(raw_data, info, verbose=False)

    # Add event annotations every 0.5s (40 events)
    onsets = np.arange(0.5, duration - 0.5, 0.5)[:40]
    durations = np.zeros(len(onsets))
    descriptions = [str(1 + i % 2) for i in range(len(onsets))]  # alternating "1" and "2"
    raw.set_annotations(mne.Annotations(onsets, durations, descriptions))

    fif_path = str(tmp_path / "test_raw.fif")
    raw.save(fif_path, overwrite=True, verbose=False)

    # Register in DB
    study = data_svc.studies.get_or_create("train_test")

    # Start training (file_path replaces dataset_id)
    result = await ml_svc.start_training({
        "study_id": study["study_id"],
        "file_path": fif_path,
        "model_type": "EEGNet",
        "num_classes": 2,
        "epochs": 2,
        "batch_size": 8,
        "learning_rate": 0.01,
        "validation_split": 0.2,
        "epoch_tmin": 0.0,
        "epoch_tmax": 0.5,
        "event_mapping": {1: "left", 2: "right"},
        "label_mapping": {"left": 0, "right": 1},
    })
    assert result["job_id"] is not None

    # Wait for training to finish
    import asyncio
    task = ml_svc._active_jobs.get(result["job_id"])
    if task:
        await asyncio.wait_for(task, timeout=60)

    job = ml_svc.get_job(result["job_id"])
    assert job["status"] == "completed"
    assert job["result_json"] is not None


async def test_start_training_invalid_study(ml_svc):
    with pytest.raises(ValueError, match="Study 9999 not found"):
        await ml_svc.start_training({
            "study_id": 9999,
            "model_type": "EEGNet",
        })


async def test_start_training_invalid_model(ml_svc, data_svc):
    study = data_svc.studies.get_or_create("test")
    with pytest.raises(ValueError, match="Unknown model type"):
        await ml_svc.start_training({
            "study_id": study["study_id"],
            "model_type": "FakeModel",
        })


# --- Executor cleanup ---


def test_cleanup_sync_shuts_down_executor(ml_svc):
    """cleanup_sync() shuts down the ProcessPoolExecutor if it was created."""
    executor = ml_svc._get_executor()
    assert executor is not None
    assert ml_svc._executor is not None

    ml_svc.cleanup_sync()

    assert ml_svc._executor is None


def test_cleanup_sync_safe_when_no_executor(ml_svc):
    """cleanup_sync() is safe to call when executor was never created."""
    assert ml_svc._executor is None
    ml_svc.cleanup_sync()  # Should not raise
    assert ml_svc._executor is None
