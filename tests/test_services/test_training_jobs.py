"""Tests for TrainingJobRepository — CRUD, status, result, linking."""

import json

import pytest

from dendrite.data.storage.database import Database, TrainingJobRepository
from dendrite.web.services.data_service import DataService


@pytest.fixture
def db(tmp_path):
    database = Database(str(tmp_path / "test.db"))
    database.init_db()
    return database


@pytest.fixture
def repo(db):
    return TrainingJobRepository(db)


@pytest.fixture
def svc(tmp_path):
    return DataService(db_path=str(tmp_path / "test.db"))


# --- Create & Get ---


def test_create_job(repo):
    job_id = repo.create_job(None, "EEGNet", '{"epochs": 10}')
    assert job_id is not None
    job = repo.get_by_id(job_id)
    assert job["model_type"] == "EEGNet"
    assert job["status"] == "pending"
    assert job["config_json"] == '{"epochs": 10}'


def test_create_job_with_study(repo, svc):
    study = svc.studies.get_or_create("test")
    job_id = repo.create_job(study["study_id"], "CSP+LDA", '{}')
    job = repo.get_by_id(job_id)
    assert job["study_id"] == study["study_id"]


def test_create_job_custom_type(repo):
    job_id = repo.create_job(None, "EEGNet", '{}', job_type="evaluation")
    job = repo.get_by_id(job_id)
    assert job["job_type"] == "evaluation"


def test_get_nonexistent(repo):
    assert repo.get_by_id(9999) is None


# --- List & Filter ---


def test_list_empty(repo):
    assert repo.list_jobs() == []


def test_list_all(repo):
    repo.create_job(None, "EEGNet", '{}')
    repo.create_job(None, "CSP+LDA", '{}')
    jobs = repo.list_jobs()
    assert len(jobs) == 2


def test_list_by_study(repo, svc):
    s1 = svc.studies.get_or_create("s1")
    s2 = svc.studies.get_or_create("s2")
    repo.create_job(s1["study_id"], "EEGNet", '{}')
    repo.create_job(s2["study_id"], "CSP+LDA", '{}')
    jobs = repo.list_jobs(study_id=s1["study_id"])
    assert len(jobs) == 1
    assert jobs[0]["model_type"] == "EEGNet"


def test_list_by_type(repo):
    repo.create_job(None, "EEGNet", '{}', job_type="training")
    repo.create_job(None, "EEGNet", '{}', job_type="evaluation")
    jobs = repo.list_jobs(job_type="evaluation")
    assert len(jobs) == 1
    assert jobs[0]["job_type"] == "evaluation"


def test_list_order_desc(repo):
    id1 = repo.create_job(None, "A", '{}')
    id2 = repo.create_job(None, "B", '{}')
    jobs = repo.list_jobs()
    assert jobs[0]["job_id"] == id2  # newest first
    assert jobs[1]["job_id"] == id1


# --- Status Updates ---


def test_update_status(repo):
    job_id = repo.create_job(None, "EEGNet", '{}')
    assert repo.update_status(job_id, "running", started_at="2025-01-01T00:00:00")
    job = repo.get_by_id(job_id)
    assert job["status"] == "running"
    assert job["started_at"] == "2025-01-01T00:00:00"


def test_update_status_with_error(repo):
    job_id = repo.create_job(None, "EEGNet", '{}')
    repo.update_status(job_id, "failed", error_message="OOM", completed_at="2025-01-01T01:00:00")
    job = repo.get_by_id(job_id)
    assert job["status"] == "failed"
    assert job["error_message"] == "OOM"
    assert job["completed_at"] is not None


def test_update_nonexistent(repo):
    assert not repo.update_status(9999, "running")


# --- Result ---


def test_set_and_get_result(repo):
    job_id = repo.create_job(None, "EEGNet", '{}')
    result = {"accuracy": 0.85, "path": "/tmp/model.pt"}
    repo.set_result(job_id, json.dumps(result))
    job = repo.get_by_id(job_id)
    assert json.loads(job["result_json"]) == result


def test_set_result_nonexistent(repo):
    assert not repo.set_result(9999, '{}')


# --- Link Decoder ---


def test_link_decoder(repo, svc):
    study = svc.studies.get_or_create("test")
    job_id = repo.create_job(study["study_id"], "EEGNet", '{}')
    dec_id = svc.decoders.add_decoder(
        study_id=study["study_id"],
        decoder_name="my_dec",
        decoder_path="/tmp/dec.json",
        model_type="EEGNet",
    )
    assert repo.link_decoder(job_id, dec_id)
    job = repo.get_by_id(job_id)
    assert job["decoder_id"] == dec_id


# --- Delete ---


def test_delete_job(repo):
    job_id = repo.create_job(None, "EEGNet", '{}')
    assert repo.delete_job(job_id)
    assert repo.get_by_id(job_id) is None


def test_delete_nonexistent(repo):
    assert not repo.delete_job(9999)
