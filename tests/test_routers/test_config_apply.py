"""Tests for /api/config/apply and study auto-create on /api/config/load."""

import json
import shutil
import uuid

import pytest

from dendrite.constants import STUDIES_DIR
from dendrite.web.deps import get_config_service, get_data_service


def _minimal_config(study_name: str) -> dict:
    """Build a minimal valid PipelineConfig body."""
    return {
        "study_name": study_name,
        "subject_id": "01",
        "session_id": "01",
        "recording_name": "rec",
    }


@pytest.fixture
def study_cleanup():
    """Track studies created during the test and tear them down afterwards."""
    created: list[str] = []
    yield created
    for name in created:
        study_dir = STUDIES_DIR / name
        if study_dir.exists():
            shutil.rmtree(study_dir, ignore_errors=True)
        try:
            svc = get_data_service()
            for row in svc.studies.get_all_studies():
                if row["study_name"] == name:
                    svc.studies.delete_study(row["study_id"])
        except RuntimeError:
            pass


async def test_apply_applies_and_creates_study(client, study_cleanup):
    study_name = f"apply_{uuid.uuid4().hex[:8]}"
    study_cleanup.append(study_name)

    resp = await client.post("/api/config/apply", json=_minimal_config(study_name))
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "loaded"
    assert data["warnings"] == []

    # Applied to running ConfigService
    assert get_config_service().study_name == study_name
    # Study row was auto-created
    assert any(
        r["study_name"] == study_name
        for r in get_data_service().studies.get_all_studies()
    )
    # No file written under studies/{name}/config/ — apply does not persist
    assert not (STUDIES_DIR / study_name / "config").exists()


async def test_apply_accepts_unknown_top_level_fields(client):
    resp = await client.post(
        "/api/config/apply",
        json={"unknown_field": 42},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "loaded"
    # Per-instance validation issues would come back via warnings.


async def test_load_auto_creates_study(client, study_cleanup):
    """Loading a config naming a brand-new study should auto-create the DB row."""
    study_name = f"load_{uuid.uuid4().hex[:8]}"
    study_cleanup.append(study_name)

    # Hand-write a config file to disk so /load has something to read.
    cfg_dir = STUDIES_DIR / study_name / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    file_path = cfg_dir / "seed.json"
    file_path.write_text(json.dumps(_minimal_config(study_name)))

    svc = get_data_service()
    # Make sure no stale row exists for this random name.
    for row in svc.studies.get_all_studies():
        if row["study_name"] == study_name:
            svc.studies.delete_study(row["study_id"])

    resp = await client.post(f"/api/config/load?file_path={file_path}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "loaded"
    assert any(
        r["study_name"] == study_name for r in svc.studies.get_all_studies()
    )
