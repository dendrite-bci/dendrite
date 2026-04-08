"""Tests for DataService — repository access + H5 inspection."""

import pytest

from dendrite.web.services.data_service import DataService


@pytest.fixture
def svc(tmp_path):
    """DataService with an isolated temp database."""
    return DataService(db_path=str(tmp_path / "test.db"))


# --- Studies ---


def test_list_studies_empty(svc):
    assert svc.studies.get_all_studies() == []


def test_create_study(svc):
    study = svc.studies.get_or_create("my_study", "a description")
    assert study["study_name"] == "my_study"
    assert study["description"] == "a description"
    assert study["study_id"] is not None


def test_create_study_idempotent(svc):
    s1 = svc.studies.get_or_create("my_study")
    s2 = svc.studies.get_or_create("my_study")
    assert s1["study_id"] == s2["study_id"]


def test_get_study(svc):
    created = svc.studies.get_or_create("test")
    got = svc.studies.get_by_id(created["study_id"])
    assert got is not None
    assert got["study_name"] == "test"


def test_get_study_not_found(svc):
    assert svc.studies.get_by_id(9999) is None


def test_update_study(svc):
    study = svc.studies.get_or_create("test")
    assert svc.studies.update_study(study["study_id"], description="updated")
    got = svc.studies.get_by_id(study["study_id"])
    assert got["description"] == "updated"


def test_delete_study(svc):
    study = svc.studies.get_or_create("test")
    assert svc.studies.delete_study(study["study_id"])
    assert svc.studies.get_by_id(study["study_id"]) is None


def test_get_study_detail_counts(svc):
    study = svc.studies.get_or_create("test")
    sid = study["study_id"]
    svc.recordings.add_recording(
        study_id=sid,
        recording_name="rec1",
        session_timestamp="20240101_120000",
        hdf5_file_path="/tmp/rec1.h5",
    )
    # Verify counts via repos directly (matches router logic)
    assert len(svc.recordings.get_recordings_by_study(sid)) == 1
    assert len(svc.decoders.get_decoders_by_study(sid)) == 0


def test_get_study_detail_not_found(svc):
    assert svc.studies.get_by_id(9999) is None


# --- Recordings ---


def test_list_recordings_empty(svc):
    assert svc.recordings.get_all_recordings() == []


def test_list_recordings(svc):
    study = svc.studies.get_or_create("test")
    svc.recordings.add_recording(
        study_id=study["study_id"],
        recording_name="rec1",
        session_timestamp="20240101_120000",
        hdf5_file_path="/tmp/rec1.h5",
    )
    recs = svc.recordings.get_all_recordings()
    assert len(recs) == 1
    assert recs[0]["recording_name"] == "rec1"


def test_list_recordings_by_study(svc):
    s1 = svc.studies.get_or_create("study1")
    s2 = svc.studies.get_or_create("study2")
    svc.recordings.add_recording(
        study_id=s1["study_id"],
        recording_name="rec1",
        session_timestamp="20240101_120000",
        hdf5_file_path="/tmp/rec1.h5",
    )
    svc.recordings.add_recording(
        study_id=s2["study_id"],
        recording_name="rec2",
        session_timestamp="20240101_130000",
        hdf5_file_path="/tmp/rec2.h5",
    )
    recs = svc.recordings.get_recordings_by_study(s1["study_id"])
    assert len(recs) == 1
    assert recs[0]["recording_name"] == "rec1"


def test_search_recordings(svc):
    study = svc.studies.get_or_create("test")
    svc.recordings.add_recording(
        study_id=study["study_id"],
        recording_name="motor_imagery",
        session_timestamp="20240101_120000",
        hdf5_file_path="/tmp/mi.h5",
        subject_id="sub01",
    )
    svc.recordings.add_recording(
        study_id=study["study_id"],
        recording_name="p300_speller",
        session_timestamp="20240101_130000",
        hdf5_file_path="/tmp/p300.h5",
        subject_id="sub02",
    )
    results = svc.recordings.search_recordings("motor")
    assert len(results) == 1
    assert results[0]["recording_name"] == "motor_imagery"


def test_get_recording(svc):
    study = svc.studies.get_or_create("test")
    rid = svc.recordings.add_recording(
        study_id=study["study_id"],
        recording_name="rec1",
        session_timestamp="20240101_120000",
        hdf5_file_path="/tmp/rec1.h5",
    )
    rec = svc.recordings.get_by_id(rid)
    assert rec is not None
    assert rec["recording_name"] == "rec1"


def test_get_recording_not_found(svc):
    assert svc.recordings.get_by_id(9999) is None


def test_delete_recording(svc):
    study = svc.studies.get_or_create("test")
    rid = svc.recordings.add_recording(
        study_id=study["study_id"],
        recording_name="rec1",
        session_timestamp="20240101_120000",
        hdf5_file_path="/tmp/rec1.h5",
    )
    assert svc.recordings.delete_recording(rid)
    assert svc.recordings.get_by_id(rid) is None


def test_delete_study_cascades_recordings(svc):
    study = svc.studies.get_or_create("test")
    sid = study["study_id"]
    svc.recordings.add_recording(
        study_id=sid,
        recording_name="rec1",
        session_timestamp="20240101_120000",
        hdf5_file_path="/tmp/rec1.h5",
    )
    svc.studies.delete_study(sid)
    assert svc.recordings.get_all_recordings() == []


def test_get_recording_file_info_not_found(svc):
    assert svc.get_recording_file_info(9999) is None


def test_get_recording_file_info_missing_file(svc):
    study = svc.studies.get_or_create("test")
    rid = svc.recordings.add_recording(
        study_id=study["study_id"],
        recording_name="rec1",
        session_timestamp="20240101_120000",
        hdf5_file_path="/nonexistent/path.h5",
    )
    with pytest.raises(FileNotFoundError, match="not found"):
        svc.get_recording_file_info(rid)


# --- Decoders ---


def test_list_decoders_empty(svc):
    assert svc.decoders.get_all_decoders() == []


def test_list_decoders(svc):
    study = svc.studies.get_or_create("test")
    svc.decoders.add_decoder(
        study_id=study["study_id"],
        decoder_name="csp_lda",
        decoder_path="/tmp/model.pkl",
        model_type="CSP_LDA",
    )
    decs = svc.decoders.get_all_decoders()
    assert len(decs) == 1
    assert decs[0]["decoder_name"] == "csp_lda"


def test_list_decoders_by_study(svc):
    s1 = svc.studies.get_or_create("study1")
    s2 = svc.studies.get_or_create("study2")
    svc.decoders.add_decoder(
        study_id=s1["study_id"],
        decoder_name="dec1",
        decoder_path="/tmp/d1.pkl",
        model_type="CSP",
    )
    svc.decoders.add_decoder(
        study_id=s2["study_id"],
        decoder_name="dec2",
        decoder_path="/tmp/d2.pkl",
        model_type="CNN",
    )
    decs = svc.decoders.get_decoders_by_study(s1["study_id"])
    assert len(decs) == 1
    assert decs[0]["decoder_name"] == "dec1"


def test_search_decoders(svc):
    study = svc.studies.get_or_create("test")
    svc.decoders.add_decoder(
        study_id=study["study_id"],
        decoder_name="csp_lda",
        decoder_path="/tmp/d1.pkl",
        model_type="CSP_LDA",
    )
    svc.decoders.add_decoder(
        study_id=study["study_id"],
        decoder_name="eegnet",
        decoder_path="/tmp/d2.pkl",
        model_type="EEGNet",
    )
    results = svc.decoders.search_decoders("eeg")
    assert len(results) == 1


def test_get_decoder(svc):
    study = svc.studies.get_or_create("test")
    did = svc.decoders.add_decoder(
        study_id=study["study_id"],
        decoder_name="csp",
        decoder_path="/tmp/csp.pkl",
        model_type="CSP",
    )
    dec = svc.decoders.get_decoder_by_id(did)
    assert dec is not None
    assert dec["decoder_name"] == "csp"


def test_delete_decoder(svc):
    study = svc.studies.get_or_create("test")
    did = svc.decoders.add_decoder(
        study_id=study["study_id"],
        decoder_name="csp",
        decoder_path="/tmp/csp.pkl",
        model_type="CSP",
    )
    assert svc.decoders.delete_decoder(did)
    assert svc.decoders.get_decoder_by_id(did) is None


# --- Import bug: duplicate timestamps should not skip files ---


def test_import_duplicate_timestamps_allowed(svc):
    """Two recordings with the same session_timestamp should both import."""
    study = svc.studies.get_or_create("test")
    sid = study["study_id"]
    r1 = svc.recordings.add_recording(
        study_id=sid,
        recording_name="rec1",
        session_timestamp="20240101_120000",
        hdf5_file_path="/tmp/rec1.h5",
    )
    r2 = svc.recordings.add_recording(
        study_id=sid,
        recording_name="rec2",
        session_timestamp="20240101_120000",
        hdf5_file_path="/tmp/rec2.h5",
    )
    assert r1 is not None
    assert r2 is not None
    assert r1 != r2


# --- Transaction semantics ---


def test_transaction_commits(svc):
    with svc.db.transaction() as conn:
        svc.studies.get_or_create("txn_study", _conn=conn)
    assert svc.studies.get_by_id(1) is not None


def test_transaction_rollback_on_error(svc):
    with pytest.raises(RuntimeError):
        with svc.db.transaction() as conn:
            svc.studies.get_or_create("rollback_study", _conn=conn)
            raise RuntimeError("force rollback")
    assert svc.studies.get_all_studies() == []
