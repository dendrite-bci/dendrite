"""Tests for PreflightService — pre-start validation checks."""

from unittest.mock import MagicMock

from dendrite.data.stream_schemas import StreamMetadata
from dendrite.web.services.config_service import ConfigService
from dendrite.web.services.mode_service import ModeService
from dendrite.web.services.preflight_service import PreflightService


def _make_stream(uid: str = "s1", stream_type: str = "EEG", rate: float = 250.0):
    return StreamMetadata(
        name=f"stream_{uid}",
        type=stream_type,
        channel_count=8,
        sample_rate=rate,
        channel_format="float32",
        source_id="src",
        uid=uid,
        labels=[f"Ch_{i}" for i in range(8)],
        channel_types=[stream_type] * 8,
        channel_units=["uV"] * 8,
        has_metadata_issues=False,
        metadata_issues={},
    )


def _make_services(
    *,
    has_streams: bool = True,
    streams: dict | None = None,
    sample_rate: int | None = 250,
    modes: dict | None = None,
    study_name: str = "my_study",
    subject_id: str = "sub01",
    session_id: str = "ses01",
    recording_name: str = "task",
) -> PreflightService:
    stream_svc = MagicMock()
    stream_svc.has_streams.return_value = has_streams
    stream_svc.get_system_sample_rate.return_value = sample_rate
    if streams is None and has_streams:
        streams = {"s1": _make_stream()}
    stream_svc.get_streams.return_value = streams or {}
    liveness = {uid: True for uid in (streams or {})}
    stream_svc.check_streams.return_value = (liveness, {})

    mode_svc = ModeService()
    if modes is None:
        mode_svc.add_instance("P300", {
            "name": "P300",
            "mode": "synchronous",
            "channel_selection": {"eeg": [0, 1, 2, 3]},
            "event_mapping": {1: "left", 2: "right"},
        })
    else:
        for name, cfg in modes.items():
            mode_svc.add_instance(name, cfg)

    config_svc = MagicMock(spec=ConfigService)
    config_svc.study_name = study_name
    config_svc.subject_id = subject_id
    config_svc.session_id = session_id
    config_svc.recording_name = recording_name

    return PreflightService(stream_svc, mode_svc, config_svc)


def _get_check(result, check_id: str):
    for c in result.checks:
        if c.id == check_id:
            return c
    return None


# --- Tests ---


def test_all_checks_pass():
    svc = _make_services()
    result = svc.run_preflight()
    assert result.ready is True
    assert all(c.passed for c in result.checks)


def test_no_streams_fails():
    svc = _make_services(has_streams=False, streams={})
    result = svc.run_preflight()
    assert result.ready is False
    check = _get_check(result, "data_stream")
    assert check is not None and check.passed is False


def test_no_data_stream_fails():
    """Event-only stream (sample_rate=0) should fail the data_stream check."""
    markers = _make_stream(stream_type="Markers", rate=0)
    svc = _make_services(streams={"s1": markers}, sample_rate=None)
    result = svc.run_preflight()
    check = _get_check(result, "data_stream")
    assert check is not None and check.passed is False


def test_non_eeg_data_stream_passes():
    """An EMG-only setup should pass preflight — no EEG required."""
    emg = _make_stream(stream_type="EMG", rate=250)
    svc = _make_services(streams={"s1": emg})
    result = svc.run_preflight()
    assert result.ready is True
    check = _get_check(result, "data_stream")
    assert check is not None and check.passed is True


def test_empty_bids_is_warning_not_blocker():
    """Empty BIDS = warning (not required), so ready is still True if streams are OK."""
    svc = _make_services(subject_id="")
    result = svc.run_preflight()
    check = _get_check(result, "bids_fields")
    assert check is not None and check.passed is False
    assert check.required is False
    assert "subject_id" in check.detail
    # ready should still be True since BIDS is not required
    assert result.ready is True


def test_empty_bids_session_fails():
    svc = _make_services(session_id="")
    result = svc.run_preflight()
    check = _get_check(result, "bids_fields")
    assert check is not None and check.passed is False
    assert "session_id" in check.detail


def test_invalid_bids_chars_fails():
    svc = _make_services(study_name="bad<name>")
    result = svc.run_preflight()
    check = _get_check(result, "bids_fields")
    assert check is not None and check.passed is False


def test_multiple_failures_all_reported():
    svc = _make_services(
        has_streams=False,
        streams={},
        sample_rate=None,
        modes={},
        subject_id="",
    )
    result = svc.run_preflight()
    assert result.ready is False
    failed = [c for c in result.checks if not c.passed]
    assert len(failed) >= 2


def test_ready_with_streams_only():
    """Minimal condition: streams configured + EEG + sample rate → ready."""
    svc = _make_services(modes={}, subject_id="", session_id="")
    result = svc.run_preflight()
    assert result.ready is True
    # Warnings should be present but not blocking
    warnings = [c for c in result.checks if not c.passed and not c.required]
    assert len(warnings) >= 1


