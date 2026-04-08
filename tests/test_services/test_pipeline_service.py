"""Tests for PipelineService — state and lifecycle."""

from dendrite.web.services.pipeline_service import PipelineService


def test_is_recording_false_by_default():
    svc = PipelineService()
    assert svc.is_recording is False


def test_elapsed_seconds_zero_when_not_recording():
    svc = PipelineService()
    assert svc.elapsed_seconds == 0.0


def test_visualization_queues_none_before_start():
    """Visualization queues are None when orchestrator not created."""
    svc = PipelineService()
    assert svc.visualization_data_queue is None
    assert svc.visualization_queue is None


def test_no_visualization_streamer_attribute():
    """After cleanup, PipelineService should not have _visualization_streamer."""
    svc = PipelineService()
    assert not hasattr(svc, "_visualization_streamer")


def test_component_states_empty_before_start():
    svc = PipelineService()
    assert svc.get_component_states() == {}


def test_mode_pids_empty_before_start():
    svc = PipelineService()
    assert svc.mode_pids == {}


def test_is_mode_running_false_before_start():
    svc = PipelineService()
    assert svc.is_mode_running("any_mode") is False
