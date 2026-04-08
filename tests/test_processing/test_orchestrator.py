"""Tests for PipelineOrchestrator health check."""

import pytest
from unittest.mock import MagicMock

from dendrite.processing.orchestrator import PipelineOrchestrator


@pytest.fixture
def orchestrator():
    return PipelineOrchestrator(shared_state=MagicMock())


def _mock_process(alive: bool = True, pid: int = 1234):
    proc = MagicMock()
    proc.is_alive.return_value = alive
    proc.pid = pid
    return proc


def test_check_mode_health_returns_dead_modes(orchestrator):
    dead_proc = _mock_process(alive=False, pid=1234)
    alive_proc = _mock_process(alive=True, pid=5678)

    orchestrator._mode_processes = {"dead_mode": dead_proc, "alive_mode": alive_proc}
    orchestrator._mode_stops = {"dead_mode": MagicMock(), "alive_mode": MagicMock()}
    orchestrator._mode_output_queues = {"dead_mode": MagicMock(), "alive_mode": MagicMock()}

    dead = orchestrator.check_mode_health()
    assert dead == ["dead_mode"]
    assert "dead_mode" not in orchestrator._mode_processes
    assert "alive_mode" in orchestrator._mode_processes
    # SharedState keys cleaned up
    orchestrator.shared_state.clear.assert_called()


def test_check_mode_health_empty_when_all_alive(orchestrator):
    proc = _mock_process(alive=True)
    orchestrator._mode_processes = {"mode1": proc}
    assert orchestrator.check_mode_health() == []


def test_check_mode_health_empty_when_no_modes(orchestrator):
    assert orchestrator.check_mode_health() == []
