"""Tests for ComponentStateMachine."""

import pytest

from dendrite.utils.component_state import (
    ComponentState,
    ComponentStateMachine,
    InvalidTransitionError,
)
from dendrite.utils.state_keys import component_error_key, component_state_key


class FakeSharedState:
    """Minimal SharedState stub for testing."""

    def __init__(self):
        self.data: dict[str, object] = {}

    def set(self, key: str, value: object) -> None:
        self.data[key] = value

    def get(self, key: str, default: object = None) -> object:
        return self.data.get(key, default)


class TestComponentStateMachine:
    def test_initial_state_is_idle(self):
        sm = ComponentStateMachine("daq")
        assert sm.state == ComponentState.IDLE

    def test_component_id(self):
        sm = ComponentStateMachine("processor")
        assert sm.component_id == "processor"

    def test_happy_path_lifecycle(self):
        sm = ComponentStateMachine("daq")
        sm.transition(ComponentState.STARTING)
        assert sm.state == ComponentState.STARTING
        sm.transition(ComponentState.RUNNING)
        assert sm.state == ComponentState.RUNNING
        sm.transition(ComponentState.STOPPING)
        assert sm.state == ComponentState.STOPPING
        sm.transition(ComponentState.STOPPED)
        assert sm.state == ComponentState.STOPPED

    def test_running_to_paused_and_back(self):
        sm = ComponentStateMachine("mode:nf_1")
        sm.transition(ComponentState.STARTING)
        sm.transition(ComponentState.RUNNING)
        sm.transition(ComponentState.PAUSED)
        assert sm.state == ComponentState.PAUSED
        sm.transition(ComponentState.RUNNING)
        assert sm.state == ComponentState.RUNNING

    def test_error_from_running(self):
        sm = ComponentStateMachine("daq")
        sm.transition(ComponentState.STARTING)
        sm.transition(ComponentState.RUNNING)
        sm.set_error("stream disconnected")
        assert sm.state == ComponentState.ERROR

    def test_error_to_stopping(self):
        sm = ComponentStateMachine("daq")
        sm.transition(ComponentState.STARTING)
        sm.transition(ComponentState.RUNNING)
        sm.set_error("crash")
        sm.transition(ComponentState.STOPPING)
        assert sm.state == ComponentState.STOPPING

    def test_error_to_stopped_directly(self):
        sm = ComponentStateMachine("daq")
        sm.transition(ComponentState.STARTING)
        sm.set_error("init failed")
        sm.transition(ComponentState.STOPPED)
        assert sm.state == ComponentState.STOPPED

    def test_invalid_transition_idle_to_running(self):
        sm = ComponentStateMachine("daq")
        with pytest.raises(InvalidTransitionError) as exc_info:
            sm.transition(ComponentState.RUNNING)
        assert exc_info.value.current == ComponentState.IDLE
        assert exc_info.value.target == ComponentState.RUNNING

    def test_invalid_transition_stopped_to_running(self):
        sm = ComponentStateMachine("daq")
        sm.transition(ComponentState.STARTING)
        sm.transition(ComponentState.RUNNING)
        sm.transition(ComponentState.STOPPING)
        sm.transition(ComponentState.STOPPED)
        with pytest.raises(InvalidTransitionError):
            sm.transition(ComponentState.RUNNING)

    def test_invalid_transition_running_to_starting(self):
        sm = ComponentStateMachine("daq")
        sm.transition(ComponentState.STARTING)
        sm.transition(ComponentState.RUNNING)
        with pytest.raises(InvalidTransitionError):
            sm.transition(ComponentState.STARTING)

    # --- SharedState publishing ---

    def test_publishes_initial_state(self):
        ss = FakeSharedState()
        ComponentStateMachine("daq", ss)
        assert ss.get(component_state_key("daq")) == "idle"

    def test_publishes_on_transition(self):
        ss = FakeSharedState()
        sm = ComponentStateMachine("processor", ss)
        sm.transition(ComponentState.STARTING)
        assert ss.get(component_state_key("processor")) == "starting"
        sm.transition(ComponentState.RUNNING)
        assert ss.get(component_state_key("processor")) == "running"

    def test_publishes_error_message(self):
        ss = FakeSharedState()
        sm = ComponentStateMachine("mode:sync_1", ss)
        sm.transition(ComponentState.STARTING)
        sm.transition(ComponentState.RUNNING)
        sm.set_error("decoder failed to load")
        assert ss.get(component_state_key("mode:sync_1")) == "error"
        assert ss.get(component_error_key("mode:sync_1")) == "decoder failed to load"

    def test_works_without_shared_state(self):
        sm = ComponentStateMachine("daq", None)
        sm.transition(ComponentState.STARTING)
        sm.transition(ComponentState.RUNNING)
        sm.set_error("test")
        assert sm.state == ComponentState.ERROR

    # --- finalize() ---

    def test_finalize_from_running(self):
        sm = ComponentStateMachine("daq")
        sm.transition(ComponentState.STARTING)
        sm.transition(ComponentState.RUNNING)
        sm.finalize()
        assert sm.state == ComponentState.STOPPED

    def test_finalize_from_error(self):
        sm = ComponentStateMachine("daq")
        sm.transition(ComponentState.STARTING)
        sm.transition(ComponentState.RUNNING)
        sm.set_error("crash")
        sm.finalize()
        assert sm.state == ComponentState.STOPPED

    def test_finalize_when_already_stopped(self):
        sm = ComponentStateMachine("daq")
        sm.transition(ComponentState.STARTING)
        sm.transition(ComponentState.RUNNING)
        sm.transition(ComponentState.STOPPING)
        sm.transition(ComponentState.STOPPED)
        sm.finalize()  # no-op
        assert sm.state == ComponentState.STOPPED

    def test_stopped_can_return_to_idle(self):
        sm = ComponentStateMachine("mode:async_1")
        sm.transition(ComponentState.STARTING)
        sm.transition(ComponentState.RUNNING)
        sm.transition(ComponentState.STOPPING)
        sm.transition(ComponentState.STOPPED)
        sm.transition(ComponentState.IDLE)
        assert sm.state == ComponentState.IDLE
