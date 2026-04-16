"""
Component State Machine

Per-component lifecycle state with cross-process observability via SharedState.
Each subprocess owns its own ComponentStateMachine instance and publishes
state transitions to SharedState for the web layer to read.
"""

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dendrite.utils.shared_state import SharedState

from dendrite.utils.state_keys import component_error_key, component_state_key


class ComponentState(StrEnum):
    """Lifecycle states for pipeline components."""

    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


# Valid state transitions
_TRANSITIONS: dict[ComponentState, set[ComponentState]] = {
    ComponentState.IDLE: {ComponentState.STARTING},
    ComponentState.STARTING: {ComponentState.RUNNING, ComponentState.ERROR, ComponentState.STOPPING},
    ComponentState.RUNNING: {
        ComponentState.PAUSED,
        ComponentState.STOPPING,
        ComponentState.ERROR,
    },
    ComponentState.PAUSED: {ComponentState.RUNNING, ComponentState.STOPPING, ComponentState.ERROR},
    ComponentState.STOPPING: {ComponentState.STOPPED},
    ComponentState.STOPPED: {ComponentState.IDLE},
    ComponentState.ERROR: {ComponentState.STOPPING, ComponentState.STOPPED},
}


class InvalidTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    def __init__(self, current: ComponentState, target: ComponentState):
        self.current = current
        self.target = target
        super().__init__(f"Invalid transition: {current.value} → {target.value}")


class ComponentStateMachine:
    """Per-component state machine that publishes to SharedState.

    Usage in a subprocess:
        sm = ComponentStateMachine("daq", shared_state)
        sm.transition(ComponentState.STARTING)
        # ... do init ...
        sm.transition(ComponentState.RUNNING)
        # ... main loop ...
        sm.transition(ComponentState.STOPPING)
        sm.transition(ComponentState.STOPPED)

    Usage from web layer (read-only):
        state = shared_state.get(component_state_key("daq"))  # "running"
        error = shared_state.get(component_error_key("daq"))  # None or error msg
    """

    def __init__(self, component_id: str, shared_state: "SharedState | None" = None):
        self._id = component_id
        self._state = ComponentState.IDLE
        self._shared_state = shared_state
        self._publish()

    @property
    def component_id(self) -> str:
        return self._id

    @property
    def state(self) -> ComponentState:
        return self._state

    def transition(self, target: ComponentState) -> None:
        """Execute a validated state transition.

        Raises:
            InvalidTransitionError: If the transition is not allowed.
        """
        valid = _TRANSITIONS.get(self._state, set())
        if target not in valid:
            raise InvalidTransitionError(self._state, target)
        self._state = target
        self._publish()

    def finalize(self) -> None:
        """Ensure component reaches STOPPED state on shutdown."""
        if self._state == ComponentState.STOPPED:
            return
        if self._state == ComponentState.ERROR:
            self.transition(ComponentState.STOPPED)
        else:
            self.transition(ComponentState.STOPPING)
            self.transition(ComponentState.STOPPED)

    def set_error(self, message: str) -> None:
        """Transition to ERROR state with a message."""
        self._state = ComponentState.ERROR
        if self._shared_state:
            self._shared_state.set(component_error_key(self._id), message)
        self._publish()

    def _publish(self) -> None:
        """Publish current state to SharedState."""
        if self._shared_state:
            self._shared_state.set(component_state_key(self._id), self._state.value)
