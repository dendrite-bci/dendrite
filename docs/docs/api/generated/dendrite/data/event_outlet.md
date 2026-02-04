---
sidebar_label: event_outlet
title: dendrite.data.event_outlet
---

LSL Event Outlet - Sends structured events as JSON strings.

Standalone module with only pylsl dependency. Can be copied for use in task apps.

**Example**:

  Complete workflow for sending task events:
  
    ```python
    from dendrite.data import EventOutlet

    # Define events with unique integer IDs
    events = \{'trial_start': 1, 'cue_left': 10, 'cue_right': 11, 'response': 20, 'trial_end': 2\}

    # Create outlet
    outlet = EventOutlet(stream_name='TaskEvents', events=events)

    # Send events during task execution
    outlet.send_event('trial_start', \{'trial_number': 1\})
    outlet.send_event('cue_left')
    outlet.send_event('response', \{'reaction_time_ms': 450\})
    outlet.send_event('trial_end')

    # Clean up when done
    outlet.close()
    ```

## EventOutlet Objects

```python
class EventOutlet()
```

LSL outlet for sending structured events as JSON strings.

#### \_\_init\_\_

```python
def __init__(stream_name: str,
             events: dict[str, int] | None = None,
             stream_id: str | None = None) -> None
```

Initialize event outlet.

**Arguments**:

- `stream_name` - Name of the LSL stream
- `events` - Event name to ID mappings
- `stream_id` - Optional unique identifier (defaults to '\{stream_name\}_id')

#### send\_event

```python
def send_event(event_type: str,
               additional_data: dict[str, Any] | None = None) -> None
```

Send event marker with optional metadata payload.

**Arguments**:

- `event_type` - Event name from configured events mapping.
- `additional_data` - Optional metadata to include in event packet.
  

**Raises**:

- `ValueError` - If event_type is not in configured events.

#### close

```python
def close() -> None
```

Close outlet and release LSL resources.

Call this when done sending events to properly clean up.

#### have\_consumers

```python
def have_consumers() -> bool
```

Check if any consumers are connected to this LSL stream.

**Returns**:

  True if at least one consumer is connected.

