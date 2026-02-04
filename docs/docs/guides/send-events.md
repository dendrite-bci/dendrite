---
id: send-events
title: Send and Log Events
sidebar_label: Send Events
---

# Send Events

Send structured events over LSL and store them in HDF5 for trial segmentation and analysis.

## Event Structure

Events are JSON-encoded strings over LSL with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | int | Numeric identifier (from event mapping) |
| `event_type` | str | Event name string |
| `*` | any | Additional metadata fields |

LSL captures the timestamp automatically when pushed. When saved to HDF5, events gain additional fields: `timestamp` (LSL server time), `local_timestamp` (receive time), and metadata stored in `extra_vars`.

## Programmatic Event Sending (Python)

### Event Mapping

The `events` parameter maps human-readable event names to numeric IDs:

```python
events = {
    'trial_start': 10,
    'cue_onset': 20,
    'response': 30,
}
```

### Basic Usage

```python
from dendrite.data import EventOutlet

# Create outlet with event mapping
outlet = EventOutlet(
    stream_name='Events',
    events={
        'trial_start': 10,
        'cue_onset': 20,
        'cue_offset': 21,
        'response': 30,
        'trial_end': 40,
    },
)

# Send event by name (with optional metadata)
outlet.send_event('trial_start', {'trial_number': 1, 'condition': 'left'})
```

## Other Languages

Events are JSON strings over LSL. Any language with LSL bindings can send them.

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | int | Numeric identifier |
| `event_type` | string | Event name |

Additional fields are stored as metadata.

### LSL Stream Configuration

| Property | Value |
|----------|-------|
| Type | `Events` |
| Channel format | `string` |
| Channel count | `1` |

### Example: MATLAB

```matlab
lib = lsl_loadlib();
info = lsl_streaminfo(lib, 'Events', 'Events', 1, 0, 'cf_string', 'events_id');
outlet = lsl_outlet(info);

event = struct('event_id', 10, 'event_type', 'trial_start', 'trial_num', 1);
outlet.push_sample({jsonencode(event)});
```

### Example: C++/Unity

```cpp
// Create LSL outlet (type='Events', format=string, channels=1)
lsl::stream_info info("Events", "Events", 1, LSL_IRREGULAR_RATE, lsl::cf_string);
lsl::stream_outlet outlet(info);

// Send event as JSON string
std::string event = R"({"event_id": 10, "event_type": "trial_start"})";
outlet.push_sample(&event);
```

## Loading Events

Use `load_events()` to load stored events as a pandas DataFrame:

```python
from dendrite.data.io import load_events

df = load_events('recording.h5')  # Loads 'Event' dataset as DataFrame
```

### DataFrame Structure

| Column | Type | Description |
|--------|------|-------------|
| `event_id` | int32 | Numeric event identifier |
| `event_type` | string | Event name |
| `timestamp` | float64 | LSL server timestamp |
| `local_timestamp` | float64 | Local receive timestamp |
| `extra_*` | varies | Additional fields (prefixed with `extra_`) |

### Example Output

```python
>>> df = load_events('recording.h5')
>>> print(df)
   event_id   event_type      timestamp  local_timestamp  extra_trial_number extra_condition
0        10  trial_start  1234567890.12   1234567889.45                    1            left
1        20   cue_onset   1234567891.23   1234567890.56                  NaN             NaN
2        30    response   1234567892.34   1234567891.67                    1            left
3        40   trial_end   1234567893.45   1234567892.78                  NaN             NaN
```

Metadata fields sent with `send_event()` become columns prefixed with `extra_`.

## See Also

- [Data Acquisition](data-acquisition.md) - Recording continuous data with events
- [Synchronous Mode](synchronous-mode.md) - Trial-based processing using events
- [Events API](../api/generated/dendrite/data/event_outlet.md) - EventOutlet technical reference
