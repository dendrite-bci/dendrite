---
id: send-events
title: Send and Log Events
sidebar_label: Send Events
---

# Send Events

Events are JSON strings sent over LSL with at minimum `event_id` (int) and `event_type` (str). Additional metadata fields are stored alongside in HDF5.

## Python

```python
from dendrite.data.streaming.event_outlet import EventOutlet

outlet = EventOutlet(
    stream_name='TaskEvents',
    events={'cue_left': 10, 'cue_right': 11, 'response': 20},
)

outlet.send_event('cue_left', {'trial': 1, 'condition': 'MI'})
outlet.send_event('response', {'rt_ms': 450})
outlet.close()
```

Send events immediately after the stimulus/response -- LSL timestamps on push, so delays become timestamp errors.

## Other Languages

Any LSL client can send events. Create a string stream with type `Events`, channel count 1, and push JSON:

**MATLAB:**
```matlab
info = lsl_streaminfo(lib, 'Events', 'Events', 1, 0, 'cf_string', 'events_id');
outlet = lsl_outlet(info);
outlet.push_sample({jsonencode(struct('event_id', 10, 'event_type', 'cue_left'))});
```

**C++:**
```cpp
lsl::stream_info info("Events", "Events", 1, LSL_IRREGULAR_RATE, lsl::cf_string);
lsl::stream_outlet outlet(info);
std::string event = R"({"event_id": 10, "event_type": "cue_left"})";
outlet.push_sample(&event);
```

## Loading Stored Events

```python
from dendrite.data.io import load_events

df = load_events('recording.h5')
# Columns: event_id, event_type, timestamp, local_timestamp, extra_*
```

## See Also

- [Data Acquisition](data-acquisition.md)
- [Synchronous Mode](synchronous-mode.md)
