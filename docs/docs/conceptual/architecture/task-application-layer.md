---
id: task-application-layer
title: Task Application Layer
sidebar_label: Task Application Layer
---

# Task Application Layer

Task applications communicate with Dendrite via LSL. They send events to mark experimental structure and receive predictions for closed-loop control.

---

## Sending Events

Events mark trial boundaries, stimulus onsets, and experimental conditions. Dendrite captures these for epoch extraction and offline analysis.

### Stream Configuration

| Property | Value |
|----------|-------|
| Type | `Events` |
| Channel format | `string` |
| Channel count | `1` |
| Sample rate | `0` (irregular) |

### Event Payload

Events are JSON strings with two required fields:

```json
{
  "event_id": 20,
  "event_type": "cue_onset",
  "trial_number": 5,
  "condition": "left"
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `event_id` | Yes | Integer for real-time filtering |
| `event_type` | Yes | Human-readable name |
| `*` | No | Application-defined metadata |

Define your own event ID scheme. Group related events (stimuli 20-29, feedback 60-69, etc.) for easier filtering.

### Timing

Send events immediately when they occur. LSL timestamps the packet on push, so delays in your code become timestamp errors. For stimulus-locked analysis, call send before or immediately after rendering.

---

## Receiving Predictions

Dendrite streams predictions and neurofeedback values over LSL. Task applications poll for packets in their update loop.

### Stream Configuration

| Property | Value |
|----------|-------|
| Type | `PredictionStream` |
| Channel format | `string` |
| Channel count | `1` |

### Prediction Payload

Packets are JSON strings. Structure varies by mode:

**Classification (Synchronous/Asynchronous):**
```json
{
  "mode_name": "Async_1",
  "mode_type": "asynchronous",
  "data": {
    "prediction": 1,
    "event_name": "left",
    "confidence": 0.85
  }
}
```

**Neurofeedback:**
```json
{
  "mode_name": "NFB_1",
  "mode_type": "neurofeedback",
  "data": {
    "channel_powers": {
      "C3": {"alpha": 12.5, "beta": 8.3}
    }
  }
}
```


---

## Supported Frameworks

Any language with LSL bindings works.

**Related Documentation:**
- **[Data Layer](data-layer.md)** - Event acquisition and storage
- **[Processing Layer](processing-layer.md)** - Mode outputs and prediction streaming

**API References:**
- **[Events API](../../api/generated/dendrite/data/event_outlet.md)** - Python helper for event streaming
- **[Send Events Guide](../../guides/send-events.md)** - Step-by-step tutorial
