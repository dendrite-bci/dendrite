---
id: task-application-layer
title: Task Application Layer
sidebar_label: Task Application Layer
---

# Task Application Layer

Task applications communicate with Dendrite via LSL. They send events to mark experimental structure and receive predictions for closed-loop control.

---

## Sending Events

Events mark trial boundaries, stimulus onsets, and experimental conditions. Dendrite captures these for epoch extraction and offline analysis. Task applications send events as JSON strings over an LSL stream of type `Events`.

For stream configuration, event format, code examples, and best practices, see the **[Send Events Guide](../../guides/send-events.md)**.

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
