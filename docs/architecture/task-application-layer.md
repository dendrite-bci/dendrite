
# Task Application Layer

External task applications (stimulus programs, games, robotics controllers) communicate with Dendrite via streaming protocols. They send events to mark experimental structure and receive predictions for closed-loop control.

---

## Sending Events

Events mark trial boundaries, stimulus onsets, and experimental conditions. Dendrite captures them for epoch extraction, online training, and offline analysis. Events are JSON strings pushed over an LSL stream of type `Events`.

For stream setup, event format, and code examples, see the **[Send Events Guide](../guides/send-events.md)**.

---

## Receiving Predictions

Dendrite streams mode outputs (predictions, neurofeedback values) to external applications. Four output protocols are supported, configured via the **Output** tab in the frontend or `PUT /api/config/output`.

### Output Protocols

| Protocol | Transport | Default Config | Use Case |
|----------|-----------|----------------|----------|
| **LSL** | Lab Streaming Layer | `PredictionStream`, source `dendrite_default` | BCI research, MATLAB/Python integration |
| **Socket** | TCP or UDP | `127.0.0.1:8080` | Game engines (Unity, Godot), simple clients |
| **ZMQ** | ZeroMQ PUB | `127.0.0.1:5556`, JSON | High-throughput pub/sub, distributed systems |
| **ROS2** | ROS2 topic | topic `bmi_predictions`, node `bmi_prediction_node` | Robotics, ROS2 ecosystems |

Each protocol is independently enabled/disabled. Multiple protocols can run simultaneously.

Source: `src/dendrite/data/streaming/output_schemas.py` (config models), `src/dendrite/data/streaming/base.py` (streamer base classes)

### Packet Structure

All protocols send the same `ModeOutputPacket` as JSON:

```json
{
  "type": "prediction",
  "mode_name": "Sync_1",
  "mode_type": "synchronous",
  "data": { ... },
  "data_timestamp": 1712345678.123
}
```

Source: `src/dendrite/processing/modes/base_mode.py` (`ModeOutputPacket`)

### Prediction Payloads

The `data` field varies by mode type:

**Synchronous (trial classification):**
```json
{
  "prediction": 1,
  "event_name": "left",
  "true_event": "left",
  "confidence": 0.85
}
```

**Asynchronous (continuous classification):**
```json
{
  "prediction": 2,
  "event_name": "right",
  "confidence": 0.72,
  "detected": true
}
```

`detected` is `true` when the dwell gate fires (N consecutive predictions of the same class exceeded the confidence threshold). Task applications should typically act only on `detected: true` packets.

**Neurofeedback (band powers):**
```json
{
  "channel_powers": {
    "C3": {"alpha": 12.5, "beta": 8.3},
    "C4": {"alpha": 9.1, "beta": 11.2}
  },
  "target_bands": {"alpha": [8.0, 12.0], "beta": [13.0, 30.0]}
}
```

### Consuming Predictions

Any language with protocol bindings works. Poll for packets in your application's update loop:

**Python (LSL):**
```python
from pylsl import StreamInlet, resolve_byprop
import json

streams = resolve_byprop("type", "PredictionStream", timeout=5)
inlet = StreamInlet(streams[0])

while running:
    sample, timestamp = inlet.pull_sample(timeout=0.0)
    if sample:
        packet = json.loads(sample[0])
        if packet["data"].get("detected"):
            print(f"Detected: {packet['data']['event_name']}")
```

---

**Related Documentation:**
- **[Data Layer](data-layer.md)** — Event acquisition and storage
- **[Processing Layer](processing-layer.md)** — Mode outputs and prediction streaming
- **[Send Events Guide](../guides/send-events.md)** — Step-by-step event sending tutorial
