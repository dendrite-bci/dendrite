# Synchronous Mode

Segments data around discrete events, collecting labeled epochs for classifier training and evaluation.

## Configuration

Add a synchronous mode instance from the **Modes** section in the control panel. Click **Configure** to open the mode settings.

**Epoch timing** defines the window extracted around each event. Set the start offset (typically 0.0s) and end offset (1.0-2.0s depending on paradigm).

**Event mappings** connect event codes to class labels (e.g., 1 → "left", 2 → "right"). Unmapped codes are ignored.

## Online Training

Enable training to update the decoder as epochs accumulate. Select a decoder type (EEGNet, CSP+LDA) and set a training interval.

Trained decoders save when the mode stops. Linked async modes (`decoder_source: "online"`) auto-load new decoders during recording sessions.

## Output

Each event triggers epoch extraction and prediction. The mode outputs per-trial classification with confidence and tracks prequential accuracy and Cohen's kappa.

```json
{
  "type": "prediction",
  "mode_name": "sync_1",
  "mode_type": "synchronous",
  "data": {
    "prediction": 1,
    "event_name": "left",
    "true_event": "left",
    "confidence": 0.85
  },
  "data_timestamp": 1705312345.123
}
```

## See Also

- [Asynchronous Mode](./asynchronous-mode)
- [Send Events](./send-events)
- [ML Training](./ml-training)
