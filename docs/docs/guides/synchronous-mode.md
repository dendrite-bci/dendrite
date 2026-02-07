---
id: synchronous-mode
title: Synchronous Mode
sidebar_label: Synchronous Mode
---

# Synchronous Mode

Synchronous mode segments data around discrete events, collecting labeled epochs for classifier training and evaluation. Use it for motor imagery with cued trials, P300 oddball paradigms, or any protocol with known event timing.

## Configuration

Add a synchronous mode from the Modes section and open its configuration dialog.

**Epoch timing** defines the window extracted around each event. Set the start offset (typically 0.0s) and end offset (1.0-2.0s depending on paradigm).

**Event mappings** connect event codes to class labels. Map event ID 1 to "left", ID 2 to "right", etc. Unmapped codes are ignored.

## Online Training

Enable training to update the decoder as epochs accumulate. Select a model type (EEGNet, CSP+LDA) and set a training interval. An interval of 10 retrains after every 10 new epochs.

Trained models save every 30 seconds. Asynchronous modes configured with `decoder_source: sync_mode` receive these updates automatically.


## Output

Each event triggers epoch extraction and prediction. The mode outputs per-trial classification with confidence and tracks accuracy and kappa. ERP visualization shows averaged waveforms per event type.

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

- [Send Events](send-events.md)
