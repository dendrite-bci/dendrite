---
id: asynchronous-mode
title: Asynchronous Mode
sidebar_label: Asynchronous Mode
---

# Asynchronous Mode

Asynchronous mode provides continuous classification using sliding windows. Use it for BCI control, mental state monitoring, or any application needing steady predictions without discrete trials. Requires a pre-trained decoder.

## Loading a Decoder

Add an asynchronous mode from the Modes section and open its configuration dialog.

**Decoder source** options: `pretrained` loads from file, `database` loads from the experiment database, `sync_mode` receives live updates from a running synchronous mode.

For file-based loading, browse to the decoder's `.json` config in `data/studies/<study>/decoders/`. Channel labels must match your live stream.

With `sync_mode`, the decoder polls for updates every 5 seconds.

## Step Size

Set **step size** (50-500ms) to control prediction frequency. A 250ms step produces 4 predictions per second.

Window length is fixed to match the decoder's training epoch length and cannot be changed.

## Output

Each prediction includes class index, label, confidence, and timestamp. Predictions stream via LSL and internal queues. When ground-truth labels are available, the mode displays accuracy.

```json
{
  "type": "prediction",
  "mode_name": "async_1",
  "mode_type": "asynchronous",
  "data": {
    "prediction": 1,
    "event_name": "left",
    "confidence": 0.85
  },
  "data_timestamp": 1705312345.123
}
```

Filter noisy predictions by confidence score in your application.

## See Also

- [Synchronous Mode](synchronous-mode.md)
