# Asynchronous Mode

Continuous classification using sliding windows. Requires a pre-trained decoder.

## Loading a Decoder

Add an asynchronous mode instance from the **Modes** section in the control panel. Click **Configure** to set the decoder source.

**Decoder source** options:
- `database` — loads a saved decoder from `data/studies/<study>/decoders/`. Select matching channels via the channel selection grid.
- `online` — receives live decoder updates from a running synchronous mode. Set `source_mode` to the sync mode's name.

## Step Size

Set **step size** (50-500ms) to control prediction frequency. A 100ms step produces 10 predictions per second.

Window length is fixed to match the decoder's training input shape and updates automatically when a new decoder is loaded.

## Output


```json
{
  "type": "prediction",
  "mode_name": "async_1",
  "mode_type": "asynchronous",
  "data": {
    "prediction": 1,
    "event_name": "left",
    "confidence": 0.85,
    "detected": false
  },
  "data_timestamp": 1705312345.123
}
```



## See Also

- [Synchronous Mode](./synchronous-mode)
- [ML Training](./ml-training)
