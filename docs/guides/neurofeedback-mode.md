# Neurofeedback Mode

Real-time band power extraction via Welch's method. No decoder required.

## Configuration

Add a neurofeedback mode instance from the **Modes** section in the control panel. Click **Configure** to set frequency bands and channels.

**Frequency bands** to extract (e.g., alpha 8-13 Hz, SMR 12-15 Hz, beta 13-30 Hz).

**Channels**: select channels relevant to your target bands and modality from the configured stream.

**Window length** (1.0-2.0s) and **step size** (100-250ms). Longer windows give better frequency resolution.

## Optional Settings

**Cluster mode** averages power across selected channels into one value per band.

**Relative power** normalizes by total power across a broad band (range depends on modality and sample rate).

## Output

Per-channel band powers with timestamps stream via LSL and internal queues.

```json
{
  "type": "neurofeedback",
  "mode_name": "nfb_1",
  "mode_type": "neurofeedback",
  "data": {
    "channel_powers": {
      "O1": {"alpha": 12.5, "beta": 8.2},
      "O2": {"alpha": 11.8, "beta": 7.9}
    },
    "target_bands": {"alpha": [8, 13], "beta": [13, 30]}
  },
  "data_timestamp": 1705312345.123
}
```


## See Also

- [Data Acquisition](./data-acquisition)
