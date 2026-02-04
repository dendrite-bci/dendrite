---
id: neurofeedback-mode
title: Neurofeedback Mode
sidebar_label: Neurofeedback Mode
---

# Neurofeedback Mode

Neurofeedback mode extracts real-time band power using Welch's method. Use it for alpha training, SMR training, or any protocol requiring spectral feedback. No decoder required.

## Configuration

Add a neurofeedback mode from the Modes section and open its configuration dialog.

**Frequency bands** to extract (e.g., alpha 8-13 Hz, SMR 12-15 Hz, beta 13-30 Hz).

**Channels**: O1/O2/Oz for alpha, C3/C4/Cz for SMR/beta, Fz/F3/F4 for frontal.

**Window length** (1.0-2.0s) and **step size** (100-250ms). Longer windows give better frequency resolution.

## Optional Settings

**Cluster mode** averages power across selected channels into one value per band.

**Relative power** normalizes by total power (1-40 Hz), reducing artifact sensitivity.

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

- [Data Acquisition](data-acquisition.md)
