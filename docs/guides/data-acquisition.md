# Data Acquisition

Dendrite connects to any LSL-compatible amplifier.

## Discovering Streams

Open the **Configure Streams** dialog from the control panel. Click **Rescan** to discover all LSL outlets on the network. Each stream shows its name, type, channel count, and sampling rate. Toggle streams on/off, then click **Apply** to finalize your selection.

You can also discover streams via the API: `POST /api/streams/discover`.

## Channel Configuration

The stream setup dialog pre-populates channel labels and types from the LSL stream descriptor. Some devices send generic labels (Ch1, Ch2...) or incorrect types. Review and correct before starting.

## Starting Acquisition

Click **Start** to begin recording. The dashboard shows per-stream latency — high latency usually indicates network congestion or an overloaded source machine.

Click **Stop** to end the session. All data is saved automatically to HDF5.

## See Also

- [Send Events](./send-events) — Send event markers from task applications
