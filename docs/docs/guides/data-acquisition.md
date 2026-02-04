---
id: data-acquisition
title: Data Acquisition
sidebar_label: Data Acquisition
---

# Data Acquisition

Dendrite connects to any LSL-compatible amplifier. LSL synchronizes timestamps across devices at different sampling rates.

## Discovering Streams

Click **Discover LSL Streams** to scan the network. The discovery dialog shows all available LSL outlets with their metadata: stream name, type, channel count, and sampling rate. Select the streams you need for your experiment.


After selecting streams, click **Apply** to finalize your configuration.

## Channel Configuration

The discovery dialog pre-populates channel labels and types from the LSL stream descriptor. Most amplifier software embeds this metadata correctly, but some devices send generic labels (Ch1, Ch2...) or incorrect types. Review the channel table and reassign any mislabeled channels before starting acquisition.

Channel types matter because the system routes data differently based on modality. 

## Starting Acquisition

Click **Start** to begin data acquisition. The system resolves each configured stream, opens LSL inlets with clock synchronization enabled, and spawns reader threads that pull samples at each stream's native rate.

The telemetry panel shows per-stream latency. High latency usually indicates network congestion or an overloaded source machine.

Click **Stop** to end acquisition and close all stream connections. All data is saved automatically.



## See Also

- [Send Events](send-events.md) - Send event markers from task applications
