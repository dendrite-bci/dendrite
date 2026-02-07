---
id: data-layer
title: Data Layer
sidebar_label: Data Layer
---

# Data Layer

Receives LSL streams, synchronizes timestamps, and saves raw data to HDF5. Separate queues handle storage and real-time distribution. 

---

## Data Acquisition

DataAcquisition (`src/dendrite/data/acquisition.py`) connects to configured LSL streams (discovered during preflight), uses LSL clock synchronization for timestamp coherence, and distributes data to downstream components.

### Architecture

Dedicated reader threads for each stream type pull samples and send to the data queue at native sampling rates. LSL inlets use `proc_clocksync` and `proc_dejitter` flags to correct clock offsets and smooth timestamp jitter.

Each reader builds per-modality payloads from its channel mapping (e.g., an EEG stream with channels typed as 'eeg', 'eog', and 'markers'). Payloads include LSL timestamps and enqueue timestamps for latency tracking.

An EEG stream is required for acquisition to start.

**Stream Types:**

LSL streams accepted by DataAcquisition, with each stream containing one or more channels classified by modality:

| LSL Stream Type | Format | Required | Purpose |
|-----------------|--------|----------|---------|
| **EEG** | Numeric | Yes | Neural signal acquisition (GUI requirement) |
| **EMG** | Numeric | No | Muscle signal acquisition |
| **EOG** | Numeric | No | Eye movement tracking |
| **ContinuousEvents** | Numeric | No | Continuous position/torque data |
| **Events** | String (JSON) | No | Discrete task events (see **[Events API](../../api/generated/dendrite/data/event_outlet.md)**) |
| **...** | Numeric/String | No | User-defined types (numerical: forwarded; string: saved only) |

Stream types are user-definable LSL type names from the stream's `type()` field. Only **Events** streams have special behavior (JSON parsing; event IDs are injected into Markers by the processor). All numerical streams are forwarded to processing at their native rates. String-formatted streams are saved to disk only.

**Channel Types:**

Channel types are user-defined labels classifying individual channels by modality. A single LSL stream can contain mixed channel types (e.g., an EEG stream with 62 EEG channels, 2 EOG channels, and 1 Markers channel). Common types: EEG, EMG, EOG, Markers, Reference.

### Markers channel

The Markers channel embeds event trigger values into data samples, enabling event-locked epoch extraction.

**Event injection** is the canonical mechanism: when events arrive from the Events stream, the Data Processor queues them and injects the `event_id` into the `markers` field of subsequent data samples. Modes detect events by reading the markers channel (non-zero values indicate event triggers). The processor ensures each configured data stream receives the marker before removing it from the queue.

When a stream has EEG channels but no hardware markers, a synthetic Markers channel is added during stream setup (initialized to zero).

> **Deprecated:** Physical marker extraction from hardware trigger channels (`marker_index`) is legacy functionality for backward compatibility with older recordings. This will be removed in a future release. Use the Events stream for all new implementations.

The storage layer preserves events separately as discrete records with precise timestamps for offline analysis, in addition to the marker values embedded in data samples.

See **[Synchronous Mode](processing-layer.md#synchronousmode)** for epoch triggering and **[Sending Events](task-application-layer.md#sending-events)** for event broadcasting.

### Timestamp Types and Semantics

The system captures multiple timestamps at different pipeline stages for timing analysis and latency tracking.

| Timestamp | Source | Layer | Purpose |
|-----------|--------|-------|---------|
| `timestamp` / `lsl_timestamp` | `inlet.pull_sample()` | Data/Processing | LSL-synchronized capture time |
| `local_timestamp` | `local_clock()` | Storage | Local receive time for HDF5 |
| `_daq_receive_ns` | `time.time_ns()` | Internal | Pipeline latency tracking |
| `data_timestamp` | Copied from `lsl_timestamp` | Mode Output | E2E latency for task apps |



### Latency & Stream Health

**LSL transmission latency** (`local_timestamp - timestamp`) measures transmission delay from device capture to DAQ receive (2-10ms typical). All streams warn if latency exceeds 50ms. Latency does not accumulate; each sample has independent delay and bounded queues prevent backlog buildup. Per-stream metrics (rolling P50 latency, last update timestamp) are published to SharedState for telemetry display.

**Stream drops**: Each stream thread operates independently. Dropped streams hold last value while others continue. TelemetryWidget shows "DROPPED" after timeout. Saved files show drops as timestamp gaps; original timestamps preserved for post-hoc alignment. Streams reconnect automatically when source resumes.

### Data Contracts

**Input (stream configuration):**

Stream configurations use `StreamMetadata` Pydantic schemas (`src/dendrite/data/stream_schemas.py`) containing stream identification (name, type, uid), channel information (count, labels, types, units), timing (sample rate), and validation tracking. Events streams use string `channel_format` instead of numeric.

**Output to Processing Layer (via data_queue):**
```python
{
  'data': {
    '<modality>': np.ndarray,        # Shape: (n_channels, 1) - per channel_type from stream
    'markers': np.ndarray,           # Shape: (1, 1) - synthetic or physical markers
  },  # Keys are dynamic based on channel_types in stream config (e.g., 'eeg', 'eog', 'emg')
  'stream_name': str,                # Stream identifier for routing
  'lsl_timestamp': float,            # LSL synchronized timestamp
  '_daq_receive_ns': int,            # DAQ receive time (latency tracking)
  '{stream_type}_latency_ms': float, # Dynamic key (e.g., eeg_latency_ms, emg_latency_ms)
}
```

**To Storage (via save_queue):**

```python
@dataclass
class DataRecord:
    """Data structure for holding samples and metadata."""
    modality: str          # Stream type: 'EEG', 'EMG', 'Event', 'String'
    sample: Any            # Timeseries: np.ndarray; Events: [json_string]; Strings: [str]
    timestamp: float       # LSL synchronized timestamp (from inlet.pull_sample())
    local_timestamp: float # Local machine timestamp when received (from local_clock())
```

**Example:**
```python
# Timeseries
DataRecord(modality='EEG', sample=np.array([0.1, 0.2]), timestamp=1234.567, local_timestamp=1234.568)

# Event
event_json = json.dumps({'event_id': 10, 'event_type': 'target_onset'})
DataRecord(modality='Event', sample=[event_json], timestamp=1234.567, local_timestamp=1234.568)
```

---

## DataSaver

DataSaver (`src/dendrite/data/storage/data_saver.py`) runs in a separate process, buffering data records and writing chunked HDF5 datasets for real-time isolation.

The saver consumes DataRecord objects from the save queue, routing each to modality-specific handlers. Chunked writes and periodic flushes provide crash resistance. The HDF5 file uses a hierarchical structure for cross-platform compatibility (Python/MATLAB), with each modality as a structured dataset containing one field per channel plus timestamps. Events use a compound datatype storing event_id, event_type, timestamps, and extra_vars (JSON metadata).

**File Structure:**
```text
<session>.h5
├── EEG                 # Structured dataset with channel fields
├── EMG                 # Structured dataset (if present)
├── Event               # Structured event dataset
└── <OtherModalities>   # Additional datasets as configured
attrs: created_timestamp, created_by, version (metadata added via Metadata records)
```

Data is stored using NumPy structured arrays (compound datatypes) with named fields for each channel plus timestamps. Timeseries datasets (EEG, EMG) use float32 channel data with float64 timestamps. Event datasets store event_id, event_type, timestamps, and extra_vars (JSON metadata).

---

## MetricsSaver

MetricsSaver (`src/dendrite/data/storage/metrics_saver.py`) persists mode outputs (predictions, confidences, timing metrics) to a separate HDF5 file for offline analysis.

The saver consumes output records from Processing Modes via `mode_metrics_queues` (dict mapping mode names to their output queues) and creates an HDF5 group for each mode using its configured name. Datasets are created dynamically based on mode output, with common metrics including predictions, confidences, and processing times. Each metric gets a parallel `{metric}_timestamps` dataset for temporal alignment.

**File Structure:**
```text
<session>_metrics.h5
├── <mode_name>/               # e.g., "sync_mode_0", "async_mode_0"
│   ├── <metric_key>           # Dynamic from mode output (predictions, confidences, etc.)
│   ├── <metric_key>_timestamps # Float64 epoch timestamps for temporal alignment
│   └── ...                    # Additional metrics as emitted by mode
```

**Note:** The metrics schema is data-driven. Datasets are created dynamically from mode output packets rather than following a fixed structure.

**Usage:** Metrics file path registered in recordings table, linking mode outputs to source session for experiment lineage tracing.

---

## Database

Database (`src/dendrite/data/storage/database.py`) tracks experiment lineage and recording metadata through an SQLite database located at `data/dendrite.db`.

The database uses a repository pattern with four tables: `studies` (master organization), `recordings` (session metadata with file paths and BIDS fields), `datasets` (imported FIF files with preprocessing parameters), and `decoders` (trained model metadata including accuracy metrics). All repositories use parameterized queries for SQL injection protection and context managers for cleanup. Tables are indexed on names and foreign keys for fast lineage queries.

**Usage:** The Main Window registers new sessions via `add_recording()`, the ML Workbench registers trained models via `add_decoder()`, and the DB Explorer provides a GUI for browsing the database.

See **[Auxiliary Layer](auxiliary-layer.md#database-explorer)** for complete schema documentation.



## Data Dimension Standard

Dendrite uses a consistent **(channels, times)** format throughout the entire processing pipeline:

```
Data Processor: Extracts channels by type → (channels, samples) chunks
                ↓ Send individual samples as (n_channels, 1)
Mode Buffers:   Accumulate → [(n_channels, 1), (n_channels, 1), ...]
                ↓ Direct concatenation along time axis
Mode Output:    np.concatenate(axis=1) → (n_channels, n_times)
                ↓ Add batch dimension in decoder
Decoder Input:  (batch, n_channels, n_times)
                ↓ Model-specific transforms (STFT, etc.)
Model Input:    • Time-series: (batch, 1, n_channels, n_times)
                • Time-frequency: (batch, n_channels, n_frequencies, n_times)
```

---

**Related Documentation:**
- **[Task Application Layer](task-application-layer.md)** - Event creation and broadcasting
- **[Processing Layer](processing-layer.md)** - Real-time data processing workflows
- **[Auxiliary Layer](auxiliary-layer.md)** - Offline analysis and stream management

**API References:**
- **[Events API](../../api/generated/dendrite/data/event_outlet.md)** - Event stream format and integration
