# Data Layer

Receives LSL streams, writes raw samples to shared memory ring buffers for zero-copy distribution, and saves data to HDF5 for persistence and offline analysis.

---

## Data Acquisition

DataAcquisition (`src/dendrite/data/acquisition.py`) connects to configured LSL streams, writes samples to shared ring buffers, and forwards events to the event queue for DataSaver persistence.

Dedicated reader threads per stream type pull samples at native rates, writing directly to a `SharedRingBuffer` with markers in the last column. LSL inlets use `proc_clocksync` and `proc_dejitter` for clock correction and jitter smoothing.

At least one numerical data stream (non-Events) is required for acquisition to start.

**Stream Types:**

| LSL Stream Type | Format | Purpose |
|-----------------|--------|---------|
| **EEG** | Numeric | Neural signals |
| **EMG** | Numeric | Muscle signals |
| **EOG** | Numeric | Eye movement signals |
| **ContinuousEvents** | Numeric | Continuous position/torque data |
| **Events** | String (JSON) | Discrete task events (see **[Send Events](/guides/send-events)**) |
| **...** | Numeric/String | User-defined types (numerical: forwarded; string: saved only) |

Multiple streams of the same type are supported. Each gets a unique `stream_key` for pipeline indexing: the first stream of a type gets the type as key (e.g. `"EEG"`), duplicates get `"EEG_2"`, `"EEG_3"`, etc.

Stream types come from the LSL `type()` field. Only **Events** streams have special behavior (JSON parsing; event IDs injected into the markers column). Numerical streams are written to ring buffers at native rates. String streams are saved to disk only.

**Channel Types:**

Channel types are user-defined labels classifying individual channels by modality. A single LSL stream can contain mixed channel types (e.g., an EEG stream with 62 EEG channels, 2 EOG channels, and 1 Markers channel). Common types: EEG, EMG, EOG, Markers, Reference.

### Markers Column

**Creation:** The orchestrator creates each ring buffer with `raw_channels + 1` columns. The last column is the markers column (initialized to zero), invisible to stream configs.

**Event injection:** DAQ's `_events_reader()` broadcasts each `event_id` to per-stream deques. Each reader thread pops from its deque and writes the event code into the markers column, keeping markers synchronized across all ring buffers. **Event codes must be positive integers (> 0) for real-time detection.** The markers column uses `0.0` as the "no event" sentinel, so code 0 is indistinguishable from no event. Modes filter on `markers > 0`; visualization skips `marker == 0`. Events with code 0 or negative codes are still saved to HDF5 via the separate event queue (and available for offline analysis), but are invisible to real-time modes and visualization. The MOABB loader auto-shifts codes if any would be 0.

**Persistence boundary:** The markers column exists only in the ring buffer — DataSaver does not write it to HDF5. Events are persisted separately in the Event dataset with their original LSL timestamps, which are more accurate for offline epoch extraction than the marker injection point (up to 1 sample delayed). Modes use the marker column for real-time detection; offline loading uses the Event dataset.

See **[Synchronous Mode](/guides/synchronous-mode)** for epoch triggering and **[Send Events](/guides/send-events)** for event broadcasting.

### Timestamp Types and Semantics

Multiple timestamps captured at different pipeline stages:

| Timestamp | Source | Layer | Purpose |
|-----------|--------|-------|---------|
| `timestamp` / `lsl_timestamp` | `inlet.pull_sample()` | Data/Processing | LSL-synchronized capture time |
| `local_timestamp` | `local_clock()` | Data/Storage | Local machine receive time (LSL clock domain) |
| `receive_timestamp` | `time.time()` | Storage | Wall-clock receive time (system clock domain) |

**Internal timing (not persisted):** `_receive_ns` (nanosecond pipeline latency tracking via `time.time_ns()`) and `data_timestamp` (copy of `lsl_timestamp` forwarded in mode output for end-to-end latency).

### Latency & Stream Health

**LSL transmission latency** (`local_timestamp - timestamp`): 2-10ms typical, warns if >50ms. Each sample has independent delay. Per-stream metrics (rolling P50 latency, last update) published to SharedState for telemetry.

**Stream drops**: Stream threads operate independently. Dropped streams hold last value while others continue. TelemetryWidget shows "DROPPED" after timeout. Saved files show drops as timestamp gaps. Streams reconnect automatically when source resumes.

### Data Flow

DAQ writes each sample as a flat float32 row `[raw_channels..., marker]` to the ring buffer. Consumers use the channel map to slice modality data. Events flow separately through `event_queue` as `EventRecord` objects to DataSaver.

Source: `src/dendrite/data/stream_schemas.py` (StreamMetadata), `src/dendrite/data/acquisition.py` (EventRecord)

---

## Shared Ring Buffer

SharedRingBuffer (`src/dendrite/data/shared_buffers.py`) is a lock-free SPMC ring buffer backed by `multiprocessing.shared_memory`.

One ring buffer per numerical stream (keyed by `stream_key`). DAQ writes; all consumers connect and read independently with zero-copy access.

**Memory Layout:**
```text
[Header 64B][Data float32 (max_samples × n_channels)][LSL_ts float64 (max_samples)][Local_ts float64 (max_samples)][Receive_ns uint64 (max_samples)]
```

Header stores: write position (u64), channel count (u32), max samples (u32), sample rate (f64). Three timestamp arrays alongside data: LSL-synchronized, local clock, and wall-clock receive (nanoseconds).

Default buffer duration is 30 seconds (configurable via `compute_max_samples()`). At 500Hz with 65 channels, a buffer uses ~4MB of shared memory.

Buffer names derive from `stream_key`: `dendrite_rb_{stream_key.lower()}`. All consumers unpack `read_new()` the same way:

```python
data, lsl_ts, local_ts, receive_ns, new_pos = rb.read_new(last_read_pos)
```

Returns all samples written since `last_read_pos`. If the reader falls too far behind, `OverrunError` is raised.

Source: `src/dendrite/data/shared_buffers.py`

### Channel Map

The orchestrator builds a channel map per ring buffer:

```python
{
  "buffer_name": "dendrite_rb_eeg",       # derived from stream_key
  "modalities": {"eeg": [0,1,...,61], "eog": [62,63]},  # column indices by channel type
  "modality_labels": {"eeg": ["Fp1", ...], "eog": ["VEOG", "HEOG"]},
  "marker_col": 64,
  "sample_rate": 500.0,
}
```

---

## DataSaver

DataSaver (`src/dendrite/data/storage/data_saver.py`) is the single HDF5 writer for both archival and online training. Runs as a separate process, reads from shared ring buffers, and writes chunked structured datasets to disk.

Drains new samples every poll cycle, buffers in memory, and writes chunks to HDF5. Only modality channels are saved -- the markers column is excluded. Events arrive via `event_queue` from DAQ as `EventRecord` objects. Flushes every 2s for crash resistance.

**SWMR** mode is enabled after the first flush, allowing MLService to read epochs concurrently during live recording.

**File Structure:**
```text
<file_identifier>_raw.h5
├── EEG                 # Structured dataset: [Fp1 f32, Fp2 f32, ..., timestamp f64, local_timestamp f64, receive_timestamp f64]
├── EEG_2               # Second EEG stream (if present, keyed by stream_key)
├── EMG                 # Structured dataset (if present)
├── Event               # Compound dataset: [event_id i32, event_type str, timestamp f64, local_timestamp f64, receive_timestamp f64, extra_vars str]
└── <OtherStreams>      # Additional datasets as configured
attrs: created_timestamp, created_by, version, schema_version, study_name, ...,
       stream_index (JSON: {"EEG": "EEG", "EEG_2": "EEG", "EMG": "EMG"})
```

Dataset names match the `stream_key`. The `stream_index` root attribute maps each dataset name to its stream type, enabling `find_dataset()` to locate datasets by type even when names are disambiguated.

Timeseries datasets use NumPy structured arrays (one float32 per channel + three float64 timestamps). Dataset attrs preserve stream metadata for downstream channel type resolution. Events store the full event dict as JSON in `extra_vars`.

---

## MetricsSaver

MetricsSaver (`src/dendrite/data/storage/metrics_saver.py`) persists mode outputs (predictions, confidences, timing) to a separate HDF5 file.

Consumes output records from modes via `shared_metrics_queue`, creating one HDF5 group per mode. Datasets are created dynamically from mode output. Each metric gets a parallel `{metric}_timestamps` dataset.

**File Structure:**
```text
<session>_metrics.h5
├── <mode_name>/               # e.g., "sync_mode_0", "async_mode_0"
│   ├── <metric_key>           # Dynamic from mode output (predictions, confidences, etc.)
│   ├── <metric_key>_timestamps # Float64 epoch timestamps for temporal alignment
│   └── ...                    # Additional metrics as emitted by mode
```

**Usage:** Metrics file path registered in recordings table, linking mode outputs to source session.

---

## Database

Database (`src/dendrite/data/storage/database.py`) tracks experiment lineage and recording metadata via SQLite at `data/dendrite.db`.

Four tables: `studies` (organization), `recordings` (session metadata, file paths, BIDS fields), `decoders` (trained model metadata, accuracy), `training_jobs` (async training status/results). Repository pattern with parameterized queries.

PipelineService registers sessions at pipeline start; ML workbench registers trained models; Data Explorer provides a browse interface.

## Data Dimension Standard

**(channels, times)** format throughout the pipeline:

```
Ring Buffer:    DAQ writes flat rows → (n_channels,) per sample
                ↓ Modes slice by channel map
Mode Input:     Per-sample dicts with (n_channels, 1) arrays
                ↓ Buffer accumulates
Mode Buffers:   [(n_channels, 1), (n_channels, 1), ...]
                ↓ Direct concatenation along time axis
Mode Output:    np.concatenate(axis=1) → (n_channels, n_times)
                ↓ Add batch dimension in decoder
Decoder Input:  (batch, n_channels, n_times)
                ↓ Model-specific transforms (STFT, etc.)
Model Input:    • Time-series: (batch, 1, n_channels, n_times)
                • Time-frequency: (batch, n_channels, n_frequencies, n_times)
```

---

## File Loaders

Three loaders (`src/dendrite/data/loaders/`) read offline data into `RawData` format: **RawH5Loader** (`.h5`), **FIFLoader** (`.fif`), and **MOABBLoader** (MOABB public datasets). All return `RawData` with data in (channels, samples) format, lowercase channel types, and sample-indexed events.

`RawH5Loader` uses `find_dataset()` to locate data datasets by type via the `stream_index` attribute. Events are recovered from the Event dataset using `np.searchsorted` for sub-sample precision. HDF5 stores (samples, channels); loaders transpose on read.

MNE interop: `to_mne_raw()` on RawData, or `to_mne_raw()`/`export_to_fif()` directly from H5 files. Event metadata preserved via JSON in `raw.info['description']`.

Source: `src/dendrite/data/loaders/_types.py` (RawData, EpochedData), `src/dendrite/data/io/` (H5 explorer, MNE export)

### Training Data Pipeline

`load_moabb_for_training()` and `load_epochs()` compose loaders with epoching and preprocessing. Both apply configurable bandpass/CAR and reject bad epochs (NaN, flat, outlier). `load_epochs()` supports SWMR HDF5 reads for online training during live recording.

Source: `src/dendrite/data/loaders/_training_data.py`

---

## Channel Quality Monitoring

`ChannelQualityMonitor` (`src/dendrite/data/quality.py`) detects bad channels using rolling-window (5s) iterative MAD refinement with hysteresis (2 of 3 evaluations must exceed threshold). Per-channel status ("good"/"warning"/"bad") drives both the real-time quality display in the visualization bridge and the automatic interpolation pipeline.

---

**Related Documentation:**
- **[Send Events](/guides/send-events)** — Event creation and broadcasting
- **[Processing Layer](./processing-layer)** — Real-time data processing workflows
