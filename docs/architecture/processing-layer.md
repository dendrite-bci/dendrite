
# Processing Layer

Runs processing modes that classify signals or extract features. Modes run independently or together.

All consumers read from shared memory ring buffers written by DAQ. Each mode runs as an independent process with its own buffer and per-mode preprocessing. Outputs flow through FanOutQueue to MetricsSaver and the WebSocket visualization bridge.

**Processing Flow:**
```text
DAQ → SharedRingBuffer (per stream, zero-copy shared memory)
            ↑ read_new()
            ├─ SynchronousMode  (event-driven training)
            ├─ AsynchronousMode (continuous inference)
            ├─ NeurofeedbackMode (band power extraction)
            ├─ DataSaver         (SWMR HDF5 for archival + online training)
            └─ run_visualization_bridge() (WebSocket broadcast + quality monitoring)
```

---

## Data Distribution

DAQ writes raw samples (all channels + markers column) to one `SharedRingBuffer` per stream type (see **[SharedRingBuffer](data-layer.md#shared-ring-buffer)**). All consumers connect to the same buffer and read independently -- zero-copy, no serialization.

`PipelineOrchestrator` (`src/dendrite/processing/orchestrator.py`) creates ring buffers, passes names + channel maps to every consumer, and manages pipeline subprocess lifecycle.

**Channel Maps**

```python
{
  "buffer_name": "dendrite_rb_eeg",                        # derived from stream_key
  "modalities": {"eeg": [0,1,...,61], "eog": [62,63]},    # column indices by channel type
  "modality_labels": {"eeg": ["Fp1", ...], "eog": ["VEOG", "HEOG"]},
  "marker_col": 64,                                        # last column
  "sample_rate": 500.0,
}
```

All consumers use this map to slice columns from the shared data array.

---

## Preprocessing

OnlinePreprocessor (`src/dendrite/processing/preprocessing/preprocessor.py`) applies modality-specific filtering without modifying raw storage.

Runs **inside each mode** (not centrally). BaseMode lazily creates the preprocessor on first sample, using actual channel counts. CAR operates on all channels before channel selection narrows to the mode's subset.

**Per-Mode Lifecycle:**
1. `_setup_preprocessor()` — reads config, creates `SamplePreprocessor` (`mode_utils.py`)
2. `SamplePreprocessor._ensure_preprocessor(data_dict)` — lazily creates `OnlinePreprocessor` on first sample using actual channel counts
3. `_preprocess_sample(sample)` — delegates to `SamplePreprocessor`, applies preprocessing on all channels, then selects subset

**Modality Processors**

Single `ModalityProcessor` class with config-driven behavior: bandpass filter (lowcut/highcut/filter_order), notch filter (line_freq/notch_width), common average reference (apply_rereferencing), interpolation (channel_labels), and anti-aliased downsampling (downsample_factor). Omitting filter keys gives passthrough. Markers bypass preprocessing. `ChannelScaler` (`preprocessing/scalers.py`) provides optional per-channel z-score normalization.

**Bad Channel Detection & Interpolation**

`ChannelQualityMonitor` (MAD-based z-score with hysteresis) runs in the visualization bridge. After 10s warmup, the bad channel list is frozen and a correlation-based interpolation matrix W is precomputed from the warmup data's pairwise Pearson correlations. Applied per-chunk as `data[bad] = W @ data[good]` before CAR. No montage or 3D electrode positions required.

Operators can manually flag/unflag channels via dashboard. Manual flags merge with auto-detected: `effective_bad = (auto UNION flagged) MINUS unflagged`. Changes trigger W recomputation via `interp_version` in SharedState.

Interpolation pipeline:
1. **Detection**: `ChannelQualityMonitor` on raw signal data (rolling 5s window, iterative MAD z-score, hysteresis)
2. **Freeze**: After 10s warmup, list locked. Override via `PUT /api/pipeline/channel-flags`
3. **Interpolation**: `CorrelationInterpolationMatrix` computes W from pairwise channel correlations in warmup data. `InterpolationApplicator` applies W per chunk before CAR
4. **Training parity**: `load_recording_epochs()` applies the same interpolation to raw HDF5 data before offline preprocessing

Processing order per chunk: **interpolate bad channels → CAR (all channels) → bandpass filter → downsample**

**Epoch Quality Control**

`EpochQualityChecker` (`src/dendrite/data/quality.py`) rejects epochs with NaN/Inf, flat signals (variance < 1e-12), or extreme outliers (MAD z-score > 50). Applied by `RawData.epoch()` (enabled by default).

**Preprocessing Parity**

OnlinePreprocessor uses causal IIR filtering (`scipy.signal.lfilter` with state preservation) for both:
- **Online**: sample-by-sample processing with maintained filter state
- **Offline**: `apply_preprocessing_offline()` (`preprocessing/offline_adapter.py`) processes data in chunks (default 250 samples), simulating online streaming. Used by MLService when loading training data.

---

## Processing Modes

Each mode can be powered by a Decoder that wraps a predictive Model. Modes are independently optional and can run concurrently. All inherit from BaseMode (`src/dendrite/processing/modes/base_mode.py`) and run as independent `multiprocessing.Process` instances.

**Data Flow**

Each mode reads samples from its assigned ring buffer via `_get_next_sample()`. Samples arrive as dicts with dynamic modality keys (e.g. `'eeg'`, `'emg'`) containing `(n_channels, 1)` arrays, plus `'markers'` and timestamps. Modality keys come from the channel map — not hardcoded.

Samples accumulate in `Buffer` (per-modality deques), which provides `extract_window()` for continuous inference and `extract_epoch_at_event()` for event-locked epochs. Outputs route through `FanOutQueue` to MetricsSaver and the visualization bridge — dropped (not blocked) if queues are full.

Source: `src/dendrite/processing/modes/base_mode.py`, `src/dendrite/processing/modes/mode_utils.py`

---

## SynchronousMode

SynchronousMode (`src/dendrite/processing/modes/synchronous_mode.py`) collects event-locked epochs and trains models online.

Monitors event markers to extract time-locked windows (default 0.0-2.0s post-stimulus, configurable via `epoch_tmin`/`epoch_tmax`). Training triggered at configurable intervals (e.g., every 10 epochs) via `training_queue` -- MLService pulls epoch data from SWMR HDF5.

**Output:** Predictions with confidence, accuracy, Cohen's kappa, and averaged ERP waveforms per event type.

---

## AsynchronousMode

AsynchronousMode (`src/dendrite/processing/modes/asynchronous_mode.py`) applies pre-trained decoders continuously for real-time BCI control.

Processes sliding windows at regular intervals. Supports decoder hot-swapping from linked SynchronousMode or disk. Polls SharedState ~1Hz for new decoders, loads in background thread. When ground truth events are available, temporal evaluation calculates accuracy with separate background handling.

**Output:** Predictions with confidence, accuracy metrics when ground truth available.

---

## NeurofeedbackMode

NeurofeedbackMode (`src/dendrite/processing/modes/neurofeedback_mode.py`) extracts spectral band power.

Sliding windows with Welch's method per channel, extracting power in configurable frequency bands (alpha, beta, SMR, etc.). Optional baseline normalization (percent-change) and channel clustering for regional feedback.

**Output:** Per-channel band power per frequency band, sent to both `prediction_queue` and `output_queue`.

---

## Visualization Bridge

`run_visualization_bridge()` (`src/dendrite/web/ws/visualization_bridge.py`) reads raw signal data from the primary stream's ring buffer, preprocesses, monitors quality, and broadcasts via QueueBridge.

**Raw Data Path:**
1. Reads from stream ring buffer (zero-copy)
2. Applies CAR + bandpass
3. Publishes channel quality to SharedState every 2s
4. Decimates 5x before WebSocket broadcast
5. Events bypass decimation to preserve timing

**Mode Output Path:**
Drains `visualization_queue` and broadcasts to `mode_data` WebSocket channel.

**Architecture:**
```text
SharedRingBuffer ──► VizBridge ──► QueueBridge ──► /ws/visualization (raw signal data)
                     (preproc +     (msgpack)
                      quality)

visualization_queue ──► VizBridge ──► QueueBridge ──► /ws/mode_data (mode outputs)
```

The bridge reconfigures preprocessing dynamically when settings change during recording.

**Generic Output Streamers**

Alternative output protocols inheriting from `BaseOutputStreamer` (`src/dendrite/data/streaming/`):

| Streamer | Protocol | Use Case |
|----------|----------|----------|
| **LSLStreamer** | Generic LSL | Configurable LSL output for custom applications |
| **SocketStreamer** | TCP/UDP | Low-latency local networking (Python/MATLAB clients) |
| **ZMQStreamer** | ZeroMQ | High-performance pub/sub for distributed systems |
| **ROS2Streamer** | ROS2 Topics | Robotic control integration (wheelchair, prosthetics, exoskeletons) |

Each streamer runs in a separate process, consuming mode output queues asynchronously.

---

**Related Documentation:**
- **[ML Layer](./ml-layer)** — Decoders, models, and training infrastructure
- **[Data Layer](./data-layer)** — Data acquisition, ring buffers, and storage
- **[Web Layer](./web-layer)** — WebSocket bridge, frontend architecture
