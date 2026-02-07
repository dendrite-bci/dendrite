---
id: processing-layer
title: Processing Layer
sidebar_label: Processing Layer
---

# Processing Layer

Runs processing modes that classify signals or extract features. Modes run independently or together.

DataProcessor distributes samples to all enabled modes via dedicated queues. Each mode runs as an independent process with its own buffer. Optional preprocessing applies modality-specific filtering before distribution. Mode outputs flow through FanOutQueue to MetricsSaver and VisualizationStreamer.

**Processing Flow:**
```text
Data Acquisition → Data Processor → ┬─ SynchronousMode (Event-driven training)
                   (Preprocessing    ├─ AsynchronousMode (Continuous inference)
                    + Fan-out)       └─ NeurofeedbackMode (Band power extraction)
```

---

## Data Processor

DataProcessor (`src/dendrite/processing/processor.py`) receives incoming samples and distributes synchronized data to all enabled modes with backpressure handling.

The processor receives per-sample payloads from **[`DataAcquisition`](data-layer.md#data-acquisition)** via `data_queue`, building channel-type index mapping (EEG/EOG/EMG/MARKERS) from shared state. Reshapes each sample into modality dictionaries `(n_channels,)`, applies optional **[Preprocessing](#preprocessing)**, then fans out to all **[enabled mode queues](#processing-modes)**. Additionally, samples are transmitted to dashboard plot queue for real-time visualization (decimated 5× for bandwidth reduction, e.g., 500Hz → 100Hz).

Ensures all enabled processing modes receive identical samples in identical order for consistent multi-mode operation. Samples include `_daq_receive_ns` timestamp for latency tracking across pipeline stages. Non-blocking queues drop new samples when full to prevent processing stalls.

**Data Contracts**

**Input from [`DataAcquisition`](data-layer.md#data-contracts):**
```python
{
  'data': {
    'eeg': np.ndarray,               # Shape: (n_channels, 1) - EEG modality channels
    'eog': np.ndarray,               # Shape: (n_channels, 1) - EOG modality channels
    'emg': np.ndarray,               # Shape: (n_channels, 1) - EMG modality channels
    'markers': np.ndarray,           # Shape: (1, 1) - Event code (-1=background, >0=event IDs)
  },
  'lsl_timestamp': float,            # LSL synchronized timestamp
  '_daq_receive_ns': int,            # DAQ receive time (latency tracking)
  'eeg_latency_ms': float,           # Per-stream LSL transmission latency
}
```

**Output to Processing Modes:**
```python
{
  # Filtered modality data (simple modality keys)
  'eeg': np.ndarray,                 # Shape: (n_channels, 1) - if mode requires 'eeg'
  'markers': np.ndarray,             # Shape: (1, 1) - Event code (-1=background, >0=event IDs)

  # Metadata (prefixed with underscore)
  'lsl_timestamp': float,            # LSL synchronized timestamp
  '_daq_receive_ns': int,            # DAQ receive time
  '_stream_name': str,               # Source stream name for routing
  '_eeg_latency_ms': float,          # Latency keys get underscore prefix
}
```

All enabled modes receive identical samples in identical order via synchronized fan-out. Consistent `(channels, times)` format throughout - reshapes chunks to (channels, samples), slices to (channels, 1) per sample, modes concatenate into (channels, n_times) buffers without transpose operations.

**Markers Channel:** Modes receive **[Markers channel](data-layer.md#markers-channel)** as `np.ndarray` shape `(1, 1)` with event codes in per-sample dictionaries. SynchronousMode uses Markers for event-triggered epoch extraction, AsynchronousMode for temporal performance evaluation. Markers bypass preprocessing and are attached directly by DataProcessor.

**Fan-Out Architecture**

The DataProcessor uses a fan-out pattern to distribute samples to all enabled modes simultaneously. Each mode instance gets a dedicated queue with modality filtering (only required modalities + metadata), optional stream routing (filter to specific input streams), and non-blocking distribution (samples dropped when full to maintain real-time priority). Independent failure isolation ensures a full queue in one mode doesn't affect others.

---

## Preprocessing

OnlinePreprocessor (`src/dendrite/processing/preprocessing/preprocessor.py`) applies modality-specific filtering without modifying raw data storage.

**Modality Processors**

Processors apply modality-specific signal conditioning: EEGProcessor (re-referencing, bandpass filtering, downsampling), EMGProcessor (notch filter, bandpass, downsampling), EOGProcessor (bandpass, downsampling), and PassthroughProcessor (type conversion only). Markers bypass preprocessing entirely to maintain timing fidelity. Configuration includes global target sample rate, per-modality filter parameters, and optional artifact rejection.

**Preprocessing Parity**

Minimal preprocessing is currently applied. Identical preprocessing between offline training and online deployment prevents covariate shift.

OnlinePreprocessor uses causal IIR filtering (`scipy.signal.lfilter` with state preservation) for both:
- **Online**: Real-time sample-by-sample processing with maintained filter state
- **Offline**: Chunk-based processing that simulates real-time behavior
---

## Processing Modes

Modes define the primary operational state and determine the data processing workflow and output structure. Each mode can be powered by a Decoder, which manages a data processing Pipeline. The final stage of every Pipeline is a predictive Model that translates neural signals into commands.

Modes are independently optional: enable SynchronousMode for event-driven training, AsynchronousMode for continuous inference, NeurofeedbackMode for spectral feedback, or run multiple modes concurrently. All modes inherit from BaseMode (`src/dendrite/processing/modes/base_mode.py`) and run as independent `multiprocessing.Process` instances. Process isolation prevents crashes in one mode from affecting others. All modes share unified buffering with consistent (channels, times) format and dual-queue output management for visualization (`output_queue`) and external control (`prediction_queue`).

**Data Contracts**

**Input:** Per-sample dictionaries from [DataProcessor](#data-processor) (see **Output to Processing Modes** above), filtered to `required_modalities`.

**Output (ModeOutputPacket):**
```python
{
  'mode_name': str,                 # Instance identifier (e.g., "Synchronous_1")
  'mode_type': str,                 # 'synchronous'/'asynchronous'/'neurofeedback'
  'type': str,                      # Output category ('performance', 'erp', 'prediction', etc.)
  'data': dict,                     # Mode-specific payload (see mode sections below)
  'data_timestamp': float           # LSL timestamp for E2E latency measurement
}
```

**Mode Buffering**

Modes use `Buffer` (`src/dendrite/processing/modes/mode_utils.py`) to accumulate samples into analysis windows:

```python
# Per-sample input: Full sample dictionary with all modalities
sample = {'eeg': np.ndarray, 'markers': int, 'lsl_timestamp': float, ...}
buffer.add_sample(sample)  # Stores all modalities in separate deques

# Window extraction: Returns dict of (n_channels, n_times) per modality
X = buffer.extract_window()  # Full window for continuous inference
epoch = buffer.extract_epoch_at_event(event_idx)  # Epoch around event marker
```

The buffer maintains separate deques per modality plus timestamps for latency tracking. Window extraction concatenates buffered samples along axis=1, producing the standard `(channels, times)` format. Markers are excluded from data extraction by default.

**Output Routing**

Modes emit `ModeOutputPacket` (see **Data Contracts** above) through `OutputQueueManager`. **FanOutQueue** routes outputs to multiple destinations without blocking:

| Queue | Priority | Purpose | On Full |
|-------|----------|---------|---------|
| Primary | Critical | MetricsSaver (HDF5) | Warn + drop |
| Secondary | Best-effort | VisualizationStreamer | Silent drop |
| Monitoring | Optional | Telemetry | Silent drop |

FanOutQueue uses `put_nowait()` for real-time operation (outputs are dropped rather than blocking). Mode output queues use blocking `put()` to ensure delivery.

---

## SynchronousMode

SynchronousMode (`src/dendrite/processing/modes/synchronous_mode.py`) collects event-locked epochs from experimental trials and trains classification models online.

The mode monitors event markers to extract time-locked windows around triggers (default 0.0s to 2.0s after stimulus onset, configurable via `start_offset` and `end_offset`). Labeled epochs accumulate in a dataset, and the decoder retrains at configurable intervals (e.g., every 20 epochs). Optional features include background class sampling for idle-state detection, quality checking to reject noisy epochs, and hyperparameter search across model architectures.

**Output:** Predictions with confidence scores, performance metrics (accuracy, Cohen's kappa), and averaged ERP waveforms per event type.

---

## AsynchronousMode

AsynchronousMode (`src/dendrite/processing/modes/asynchronous_mode.py`) applies pre-trained decoders continuously for real-time BCI control.

The mode processes sliding windows at regular intervals, streaming predictions with confidence scores. It supports decoder hot-swapping from linked SynchronousMode or disk-based loading. Models update automatically when SynchronousMode saves to the shared directory (`data/studies/{study_name}/decoders/shared/`). When ground truth events are available, temporal evaluation calculates accuracy with separate background handling for idle periods.

**Output:** Predictions with confidence, and accuracy metrics when ground truth available.

---

## NeurofeedbackMode

NeurofeedbackMode (`src/dendrite/processing/modes/neurofeedback_mode.py`) extracts spectral band power for neurofeedback protocols.

The mode maintains sliding windows, applies Welch's method to each channel, and extracts power within configurable frequency bands (alpha, beta, SMR, etc.). Optional baseline normalization converts absolute power to percent-change metrics, and channel clustering averages power across electrode groups for regional feedback.

**Output:** Per-channel band power values for each configured frequency band, sent to both queues (`prediction_queue` for external control, `output_queue` for visualization).

---

## Output Streaming

VisualizationStreamer (`src/dendrite/data/streaming/visualization.py`) publishes mode outputs and raw data via LSL. Additional streamers support Socket, ZMQ, and ROS2 protocols.

**VisualizationStreamer**

VisualizationStreamer publishes a unified LSL stream for real-time dashboard visualization. Consumes plot_queue (raw biosignal samples from DataProcessor) and mode_output_queues (predictions, performance metrics, ERP data) to create structured JSON payloads with three distinct types.

**Payload Types:**
- **RawDataPayload** - Per-sample biosignal data (EEG, EMG, EOG) with channel labels for real-time traces
- **ModeOutputPayload** - Mode predictions, performance metrics, ERP averages, neurofeedback features
- **ModeHistoryPayload** - Historical data sent to new consumers for instant plot initialization

**History Management:**

Maintains bounded buffer for mode packet history (1000 packets per mode). Sends ModeHistoryPayload (last 100 packets) to new consumers so dashboards populate immediately.

**Queue Architecture:**
```
DataProcessor → plot_queue → VisualizationStreamer → LSL Stream → Dashboard
             ↓
    mode_queues → Modes → FanOutQueue ├→ MetricsSaver (primary)
                                      └→ VisualizationStreamer (secondary)
```

**Generic Output Streamers**

Alternative output protocols for non-LSL ecosystems, all inheriting from `BaseOutputStreamer` with unified multiprocessing architecture:

| Streamer | Protocol | Use Case |
|----------|----------|----------|
| **LSLStreamer** | Generic LSL | Configurable LSL output for custom applications |
| **SocketStreamer** | TCP/UDP | Low-latency local networking (Python/MATLAB clients) |
| **ZMQStreamer** | ZeroMQ | High-performance pub/sub for distributed systems |
| **ROS2Streamer** | ROS2 Topics | Robotic control integration (wheelchair, prosthetics, exoskeletons) |

Each streamer operates in a separate process, consuming mode output queues asynchronously to prevent blocking real-time processing.

**Data Contracts**

**Unified LSL Stream Output (JSON String Channel):**
```python
# RawDataPayload - Per-sample biosignal data
{
  'type': 'raw_data',
  'timestamp': float,
  'sample_rate': int,                # Effective sample rate after decimation
  'data': {
    'eeg': [ch1, ch2, ...],          # Per-modality arrays (lowercase keys)
    'emg': [...]
  },
  'channel_labels': {
    'eeg': ['C3', 'Cz', 'C4', ...],
    'emg': ['Flexor', 'Extensor']
  }
}

# ModeOutputPayload - Mode predictions and metrics
{
  'type': 'performance' | 'erp' | 'neurofeedback_features',
  'timestamp': float,
  'mode_name': 'Synchronous_1',  # Unique mode instance identifier
  'mode_type': 'synchronous' | 'asynchronous' | 'neurofeedback',
  'data': {...},  # Mode-specific payload (predictions, accuracy, ERP, features)
  'data_timestamp': float  # LSL timestamp for E2E latency measurement
}

# ModeHistoryPayload - Initialization data for new consumers
{
  'type': 'mode_history',
  'timestamp': float,
  'mode_name': 'Synchronous_1',
  'mode_type': 'synchronous',
  'data': {'latest_output': {...}},  # Most recent mode output
  'packets': [...],  # Last 100 packets
  'packet_count': int
}
```

---

**Related Documentation:**
- **[ML Layer](ml-layer.md)** - Decoders, models, and training infrastructure
- **[Data Layer](data-layer.md)** - Data acquisition and synchronized distribution

**API References:**
- **[Decoders API](../../api/generated/dendrite/ml/decoders/decoder.md)** - Decoder interface and usage
