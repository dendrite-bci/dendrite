# Core Concepts

## Streams

Dendrite acquires data over [LSL](https://labstreaminglayer.org) (Lab Streaming Layer). Any LSL-compatible amplifier or software outlet can be used. Each stream has a type (EEG, EMG, EOG, Events), channel count, and sample rate. Multiple streams can be active simultaneously and are recorded together into a single session.

For development without hardware, the [Stream Manager](./stream-replay) can replay recorded files or public datasets as live LSL streams.

## Sessions and Recordings

A **session** starts when you click Start and ends when you click Stop. During a session, all incoming stream data is saved continuously to an HDF5 file alongside event markers and metadata. Each recording is stored under a study and tagged with subject, session, and run identifiers following BIDS-inspired naming.

## Modes

Modes are the processing units that run during a session. Each mode is a separate process that reads from the shared data buffers, applies its logic, and outputs results. You can run multiple mode instances simultaneously.

| Mode | Purpose | Requires decoder |
|------|---------|:---:|
| **[Synchronous](./synchronous-mode)** | Segments data around events into epochs, requests decoder training, tracks prequential accuracy | No (requests training from ML service) |
| **[Asynchronous](./asynchronous-mode)** | Continuous sliding-window classification with dwell-based detection | Yes |
| **[Neurofeedback](./neurofeedback-mode)** | Real-time band power extraction via Welch's method | No |

Modes are composable: a synchronous mode can train a decoder that an asynchronous mode uses in the same session (`decoder_source: "online"`).

## Decoders

A **decoder** is a trained classifier that maps preprocessed EEG (or other signal) epochs to class labels. Decoders wrap either neural networks (EEGNet, etc.) or classical pipelines (CSP+LDA) behind a unified sklearn-compatible interface.

Decoders can be trained in two ways:
- **Online** — a synchronous mode requests training from the ML service as epochs accumulate during a live session
- **Offline** — the ML Workbench loads recordings or public datasets and trains in the background

A saved decoder is a JSON config (input shape, class mapping, model type) paired with a model file (`.pt` for neural, `.joblib` for classical).

## Events

Events are timestamped markers sent over LSL during a session — typically from a task application. Each event has a numeric `event_id` and a string `event_type`. Events drive epoch segmentation in synchronous mode and provide ground truth for asynchronous mode metrics. See [Send Events](./send-events).

## Output Protocols

Mode outputs (predictions, band powers) can be forwarded to external applications over LSL, TCP socket, ZMQ, or ROS2. This lets task applications close the loop — receiving classifications and adapting the paradigm in real time. See [Task Application Layer](../architecture/task-application-layer).
