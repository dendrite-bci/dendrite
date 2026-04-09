# Changelog

All notable changes to Dendrite will be documented here.

---

## v0.10.0 - Web Platform Migration

Full platform migration from PyQt6 desktop app to FastAPI + Vue 3 SPA. All four auxiliary tools (Dashboard, Stream Manager, ML Workbench, DB Explorer) are now integrated into a single web application.

### Platform

- FastAPI backend with REST API for configuration/control and WebSocket for real-time data
- Vue 3 SPA with Pinia stores, Tailwind CSS, dark theme with custom design tokens
- Four WS channels: `/ws/telemetry` (1Hz JSON), `/ws/visualization` (~100 Hz msgpack), `/ws/mode_data` (msgpack), `/ws/training` (JSON)
- Singleton service layer (`deps.py`) with dependency injection
- `QueueBridge` drains multiprocessing queues and fans out to WebSocket subscribers
- Single-port production deployment (FastAPI serves built frontend static files)

### Processing Pipeline

- Dynamic mode control: REST API for adding, removing, starting, and stopping modes at runtime without restarting the pipeline
- Online bad channel interpolation via correlation-based weighting (no electrode positions required)
- `SharedRingBuffer` replaces per-mode queues — all consumers read shared memory directly (zero-copy, no pickle overhead)
- Per-mode lazy-init preprocessing (`BaseMode._setup_preprocessor()`, `SamplePreprocessor`)
- Visualization bridge reads ring buffer directly, applies preprocessing + channel quality monitoring
- Bad channel tracking via SharedState with stable rereferencing (no dynamic exclusion to prevent step artifacts)
- Epoch quality control: auto-reject NaN, flat, and extreme outlier epochs during training data prep
- `ComponentStateMachine` for process lifecycle management
- Dead mode detection and automatic cleanup via `check_mode_health()`
- Normalized DAQ latency key to flat `latency_ms` + `stream_type`
- Lazy ML imports (deferred torch loading in decoders)
- Multi-modality refactoring: removed hardcoded EEG references, generalized stream-modality hierarchy
- Online training from live recording data via training queue (sync mode → MLService)

### Data & Storage

- `DataSaver` (SWMR HDF5) is the single writer for session-wide data + epoch storage
- Shared memory ring buffers (`SharedRingBuffer`) for zero-copy DAQ → consumer data flow
- `ReplayStreamer` for offline stream replay (replaces `OfflineDataStreamer`)
- Dataset import from internal DB and MOABB
- Recording protection (prevent deletion during active sessions)
- Continuous channel quality monitoring (MAD-based, replaces one-shot detector)

### ML Workbench 

- Online training offloaded to subprocess 
- Recording-based training: load recordings from database with RecordingBrowser
- Job management: save decoder from completed job, delete job records

### UI/UX

- Embedded live dashboard (EEG, PSD, events, mode plots) directly in Control page
- Telemetry sidebar auto-shows on recording start, hidden when idle
- Stream manager integrated with indicator in nav showing active offline stream count
- Data explorer integrated with full CRUD for studies, recordings, decoders, datasets
- Channel quality summary and total system CPU/RAM in telemetry sidebar
- Core process PIDs (DAQ, DataSaver, MetricsSaver) tracked in telemetry resources
- Mode info panels in dashboard with color-coded latency (Proc/Inf)
- CreateStudyDialog with folder import for bulk .h5 recording registration
- RecordingBrowser component for ML training data selection from database
- Toast notifications and tab close warning during recording
- StreamSetupDialog UX improvements

---

## v0.9.0 - Initial Release

PyQt6 desktop application with real-time EEG/BCI processing pipeline. Three composable processing modes (Synchronous, Asynchronous, Neurofeedback), multi-modality LSL data acquisition, HDF5/SQLite storage, full ML layer (EEGNet, Braindecode, CSP+LDA/SVM), and output over LSL, TCP, ZeroMQ, and ROS2. Separate auxiliary tools: Dashboard, Stream Manager, ML Workbench, DB Explorer.
