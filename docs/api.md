# API Reference

Dendrite exposes a REST API for configuration and control, and WebSocket channels for real-time data. When the backend is running, interactive API docs are available at `/docs` (Swagger UI).

## Pipeline

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/pipeline/start` | POST | Start processing pipeline |
| `/api/pipeline/stop` | POST | Stop pipeline |
| `/api/pipeline/status` | GET | Pipeline state (recording, elapsed, PIDs) |
| `/api/pipeline/preflight` | GET | Run pre-start validation checks |
| `/api/pipeline/debug` | GET | Data flow diagnostics |
| `/api/pipeline/viz-preprocessing` | GET/PUT | Visualization preprocessing config |
| `/api/pipeline/channel-flags` | GET/PUT | Manual bad channel flags for interpolation |
| `/api/pipeline/session-events` | GET | Unique event codes seen in current session |

## Configuration

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/config` | GET | Get full current configuration |
| `/api/config/general` | GET/PUT | Study name, subject ID, session ID |
| `/api/config/output` | GET/PUT | Output protocol settings |
| `/api/config/output/availability` | GET | Which output protocols have dependencies installed |
| `/api/config/output/defaults` | GET | Default config values for each output protocol |
| `/api/config/load?file_path=...` | POST | Load config from JSON |
| `/api/config/save` | POST | Save config to JSON |
| `/api/config/next-run` | GET | Next auto-incremented run number for subject/session |
| `/api/config/list` | GET | List saved config files |

## Streams

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/streams` | GET | Get configured streams |
| `/api/streams/discover` | POST | Discover LSL streams on network |
| `/api/streams/configure` | POST | Configure selected streams |
| `/api/streams/liveness` | GET | Check if configured streams are still available |

## Modes

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/modes` | GET/POST | List/create mode instances |
| `/api/modes/{name}` | GET/PUT/DELETE | Get/update/remove mode instance |
| `/api/modes/{name}/start` | POST | Start mode during recording |
| `/api/modes/{name}/stop` | POST | Stop mode during recording |
| `/api/modes/{name}/state` | GET | Get mode component state |
| `/api/modes/{name}/rename` | POST | Rename mode instance |

## WebSocket Channels

| Endpoint | Format | Description |
|----------|--------|-------------|
| `/ws/visualization` | msgpack (~100Hz) | Raw signal data (decimated to ~100 Hz, preprocessed) |
| `/ws/telemetry` | JSON (1Hz) | CPU, memory, stream health, latency, channel quality |
| `/ws/mode_data` | msgpack | Mode output data (predictions, ERP, band power) |
| `/ws/training` | JSON | Decoder training progress |

## Data Explorer

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/data/studies` | GET/POST | List/create studies |
| `/api/data/studies/{id}` | GET/PUT/DELETE | Study CRUD |
| `/api/data/studies/import-folder` | POST | Scan folder for .h5 files, register under study |
| `/api/data/studies/pick-folder` | POST | Native OS folder picker for study import |
| `/api/data/recordings` | GET | List recordings (filterable, searchable) |
| `/api/data/recordings/{id}` | GET/DELETE | Recording detail/delete |
| `/api/data/recordings/{id}/file-info` | GET | HDF5 file structure |
| `/api/data/recordings/{id}/channels` | GET | Channel labels, count, sample rate |
| `/api/data/recordings/{id}/signal-preview` | GET | Signal preview data |
| `/api/data/recordings/{id}/erp` | GET | ERP preview (epoch/filter params) |
| `/api/data/recordings/{id}/qc-preview` | GET | Signal quality preview |
| `/api/data/recordings/{id}/event-summary` | GET | Event summary |
| `/api/data/recordings/{id}/summary` | GET | Session summary |
| `/api/data/recordings/{id}/telemetry` | GET | Historical telemetry data |
| `/api/data/recordings/{id}/mode-performance` | GET | Mode performance metrics |
| `/api/data/decoders` | GET | List decoders (filterable, searchable) |
| `/api/data/decoders/{id}` | GET/DELETE | Decoder detail/delete |
| `/api/data/decoders/{id}/metadata` | GET | Full decoder metadata (event/label mappings) from saved JSON |

## ML Training

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ml/models` | GET | List available model types |
| `/api/ml/models/{type}/schema` | GET | Model config JSON schema |
| `/api/ml/load-recording` | POST | Load recording for ML training |
| `/api/ml/data/loaded` | GET | Loaded data metadata |
| `/api/ml/moabb/datasets` | GET | List MOABB datasets |
| `/api/ml/moabb/load` | POST | Load MOABB dataset |
| `/api/ml/train` | POST | Start training job |
| `/api/ml/train/{job_id}/cancel` | POST | Cancel training job |
| `/api/ml/jobs` | GET | List training jobs |
| `/api/ml/jobs/{id}` | GET | Job status |
| `/api/ml/jobs/{id}` | DELETE | Delete a job record |
| `/api/ml/jobs/{id}/reaggregate` | POST | Re-aggregate eval results with new decision gate |
| `/api/ml/jobs/{id}/save-decoder` | POST | Save trained model as named decoder |
| `/api/ml/search-categories/{decoder_type}` | GET | Available search categories for a decoder type |
| `/api/ml/evaluate` | POST | Epoch-by-epoch decoder evaluation |
| `/api/ml/benchmark` | POST | K-fold CV benchmark across models |

## Stream Manager

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stream-manager/start` | POST | Start file/MOABB replay stream |
| `/api/stream-manager/stop/{id}` | POST | Stop a replay stream |
| `/api/stream-manager/status` | GET | Active stream status |
| `/api/stream-manager/pick-file` | POST | Open native file picker |
| `/api/stream-manager/file-info` | POST | Get file metadata |
| `/api/stream-manager/moabb` | GET | List MOABB datasets |
| `/api/stream-manager/datasets` | GET | List internal datasets |

## Health

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
