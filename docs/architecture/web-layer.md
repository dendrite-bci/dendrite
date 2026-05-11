# Web Layer

The web layer connects the Vue 3 frontend to the processing pipeline via FastAPI.

## Backend Services

Source: `src/dendrite/web/deps.py` (singleton wiring), `src/dendrite/web/services/`.

Services are singletons created at startup and manage distinct concerns:

| Service | Responsibility |
|---------|---------------|
| **ConfigService** | Holds configuration state, builds `PipelineConfig` |
| **PipelineService** | Pipeline lifecycle (start/stop), mode control, viz config |
| **StreamService** | LSL stream discovery and configuration |
| **ModeService** | Mode instance CRUD |
| **PreflightService** | Pre-start validation checks |
| **MLService** | Data loading, online training (subprocess), decoder management, evaluation reaggregation |
| **DataService** | Database access (studies, recordings, decoders, datasets) |
| **StreamManagerService** | File/MOABB replay stream management |

Services communicate through direct references. `ConfigService` aggregates state from `StreamService` and `ModeService` to build the pipeline configuration.

## WebSocket Channels

Source: `src/dendrite/web/ws/` (handlers, `QueueBridge`, viz/telemetry bridges).

Real-time data flows from the pipeline to browser clients via `QueueBridge`:

```
SharedRingBuffer ──► run_visualization_bridge() ──► QueueBridge ──► WebSocket ──► Browser
visualization_queue ───────────────────────────┘
SharedState ──────────► run_telemetry_poller() ──► QueueBridge ──► WebSocket ──► Browser
```

| Channel | Format | Rate | Content |
|---------|--------|------|---------|
| `/ws/visualization` | msgpack | ~100Hz | Raw signal data (decimated to ~100 Hz, preprocessed) |
| `/ws/telemetry` | JSON | 1Hz | CPU, memory, stream latency, mode metrics, channel quality |
| `/ws/mode_data` | msgpack | event-driven | Mode output data (predictions, ERP, band power) |
| `/ws/training` | JSON | per-epoch | Training progress, completion, errors |

**QueueBridge** (`ws/bridge.py`) drains multiprocessing queues from a thread pool and fans out to WebSocket subscribers. Slow clients get frames dropped (no backpressure). `enable_history()` lets new subscribers receive recent frames on connect.

Three long-lived background tasks run in the FastAPI event loop (not inside the pipeline subprocess) so they can serve HTTP/WebSocket clients and touch SQLite while the pipeline runs independently — heavy lifting (model training, HPO trials) is still offloaded to subprocesses from within them, so a runaway training job can't freeze the API or the UI.

**`run_visualization_bridge()`** (`ws/visualization_bridge.py`) — async task spawned/cancelled on recording start/stop. Reads raw signal data from the primary stream's ring buffer, applies preprocessing (CAR + bandpass), monitors channel quality, broadcasts via QueueBridge. Mode outputs drained from `visualization_queue` to `mode_data` channel.

**Online Training Task** (`web/services/pipeline_service.py`) -- Spawned by `PipelineService.start()` when a pipeline is active; drains `training_queue` (from SynchronousMode or manual requests) by invoking `MLService.run_online_training_loop()`. MLService loads data from SWMR HDF5, trains in a subprocess, publishes decoder path to SharedState for hot-swap.

**Background Optimizer** (`ml/search/optuna_runner.py`, managed by `ml_service.py`) -- Per-mode asyncio tasks running Optuna search during recording. Loads fresh session data, searches within mode's model type, promotes winners (5%+ improvement) via SharedState. Profiles (quick/balanced/full) control trial count and scope.

## Frontend Architecture

- **Pinia stores**: config, pipeline, streams, modes, data, ml, streamManager, telemetry, visualization
- **Composables**: `useWebSocket`, `useRingBuffer`, `useToast`, `useDecoderPicker`, `useSessionEvents`
- **uPlot**: real-time signal visualization. The backend broadcasts `/ws/visualization` at ~100 Hz; the browser coalesces incoming frames into ~30 fps paints via `requestAnimationFrame`, so extra samples buffer in a ring between frames rather than triggering per-sample re-renders.
- **PrimeIcons**, **Tailwind CSS**

### Views & Components

Three routed views: **ControlView** (`/`), **DataExplorerView** (`/data`), **MLWorkbenchView** (`/ml`). ControlView embeds `DashboardView` as the live monitoring panel. Components are organized in feature-based subdirectories: `common/`, `config/`, `dashboard/`, `data/`, `layout/`, `ml/`, `stream-manager/`.

### Data Flow

```
REST API → Pinia store → Vue component (reactive)
WebSocket → composable → ring buffer → uPlot (real-time)
```

Config flow: UI form -> store action -> `PUT /api/config/*` -> service state. Pipeline reads at start via `ConfigService.build_configuration()`.

## Pipeline Integration

```
Frontend            Backend              Pipeline Subprocess
  │                    │                        │
  ├─POST /start──────►│                        │
  │                    ├──_run_session_io()─────►│ (DB, files)
  │                    ├──orchestrator.start()──►│ (DAQ, saver, session, metrics)
  │                    ├──start_modes()─────────►│ (sync, async, nfb)
  │                    │                        │
  │◄──/ws/telemetry───┤◄──SharedState──────────┤ (1Hz polling)
  │◄──/ws/visualization┤◄──ring buffer─────────┤ (VizBridge reads + preprocesses)
  │◄──/ws/mode_data───┤◄──viz_queue────────────┤ (mode outputs)
  │                    │                        │
  ├─POST /stop───────►│                        │
  │                    ├──orchestrator.stop()───►│ (graceful shutdown)
```
