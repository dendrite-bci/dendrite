# System Architecture

Dendrite is a multi-process BCI platform. A **Vue 3 frontend** communicates with a **FastAPI backend** over REST and WebSocket. The backend manages configuration, serves real-time data to the browser, and spawns a **pipeline subprocess** that runs data acquisition, recording, signal processing, and BCI modes as independent processes. Raw samples flow between processes through shared memory ring buffers (zero-copy); mode outputs and control signals use multiprocessing queues.

Source: `src/dendrite/web/app.py` (FastAPI entry + pipeline startup), `src/dendrite/processing/orchestrator.py` (pipeline subprocess entry).

## Layers

| Layer | Scope |
|-------|-------|
| **[Web](./web-layer)** | FastAPI services, REST/WebSocket API, Vue 3 frontend |
| **[Data](./data-layer)** | LSL acquisition, HDF5 recording, stream replay, dataset loading |
| **[Processing](./processing-layer)** | Online preprocessing, pipeline orchestration, composable BCI modes |
| **[ML](./ml-layer)** | Decoder training (online and offline), evaluation, hyperparameter search |
| **[Task Application](./task-application-layer)** | Output to external applications via LSL, TCP socket, ZMQ, or ROS2 |
