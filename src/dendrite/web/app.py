"""
FastAPI application factory.
"""

import asyncio
import multiprocessing
import os

# Disable HDF5 file locking — MUST be set before h5py import.
# On Windows, HDF5's OS-level file locks conflict with SWMR concurrent access,
# causing the writer to crash with "Permission denied" when readers open the file.
# SWMR's internal consistency protocol (atomic flushes) still works without OS locks.
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dendrite.web.deps import cleanup_services, get_ml_service, get_pipeline_service, init_services
from dendrite.web.routers import config, data, ml, modes, pipeline, stream_manager, streams
from dendrite.web.ws.bridge import QueueBridge
from dendrite.web.ws.handlers import router as ws_router
from dendrite.web.ws.telemetry_poller import run_telemetry_poller
from dendrite.web.ws.visualization_bridge import run_visualization_bridge


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan — startup and shutdown."""
    # Set multiprocessing start method (must be done before any Process is created).
    # Always use "spawn" — fork + asyncio is unsafe and deprecated since Python 3.12.
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass  # Already set

    # Initialize database tables
    from dendrite.data.storage.database import Database
    db = Database()
    db.init_db()

    init_services()

    # Create QueueBridge and store in app state
    bridge = QueueBridge()
    bridge.enable_history("mode_data")
    app.state.queue_bridge = bridge

    # Wire bridge to ML service for training progress broadcasting
    get_ml_service().set_bridge(bridge)

    # Start background tasks
    bg_tasks = [
        asyncio.create_task(run_telemetry_poller(bridge, get_pipeline_service)),
        asyncio.create_task(run_visualization_bridge(bridge, get_pipeline_service)),
    ]

    yield

    # Shutdown: stop pipeline if running (with timeout)
    try:
        service = get_pipeline_service()
        if service.is_recording:
            await asyncio.wait_for(service.stop(), timeout=10)
    except (RuntimeError, TimeoutError):
        pass

    # Cancel background tasks and wait for them to finish
    for task in bg_tasks:
        task.cancel()
    await asyncio.gather(*bg_tasks, return_exceptions=True)

    try:
        await asyncio.wait_for(bridge.shutdown(), timeout=5)
    except TimeoutError:
        pass
    cleanup_services()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Dendrite",
        description="Real-time neural signal processing and brain-computer interfaces",
        version="2.0.0",
        lifespan=lifespan,
    )

    # CORS for local development (Vue dev server on different port)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # REST routers
    app.include_router(pipeline.router)
    app.include_router(config.router)
    app.include_router(streams.router)
    app.include_router(modes.router)
    app.include_router(stream_manager.router)
    app.include_router(data.router)
    app.include_router(ml.router)

    # WebSocket router
    app.include_router(ws_router)

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    # Serve built frontend static files (if dist/ exists).
    # Mounted last so /api/* and /ws/* routes take priority.
    from pathlib import Path

    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    dist_dir = Path(__file__).resolve().parent.parent.parent.parent / "frontend" / "dist"
    if dist_dir.is_dir():
        index_html = dist_dir / "index.html"

        # SPA fallback: client-side routes serve index.html
        @app.get("/data")
        @app.get("/ml")
        async def spa_fallback():
            return FileResponse(index_html)

        app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="frontend")

    return app
