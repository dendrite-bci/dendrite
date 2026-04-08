"""
Pipeline REST endpoints — start, stop, status, preflight.
"""

import asyncio

from fastapi import APIRouter, HTTPException, Request

from dendrite.web.deps import get_config_service, get_pipeline_service, get_preflight_service
from dendrite.web.schemas import PipelineStatusResponse, PreflightResult

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

_start_lock = asyncio.Lock()


@router.get("/preflight", response_model=PreflightResult)
async def preflight_check():
    """Run pre-start validation checks. Returns all failures at once."""
    service = get_preflight_service()
    return service.run_preflight()


@router.post("/start", status_code=200)
async def start_pipeline():
    """Start the processing pipeline using the current configuration state.

    Configure streams, modes, preprocessing, and output via their respective
    endpoints before calling this. Runs preflight validation first.
    """
    if _start_lock.locked():
        raise HTTPException(409, "Pipeline start already in progress")

    async with _start_lock:
        pipeline = get_pipeline_service()
        config_service = get_config_service()
        preflight_service = get_preflight_service()

        # Validate before start
        result = preflight_service.run_preflight()
        if not result.ready:
            failed = [c.model_dump() for c in result.checks if not c.passed]
            raise HTTPException(422, {"message": "Pre-start validation failed", "checks": failed})

        try:
            config = config_service.build_configuration()
            await pipeline.start(config)
            return {"status": "started"}
        except RuntimeError as e:
            raise HTTPException(409, str(e))


@router.post("/stop", status_code=200)
async def stop_pipeline():
    """Stop the processing pipeline."""
    service = get_pipeline_service()
    await service.stop()
    return {"status": "stopped"}


@router.get("/status", response_model=PipelineStatusResponse)
async def pipeline_status():
    """Get current pipeline status with per-component states."""
    service = get_pipeline_service()
    return PipelineStatusResponse(
        recording=service.is_recording,
        recording_id=service.recording_id,
        elapsed_seconds=service.elapsed_seconds,
        log_file=service.log_file,
        mode_pids=service.mode_pids,
        system_pids=service.system_pids,
        component_states=service.get_component_states(),
    )


@router.put("/viz-preprocessing")
async def set_viz_preprocessing(request: Request):
    """Update visualization preprocessing config (safe during recording).

    Body: per-modality config, e.g. {"eeg": {"filter_low": 1.0, "filter_high": 40.0, "apply_rereferencing": true}}
    Empty dict {} resets to defaults.
    """
    body = await request.json()
    service = get_pipeline_service()
    service.set_viz_preproc_config(body)
    return {"status": "ok", "config": body}


@router.get("/viz-preprocessing")
async def get_viz_preprocessing():
    """Get current visualization preprocessing config."""
    service = get_pipeline_service()
    return service.viz_preproc_config


@router.put("/channel-flags")
async def set_channel_flags(request: Request):
    """Set manual bad channel flags (safe during recording).

    Body: {"flagged": {"eeg": [3, 17]}, "unflagged": {"eeg": [55]}}
    Merged with auto-detected bad channels by viz bridge on next quality cycle.
    """
    from dendrite.utils.state_keys import manual_bad_channels_key

    body = await request.json()
    service = get_pipeline_service()
    if service.shared_state:
        service.shared_state.set(manual_bad_channels_key(), body)
    return {"status": "ok", "flags": body}


@router.get("/channel-flags")
async def get_channel_flags():
    """Get current manual bad channel flags."""
    from dendrite.utils.state_keys import manual_bad_channels_key

    service = get_pipeline_service()
    if service.shared_state:
        return service.shared_state.get(manual_bad_channels_key()) or {}
    return {}


@router.get("/session-events")
async def get_session_events():
    """Return unique event codes and names from current recording session."""
    service = get_pipeline_service()
    return await asyncio.to_thread(service.get_session_events)


@router.get("/debug")
async def pipeline_debug(request: Request):
    """Diagnostic endpoint — shows data flow stats for debugging."""
    service = get_pipeline_service()
    bridge = request.app.state.queue_bridge

    # Queue sizes (only available when recording)
    queues = {}
    mode_q = service.visualization_queue
    if mode_q is not None:
        try:
            queues["visualization_queue_size"] = mode_q.qsize()
        except NotImplementedError:
            queues["visualization_queue_size"] = -1

    return {
        "recording": service.is_recording,
        "elapsed_seconds": service.elapsed_seconds,
        "bridge_stats": bridge.get_stats(),
        "drain_tasks_active": getattr(bridge, "_viz_drain_task_count", 0),
        "queues": queues,
    }
