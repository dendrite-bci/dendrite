"""
Mode instance CRUD REST endpoints with live mode control.

CRUD (add/update/delete) is allowed during recording if the target mode is not running.
Live control (start/stop) is available during recording for dynamic mode management.
"""

from typing import Any

from fastapi import APIRouter, HTTPException

from dendrite.web.deps import (
    get_mode_service,
    get_pipeline_service,
    get_preflight_service,
    require_mode_not_running,
)
from dendrite.web.schemas import ModeInstanceRequest, ModeRenameRequest

router = APIRouter(prefix="/api/modes", tags=["modes"])


def _build_config(name: str, request: ModeInstanceRequest) -> dict[str, Any]:
    return {**request.config, "name": name, "mode": request.mode}


def _require_mode_editable(name: str) -> None:
    """Allow mode mutation if not recording, OR if recording but mode is stopped."""
    pipeline = get_pipeline_service()
    if pipeline.is_recording:
        require_mode_not_running(name)


@router.get("")
async def list_modes():
    """List all mode instances."""
    service = get_mode_service()
    return {
        "instances": service.get_all_instances(),
        "names": service.get_all_instance_names(),
    }


@router.post("")
async def add_mode(request: ModeInstanceRequest):
    """Add a new mode instance. Allowed during recording."""
    service = get_mode_service()

    name = request.name or service.generate_unique_name(
        request.mode.title(), sanitize=True
    )
    config = _build_config(name, request)

    try:
        if not service.add_instance(name, config):
            raise HTTPException(409, f"Instance '{name}' already exists")
    except ValueError as e:
        raise HTTPException(422, str(e)) from e

    return {"name": name, "config": service.get_instance(name)}


@router.get("/{name}")
async def get_mode(name: str):
    """Get a specific mode instance."""
    service = get_mode_service()
    instance = service.get_instance(name)
    if instance is None:
        raise HTTPException(404, f"Instance '{name}' not found")
    return instance


@router.put("/{name}")
async def update_mode(name: str, request: ModeInstanceRequest):
    """Update a mode instance. Allowed during recording if mode is stopped."""
    _require_mode_editable(name)

    service = get_mode_service()
    config = _build_config(name, request)

    try:
        if not service.update_instance(name, config):
            raise HTTPException(404, f"Instance '{name}' not found")
    except ValueError as e:
        raise HTTPException(422, str(e)) from e

    return service.get_instance(name)


@router.delete("/{name}")
async def delete_mode(name: str):
    """Delete a mode instance. Allowed during recording if mode is stopped."""
    _require_mode_editable(name)

    service = get_mode_service()
    if not service.remove_instance(name):
        raise HTTPException(404, f"Instance '{name}' not found")
    return {"ok": True}


@router.post("/{name}/rename")
async def rename_mode(name: str, request: ModeRenameRequest):
    """Rename a mode instance. Allowed during recording if mode is stopped."""
    _require_mode_editable(name)

    service = get_mode_service()
    if not service.rename_instance(name, request.new_name):
        raise HTTPException(409, f"Cannot rename '{name}' to '{request.new_name}'")
    return {"old_name": name, "new_name": request.new_name}


# ------------------------------------------------------------------
# Live mode control (during recording)
# ------------------------------------------------------------------


@router.post("/{name}/start")
async def start_mode(name: str):
    """Start a mode during a recording session.

    Validates the mode config, runs per-mode preflight, and spawns the mode.
    """
    pipeline = get_pipeline_service()
    if not pipeline.is_recording:
        raise HTTPException(409, "Cannot start mode: pipeline not recording")

    mode_service = get_mode_service()
    instance = mode_service.get_instance(name)
    if instance is None:
        raise HTTPException(404, f"Instance '{name}' not found")

    if pipeline.is_mode_running(name):
        raise HTTPException(409, f"Mode '{name}' is already running")

    # Per-mode preflight
    preflight = get_preflight_service()
    result = preflight.run_mode_preflight(name)
    if not result.ready:
        failed = [c for c in result.checks if not c.passed]
        details = "; ".join(f"{c.label}: {c.detail}" for c in failed)
        raise HTTPException(422, f"Mode preflight failed: {details}")

    pid = await pipeline.start_mode(name, instance)
    if pid is None:
        raise HTTPException(500, f"Failed to start mode '{name}'")

    return {"status": "started", "name": name, "pid": pid}


@router.post("/{name}/stop")
async def stop_mode(name: str):
    """Stop a mode during a recording session."""
    pipeline = get_pipeline_service()
    if not pipeline.is_recording:
        raise HTTPException(409, "Cannot stop mode: pipeline not recording")

    if not pipeline.is_mode_running(name):
        raise HTTPException(409, f"Mode '{name}' is not running")

    await pipeline.stop_mode(name)
    return {"status": "stopped", "name": name}


@router.get("/{name}/state")
async def get_mode_state(name: str):
    """Get the component state for a running mode."""
    pipeline = get_pipeline_service()
    states = pipeline.get_component_states()
    mode_key = f"mode:{name}"
    state = states.get(mode_key, "idle")
    return {"name": name, "state": state, "running": pipeline.is_mode_running(name)}
