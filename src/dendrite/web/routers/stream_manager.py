"""Stream Manager REST endpoints."""

import asyncio

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator

from dendrite.web.deps import get_stream_manager_service, require_local
from dendrite.web.routers.data import EEG_FILE_FILTER, open_native_picker

router = APIRouter(prefix="/api/stream-manager", tags=["stream-manager"])


class StartStreamRequest(BaseModel):
    source: str = "file"  # file, moabb
    # File source
    path: str = ""
    # MOABB source
    dataset: str = ""
    subject: int | None = None
    session: str | None = None
    # Options
    stream_name: str | None = None
    enable_events: bool = False


class FileInfoRequest(BaseModel):
    path: str

    @field_validator("path")
    @classmethod
    def reject_traversal(cls, v: str) -> str:
        from dendrite.web.schemas import validate_no_path_traversal

        return validate_no_path_traversal(v)


@router.post("/start")
async def start_stream(request: StartStreamRequest):
    service = get_stream_manager_service()
    stream_id = await asyncio.to_thread(service.start_stream, request.model_dump())
    return {"id": stream_id}


@router.post("/stop/{stream_id}")
async def stop_stream(stream_id: str):
    service = get_stream_manager_service()
    await asyncio.to_thread(service.stop_stream, stream_id)
    return {"stopped": stream_id}


@router.get("/status")
async def get_status():
    service = get_stream_manager_service()
    return {"streams": service.get_status()}


@router.get("/moabb")
async def list_moabb():
    service = get_stream_manager_service()
    datasets = await asyncio.to_thread(service.list_moabb_datasets)
    return {"datasets": datasets}


@router.get("/datasets")
async def list_datasets():
    """List internal datasets and recordings from the database."""
    service = get_stream_manager_service()
    return {"datasets": service.list_internal_datasets()}


@router.post("/file-info")
async def get_file_info(request: FileInfoRequest):
    service = get_stream_manager_service()
    info = await asyncio.to_thread(service.get_file_info, request.path)
    if "error" in info:
        raise HTTPException(400, info["error"])
    return info


@router.post("/pick-file", dependencies=[Depends(require_local)])
async def pick_file():
    """Open native OS file picker and return selected path."""
    try:
        path = await asyncio.wait_for(
            asyncio.to_thread(open_native_picker, "file", "Select recording file", EEG_FILE_FILTER),
            timeout=120,
        )
    except (TimeoutError, asyncio.CancelledError):
        return {"path": None}
    return {"path": path or None}
