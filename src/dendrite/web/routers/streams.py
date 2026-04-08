"""
Stream discovery and configuration REST endpoints.
"""

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from dendrite.data.stream_schemas import StreamMetadata
from dendrite.web.deps import get_stream_service, require_not_recording
from dendrite.web.schemas import StreamConfigureRequest, StreamMetadataResponse

router = APIRouter(prefix="/api/streams", tags=["streams"])


def _serialize_streams(streams: dict[str, StreamMetadata]) -> dict[str, dict]:
    """Convert StreamMetadata objects to API-safe dicts via StreamMetadataResponse."""
    return {
        uid: StreamMetadataResponse.from_stream_metadata(s).model_dump()
        for uid, s in streams.items()
    }


@router.post("/discover")
async def discover_streams(timeout: float = 2.0, _=Depends(require_not_recording)):
    """Discover all available LSL streams on the network.

    This may take up to `timeout` seconds. Runs in a background thread.
    """
    service = get_stream_service()
    discovered = await asyncio.to_thread(service.discover_and_cache, timeout)

    return {
        "streams": _serialize_streams(discovered),
        "count": len(discovered),
    }


@router.post("/configure")
async def configure_streams(request: StreamConfigureRequest, _=Depends(require_not_recording)):
    """Select streams from discovery results and configure them."""
    service = get_stream_service()

    # Use cached discovery if recent, else re-discover
    discovered = service.get_cached_discovery()
    if discovered is None:
        discovered = await asyncio.to_thread(service.discover_and_cache, 2.0)

    # Validate requested UIDs exist
    missing = [uid for uid in request.selected_uids if uid not in discovered]
    if missing:
        raise HTTPException(404, f"Streams not found: {missing}. Re-run discovery.")

    result = service.configure_streams(
        request.selected_uids, discovered, request.channel_overrides or None
    )

    return {
        "configured": _serialize_streams(result["streams"]),
        "issues": result["issues"],
        "modalities_by_stream": service.get_modalities_by_stream(),
    }


@router.get("/liveness")
async def check_stream_liveness():
    """Quick check if configured streams are still available on the network."""
    service = get_stream_service()
    if not service.has_streams():
        return {"liveness": {}}
    liveness = await asyncio.to_thread(service.check_liveness)
    return {"liveness": liveness}


@router.get("")
async def get_streams():
    """Get currently configured streams."""
    service = get_stream_service()
    streams = service.get_streams()
    return {
        "streams": _serialize_streams(streams),
        "has_streams": service.has_streams(),
        "modalities_by_stream": service.get_modalities_by_stream(),
    }
