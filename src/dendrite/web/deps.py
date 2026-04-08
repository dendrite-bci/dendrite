"""
Dependency injection — singleton services.

Services are created once and shared across all requests.
"""

import logging
import socket

from fastapi import HTTPException, Request

from dendrite.data.storage.database import Database
from dendrite.web.services.config_service import ConfigService
from dendrite.web.services.data_service import DataService
from dendrite.web.services.ml_service import MLService
from dendrite.web.services.mode_service import ModeService
from dendrite.web.services.pipeline_service import PipelineService
from dendrite.web.services.preflight_service import PreflightService
from dendrite.web.services.stream_manager_service import StreamManagerService
from dendrite.web.services.stream_service import StreamService

logger = logging.getLogger(__name__)

_pipeline_service: PipelineService | None = None
_stream_service: StreamService | None = None
_mode_service: ModeService | None = None
_config_service: ConfigService | None = None
_preflight_service: PreflightService | None = None
_stream_manager_service: StreamManagerService | None = None
_data_service: DataService | None = None
_ml_service: MLService | None = None


def init_services() -> None:
    """Initialize all singleton services. Called from app lifespan."""
    global _pipeline_service, _stream_service, _mode_service, _config_service
    global _preflight_service, _stream_manager_service, _data_service, _ml_service

    _stream_service = StreamService()
    _mode_service = ModeService(_stream_service)
    _config_service = ConfigService(_stream_service, _mode_service)
    _preflight_service = PreflightService(_stream_service, _mode_service, _config_service)
    _pipeline_service = PipelineService()
    db = Database()
    db.init_db()
    _data_service = DataService(db=db)
    _stream_manager_service = StreamManagerService(recording_repo=_data_service.recordings)
    _ml_service = MLService(_data_service)


def get_pipeline_service() -> PipelineService:
    if _pipeline_service is None:
        raise RuntimeError("Services not initialized")
    return _pipeline_service


def get_stream_service() -> StreamService:
    if _stream_service is None:
        raise RuntimeError("Services not initialized")
    return _stream_service


def get_mode_service() -> ModeService:
    if _mode_service is None:
        raise RuntimeError("Services not initialized")
    return _mode_service


def get_config_service() -> ConfigService:
    if _config_service is None:
        raise RuntimeError("Services not initialized")
    return _config_service


def get_preflight_service() -> PreflightService:
    if _preflight_service is None:
        raise RuntimeError("Services not initialized")
    return _preflight_service


def get_stream_manager_service() -> StreamManagerService:
    if _stream_manager_service is None:
        raise RuntimeError("Services not initialized")
    return _stream_manager_service


def get_data_service() -> DataService:
    if _data_service is None:
        raise RuntimeError("Services not initialized")
    return _data_service


def get_ml_service() -> MLService:
    if _ml_service is None:
        raise RuntimeError("Services not initialized")
    return _ml_service


def require_not_recording() -> None:
    """FastAPI dependency that blocks config mutations during recording.

    Used for DAQ/Processor-level config (streams, preprocessing).
    For mode-level config, use require_mode_not_running() instead.
    """
    svc = get_pipeline_service()
    if svc.is_recording:
        raise HTTPException(
            status_code=409,
            detail="Cannot modify configuration while recording. Stop the pipeline first.",
        )


def require_mode_not_running(name: str) -> None:
    """Block mutation of a specific mode if it's currently running.

    Allows mode CRUD during recording as long as the target mode is stopped.
    """
    svc = get_pipeline_service()
    if svc.is_mode_running(name):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot modify mode '{name}' while it is running. Stop the mode first.",
        )


_local_ips: set[str] | None = None


def _get_local_ips() -> set[str]:
    """Return all IP addresses belonging to this machine (cached)."""
    global _local_ips
    if _local_ips is not None:
        return _local_ips
    ips = {"127.0.0.1", "::1"}
    try:
        ips.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    _local_ips = ips
    logger.info("Local IPs for require_local: %s", ips)
    return ips


def require_local(request: Request) -> None:
    """FastAPI dependency that blocks non-local clients (e.g. for OS file dialogs)."""
    host = request.client.host if request.client else ""
    # Strip IPv4-mapped IPv6 prefix so ::ffff:127.0.0.1 matches 127.0.0.1
    if host.startswith("::ffff:"):
        host = host[7:]
    if host not in _get_local_ips():
        raise HTTPException(403, "Only available on the server machine")


def cleanup_services() -> None:
    """Cleanup all services. Called from app lifespan shutdown."""
    global _pipeline_service, _stream_service, _mode_service, _config_service
    global _preflight_service, _stream_manager_service, _data_service, _ml_service
    if _ml_service:
        _ml_service.cleanup_sync()
    if _pipeline_service:
        _pipeline_service.cleanup()
    if _stream_manager_service:
        _stream_manager_service.stop_all()
    _pipeline_service = None
    _stream_service = None
    _mode_service = None
    _config_service = None
    _preflight_service = None
    _stream_manager_service = None
    _data_service = None
    _ml_service = None
