"""
Configuration REST endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from dendrite.data.streaming.output_schemas import (
    DEFAULT_LSL_CONFIG,
    DEFAULT_ROS2_CONFIG,
    DEFAULT_SOCKET_CONFIG,
    DEFAULT_ZMQ_CONFIG,
    LSLConfig,
    ROS2Config,
    SocketConfig,
    ZMQConfig,
)
from dendrite.web.deps import get_config_service, get_data_service, require_not_recording
from dendrite.web.schemas import (
    GeneralConfigRequest,
    OutputConfigRequest,
    validate_no_path_traversal,
)

router = APIRouter(prefix="/api/config", tags=["config"])

_PROTOCOL_VALIDATORS = {
    "lsl": LSLConfig,
    "socket": SocketConfig,
    "zmq": ZMQConfig,
    "ros2": ROS2Config,
}


@router.get("")
async def get_full_config():
    """Get the full current configuration."""
    service = get_config_service()
    return service.build_configuration()


@router.put("/general")
async def update_general(request: GeneralConfigRequest, _=Depends(require_not_recording)):
    """Update general parameters (study name, subject ID, etc.)."""
    service = get_config_service()
    try:
        service.set_general_config(request.model_dump())
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return service.get_general_config()


@router.get("/general")
async def get_general():
    """Get general parameters."""
    return get_config_service().get_general_config()


@router.put("/output")
async def update_output(request: OutputConfigRequest, _=Depends(require_not_recording)):
    """Update output protocol configuration with validation."""
    errors: dict[str, list[dict]] = {}
    for proto_key, proto_data in request.protocols.items():
        if not proto_data.get("enabled", False):
            continue
        validator = _PROTOCOL_VALIDATORS.get(proto_key)
        if validator and proto_data.get("config"):
            try:
                validator(**proto_data["config"])
            except ValidationError as e:
                errors[proto_key] = [
                    {"field": ".".join(str(loc) for loc in err["loc"]), "msg": err["msg"]}
                    for err in e.errors()
                ]
    if errors:
        raise HTTPException(422, detail={"protocol_errors": errors})
    service = get_config_service()
    service.output = request.protocols
    return {"output": service.output}


@router.get("/output")
async def get_output():
    """Get output protocol configuration."""
    return {"output": get_config_service().output}


@router.get("/output/availability")
async def get_output_availability():
    """Report which output protocols have their dependencies installed."""
    from dendrite.data.streaming import HAS_ROS2, HAS_ZMQ

    return {
        "lsl": True,
        "socket": True,
        "zmq": HAS_ZMQ,
        "ros2": HAS_ROS2,
    }


@router.get("/output/defaults")
async def get_output_defaults():
    """Return default config values for each output protocol."""
    return {
        "lsl": DEFAULT_LSL_CONFIG,
        "socket": DEFAULT_SOCKET_CONFIG,
        "zmq": DEFAULT_ZMQ_CONFIG,
        "ros2": DEFAULT_ROS2_CONFIG,
    }


@router.get("/next-run")
async def get_next_run(subject_id: str, session_id: str, recording_name: str):
    """Get the next auto-incremented run number for this subject/session/recording."""
    svc = get_data_service()
    run = svc.recordings.get_next_run_number(subject_id, session_id, recording_name)
    return {"run_number": run}


@router.get("/list")
async def list_configs():
    """List all saved config files across studies."""
    svc = get_config_service()
    return {"configs": svc.list_configs(), "study_names": svc.list_study_names()}


@router.post("/load")
async def load_config(file_path: str, _=Depends(require_not_recording)):
    """Load configuration from a JSON file."""
    try:
        validate_no_path_traversal(file_path)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    service = get_config_service()
    try:
        result = service.load_configuration(file_path)
        return {"status": "loaded", "file_path": file_path, "warnings": result["warnings"]}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except PermissionError as e:
        raise HTTPException(403, str(e)) from e
    except Exception as e:
        raise HTTPException(400, str(e)) from e


@router.post("/save")
async def save_config(file_path: str | None = None):
    """Save current configuration to a JSON file."""
    if file_path is not None:
        try:
            validate_no_path_traversal(file_path)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
    service = get_config_service()
    saved_path = service.save_configuration(file_path)
    return {"status": "saved", "file_path": saved_path}
