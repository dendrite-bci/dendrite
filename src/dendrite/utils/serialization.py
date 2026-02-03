"""
JSON serialization utilities for Dendrite system.

Provides safe JSON encoding for complex types including datetime, numpy arrays,
and Pydantic models.
"""

import json
from datetime import datetime
from typing import Any

import numpy as np


def jsonify(obj: Any) -> dict | list | str | int | float | bool | None:
    """
    Recursively convert object to JSON-serializable format.

    Handles: dict, list, datetime, numpy arrays/scalars, Pydantic models.
    Falls back to str() for unknown types.

    Args:
        obj: Any Python object

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        return {k: jsonify(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [jsonify(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif hasattr(obj, "model_dump") and callable(obj.model_dump):
        return obj.model_dump()
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def write_json(data: dict[str, Any], filepath: str) -> None:
    """
    Write data to JSON file.

    Args:
        data: Dictionary to serialize
        filepath: Path to write JSON file
    """
    serializable = jsonify(data)
    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)
