"""
Unit tests for dendrite/utils/serialization.py.

Tests jsonify() and write_json() with numpy arrays, datetime,
Pydantic models, and nested structures.
"""

import json
import os
import tempfile
import pytest
import numpy as np
from datetime import datetime
from pydantic import BaseModel

from dendrite.utils.serialization import jsonify, write_json


class SamplePydanticModel(BaseModel):
    """Test Pydantic model for serialization tests."""
    name: str
    value: float


class TestJsonify:
    """Test suite for jsonify() function."""

    def test_numpy_array_1d(self):
        """Test 1D numpy array conversion to list."""
        arr = np.array([1, 2, 3])
        result = jsonify(arr)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_numpy_array_2d(self):
        """Test 2D numpy array conversion to nested list."""
        arr = np.array([[1, 2], [3, 4]])
        result = jsonify(arr)
        assert result == [[1, 2], [3, 4]]

    def test_numpy_scalar_int(self):
        """Test numpy int64 conversion to Python int."""
        val = np.int64(42)
        result = jsonify(val)
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_scalar_float(self):
        """Test numpy float64 conversion to Python float."""
        val = np.float64(3.14)
        result = jsonify(val)
        assert result == 3.14
        assert isinstance(result, float)

    def test_datetime_to_isoformat(self):
        """Test datetime conversion to ISO format string."""
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = jsonify(dt)
        assert result == "2024-01-15T10:30:00"
        assert isinstance(result, str)

    def test_pydantic_model(self):
        """Test Pydantic model conversion via model_dump()."""
        model = SamplePydanticModel(name="test", value=1.5)
        result = jsonify(model)
        assert result == {"name": "test", "value": 1.5}
        assert isinstance(result, dict)

    def test_nested_dict(self):
        """Test nested dictionary with mixed types."""
        data = {
            "array": np.array([1, 2]),
            "timestamp": datetime(2024, 1, 1),
            "nested": {"value": np.float64(3.14)}
        }
        result = jsonify(data)
        assert result == {
            "array": [1, 2],
            "timestamp": "2024-01-01T00:00:00",
            "nested": {"value": 3.14}
        }

    def test_list_with_numpy(self):
        """Test list containing numpy arrays."""
        data = [np.array([1, 2]), np.int64(3)]
        result = jsonify(data)
        assert result == [[1, 2], 3]

    def test_primitives_passthrough(self):
        """Test that primitives pass through unchanged."""
        assert jsonify(42) == 42
        assert jsonify(3.14) == 3.14
        assert jsonify("hello") == "hello"
        assert jsonify(True) is True
        assert jsonify(None) is None

    def test_unknown_type_to_str(self):
        """Test that unknown types fall back to str()."""
        class CustomClass:
            def __str__(self):
                return "custom_repr"

        result = jsonify(CustomClass())
        assert result == "custom_repr"
        assert isinstance(result, str)


class TestWriteJson:
    """Test suite for write_json() function."""

    def test_writes_valid_json(self):
        """Test that write_json creates a valid JSON file."""
        data = {"key": "value", "number": 42}

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        try:
            write_json(data, path)

            with open(path, 'r') as f:
                loaded = json.load(f)

            assert loaded == data
        finally:
            os.unlink(path)

    def test_handles_numpy_data(self):
        """Test that write_json handles numpy arrays."""
        data = {"array": np.array([1, 2, 3]), "scalar": np.float64(2.5)}

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        try:
            write_json(data, path)

            with open(path, 'r') as f:
                loaded = json.load(f)

            assert loaded == {"array": [1, 2, 3], "scalar": 2.5}
        finally:
            os.unlink(path)

    def test_indentation(self):
        """Test that output is indented (human-readable)."""
        data = {"nested": {"key": "value"}}

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        try:
            write_json(data, path)

            with open(path, 'r') as f:
                content = f.read()

            # Should have newlines from indentation
            assert '\n' in content
            assert '  ' in content  # 2-space indent
        finally:
            os.unlink(path)
