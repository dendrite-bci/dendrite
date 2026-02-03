"""Shared fixtures for streamer tests."""

import multiprocessing
import socket

import numpy as np
import pytest
from unittest.mock import Mock

from dendrite.data.lsl_helpers import StreamConfig


@pytest.fixture
def mock_lsl_stream_config():
    """Create a mock StreamConfig for LSL testing."""
    config = Mock(spec=StreamConfig)
    config.name = "TestLSLStream"
    config.type = "Predictions"
    config.source_id = "test_lsl_source"
    config.channels = 1
    config.sample_rate = 100.0
    config.channel_format = "string"
    return config


@pytest.fixture
def mock_visualization_stream_config():
    """Create a mock StreamConfig for visualization testing."""
    config = Mock(spec=StreamConfig)
    config.name = "VisualizationStream"
    config.type = "Visualization"
    config.source_id = "viz_source"
    config.channels = 1
    config.sample_rate = 100.0
    config.channel_format = "string"
    return config


@pytest.fixture
def input_queue():
    """Create a multiprocessing Queue for streamer input."""
    queue = multiprocessing.Queue(maxsize=100)
    yield queue
    try:
        while not queue.empty():
            queue.get_nowait()
    except Exception:
        pass


@pytest.fixture
def plot_queue():
    """Create a multiprocessing Queue for plot data."""
    queue = multiprocessing.Queue(maxsize=100)
    yield queue
    try:
        while not queue.empty():
            queue.get_nowait()
    except Exception:
        pass


@pytest.fixture
def output_queue():
    """Create a single queue for mode outputs."""
    q = multiprocessing.Queue(maxsize=50)
    yield q
    try:
        while not q.empty():
            q.get_nowait()
    except Exception:
        pass


@pytest.fixture
def sample_numpy_data():
    """Create sample data with numpy arrays for JSON serialization testing."""
    return {
        'array_2d': np.array([[1, 2], [3, 4]]),
        'array_1d': np.array([1.5, 2.5, 3.5]),
        'scalar': np.float64(42.0),
        'nested': {
            'inner_array': np.array([7, 8, 9]),
            'value': np.int32(123)
        }
    }


@pytest.fixture
def free_port():
    """Find a free port for socket testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port
