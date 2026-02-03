"""
Pytest configuration and fixtures for the BMI test suite.

This file provides common fixtures used across all test modules.
"""

import pytest
import tempfile
import os
import multiprocessing
import time
import shutil
from pathlib import Path

from dendrite.data.stream_schemas import StreamMetadata


def drain_queue(queue, timeout=0.1):
    """
    Drain all items from a multiprocessing queue without blocking.

    Prevents queue feeder thread hangs by consuming all pending items.
    """
    try:
        while True:
            queue.get_nowait()
    except Exception:
        pass  # Queue empty or closed


def cleanup_process(process, timeout=2, queues_to_drain=None):
    """
    Properly cleanup a multiprocessing.Process and associated queues.

    Args:
        process: The multiprocessing.Process to cleanup
        timeout: Seconds to wait for graceful shutdown
        queues_to_drain: Optional list of queues to drain before terminating
    """
    if process and process.is_alive():
        # Try graceful shutdown first
        if hasattr(process, 'stop_event'):
            process.stop_event.set()

        # Wait for graceful shutdown
        process.join(timeout=timeout)

        # Drain queues to allow feeder threads to exit
        if queues_to_drain:
            for queue in queues_to_drain:
                if queue:
                    drain_queue(queue)
                    try:
                        queue.close()
                        queue.cancel_join_thread()  # Don't wait for feeder thread
                    except Exception:
                        pass

        # Force terminate if still alive
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)

        # Final check
        if process.is_alive():
            try:
                process.kill()
            except Exception:
                pass


@pytest.fixture
def stop_event():
    """Create a multiprocessing Event for stopping processes."""
    event = multiprocessing.Event()
    yield event
    # Ensure event is set after test to prevent hanging processes
    event.set()


@pytest.fixture
def data_queue():
    """Create a multiprocessing Queue for data transmission."""
    queue = multiprocessing.Queue(maxsize=1000)
    yield queue
    # Clean up queue after test to prevent hanging
    drain_queue(queue)
    try:
        queue.close()
        queue.cancel_join_thread()  # Prevent hang waiting for feeder thread
    except Exception:
        pass


@pytest.fixture
def save_queue():
    """Create a multiprocessing Queue for save operations."""
    queue = multiprocessing.Queue(maxsize=1000)
    yield queue
    # Clean up queue after test to prevent hanging
    drain_queue(queue)
    try:
        queue.close()
        queue.cancel_join_thread()  # Prevent hang waiting for feeder thread
    except Exception:
        pass


@pytest.fixture
def shared_state():
    """Create a SharedState for cross-process communication."""
    from dendrite.utils import SharedState
    state = SharedState()
    yield state
    state.cleanup()


@pytest.fixture
def temp_h5_file():
    """Create a temporary HDF5 file path."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup with retry
    max_retries = 5
    for i in range(max_retries):
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            break
        except PermissionError:
            if i < max_retries - 1:
                time.sleep(0.1)  # Wait a bit and retry
            else:
                pass  # Give up silently


@pytest.fixture
def temp_directory():
    """Create a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_stream_config():
    """Create a mock stream configuration for testing."""
    return StreamMetadata(
        type='EEG',
        name='TestEEG',
        channel_count=32,
        sample_rate=500.0,
        labels=[f'EEG_{i+1}' for i in range(31)] + ['Markers'],
        channel_types=['EEG'] * 31 + ['Markers'],
        channel_units=['microvolts'] * 31 + ['integer'],
        source_id='mock_eeg_123',
        uid='mock_eeg_uid_123'
    )


@pytest.fixture
def mock_emg_stream_config():
    """Create a mock EMG stream configuration for testing."""
    return StreamMetadata(
        type='EMG',
        name='TestEMG',
        channel_count=8,
        sample_rate=500.0,
        labels=[f'EMG_{i+1}' for i in range(8)],
        channel_types=['EMG'] * 8,
        channel_units=['microvolts'] * 8,
        source_id='mock_emg_456',
        uid='mock_emg_uid_456'
    )


@pytest.fixture
def mock_events_stream_config():
    """Create a mock Events stream configuration for testing."""
    return StreamMetadata(
        type='Events',
        name='TestEvents',
        channel_count=1,
        sample_rate=0,  # Irregular rate
        labels=['Event'],
        channel_types=['Event'],
        channel_units=['string'],
        source_id='mock_events_789',
        uid='mock_events_uid_789'
    )


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return {
        'sample_rate': 500,
        'channel_count': 32,
        'recording_name': 'test_recording',
        'study_name': 'test_study',
        'session_name': 'test_session',
        'timestamp': time.time()
    }


@pytest.fixture
def sample_eeg_data():
    """Create sample EEG data for testing."""
    import numpy as np
    return np.random.randn(32).astype(np.float64)


@pytest.fixture
def sample_emg_data():
    """Create sample EMG data for testing."""
    import numpy as np
    return np.random.randn(8).astype(np.float64)


@pytest.fixture
def sample_event_data():
    """Create sample event data for testing."""
    return {
        'event_id': 100,
        'event_type': 'TestEvent',
        'extra_vars': '{"param": "value"}'
    }


@pytest.fixture
def basic_decoder_config():
    """Create basic decoder configuration for mode testing."""
    return {
        'model_config': {
            'model_type': 'EEGNet',
            'num_classes': 2,
            'learning_rate': 0.001,
            'buffer_size': 50
        },
        'forgetting_factor': 0.95,
        'window_size': 30
    }


@pytest.fixture
def event_mapping():
    """Create basic event mapping for mode testing."""
    return {
        1: 'left_hand',
        2: 'right_hand',
        3: 'rest',
        4: 'feet'
    }


@pytest.fixture
def channel_selection():
    """Create basic channel selection for mode testing."""
    return {
        'EEG': [0, 1, 2, 3, 4, 5, 6, 7],  # First 8 EEG channels
        'EMG': [0, 1, 2, 3]                # First 4 EMG channels
    }


@pytest.fixture
def synchronous_mode_config(basic_decoder_config, event_mapping, channel_selection):
    """Create complete SynchronousMode configuration."""
    return {
        'decoder_config': basic_decoder_config,
        'event_mapping': event_mapping,
        'channel_selection': channel_selection,
        'start_offset': -1.0,
        'end_offset': 2.0,
        'training_interval': 10,
        'study_name': 'test_study',
        'file_identifier': 'test_session'
    }


@pytest.fixture
def mock_decoder():
    """Create a mock decoder for mode testing."""
    from unittest.mock import Mock
    
    decoder = Mock()
    decoder.fit = Mock()
    decoder.predict_sample = Mock(return_value=(0, 0.7))
    decoder.save_model = Mock()
    decoder.load_model = Mock(return_value=True)
    decoder.is_fitted = True
    return decoder


@pytest.fixture
def mock_dataset():
    """Create a mock BMI dataset for mode testing."""
    from unittest.mock import Mock
    import numpy as np
    
    dataset = Mock()
    dataset.add_sample = Mock()
    dataset.to_decoder_format = Mock(return_value={
        'X': [np.random.randn(100, 8), np.random.randn(100, 8)],
        'y': [0, 1]
    })
    dataset.samples = []
    return dataset


@pytest.fixture(autouse=True)
def cleanup_processes():
    """Automatically cleanup any remaining processes after each test."""
    yield

    # Clean up any remaining multiprocessing resources
    try:
        for process in multiprocessing.active_children():
            if process.is_alive():
                cleanup_process(process)
    except Exception:
        pass  # Ignore cleanup errors
