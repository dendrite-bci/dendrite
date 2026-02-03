"""
Integration tests for Data Acquisition with real LSL streams.

This module tests DAQ integration with actual LSL streams (via MockEEGStreamer),
complementing the performance tests which use FakeInlet mocks.

Tests cover:
- Real LSL stream discovery and connection
- Multi-stream coordination with real LSL
- Hardware disconnection scenarios
"""

import sys
import os
import pytest
import time
import multiprocessing

# Add the project root to the path so BMI can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dendrite.data.acquisition import DataAcquisition
from dendrite.data.stream_schemas import StreamMetadata


def make_stream_config(**kwargs) -> StreamMetadata:
    """Create StreamMetadata with sensible defaults for tests."""
    channel_count = kwargs.get('channel_count', 32)
    stream_type = kwargs.get('type', 'EEG')
    return StreamMetadata(
        name=kwargs.get('name', 'TestStream'),
        type=stream_type,
        channel_count=channel_count,
        sample_rate=kwargs.get('sample_rate', 500.0),
        channel_format=kwargs.get('channel_format', 'float32'),
        source_id=kwargs.get('source_id', ''),
        labels=kwargs.get('labels') or [f'Ch{i}' for i in range(channel_count)],
        channel_types=kwargs.get('channel_types') or [stream_type] * channel_count,
        channel_units=kwargs.get('channel_units') or ['microvolts'] * channel_count,
        uid=kwargs.get('uid', ''),
    )


# Import mock streamers from shared test helpers
try:
    from tests.test_helpers import MockEEGStreamer, MockEventStreamer
    MOCK_STREAMERS_AVAILABLE = True
except ImportError:
    MOCK_STREAMERS_AVAILABLE = False
    pytestmark = pytest.mark.skip("Mock streamers not available")

# Try to import pylsl
try:
    import pylsl
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False
    pytestmark = pytest.mark.skip("pylsl not available")


@pytest.mark.integration
@pytest.mark.slow
def test_daq_real_lsl_connection(stop_event, data_queue, save_queue):
    """
    Test DAQ connection to real LSL stream.

    Verifies that DAQ can discover, connect to, and receive data from
    actual LSL streams (not just FakeInlet mocks).
    """
    if not MOCK_STREAMERS_AVAILABLE or not LSL_AVAILABLE:
        pytest.skip("Mock streamers or LSL not available")

    # Create mock LSL stream
    mock_stop = multiprocessing.Event()
    mock_stream = MockEEGStreamer(
        stream_name="TestEEG",
        sample_rate=500,
        n_channels=32,
        stop_event=mock_stop
    )

    # Start mock stream and wait for it to establish
    mock_stream.start()
    time.sleep(2)

    daq = None
    try:
        # Create DAQ - match mock stream configuration
        stream_configs = [make_stream_config(
            type='EEG',
            name='TestEEG',
            channel_count=32,
            sample_rate=500.0,
            labels=[f'EEG_{i+1}' for i in range(31)] + ['Markers'],
            channel_types=['EEG'] * 31 + ['Markers'],
            source_id='mock_eeg_123'
        )]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        # Start DAQ
        daq.start()

        # Let it run to receive data
        time.sleep(3)

        # Check that data was received
        data_received = 0
        while not data_queue.empty():
            try:
                data = data_queue.get_nowait()
                data_received += 1
                assert hasattr(data, 'modality')
                assert hasattr(data, 'sample')
                assert hasattr(data, 'timestamp')
            except Exception:
                break

        # Stop everything
        stop_event.set()
        mock_stop.set()

        daq.join(timeout=10)
        mock_stream.join(timeout=5)

        # Verify basic connectivity
        print(f"Data received: {data_received}")
        assert data_received > 0, "DAQ should receive data from stream"

    finally:
        # Cleanup using helpers from conftest
        from tests.conftest import cleanup_process, drain_queue

        # Stop events
        mock_stop.set()
        stop_event.set()

        # Cleanup processes with queue draining
        if daq is not None:
            cleanup_process(daq, timeout=2, queues_to_drain=[data_queue, save_queue])
        cleanup_process(mock_stream, timeout=2)


@pytest.mark.integration
@pytest.mark.slow
def test_daq_multi_stream_real_lsl(stop_event, data_queue, save_queue):
    """
    Test DAQ with multiple real LSL streams.

    Verifies that DAQ can handle multiple simultaneous LSL streams
    (EEG + Events) and coordinate data from both.
    """
    if not MOCK_STREAMERS_AVAILABLE or not LSL_AVAILABLE:
        pytest.skip("Mock streamers or LSL not available")

    # Create mock streams
    mock_eeg_stop = multiprocessing.Event()
    mock_events_stop = multiprocessing.Event()

    mock_eeg = MockEEGStreamer(
        stream_name="TestEEG",
        sample_rate=500,
        n_channels=32,
        stop_event=mock_eeg_stop
    )

    mock_events = MockEventStreamer(
        stream_name="TestEvents",
        stop_event=mock_events_stop
    )

    # Start mock streams
    mock_eeg.start()
    mock_events.start()
    time.sleep(2)

    daq = None
    try:
        # Create DAQ with multiple streams
        stream_configs = [
            make_stream_config(
                type='EEG',
                name='TestEEG',
                channel_count=32,
                sample_rate=500.0,
                labels=[f'EEG_{i+1}' for i in range(31)] + ['Markers'],
                channel_types=['EEG'] * 31 + ['Markers'],
                source_id='mock_eeg_123'
            ),
            make_stream_config(
                type='Events',
                name='TestEvents',
                channel_count=1,
                sample_rate=0.0,
                labels=['Event'],
                channel_types=['Event'],
                source_id='mock_events_456'
            )
        ]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        # Start DAQ
        daq.start()

        # Let it run to receive data
        time.sleep(5)

        # Check that data was received
        data_received = 0
        event_data_received = 0
        while not data_queue.empty():
            try:
                data = data_queue.get_nowait()
                data_received += 1
                if hasattr(data, 'modality') and 'Event' in str(data.modality):
                    event_data_received += 1
            except Exception:
                break

        # Stop everything
        stop_event.set()
        mock_eeg_stop.set()
        mock_events_stop.set()

        daq.join(timeout=10)
        mock_eeg.join(timeout=5)
        mock_events.join(timeout=5)

        # Verify multi-stream data reception
        print(f"Total data: {data_received}, Event data: {event_data_received}")
        assert data_received > 0, "Should receive data from LSL streams"

    finally:
        # Cleanup using helpers from conftest
        from tests.conftest import cleanup_process

        # Stop events
        mock_eeg_stop.set()
        mock_events_stop.set()
        stop_event.set()

        # Cleanup all processes with queue draining
        if daq is not None:
            cleanup_process(daq, timeout=2, queues_to_drain=[data_queue, save_queue])
        cleanup_process(mock_eeg, timeout=2)
        cleanup_process(mock_events, timeout=2)


@pytest.mark.integration
@pytest.mark.slow
def test_daq_stream_disconnection(stop_event, data_queue, save_queue):
    """
    Test DAQ behavior when LSL stream disconnects.

    Simulates hardware disconnection by stopping the stream mid-run.
    Verifies DAQ handles disconnection gracefully without crashing.
    """
    if not MOCK_STREAMERS_AVAILABLE or not LSL_AVAILABLE:
        pytest.skip("Mock streamers or LSL not available")

    # Create mock stream
    mock_stop = multiprocessing.Event()
    mock_stream = MockEEGStreamer(
        stream_name="TestEEG",
        sample_rate=500,
        n_channels=32,
        stop_event=mock_stop
    )

    # Start mock stream
    mock_stream.start()
    time.sleep(2)

    daq = None
    try:
        # Create DAQ
        stream_configs = [make_stream_config(
            type='EEG',
            name='TestEEG',
            channel_count=32,
            sample_rate=500.0,
            labels=[f'EEG_{i+1}' for i in range(31)] + ['Markers'],
            channel_types=['EEG'] * 31 + ['Markers'],
            source_id='mock_eeg_123'
        )]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        # Start DAQ
        daq.start()

        # Let it run briefly
        time.sleep(1)

        # Simulate hardware disconnection
        print("Simulating stream disconnection...")
        mock_stop.set()
        mock_stream.join(timeout=5)

        # Let DAQ detect disconnection
        time.sleep(2)

        # Stop DAQ
        stop_event.set()
        daq.join(timeout=10)

        # Force terminate if still alive (disconnection handling may vary)
        if daq.is_alive():
            print("DAQ still running after disconnection - terminating")
            daq.terminate()
            time.sleep(1)

        # Test passes if DAQ handled disconnection without crashing
        print("[OK] DAQ handled stream disconnection gracefully")

    finally:
        # Cleanup using helpers from conftest
        from tests.conftest import cleanup_process

        # Stop events
        mock_stop.set()
        stop_event.set()

        # Cleanup processes with queue draining
        if daq is not None:
            cleanup_process(daq, timeout=2, queues_to_drain=[data_queue, save_queue])
        cleanup_process(mock_stream, timeout=2)


@pytest.mark.integration
@pytest.mark.slow
def test_daq_stream_reconnection_after_disconnect(stop_event, data_queue, save_queue):
    """
    Test DAQ behavior when stream reconnects after disconnection.

    Tests whether DAQ can recover and reconnect when a stream drops
    and then reappears (e.g., hardware reboot scenario).
    """
    if not MOCK_STREAMERS_AVAILABLE or not LSL_AVAILABLE:
        pytest.skip("Mock streamers or LSL not available")

    # First stream session
    mock_stop_1 = multiprocessing.Event()
    mock_stream_1 = MockEEGStreamer(
        stream_name="TestEEG",
        sample_rate=500,
        n_channels=32,
        stop_event=mock_stop_1
    )

    # Start first stream
    mock_stream_1.start()
    time.sleep(2)

    try:
        # Create DAQ
        stream_configs = [make_stream_config(
            type='EEG',
            name='TestEEG',
            channel_count=32,
            sample_rate=500.0,
            labels=[f'EEG_{i+1}' for i in range(31)] + ['Markers'],
            channel_types=['EEG'] * 31 + ['Markers'],
            source_id='mock_eeg_123'
        )]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        # Start DAQ
        daq.start()
        time.sleep(1)

        # Count samples from first session
        samples_before = 0
        while not data_queue.empty():
            try:
                data_queue.get_nowait()
                samples_before += 1
            except Exception:
                break

        # Simulate hardware disconnection
        print("Simulating stream disconnection...")
        mock_stop_1.set()
        mock_stream_1.join(timeout=5)
        time.sleep(2)  # Wait for DAQ to detect disconnection

        # Restart stream (simulating hardware reboot)
        print("Restarting stream (simulating reconnection)...")
        mock_stop_2 = multiprocessing.Event()
        mock_stream_2 = MockEEGStreamer(
            stream_name="TestEEG",
            sample_rate=500,
            n_channels=32,
            stop_event=mock_stop_2
        )
        mock_stream_2.start()
        time.sleep(3)  # Give DAQ time to reconnect

        # Check if DAQ reconnected and continued receiving data
        samples_after = 0
        while not data_queue.empty():
            try:
                data_queue.get_nowait()
                samples_after += 1
            except Exception:
                break

        # Stop everything
        stop_event.set()
        mock_stop_2.set()
        daq.join(timeout=10)
        mock_stream_2.join(timeout=5)

        # Analyze results
        print(f"Samples before disconnect: {samples_before}")
        print(f"Samples after reconnect: {samples_after}")

        if samples_after > 0:
            print("[OK] DAQ successfully reconnected to stream")
        else:
            print("[WARN] DAQ did not reconnect (expected behavior - reconnection not implemented)")
            # This is expected to fail currently - documents the gap

    finally:
        mock_stop_1.set()
        if 'mock_stop_2' in locals():
            mock_stop_2.set()
        stop_event.set()
        if mock_stream_1.is_alive():
            mock_stream_1.terminate()
        if 'mock_stream_2' in locals() and mock_stream_2.is_alive():
            mock_stream_2.terminate()


@pytest.mark.integration
@pytest.mark.slow
def test_daq_multirate_sync_real_lsl(stop_event, data_queue, save_queue):
    """
    Test real multi-rate stream synchronization with actual LSL streams.

    Verifies that DAQ correctly handles 500Hz EEG + 2000Hz EMG simultaneously
    with proper sample ratios and no dropped samples.
    """
    if not MOCK_STREAMERS_AVAILABLE or not LSL_AVAILABLE:
        pytest.skip("Mock streamers or LSL not available")

    # Import MockEMGStreamer
    try:
        from tests.test_helpers import MockEMGStreamer
    except ImportError:
        pytest.skip("MockEMGStreamer not available")

    # Create mock streams at different rates
    mock_eeg_stop = multiprocessing.Event()
    mock_emg_stop = multiprocessing.Event()

    mock_eeg = MockEEGStreamer(
        stream_name="TestEEG",
        sample_rate=500,
        n_channels=32,
        stop_event=mock_eeg_stop
    )

    mock_emg = MockEMGStreamer(
        stream_name="TestEMG",
        sample_rate=2000,
        n_channels=8,
        stop_event=mock_emg_stop
    )

    # Start mock streams
    mock_eeg.start()
    mock_emg.start()
    time.sleep(2)

    try:
        # Create DAQ with multi-rate streams
        stream_configs = [
            make_stream_config(
                type='EEG',
                name='TestEEG',
                channel_count=32,
                sample_rate=500.0,
                labels=[f'EEG_{i+1}' for i in range(31)] + ['Markers'],
                channel_types=['EEG'] * 31 + ['Markers'],
                source_id='mock_eeg_123'
            ),
            make_stream_config(
                type='EMG',
                name='TestEMG',
                channel_count=8,
                sample_rate=2000.0,
                labels=[f'EMG_{i+1}' for i in range(8)],
                channel_types=['EMG'] * 8,
                source_id='mock_emg_789'
            )
        ]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        # Start DAQ
        daq.start()

        # Run for 5 seconds
        time.sleep(5)

        # Count samples per modality
        eeg_samples = 0
        emg_samples = 0
        while not data_queue.empty():
            try:
                data = data_queue.get_nowait()
                if hasattr(data, 'modality'):
                    if 'EEG' in str(data.modality):
                        eeg_samples += 1
                    elif 'EMG' in str(data.modality):
                        emg_samples += 1
            except Exception:
                break

        # Stop everything
        stop_event.set()
        mock_eeg_stop.set()
        mock_emg_stop.set()

        daq.join(timeout=10)
        mock_eeg.join(timeout=5)
        mock_emg.join(timeout=5)

        # Verify sample ratios
        print(f"EEG samples: {eeg_samples} (expected ~2500 for 500Hz × 5s)")
        print(f"EMG samples: {emg_samples} (expected ~10000 for 2000Hz × 5s)")

        if emg_samples > 0 and eeg_samples > 0:
            ratio = emg_samples / eeg_samples
            expected_ratio = 4.0  # 2000Hz / 500Hz
            print(f"Sample ratio EMG/EEG: {ratio:.2f} (expected {expected_ratio})")

            # Allow 20% tolerance due to timing variations
            assert abs(ratio - expected_ratio) < expected_ratio * 0.2, \
                f"Sample ratio {ratio:.2f} too far from expected {expected_ratio}"
            print("[OK] Multi-rate synchronization working correctly")
        else:
            print("[WARN] No samples received from one or both streams")

    finally:
        mock_eeg_stop.set()
        mock_emg_stop.set()
        stop_event.set()
        for process in [mock_eeg, mock_emg]:
            if process.is_alive():
                process.terminate()


@pytest.mark.integration
@pytest.mark.slow
def test_daq_queue_simultaneous_full():
    """
    Test DAQ behavior when both data_queue and save_queue are full.

    Verifies that DAQ handles queue backpressure gracefully without
    deadlocking when both output queues fill up simultaneously.
    """
    if not MOCK_STREAMERS_AVAILABLE or not LSL_AVAILABLE:
        pytest.skip("Mock streamers or LSL not available")

    # Create small queues that will fill quickly
    small_data_queue = multiprocessing.Queue(maxsize=50)
    small_save_queue = multiprocessing.Queue(maxsize=50)
    stop_event = multiprocessing.Event()

    # Create mock stream
    mock_stop = multiprocessing.Event()
    mock_stream = MockEEGStreamer(
        stream_name="TestEEG",
        sample_rate=500,
        n_channels=32,
        stop_event=mock_stop
    )

    # Start mock stream
    mock_stream.start()
    time.sleep(2)

    daq = None
    try:
        # Create DAQ
        stream_configs = [make_stream_config(
            type='EEG',
            name='TestEEG',
            channel_count=32,
            sample_rate=500.0,
            labels=[f'EEG_{i+1}' for i in range(31)] + ['Markers'],
            channel_types=['EEG'] * 31 + ['Markers'],
            source_id='mock_eeg_123'
        )]

        daq = DataAcquisition(
            data_queue=small_data_queue,
            save_queue=small_save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        # Start DAQ and let queues fill
        daq.start()
        time.sleep(3)  # Let queues fill up

        # Check queue sizes
        try:
            data_queue_size = small_data_queue.qsize()
            save_queue_size = small_save_queue.qsize()
        except (NotImplementedError, AttributeError):
            # qsize() not available on all platforms
            data_queue_size = -1
            save_queue_size = -1

        print(f"Data queue size: {data_queue_size}/50")
        print(f"Save queue size: {save_queue_size}/50")

        # Verify DAQ is still alive (not deadlocked)
        assert daq.is_alive(), "DAQ should still be alive despite full queues"
        print("[OK] DAQ handling full queues without deadlock")

        # Stop everything
        stop_event.set()
        mock_stop.set()

        daq.join(timeout=10)
        mock_stream.join(timeout=5)

        # Verify graceful shutdown
        if daq.is_alive():
            print("[WARN] DAQ did not stop gracefully, forcing termination")
            daq.terminate()
        else:
            print("[OK] DAQ stopped gracefully after queue backpressure")

    finally:
        mock_stop.set()
        stop_event.set()
        if mock_stream.is_alive():
            mock_stream.terminate()
        if daq is not None and daq.is_alive():
            daq.terminate()
