"""
Integration tests for DataSaver graceful shutdown behavior.

Tests that DataSaver properly handles termination signals (SIGTERM, SIGHUP)
and ensures data is flushed to HDF5 files before exiting.
"""

import os
import sys
import time
import json
import signal
import pytest
import numpy as np
import multiprocessing as mp

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dendrite.data.storage.data_saver import DataSaver
from dendrite.data.acquisition import DataRecord


@pytest.mark.integration
def test_datasaver_sigterm_graceful_shutdown(tmp_path):
    """
    Test that DataSaver flushes HDF5 file on SIGTERM.

    Verifies that when DataSaver subprocess receives SIGTERM signal,
    it properly flushes buffered data and closes the HDF5 file cleanly.
    """
    save_queue = mp.Queue(maxsize=1000)
    stop_event = mp.Event()
    out_file = tmp_path / "sigterm_test.h5"

    # Create DataSaver subprocess
    saver = DataSaver(
        filename=str(out_file),
        save_queue=save_queue,
        stop_event=stop_event,
        chunk_size=100  # Large chunk to ensure buffering
    )
    saver.start()

    # Give saver time to initialize (longer for CI environments)
    time.sleep(0.5)

    try:
        # Send EEG metadata
        eeg_metadata = {
            'channel_count': 16,
            'sample_rate': 500,
            'labels': [f'EEG_{i+1}' for i in range(16)],
            'channel_types': ['EEG'] * 16
        }
        metadata_record = DataRecord(
            modality='EEG_Metadata',
            sample=json.dumps(eeg_metadata),
            timestamp=time.time(),
            local_timestamp=time.time()
        )
        save_queue.put(metadata_record)

        # Send some EEG data samples (less than chunk_size to keep them buffered)
        num_samples = 50
        for i in range(num_samples):
            eeg_data = np.random.randn(16).astype(np.float64)
            eeg_record = DataRecord(
                modality='EEG',
                sample=eeg_data,
                timestamp=time.time() + i * 0.002,
                local_timestamp=time.time() + i * 0.002
            )
            save_queue.put(eeg_record)

        # Give saver time to process metadata and start buffering data
        time.sleep(0.3)

        # Send SIGTERM to saver subprocess
        os.kill(saver.pid, signal.SIGTERM)

        # Wait for process to exit gracefully
        saver.join(timeout=5)

        # Verify process exited
        assert not saver.is_alive(), "Saver process should have terminated"

    finally:
        # Cleanup: ensure process is terminated
        if saver.is_alive():
            saver.terminate()
            saver.join(timeout=2)
            if saver.is_alive():
                saver.kill()

    # Verify HDF5 file exists
    assert out_file.exists(), "HDF5 file should exist after SIGTERM"

    # Verify file is readable and contains data
    import h5py
    with h5py.File(out_file, 'r') as f:
        # Check EEG dataset exists
        assert 'EEG' in f, "EEG dataset should exist in file"

        # Check data was flushed (should have some samples, maybe not all due to timing)
        eeg_dataset = f['EEG']
        assert eeg_dataset.shape[0] > 0, "EEG dataset should have at least some samples"

        # Check metadata attributes
        assert 'channel_labels' in eeg_dataset.attrs, "Channel labels should be saved"
        assert 'sampling_frequency' in eeg_dataset.attrs, "Sample rate should be saved"
        assert eeg_dataset.attrs['sampling_frequency'] == 500


@pytest.mark.integration
@pytest.mark.skipif(not hasattr(signal, 'SIGHUP'), reason="SIGHUP not available on Windows")
def test_datasaver_sighup_graceful_shutdown(tmp_path):
    """
    Test that DataSaver flushes HDF5 file on SIGHUP (Unix only).

    SIGHUP is sent when terminal is closed. Verifies data is saved
    even when terminal session ends.
    """
    save_queue = mp.Queue(maxsize=1000)
    stop_event = mp.Event()
    out_file = tmp_path / "sighup_test.h5"

    # Create DataSaver subprocess
    saver = DataSaver(
        filename=str(out_file),
        save_queue=save_queue,
        stop_event=stop_event,
        chunk_size=100
    )
    saver.start()

    # Give saver time to initialize (longer for CI environments)
    time.sleep(0.5)

    try:
        # Send EMG metadata
        emg_metadata = {
            'channel_count': 8,
            'sample_rate': 1000,
            'labels': [f'EMG_{i+1}' for i in range(8)],
            'channel_types': ['EMG'] * 8
        }
        metadata_record = DataRecord(
            modality='EMG_Metadata',
            sample=json.dumps(emg_metadata),
            timestamp=time.time(),
            local_timestamp=time.time()
        )
        save_queue.put(metadata_record)

        # Send EMG data
        num_samples = 30
        for i in range(num_samples):
            emg_data = np.random.randn(8).astype(np.float64)
            emg_record = DataRecord(
                modality='EMG',
                sample=emg_data,
                timestamp=time.time() + i * 0.001,
                local_timestamp=time.time() + i * 0.001
            )
            save_queue.put(emg_record)

        # Give saver time to process
        time.sleep(0.3)

        # Send SIGHUP to saver subprocess
        os.kill(saver.pid, signal.SIGHUP)

        # Wait for process to exit gracefully
        saver.join(timeout=5)

        # Verify process exited
        assert not saver.is_alive(), "Saver process should have terminated"

    finally:
        if saver.is_alive():
            saver.terminate()
            saver.join(timeout=2)
            if saver.is_alive():
                saver.kill()

    # Verify HDF5 file exists and contains data
    assert out_file.exists(), "HDF5 file should exist after SIGHUP"

    import h5py
    with h5py.File(out_file, 'r') as f:
        assert 'EMG' in f, "EMG dataset should exist in file"
        assert f['EMG'].shape[0] > 0, "EMG dataset should have samples"
        assert 'sampling_frequency' in f['EMG'].attrs


@pytest.mark.integration
def test_datasaver_normal_shutdown_vs_signal(tmp_path):
    """
    Compare normal shutdown vs signal-based shutdown.

    Verifies that SIGTERM shutdown produces equivalent results
    to normal stop_event shutdown.
    """
    num_samples = 40

    # Test 1: Normal shutdown
    save_queue_normal = mp.Queue(maxsize=1000)
    stop_event_normal = mp.Event()
    file_normal = tmp_path / "normal_shutdown.h5"

    saver_normal = DataSaver(
        filename=str(file_normal),
        save_queue=save_queue_normal,
        stop_event=stop_event_normal,
        chunk_size=100
    )
    saver_normal.start()
    time.sleep(0.5)

    # Send data
    metadata = {
        'channel_count': 4,
        'sample_rate': 250,
        'labels': [f'CH_{i+1}' for i in range(4)],
        'channel_types': ['EEG'] * 4
    }
    save_queue_normal.put(DataRecord(
        modality='EEG_Metadata',
        sample=json.dumps(metadata),
        timestamp=time.time(),
        local_timestamp=time.time()
    ))

    for i in range(num_samples):
        save_queue_normal.put(DataRecord(
            modality='EEG',
            sample=np.random.randn(4).astype(np.float64),
            timestamp=time.time() + i * 0.004,
            local_timestamp=time.time() + i * 0.004
        ))

    time.sleep(0.3)
    stop_event_normal.set()
    saver_normal.join(timeout=5)

    # Test 2: SIGTERM shutdown
    save_queue_signal = mp.Queue(maxsize=1000)
    stop_event_signal = mp.Event()
    file_signal = tmp_path / "signal_shutdown.h5"

    saver_signal = DataSaver(
        filename=str(file_signal),
        save_queue=save_queue_signal,
        stop_event=stop_event_signal,
        chunk_size=100
    )
    saver_signal.start()
    time.sleep(0.5)

    # Send same data
    save_queue_signal.put(DataRecord(
        modality='EEG_Metadata',
        sample=json.dumps(metadata),
        timestamp=time.time(),
        local_timestamp=time.time()
    ))

    for i in range(num_samples):
        save_queue_signal.put(DataRecord(
            modality='EEG',
            sample=np.random.randn(4).astype(np.float64),
            timestamp=time.time() + i * 0.004,
            local_timestamp=time.time() + i * 0.004
        ))

    time.sleep(0.3)
    os.kill(saver_signal.pid, signal.SIGTERM)
    saver_signal.join(timeout=5)

    # Compare results
    import h5py

    with h5py.File(file_normal, 'r') as f_normal:
        samples_normal = f_normal['EEG'].shape[0]

    with h5py.File(file_signal, 'r') as f_signal:
        samples_signal = f_signal['EEG'].shape[0]

    # Both should have saved data (exact count may vary due to buffering/timing)
    assert samples_normal > 0, "Normal shutdown should save data"
    assert samples_signal > 0, "Signal shutdown should save data"

    # Signal shutdown should save most/all data (might be slightly less due to timing)
    assert samples_signal >= samples_normal * 0.8, \
        f"Signal shutdown should save most data: {samples_signal} vs {samples_normal}"
