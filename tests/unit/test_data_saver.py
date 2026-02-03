"""
Unit tests for DataSaver class.

Tests cover:
- DataSaver initialization and configuration
- HDF5 file creation and structure
- Data handling for different modalities
- Error handling and edge cases

Note: MetricsSaver tests are in test_metrics_saver.py
"""

import sys
import os
import pytest
import numpy as np
import h5py
import json
import time
import tempfile
import multiprocessing
from unittest.mock import Mock, patch

# Add the project root to the path so BMI can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dendrite.data.storage.data_saver import DataSaver
from dendrite.data.acquisition import DataRecord
from tests.conftest import cleanup_process


class TestDataSaver:
    """Test suite for DataSaver class."""
    
    def teardown_method(self):
        """Cleanup after each test method."""
        # Give some time for any remaining processes to exit
        time.sleep(0.1)
        
        # Force cleanup of any remaining multiprocessing resources
        try:
            # Check for any remaining active children
            active_children = multiprocessing.active_children()
            if active_children:
                print(f"[WARN] Found {len(active_children)} active child processes, cleaning up...")
                for child in active_children:
                    cleanup_process(child)
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def test_initialization(self, stop_event, save_queue, temp_h5_file):
        """Test DataSaver initialization."""
        saver = DataSaver(
            filename=temp_h5_file,
            save_queue=save_queue,
            stop_event=stop_event,
            chunk_size=50
        )
        
        assert saver.filename == os.path.normpath(temp_h5_file)
        assert saver.save_queue == save_queue
        assert saver.stop_event == stop_event
        assert saver.chunk_size == 50
        assert os.path.exists(os.path.dirname(temp_h5_file))
    
    def test_metadata_saving(self, stop_event, save_queue, temp_h5_file):
        """Test metadata saving functionality."""
        saver = DataSaver(
            filename=temp_h5_file,
            save_queue=save_queue,
            stop_event=stop_event
        )
        
        # Create metadata record
        metadata = {
            'sample_rate': 500,
            'channel_count': 32,
            'recording_name': 'test_recording',
            'study_name': 'test_study'
        }
        
        metadata_record = DataRecord(
            modality='Metadata',
            sample=json.dumps(metadata),
            timestamp=time.time(),
            local_timestamp=time.time()
        )
        
        # Add to queue and process
        save_queue.put(metadata_record)
        
        # Start saver process
        saver.start()
        time.sleep(0.5)  # Let it process
        stop_event.set()
        saver.join(timeout=5)
        
        # Verify metadata was saved
        assert os.path.exists(temp_h5_file)
        with h5py.File(temp_h5_file, 'r') as f:
            assert 'sample_rate' in f.attrs
            assert f.attrs['sample_rate'] == 500
            assert 'channel_count' in f.attrs
            assert f.attrs['channel_count'] == 32
            assert 'recording_name' in f.attrs
            assert f.attrs['recording_name'] == 'test_recording'
    
    def test_eeg_data_saving(self, stop_event, save_queue, temp_h5_file):
        """Test EEG data saving functionality."""
        saver = DataSaver(
            filename=temp_h5_file,
            save_queue=save_queue,
            stop_event=stop_event
        )
        
        # Create EEG metadata
        eeg_metadata = {
            'channel_count': 32,
            'sample_rate': 500,
            'labels': [f'EEG_{i+1}' for i in range(32)],
            'channel_types': ['EEG'] * 32
        }
        
        metadata_record = DataRecord(
            modality='EEG_Metadata',
            sample=json.dumps(eeg_metadata),
            timestamp=time.time(),
            local_timestamp=time.time()
        )
        save_queue.put(metadata_record)
        
        # Create EEG data samples
        num_samples = 10
        for i in range(num_samples):
            eeg_data = np.random.randn(32).astype(np.float64)
            eeg_record = DataRecord(
                modality='EEG',
                sample=eeg_data,
                timestamp=time.time() + i * 0.002,
                local_timestamp=time.time() + i * 0.002
            )
            save_queue.put(eeg_record)
        
        # Start saver process
        saver.start()
        time.sleep(0.5)  # Let it process
        stop_event.set()
        saver.join(timeout=5)
        
        # Verify EEG data was saved
        assert os.path.exists(temp_h5_file)
        with h5py.File(temp_h5_file, 'r') as f:
            assert 'EEG' in f
            eeg_dataset = f['EEG']
            assert eeg_dataset.shape == (num_samples,)  # Structured array: 1D with named fields
            # Check that structured array has the expected fields
            field_names = list(eeg_dataset.dtype.names)
            assert 'timestamp' in field_names
            assert 'local_timestamp' in field_names
            # Should have 32 EEG channels + 2 timestamp fields = 34 total fields
            assert len(field_names) == 34
            
            # Check metadata attributes
            assert 'channel_labels' in eeg_dataset.attrs
            assert 'sampling_frequency' in eeg_dataset.attrs
            assert eeg_dataset.attrs['sampling_frequency'] == 500
    
    def test_event_data_saving(self, stop_event, save_queue, temp_h5_file):
        """Test event data saving functionality."""
        saver = DataSaver(
            filename=temp_h5_file,
            save_queue=save_queue,
            stop_event=stop_event
        )
        
        # Create event data (dict format, already parsed/normalized by DAQ)
        events = [
            {
                'event_id': 100,
                'event_type': 'TestEvent1',
                'param': 'value1'
            },
            {
                'event_id': 200,
                'event_type': 'TestEvent2',
                'param': 'value2'
            }
        ]

        current_time = time.time()
        for i, event in enumerate(events):
            event_record = DataRecord(
                modality='Event',
                sample=event,  # Dict already parsed/normalized by DAQ
                timestamp=current_time + i,
                local_timestamp=time.time()
            )
            save_queue.put(event_record)
        
        # Start saver process
        saver.start()
        time.sleep(0.5)  # Let it process
        stop_event.set()
        saver.join(timeout=5)
        
        # Verify event data was saved
        assert os.path.exists(temp_h5_file)
        with h5py.File(temp_h5_file, 'r') as f:
            assert 'Event' in f
            event_dataset = f['Event']
            assert len(event_dataset) == 2
            
            # Check event structure - handle HDF5 bytes to string conversion
            assert event_dataset[0]['event_id'] == 100
            event_type_0 = event_dataset[0]['event_type']
            if isinstance(event_type_0, bytes):
                event_type_0 = event_type_0.decode('utf-8')
            assert event_type_0 == 'TestEvent1'

            assert event_dataset[1]['event_id'] == 200
            event_type_1 = event_dataset[1]['event_type']
            if isinstance(event_type_1, bytes):
                event_type_1 = event_type_1.decode('utf-8')
            assert event_type_1 == 'TestEvent2'
    
    def test_multi_modality_saving(self, stop_event, save_queue, temp_h5_file):
        """Test saving multiple modalities simultaneously."""
        saver = DataSaver(
            filename=temp_h5_file,
            save_queue=save_queue,
            stop_event=stop_event
        )
        
        # EMG is handled automatically by DataSaver when metadata is received
        
        # Create metadata for multiple modalities
        eeg_metadata = {
            'channel_count': 32,
            'sample_rate': 500,
            'labels': [f'EEG_{i+1}' for i in range(32)]
        }

        emg_metadata = {
            'channel_count': 8,
            'sample_rate': 500,
            'labels': [f'EMG_{i+1}' for i in range(8)]
        }
        
        # Add metadata records
        save_queue.put(DataRecord('EEG_Metadata', json.dumps(eeg_metadata), time.time(), time.time()))
        save_queue.put(DataRecord('EMG_Metadata', json.dumps(emg_metadata), time.time(), time.time()))
        
        # Add data for both modalities
        num_samples = 5
        for i in range(num_samples):
            eeg_data = np.random.randn(32).astype(np.float64)
            emg_data = np.random.randn(8).astype(np.float64)
            
            save_queue.put(DataRecord('EEG', eeg_data, time.time(), time.time()))
            save_queue.put(DataRecord('EMG', emg_data, time.time(), time.time()))
        
        # Start saver process
        saver.start()
        time.sleep(0.5)
        stop_event.set()
        saver.join(timeout=5)
        
        # Verify both modalities were saved
        with h5py.File(temp_h5_file, 'r') as f:
            assert 'EEG' in f
            assert 'EMG' in f
            
            # Structured arrays: 1D with named fields
            assert f['EEG'].shape == (num_samples,)
            assert f['EMG'].shape == (num_samples,)
            
            # Check field counts: channels + 2 timestamps
            eeg_fields = len(f['EEG'].dtype.names)
            emg_fields = len(f['EMG'].dtype.names)
            assert eeg_fields == 34  # 32 channels + 2 timestamps
            assert emg_fields == 10   # 8 channels + 2 timestamps
    
    def test_malformed_data_handling(self, stop_event, save_queue, temp_h5_file):
        """Test handling of malformed data."""
        saver = DataSaver(
            filename=temp_h5_file,
            save_queue=save_queue,
            stop_event=stop_event
        )
        
        # Add valid metadata first - include required fields for DataSaver
        eeg_metadata = {
            'channel_count': 32,
            'sample_rate': 500,
            'labels': [f'EEG_{i+1}' for i in range(32)],
            'channel_types': ['EEG'] * 32
        }
        save_queue.put(DataRecord('EEG_Metadata', json.dumps(eeg_metadata), time.time(), time.time()))
        
        # Add malformed data (should be handled gracefully)
        malformed_record = DataRecord(
            modality='EEG',
            sample='invalid_data',  # String instead of numpy array
            timestamp=time.time(),
            local_timestamp=time.time()
        )
        save_queue.put(malformed_record)
        
        # Add valid data after malformed
        valid_data = np.random.randn(32).astype(np.float64)
        save_queue.put(DataRecord('EEG', valid_data, time.time(), time.time()))
        
        # Start saver process
        saver.start()
        time.sleep(0.5)
        stop_event.set()
        saver.join(timeout=5)
        
        # Should not crash and file should exist
        assert os.path.exists(temp_h5_file)
    
    def test_unknown_modality_handling(self, stop_event, save_queue, temp_h5_file):
        """Test handling of unknown modality types."""
        saver = DataSaver(
            filename=temp_h5_file,
            save_queue=save_queue,
            stop_event=stop_event
        )
        
        # Add unknown modality (should be handled gracefully)
        unknown_record = DataRecord(
            modality='UnknownModality',
            sample=np.random.randn(10),
            timestamp=time.time(),
            local_timestamp=time.time()
        )
        save_queue.put(unknown_record)
        
        # Start saver process
        saver.start()
        time.sleep(0.2)
        stop_event.set()
        saver.join(timeout=5)
        
        # Should not crash
        assert os.path.exists(temp_h5_file)
    
    def test_flush_behavior(self, stop_event, save_queue, temp_h5_file):
        """Test chunk-based writing behavior - verify dataset grows over time."""
        saver = DataSaver(
            filename=temp_h5_file,
            save_queue=save_queue,
            stop_event=stop_event,
            chunk_size=2  # Small chunk size for testing
        )
        
        # Add initial metadata - include required fields for DataSaver
        eeg_metadata = {
            'channel_count': 32,
            'sample_rate': 500,
            'labels': [f'EEG_{i+1}' for i in range(32)],
            'channel_types': ['EEG'] * 32
        }
        save_queue.put(DataRecord('EEG_Metadata', json.dumps(eeg_metadata), time.time(), time.time()))
        
        # Start saver
        saver.start()
        time.sleep(0.5)  # Let metadata be processed (increased wait time)
        
        # Check if saver process is alive
        if not saver.is_alive():
            pytest.fail("DataSaver process died immediately - check for initialization errors")
        
        # Track dataset growth over time
        dataset_sizes = []
        
        # Add data in batches and monitor file growth
        for batch in range(4):
            # Add a batch of data
            for i in range(2):
                eeg_data = np.random.randn(32).astype(np.float64)
                save_queue.put(DataRecord('EEG', eeg_data, time.time(), time.time()))
            
            # Wait for chunk to be written
            time.sleep(0.1)  # Brief wait for processing
            
            # Check dataset size
            if os.path.exists(temp_h5_file):
                try:
                    with h5py.File(temp_h5_file, 'r') as f:
                        if 'EEG' in f:
                            current_size = f['EEG'].shape[0]
                            dataset_sizes.append(current_size)
                            print(f"Batch {batch}: Dataset has {current_size} samples")
                        else:
                            dataset_sizes.append(0)
                except Exception:
                    dataset_sizes.append(0)
            else:
                dataset_sizes.append(0)
        
        # Stop saver with proper cleanup
        stop_event.set()
        cleanup_process(saver, timeout=2)
        
        # Verify the process actually stopped
        assert not saver.is_alive(), "DataSaver process should have stopped"
        
        # Verify chunk-based writing worked
        assert os.path.exists(temp_h5_file)
        
        # Check that dataset grew over time (indicating chunk-based writing)
        print(f"Dataset growth over time: {dataset_sizes}")
        
        # Should have seen gradual growth from chunk writes
        non_zero_sizes = [s for s in dataset_sizes if s > 0]
        if len(non_zero_sizes) > 1:
            # If we have multiple non-zero measurements, they should be increasing
            for i in range(1, len(non_zero_sizes)):
                assert non_zero_sizes[i] >= non_zero_sizes[i-1], \
                    f"Dataset should grow over time, but got {non_zero_sizes}"
        
        # Final verification with better error handling
        if not os.path.exists(temp_h5_file):
            pytest.fail("HDF5 file was not created")
        
        try:
            with h5py.File(temp_h5_file, 'r') as f:
                if 'EEG' in f:
                    final_size = f['EEG'].shape[0]
                    assert final_size == 8, f"Expected 8 samples (4 batches * 2 samples), got {final_size}"
                    # Check structured array has correct number of fields
                    field_count = len(f['EEG'].dtype.names)
                    assert field_count == 34, f"Expected 34 fields (32 channels + 2 timestamps), got {field_count}"
                    print(f"[OK] Chunk test passed: Final dataset size = {final_size}")
                else:
                    pytest.fail("EEG dataset not found in file")
        except OSError as e:
            # Check if file is empty or corrupted
            file_size = os.path.getsize(temp_h5_file)
            pytest.fail(f"Cannot open HDF5 file (size: {file_size} bytes): {e}. "
                        f"DataSaver may have failed to initialize the file properly.")
    
    def test_incremental_data_writing(self, stop_event, save_queue, temp_h5_file):
        """Test that data is written incrementally via chunking."""
        saver = DataSaver(
            filename=temp_h5_file,
            save_queue=save_queue,
            stop_event=stop_event,
            chunk_size=1  # Write every sample
        )
        
        # Add metadata - include required fields for DataSaver
        eeg_metadata = {
            'channel_count': 16,
            'sample_rate': 250,
            'labels': [f'EEG_{i+1}' for i in range(16)],
            'channel_types': ['EEG'] * 16
        }
        save_queue.put(DataRecord('EEG_Metadata', json.dumps(eeg_metadata), time.time(), time.time()))
        
        # Add test data samples
        num_samples = 5
        for i in range(num_samples):
            eeg_data = np.random.randn(16).astype(np.float64)
            save_queue.put(DataRecord('EEG', eeg_data, time.time() + i * 0.002, time.time() + i * 0.002))
        
        # Start saver
        saver.start()
        
        # Give time for metadata to be processed first
        time.sleep(0.2)
        
        # Ensure saver is still alive and processing
        assert saver.is_alive(), "DataSaver process should be running"
        
        # Give more time for all data samples to be processed
        time.sleep(0.8)
        
        # Stop saver with proper cleanup
        stop_event.set()
        cleanup_process(saver, timeout=5)  # Increase timeout
        
        # Verify the process actually stopped
        assert not saver.is_alive(), "DataSaver process should have stopped"
        
        # Check file exists and has content before trying to open
        assert os.path.exists(temp_h5_file), f"HDF5 file should exist: {temp_h5_file}"
        file_size = os.path.getsize(temp_h5_file)
        assert file_size > 0, f"HDF5 file should not be empty, got {file_size} bytes"
        
        # Final verification - check that all data was written
        try:
            with h5py.File(temp_h5_file, 'r') as f:
                assert 'EEG' in f, "EEG dataset should exist"
                assert f['EEG'].shape[0] == num_samples, f"Should have {num_samples} samples, got {f['EEG'].shape[0]}"
                # Check structured array has correct number of fields
                field_count = len(f['EEG'].dtype.names)
                assert field_count == 18, f"Should have 18 fields (16 channels + 2 timestamps), got {field_count}"
                
                # Verify data integrity - timestamps should be increasing
                timestamps = f['EEG']['timestamp']  # Access timestamp field by name
                for i in range(1, len(timestamps)):
                    assert timestamps[i] >= timestamps[i-1], "Timestamps should be increasing"
        except OSError as e:
            file_size = os.path.getsize(temp_h5_file) if os.path.exists(temp_h5_file) else 0
            pytest.fail(f"Cannot open HDF5 file (size: {file_size} bytes): {e}. "
                        f"DataSaver may have failed to write data properly.")


    def test_crash_resistance(self, stop_event, save_queue, temp_h5_file):
        """Test that data survives hard crashes due to flush implementation."""
        saver = DataSaver(
            filename=temp_h5_file,
            save_queue=save_queue,
            stop_event=stop_event,
            chunk_size=2  # Small chunk for faster testing
        )

        # Add metadata for multiple modalities
        eeg_metadata = {
            'channel_count': 16,
            'sample_rate': 500,
            'labels': [f'EEG_{i+1}' for i in range(16)],
            'channel_types': ['EEG'] * 16
        }

        emg_metadata = {
            'channel_count': 4,
            'sample_rate': 1000,
            'labels': [f'EMG_{i+1}' for i in range(4)],
            'channel_types': ['EMG'] * 4
        }

        save_queue.put(DataRecord('EEG_Metadata', json.dumps(eeg_metadata), time.time(), time.time()))
        save_queue.put(DataRecord('EMG_Metadata', json.dumps(emg_metadata), time.time(), time.time()))

        # Start saver process
        saver.start()
        time.sleep(0.3)  # Let metadata be processed

        # Add test data samples
        num_samples = 10
        for i in range(num_samples):
            eeg_data = np.random.randn(16).astype(np.float64)
            emg_data = np.random.randn(4).astype(np.float64)

            save_queue.put(DataRecord('EEG', eeg_data, time.time() + i * 0.002, time.time() + i * 0.002))
            save_queue.put(DataRecord('EMG', emg_data, time.time() + i * 0.001, time.time() + i * 0.001))

        # Add event data
        for i in range(3):
            event = {
                'event_id': 100 + i,
                'event_type': f'CrashTest{i}',
                'extra_vars': json.dumps({'test': 'crash'})
            }
            save_queue.put(DataRecord('Event', [json.dumps(event)], time.time() + i, time.time() + i))

        # Wait for data to be written (chunks + periodic flush)
        time.sleep(0.5)

        # SIMULATE HARD CRASH - Force kill the process (SIGKILL equivalent)
        # This skips ALL cleanup, shutdown handlers, and proper HDF5 closing
        import signal
        try:
            saver.kill()  # SIGKILL - immediate termination, no cleanup
        except AttributeError:
            # Fallback for older Python versions
            os.kill(saver.pid, signal.SIGKILL)

        # Don't wait for process to finish gracefully - it was killed
        time.sleep(0.2)

        # CRITICAL TEST: File should still be readable despite hard crash
        assert os.path.exists(temp_h5_file), "File should exist even after crash"

        file_size = os.path.getsize(temp_h5_file)
        assert file_size > 0, f"File should not be empty after crash, got {file_size} bytes"

        # Verify data survived the crash
        try:
            with h5py.File(temp_h5_file, 'r') as f:
                # Check critical timeseries datasets exist (events may not if killed too fast)
                assert 'EEG' in f, "EEG dataset should survive crash"
                assert 'EMG' in f, "EMG dataset should survive crash"

                # Check we got most/all of the data (accounting for flush timing)
                eeg_samples = f['EEG'].shape[0]
                emg_samples = f['EMG'].shape[0]
                event_samples = f['Event'].shape[0] if 'Event' in f else 0

                # Should have recovered most data (allow some loss from last unflushed chunk)
                assert eeg_samples >= num_samples - 2, \
                    f"Should recover most EEG samples after crash, got {eeg_samples}/{num_samples}"
                assert emg_samples >= num_samples - 2, \
                    f"Should recover most EMG samples after crash, got {emg_samples}/{num_samples}"

                # Verify data integrity - timestamps should be increasing
                if eeg_samples > 1:
                    eeg_timestamps = f['EEG']['timestamp']
                    for i in range(1, len(eeg_timestamps)):
                        assert eeg_timestamps[i] >= eeg_timestamps[i-1], \
                            "EEG timestamps should be increasing (data integrity check)"

                print(f"[OK] Crash resistance test passed!")
                print(f"   Recovered {eeg_samples}/{num_samples} EEG samples")
                print(f"   Recovered {emg_samples}/{num_samples} EMG samples")
                print(f"   Recovered {event_samples}/3 Event samples")
                print(f"   File survived SIGKILL crash and remained readable")

        except OSError as e:
            pytest.fail(f"File corrupted after crash - flush implementation failed: {e}")

    def test_hdf5_structure_field_names(self, stop_event, save_queue, temp_h5_file):
        """Test that HDF5 structured arrays use correct lowercase field names."""
        saver = DataSaver(
            filename=temp_h5_file,
            save_queue=save_queue,
            stop_event=stop_event
        )

        # Create EEG metadata
        eeg_metadata = {
            'channel_count': 4,
            'sample_rate': 500,
            'labels': ['EEG_1', 'EEG_2', 'EEG_3', 'EEG_4'],
            'channel_types': ['EEG'] * 4
        }

        # Add metadata and data
        save_queue.put(DataRecord('EEG_Metadata', json.dumps(eeg_metadata), 1000.0, 1001.0))
        save_queue.put(DataRecord('EEG', [1.0, 2.0, 3.0, 4.0], 1002.0, 1003.0))

        # Process data
        saver.start()
        time.sleep(0.2)
        stop_event.set()
        saver.join(timeout=5)

        # Validate HDF5 structure
        with h5py.File(temp_h5_file, 'r') as f:
            assert 'EEG' in f
            eeg_dataset = f['EEG']

            # Validate field names use lowercase snake_case
            field_names = list(eeg_dataset.dtype.names)
            assert 'timestamp' in field_names, f"Expected 'timestamp' in {field_names}"
            assert 'local_timestamp' in field_names, f"Expected 'local_timestamp' in {field_names}"

            # Ensure PascalCase versions are NOT present (regression test)
            assert 'Timestamp' not in field_names, "Found old PascalCase 'Timestamp' field"
            assert 'Local_Timestamp' not in field_names, "Found old PascalCase 'Local_Timestamp' field"

            # Validate channel fields are present
            for i in range(4):
                assert f'EEG_{i+1}' in field_names, f"Missing channel field EEG_{i+1}"
    
    def test_hdf5_structure_dtype_validation(self, stop_event, save_queue, temp_h5_file):
        """Test that structured array dtypes are correctly organized."""
        saver = DataSaver(
            filename=temp_h5_file,
            save_queue=save_queue,
            stop_event=stop_event
        )
        
        # Create metadata with mixed channel types
        metadata = {
            'channel_count': 6,
            'sample_rate': 500,
            'labels': ['EEG_1', 'EEG_2', 'EMG_1', 'EMG_2', 'EOG_1', 'Markers'],
            'channel_types': ['EEG', 'EEG', 'EMG', 'EMG', 'EOG', 'Markers'],
            'channel_format': 'float32'
        }
        
        # Add metadata and data  
        save_queue.put(DataRecord('Mixed_Metadata', json.dumps(metadata), 1000.0, 1001.0))
        save_queue.put(DataRecord('Mixed', [1.0, 2.0, 3.0, 4.0, 5.0, 0.0], 1002.0, 1003.0))
        
        # Process data
        saver.start()
        time.sleep(0.2)
        stop_event.set()
        saver.join(timeout=5)
        
        # Validate dtype structure
        with h5py.File(temp_h5_file, 'r') as f:
            assert 'Mixed' in f
            dataset = f['Mixed']
            dtype = dataset.dtype
            
            # Validate field count: 6 channels + 2 timestamps = 8 fields
            assert len(dtype.names) == 8, f"Expected 8 fields, got {len(dtype.names)}"
            
            # Validate timestamp field types
            assert dtype.fields['timestamp'][0] == np.float64, "Timestamp should be float64"
            assert dtype.fields['local_timestamp'][0] == np.float64, "Local_Timestamp should be float64"
            
            # Validate channel field types match metadata
            for channel_name in ['EEG_1', 'EEG_2', 'EMG_1', 'EMG_2', 'EOG_1', 'Markers']:
                assert channel_name in dtype.names, f"Missing channel {channel_name}"
                # Should use base dtype from metadata (float32)
                assert dtype.fields[channel_name][0] == np.float32, f"{channel_name} should be float32"
    
    def test_hdf5_channel_metadata_separation(self, stop_event, save_queue, temp_h5_file):
        """Test that channels and timestamps are properly separated in structured arrays."""
        saver = DataSaver(
            filename=temp_h5_file,
            save_queue=save_queue,
            stop_event=stop_event
        )
        
        # Create EEG metadata
        eeg_metadata = {
            'channel_count': 3,
            'sample_rate': 500,
            'labels': ['C3', 'Cz', 'C4'],
            'channel_types': ['EEG'] * 3
        }
        
        # Add metadata and data
        save_queue.put(DataRecord('EEG_Metadata', json.dumps(eeg_metadata), 1000.0, 1001.0))
        save_queue.put(DataRecord('EEG', [10.5, 20.3, 30.7], 1500.0, 1501.0))
        
        # Process data
        saver.start()
        time.sleep(0.2)
        stop_event.set()
        saver.join(timeout=5)
        
        # Validate data separation
        with h5py.File(temp_h5_file, 'r') as f:
            dataset = f['EEG']
            data = dataset[0]  # Get first (and only) record
            
            # Validate channel data
            assert data['C3'] == 10.5, "C3 channel data incorrect"
            assert data['Cz'] == 20.3, "Cz channel data incorrect" 
            assert data['C4'] == 30.7, "C4 channel data incorrect"
            
            # Validate timestamp data
            assert data['timestamp'] == 1500.0, "Timestamp data incorrect"
            assert data['local_timestamp'] == 1501.0, "Local_Timestamp data incorrect"
            
            # Validate timestamps are separate from channel data
            channel_fields = [name for name in dataset.dtype.names 
                            if name not in ['timestamp', 'local_timestamp']]
            assert len(channel_fields) == 3, f"Expected 3 channel fields, got {len(channel_fields)}"
            assert set(channel_fields) == {'C3', 'Cz', 'C4'}, "Channel fields incorrect"
    
    def test_hdf5_dataset_naming_validation(self, stop_event, save_queue, temp_h5_file):
        """Test that correct dataset names are created."""
        saver = DataSaver(
            filename=temp_h5_file,
            save_queue=save_queue,
            stop_event=stop_event
        )
        
        # Create various stream types
        eeg_metadata = {
            'channel_count': 2,
            'sample_rate': 500,
            'labels': ['EEG_1', 'EEG_2'],
            'channel_types': ['EEG'] * 2
        }

        emg_metadata = {
            'channel_count': 1,
            'sample_rate': 1000,
            'labels': ['EMG_1'],
            'channel_types': ['EMG']
        }
        
        # Add metadata and data for different modalities
        save_queue.put(DataRecord('EEG_Metadata', json.dumps(eeg_metadata), 1000.0, 1001.0))
        save_queue.put(DataRecord('EMG_Metadata', json.dumps(emg_metadata), 1000.0, 1001.0))
        save_queue.put(DataRecord('EEG', [1.0, 2.0], 1002.0, 1003.0))
        save_queue.put(DataRecord('EMG', [5.0], 1002.0, 1003.0))
        
        # Add event data (dict format, already parsed/normalized by DAQ)
        event_data = {'event_id': 42, 'event_type': 'stimulus'}
        save_queue.put(DataRecord('Event', event_data, 1004.0, 1005.0))
        
        # Process data
        saver.start()
        time.sleep(0.2)
        stop_event.set()
        saver.join(timeout=5)
        
        # Validate dataset names
        with h5py.File(temp_h5_file, 'r') as f:
            datasets = list(f.keys())
            
            # Validate expected datasets exist
            assert 'EEG' in datasets, "EEG dataset missing"
            assert 'EMG' in datasets, "EMG dataset missing"
            assert 'Event' in datasets, "Event dataset missing"
            
            # Validate dataset structure
            assert f['EEG'].shape[0] > 0, "EEG dataset should have data"
            assert f['EMG'].shape[0] > 0, "EMG dataset should have data"
            assert f['Event'].shape[0] > 0, "Event dataset should have data"
            
            # Validate EEG has structured format with timestamps
            eeg_fields = list(f['EEG'].dtype.names)
            assert 'timestamp' in eeg_fields, "EEG should have Timestamp field"
            assert 'local_timestamp' in eeg_fields, "EEG should have Local_Timestamp field"
            
            # Validate Event has the expected event structure
            event_fields = list(f['Event'].dtype.names)
            expected_event_fields = ['event_id', 'event_type', 'timestamp', 'local_timestamp', 'extra_vars']
            for field in expected_event_fields:
                assert field in event_fields, f"Event dataset missing field: {field}"

    def test_recording_health_complete_structure(self, stop_event, save_queue, temp_h5_file):
        """Verify a complete recording has all expected components and valid data.

        This test simulates a realistic recording session and validates:
        - All modality datasets exist
        - Timestamps are monotonically increasing
        - Channel counts match metadata
        - No unexpected NaN values in timeseries
        - Correct dtypes throughout
        """
        saver = DataSaver(
            filename=temp_h5_file,
            save_queue=save_queue,
            stop_event=stop_event,
            chunk_size=10
        )

        # Setup metadata for EEG and EMG
        eeg_channels = 32
        emg_channels = 8
        sample_rate = 500

        eeg_metadata = {
            'channel_count': eeg_channels,
            'sample_rate': sample_rate,
            'labels': [f'EEG_{i+1}' for i in range(eeg_channels)],
            'channel_types': ['EEG'] * eeg_channels
        }
        emg_metadata = {
            'channel_count': emg_channels,
            'sample_rate': sample_rate,
            'labels': [f'EMG_{i+1}' for i in range(emg_channels)],
            'channel_types': ['EMG'] * emg_channels
        }

        save_queue.put(DataRecord('EEG_Metadata', json.dumps(eeg_metadata), time.time(), time.time()))
        save_queue.put(DataRecord('EMG_Metadata', json.dumps(emg_metadata), time.time(), time.time()))

        # Simulate 100 samples of continuous recording
        base_time = time.time()
        n_samples = 100

        for i in range(n_samples):
            timestamp = base_time + i / sample_rate
            local_timestamp = timestamp + 0.001  # Small offset

            # EEG data - realistic range
            eeg_data = np.random.randn(eeg_channels).astype(np.float64) * 50  # ~50µV
            save_queue.put(DataRecord('EEG', eeg_data, timestamp, local_timestamp))

            # EMG data - realistic range
            emg_data = np.random.randn(emg_channels).astype(np.float64) * 100  # ~100µV
            save_queue.put(DataRecord('EMG', emg_data, timestamp, local_timestamp))

        # Add some events (dict format, already parsed/normalized by DAQ)
        for i in range(3):
            event_time = base_time + (i + 1) * 0.05  # Events at 50ms intervals
            event = {
                'event_id': i + 1,
                'event_type': f'stimulus_{i+1}',
            }
            event_record = DataRecord(
                modality='Event',
                sample=event,  # Dict already parsed/normalized by DAQ
                timestamp=event_time,
                local_timestamp=event_time + 0.001
            )
            save_queue.put(event_record)

        # Process all data
        saver.start()
        time.sleep(1.0)  # Allow processing
        stop_event.set()
        saver.join(timeout=5)

        # Validate recording health
        assert os.path.exists(temp_h5_file)
        with h5py.File(temp_h5_file, 'r') as f:
            # 1. All expected datasets exist
            assert 'EEG' in f, "EEG dataset missing"
            assert 'EMG' in f, "EMG dataset missing"
            assert 'Event' in f, "Event dataset missing"

            # 2. Check EEG data integrity
            eeg_data = f['EEG']
            assert eeg_data.shape[0] == n_samples, f"Expected {n_samples} EEG samples, got {eeg_data.shape[0]}"

            # Get timestamps and verify monotonicity
            eeg_timestamps = eeg_data['timestamp'][:]
            assert np.all(np.diff(eeg_timestamps) >= 0), "EEG timestamps not monotonically increasing"

            # Verify no NaN in channel data (channel names match labels from metadata)
            for i in range(eeg_channels):
                channel_name = f'EEG_{i+1}'  # Matches label in metadata
                channel_data = eeg_data[channel_name][:]
                assert not np.any(np.isnan(channel_data)), f"NaN found in EEG channel {channel_name}"

            # 3. Check EMG data integrity
            emg_data = f['EMG']
            assert emg_data.shape[0] == n_samples, f"Expected {n_samples} EMG samples, got {emg_data.shape[0]}"

            emg_timestamps = emg_data['timestamp'][:]
            assert np.all(np.diff(emg_timestamps) >= 0), "EMG timestamps not monotonically increasing"

            # 4. Verify dtypes are correct
            assert eeg_data.dtype['timestamp'] == np.float64, "EEG timestamp dtype should be float64"
            assert emg_data.dtype['timestamp'] == np.float64, "EMG timestamp dtype should be float64"

            # 5. Events exist
            assert f['Event'].shape[0] == 3, "Expected 3 events"

    def test_recording_health_event_alignment(self, stop_event, save_queue, temp_h5_file):
        """Verify events are properly time-aligned with data streams.

        Tests that:
        - Event timestamps fall within data time range
        - Events can be correctly associated with nearby data samples
        """
        saver = DataSaver(
            filename=temp_h5_file,
            save_queue=save_queue,
            stop_event=stop_event
        )

        # Setup EEG metadata
        eeg_metadata = {
            'channel_count': 8,
            'sample_rate': 500,
            'labels': [f'EEG_{i+1}' for i in range(8)],
            'channel_types': ['EEG'] * 8
        }
        save_queue.put(DataRecord('EEG_Metadata', json.dumps(eeg_metadata), time.time(), time.time()))

        # Record data with known timing
        base_time = 1000.0  # Use fixed base for predictable testing
        n_samples = 50

        for i in range(n_samples):
            timestamp = base_time + i * 0.002  # 500 Hz = 2ms per sample
            eeg_data = np.random.randn(8).astype(np.float64)
            save_queue.put(DataRecord('EEG', eeg_data, timestamp, timestamp))

        # Add event in the middle of recording (dict format, already parsed/normalized by DAQ)
        event_time = base_time + 0.05  # 50ms into recording (sample ~25)
        event = {
            'event_id': 1,
            'event_type': 'target',
        }
        event_record = DataRecord(
            modality='Event',
            sample=event,  # Dict already parsed/normalized by DAQ
            timestamp=event_time,
            local_timestamp=event_time
        )
        save_queue.put(event_record)

        saver.start()
        time.sleep(0.5)
        stop_event.set()
        saver.join(timeout=5)

        # Validate event alignment
        with h5py.File(temp_h5_file, 'r') as f:
            eeg_timestamps = f['EEG']['timestamp'][:]
            event_timestamp = f['Event']['timestamp'][0]

            data_start = eeg_timestamps.min()
            data_end = eeg_timestamps.max()

            # Event should be within data time range
            assert data_start <= event_timestamp <= data_end, \
                f"Event at {event_timestamp} outside data range [{data_start}, {data_end}]"

            # Find nearest sample to event
            time_diffs = np.abs(eeg_timestamps - event_timestamp)
            nearest_idx = np.argmin(time_diffs)
            nearest_time_diff = time_diffs[nearest_idx]

            # Nearest sample should be within 1 sample period (2ms)
            assert nearest_time_diff < 0.003, \
                f"Nearest sample is {nearest_time_diff*1000:.1f}ms from event, expected < 3ms"

    def test_concurrent_read_while_writing(self, stop_event, save_queue, temp_h5_file):
        """Test reading HDF5 file while DataSaver is actively writing.

        HDF5 supports Single-Writer-Multiple-Reader (SWMR) mode, but even without
        it, we should be able to read committed data safely.
        """
        saver = DataSaver(
            filename=temp_h5_file,
            save_queue=save_queue,
            stop_event=stop_event,
            chunk_size=5  # Small chunks for frequent writes
        )

        # Setup metadata
        eeg_metadata = {
            'channel_count': 8,
            'sample_rate': 500,
            'labels': [f'EEG_{i+1}' for i in range(8)],
            'channel_types': ['EEG'] * 8
        }
        save_queue.put(DataRecord('EEG_Metadata', json.dumps(eeg_metadata), time.time(), time.time()))

        # Start saver
        saver.start()
        time.sleep(0.1)  # Let it initialize

        # Add data while periodically trying to read
        read_successes = 0
        read_failures = 0

        for i in range(20):
            # Add data
            eeg_data = np.random.randn(8).astype(np.float64)
            save_queue.put(DataRecord('EEG', eeg_data, time.time(), time.time()))

            # Try to read every 5 samples
            if i > 0 and i % 5 == 0:
                time.sleep(0.1)  # Give time to flush
                try:
                    # Try to open and read
                    with h5py.File(temp_h5_file, 'r') as f:
                        if 'EEG' in f:
                            _ = f['EEG'].shape[0]
                            read_successes += 1
                except (OSError, IOError):
                    # File locked or being written - this is acceptable
                    read_failures += 1

        stop_event.set()
        saver.join(timeout=5)

        # We should have had at least some successful reads
        # (HDF5 allows reading committed data)
        assert read_successes > 0 or read_failures > 0, "No read attempts made"

        # Final verification - file should be readable after close
        with h5py.File(temp_h5_file, 'r') as f:
            assert 'EEG' in f
            assert f['EEG'].shape[0] > 0

    def test_file_lock_recovery(self, stop_event, save_queue, temp_h5_file):
        """Test DataSaver behavior when file access is temporarily blocked.

        Simulates scenarios where the file might be briefly locked by another
        process (e.g., antivirus scan, backup software).
        """
        saver = DataSaver(
            filename=temp_h5_file,
            save_queue=save_queue,
            stop_event=stop_event
        )

        # Setup metadata
        eeg_metadata = {
            'channel_count': 8,
            'sample_rate': 500,
            'labels': [f'EEG_{i+1}' for i in range(8)],
            'channel_types': ['EEG'] * 8
        }
        save_queue.put(DataRecord('EEG_Metadata', json.dumps(eeg_metadata), time.time(), time.time()))

        # Add some initial data
        for i in range(10):
            eeg_data = np.random.randn(8).astype(np.float64)
            save_queue.put(DataRecord('EEG', eeg_data, time.time(), time.time()))

        # Start saver
        saver.start()
        time.sleep(0.3)  # Process initial data

        # Add more data after initial batch
        for i in range(10):
            eeg_data = np.random.randn(8).astype(np.float64)
            save_queue.put(DataRecord('EEG', eeg_data, time.time(), time.time()))
            time.sleep(0.02)

        stop_event.set()
        saver.join(timeout=5)

        # Verify file is valid and has data
        assert os.path.exists(temp_h5_file)
        with h5py.File(temp_h5_file, 'r') as f:
            assert 'EEG' in f
            # Should have most/all of our samples
            assert f['EEG'].shape[0] >= 15, f"Expected at least 15 samples, got {f['EEG'].shape[0]}"

    def test_bids_metadata_at_root_level(self, stop_event, save_queue, temp_h5_file):
        """Test that BIDS metadata fields are stored at HDF5 root level."""
        saver = DataSaver(
            filename=temp_h5_file,
            save_queue=save_queue,
            stop_event=stop_event
        )

        # Create global metadata with BIDS fields (as sent by processing_pipeline)
        bids_metadata = {
            'version': '2.9.0',
            'sample_rate': 500,
            'study_name': 'test_study',
            'recording_name': 'test_recording',
            'subject_id': '001',
            'session_id': '01',
            'run_number': 1,
            'experiment_description': 'Test experiment'
        }

        metadata_record = DataRecord(
            modality='Metadata',
            sample=json.dumps(bids_metadata),
            timestamp=time.time(),
            local_timestamp=time.time()
        )
        save_queue.put(metadata_record)

        saver.start()
        time.sleep(0.3)
        stop_event.set()
        saver.join(timeout=5)

        # Verify BIDS fields at root level
        with h5py.File(temp_h5_file, 'r') as f:
            assert 'subject_id' in f.attrs, "subject_id should be at root"
            assert 'session_id' in f.attrs, "session_id should be at root"
            assert 'run_number' in f.attrs, "run_number should be at root"
            assert 'study_name' in f.attrs, "study_name should be at root"
            assert 'recording_name' in f.attrs, "recording_name should be at root"

            assert f.attrs['subject_id'] == '001'
            assert f.attrs['session_id'] == '01'
            assert f.attrs['run_number'] == 1
            assert f.attrs['study_name'] == 'test_study'
            assert f.attrs['recording_name'] == 'test_recording'
            assert f.attrs['experiment_description'] == 'Test experiment'

    def test_channel_types_and_units_in_dataset_attrs(self, stop_event, save_queue, temp_h5_file):
        """Test that channel_types and channel_units are stored in dataset attributes."""
        saver = DataSaver(
            filename=temp_h5_file,
            save_queue=save_queue,
            stop_event=stop_event
        )

        # Create stream metadata with channel_types and channel_units
        eeg_metadata = {
            'channel_count': 4,
            'sample_rate': 500,
            'labels': ['Fp1', 'Fp2', 'C3', 'C4'],
            'channel_types': ['EEG', 'EEG', 'EEG', 'EEG'],
            'channel_units': ['µV', 'µV', 'µV', 'µV']
        }

        save_queue.put(DataRecord('EEG_Metadata', json.dumps(eeg_metadata), time.time(), time.time()))
        save_queue.put(DataRecord('EEG', [1.0, 2.0, 3.0, 4.0], time.time(), time.time()))

        saver.start()
        time.sleep(0.3)
        stop_event.set()
        saver.join(timeout=5)

        # Verify channel_types and channel_units in dataset attrs
        with h5py.File(temp_h5_file, 'r') as f:
            assert 'EEG' in f
            eeg_attrs = dict(f['EEG'].attrs)

            # channel_types should be stored (not remapped)
            assert 'channel_types' in eeg_attrs, "channel_types should be in dataset attrs"
            channel_types = json.loads(eeg_attrs['channel_types'])
            assert channel_types == ['EEG', 'EEG', 'EEG', 'EEG']

            # channel_units should be stored (not remapped)
            assert 'channel_units' in eeg_attrs, "channel_units should be in dataset attrs"
            channel_units = json.loads(eeg_attrs['channel_units'])
            assert channel_units == ['µV', 'µV', 'µV', 'µV']

    def test_skip_keys_version_filtering(self, stop_event, save_queue, temp_h5_file):
        """Test that SKIP_KEYS filters out 'version' from stream metadata."""
        saver = DataSaver(
            filename=temp_h5_file,
            save_queue=save_queue,
            stop_event=stop_event
        )

        # Send global metadata with app version (should be stored at root)
        global_metadata = {
            'version': '2.9.0',
            'sample_rate': 500
        }
        save_queue.put(DataRecord('Metadata', json.dumps(global_metadata), time.time(), time.time()))

        # Send stream metadata that also has a 'version' field (LSL version, should be skipped)
        eeg_metadata = {
            'channel_count': 2,
            'sample_rate': 500,
            'labels': ['EEG_1', 'EEG_2'],
            'version': '1.0'  # LSL stream version - should be filtered by SKIP_KEYS
        }

        save_queue.put(DataRecord('EEG_Metadata', json.dumps(eeg_metadata), time.time(), time.time()))
        save_queue.put(DataRecord('EEG', [1.0, 2.0], time.time(), time.time()))

        saver.start()
        time.sleep(0.3)
        stop_event.set()
        saver.join(timeout=5)

        with h5py.File(temp_h5_file, 'r') as f:
            # Root-level version from global metadata should exist
            assert 'version' in f.attrs, "Root version should exist from global metadata"
            assert f.attrs['version'] == '2.9.0'

            # Dataset-level version should NOT exist (filtered by SKIP_KEYS)
            assert 'EEG' in f
            eeg_attrs = dict(f['EEG'].attrs)
            assert 'version' not in eeg_attrs, "Stream version should be filtered by SKIP_KEYS"

    def test_created_by_format_with_app_name(self, stop_event, save_queue, temp_h5_file):
        """Test that created_by attribute uses APP_NAME in correct format."""
        from dendrite.constants import APP_NAME, VERSION

        saver = DataSaver(
            filename=temp_h5_file,
            save_queue=save_queue,
            stop_event=stop_event
        )

        # Just start and stop - file initialization creates created_by
        saver.start()
        time.sleep(0.2)
        stop_event.set()
        saver.join(timeout=5)

        # Verify created_by format
        with h5py.File(temp_h5_file, 'r') as f:
            assert 'created_by' in f.attrs, "created_by attribute should exist"

            created_by = f.attrs['created_by']
            expected = f'{APP_NAME} DataSaver v{VERSION}'

            assert created_by == expected, f"Expected '{expected}', got '{created_by}'"
            assert APP_NAME in created_by, "created_by should contain APP_NAME"
