"""
Unit tests for MetricsSaver class.

Tests cover:
- MetricsSaver initialization and configuration
- HDF5 metrics file creation and structure
- Data handling for different metric types
- Error handling and edge cases
"""

import sys
import os
import pytest
import numpy as np
import h5py
import time
import multiprocessing

# Add the project root to the path so BMI can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dataclasses import asdict

from dendrite.data.storage.metrics_saver import MetricsSaver
from dendrite.processing.modes.base_mode import ModeOutputPacket


class TestMetricsSaver:
    """Test suite for MetricsSaver class."""

    def test_initialization(self, stop_event, temp_h5_file):
        """Test MetricsSaver initialization."""
        script_metadata = {
            'script_version': '1.0.0',
            'sample_rate': 500,
            'mode_instances': {'test_mode': {'model_type': 'EEGNet'}}
        }

        queue1 = multiprocessing.Queue()
        queue2 = multiprocessing.Queue()
        mode_queues = {'mode1': queue1, 'mode2': queue2}

        saver = MetricsSaver(
            filename=temp_h5_file,
            stop_event=stop_event,
            script_metadata=script_metadata,
            mode_metrics_queues=mode_queues
        )

        assert saver.filename == temp_h5_file
        assert saver.stop_event == stop_event
        assert saver.script_metadata == script_metadata
        assert saver.mode_metrics_queues == mode_queues

    def test_metadata_saving(self, stop_event, temp_h5_file):
        """Test script metadata saving."""
        script_metadata = {
            'script_version': '1.0.0',
            'sample_rate': 500,
            'recording_name': 'test_recording',
            'mode_instances': {
                'test_mode': {
                    'model_type': 'EEGNet',
                    'decoder_type': 'AdaptiveNeuralNetworkDecoder'
                }
            }
        }

        saver = MetricsSaver(
            filename=temp_h5_file,
            stop_event=stop_event,
            script_metadata=script_metadata
        )

        # Start and immediately stop to save metadata
        saver.start()
        time.sleep(0.2)
        stop_event.set()
        saver.join(timeout=5)

        # Verify metadata was saved
        assert os.path.exists(temp_h5_file)
        with h5py.File(temp_h5_file, 'r') as f:
            assert 'script_metadata' in f
            metadata_group = f['script_metadata']
            assert 'script_version' in metadata_group.attrs
            assert metadata_group.attrs['script_version'] == '1.0.0'
            assert 'sample_rate' in metadata_group.attrs
            assert metadata_group.attrs['sample_rate'] == 500

    def test_multiple_classifier_queues(self, stop_event, temp_h5_file):
        """Test handling of multiple mode output queues."""
        script_metadata = {'script_version': '1.0.0'}

        # Create queues for different modes
        queue1 = multiprocessing.Queue()
        queue2 = multiprocessing.Queue()
        mode_queues = {
            'sync_mode': queue1,
            'async_mode': queue2
        }

        saver = MetricsSaver(
            filename=temp_h5_file,
            stop_event=stop_event,
            script_metadata=script_metadata,
            mode_metrics_queues=mode_queues
        )

        # Add test data to queues (using ModeOutputPacket field names)
        test_data1 = {
            'data': {
                'predictions': [0.8, 0.2],
                'confidence': 0.8,
                'mode_name': 'sync_mode'
            },
            'data_timestamp': time.time()
        }

        test_data2 = {
            'data': {
                'predictions': [0.3, 0.7],
                'confidence': 0.7,
                'mode_name': 'async_mode'
            },
            'data_timestamp': time.time()
        }

        queue1.put(test_data1)
        queue2.put(test_data2)

        # Start saver
        saver.start()
        time.sleep(0.5)  # Let it process
        stop_event.set()
        saver.join(timeout=5)

        # Verify data was saved
        assert os.path.exists(temp_h5_file)
        with h5py.File(temp_h5_file, 'r') as f:
            # Check metadata was saved
            assert 'script_metadata' in f
            metadata_group = f['script_metadata']
            assert 'script_version' in metadata_group.attrs
            # Check that some datasets were created
            assert len(f.keys()) >= 1  # At least script_metadata should exist

    def test_no_queues_handling(self, stop_event, temp_h5_file):
        """Test handling when no mode queues are provided."""
        script_metadata = {'script_version': '1.0.0'}

        saver = MetricsSaver(
            filename=temp_h5_file,
            stop_event=stop_event,
            script_metadata=script_metadata,
            mode_metrics_queues=None
        )

        # Start and stop
        saver.start()
        time.sleep(0.2)
        stop_event.set()
        saver.join(timeout=5)

        # Should not crash and file should exist
        assert os.path.exists(temp_h5_file)

    def test_empty_queues_handling(self, stop_event, temp_h5_file):
        """Test handling of empty mode queues."""
        script_metadata = {'script_version': '1.0.0'}

        # Create empty queues
        queue1 = multiprocessing.Queue()
        queue2 = multiprocessing.Queue()
        mode_queues = {
            'mode1': queue1,
            'mode2': queue2
        }

        saver = MetricsSaver(
            filename=temp_h5_file,
            stop_event=stop_event,
            script_metadata=script_metadata,
            mode_metrics_queues=mode_queues
        )

        # Start and stop with empty queues
        saver.start()
        time.sleep(0.2)
        stop_event.set()
        saver.join(timeout=5)

        # Should not crash
        assert os.path.exists(temp_h5_file)
        with h5py.File(temp_h5_file, 'r') as f:
            assert 'script_metadata' in f
            metadata_group = f['script_metadata']
            assert 'script_version' in metadata_group.attrs

    def test_mixed_data_types_handling(self, stop_event, temp_h5_file):
        """Test handling of mixed data types in metrics."""
        script_metadata = {'script_version': '1.0.0'}

        queue1 = multiprocessing.Queue()
        mode_queues = {'sync_mode': queue1}

        saver = MetricsSaver(
            filename=temp_h5_file,
            stop_event=stop_event,
            script_metadata=script_metadata,
            mode_metrics_queues=mode_queues
        )

        # Test data with different types (using ModeOutputPacket dataclass)
        test_data = ModeOutputPacket(
            type='erp',
            mode_name='sync_mode',
            mode_type='synchronous',
            data={
                'type': 'erp',  # String
                'event_type': 'aaaa',  # String
                'eeg_data': np.random.randn(250, 60),  # 2D numpy array
                'confidence': 0.85,  # Float
                'prediction': 1,  # Integer
                'is_correct': True  # Boolean
            },
            data_timestamp=time.time()
        )

        queue1.put(asdict(test_data))

        # Start saver
        saver.start()
        time.sleep(0.5)
        stop_event.set()
        saver.join(timeout=5)

        # Verify data was saved without errors
        assert os.path.exists(temp_h5_file)
        with h5py.File(temp_h5_file, 'r') as f:
            assert 'sync_mode' in f
            sync_group = f['sync_mode']

            # Check that different data types were handled correctly
            assert 'type' in sync_group
            assert 'event_type' in sync_group
            assert 'eeg_data' in sync_group
            assert 'confidence' in sync_group
            assert 'prediction' in sync_group
            assert 'is_correct' in sync_group

            # Verify shapes are correct
            assert sync_group['eeg_data'].shape == (1, 250, 60)  # 1 time entry + original shape
            assert sync_group['confidence'].shape == (1,)  # 1D for scalar

    def test_incremental_metrics_saving(self, stop_event, temp_h5_file):
        """Test incremental saving of metrics data."""
        script_metadata = {'script_version': '1.0.0'}

        queue1 = multiprocessing.Queue()
        mode_queues = {'test_mode': queue1}

        saver = MetricsSaver(
            filename=temp_h5_file,
            stop_event=stop_event,
            script_metadata=script_metadata,
            mode_metrics_queues=mode_queues
        )

        # Start saver first
        saver.start()
        time.sleep(0.1)

        # Add multiple data points incrementally (using ModeOutputPacket dataclass)
        for i in range(3):
            test_data = ModeOutputPacket(
                type='performance',
                mode_name='test_mode',
                mode_type='synchronous',
                data={
                    'accuracy': 0.7 + i * 0.1,  # Scalar that changes
                    'eeg_epoch': np.random.randn(100, 32),  # 2D array
                    'epoch_number': i + 1  # Integer counter
                },
                data_timestamp=time.time() + i
            )
            queue1.put(asdict(test_data))
            time.sleep(0.1)  # Allow processing

        time.sleep(0.3)  # Final processing
        stop_event.set()
        saver.join(timeout=5)

        # Verify incremental data was saved
        assert os.path.exists(temp_h5_file)
        with h5py.File(temp_h5_file, 'r') as f:
            assert 'test_mode' in f
            test_group = f['test_mode']

            # Check that all 3 entries were saved
            assert 'accuracy' in test_group
            assert test_group['accuracy'].shape == (3,)

            assert 'eeg_epoch' in test_group
            assert test_group['eeg_epoch'].shape == (3, 100, 32)

            assert 'epoch_number' in test_group
            assert test_group['epoch_number'].shape == (3,)

            # Verify values are correct
            accuracies = test_group['accuracy'][:]
            assert np.allclose(accuracies, [0.7, 0.8, 0.9])

            epochs = test_group['epoch_number'][:]
            assert np.array_equal(epochs, [1, 2, 3])

    def test_erp_payload_data_format(self, stop_event, temp_h5_file):
        """Test ERP payload data format (simulating synchronous mode output)."""
        script_metadata = {'script_version': '1.0.0'}

        queue1 = multiprocessing.Queue()
        mode_queues = {'synchronous_mode': queue1}

        saver = MetricsSaver(
            filename=temp_h5_file,
            stop_event=stop_event,
            script_metadata=script_metadata,
            mode_metrics_queues=mode_queues
        )

        # Simulate ERP payload from synchronous mode (using ModeOutputPacket dataclass)
        erp_data = ModeOutputPacket(
            type='erp',
            mode_name='synchronous_mode',
            mode_type='synchronous',
            data={
                'type': 'erp',
                'event_type': 'left_hand',
                'eeg_data': np.random.randn(250, 60).astype(np.float64)  # Typical ERP shape
            },
            data_timestamp=time.time()
        )

        performance_data = ModeOutputPacket(
            type='performance',
            mode_name='synchronous_mode',
            mode_type='synchronous',
            data={
                'type': 'performance',
                'accuracy': 0.75,
                'confidence': 0.82,
                'chance_level': 0.5,
                'detailed_metrics': {
                    'precision': 0.78,
                    'recall': 0.72
                }
            },
            data_timestamp=time.time()
        )

        queue1.put(asdict(erp_data))
        queue1.put(asdict(performance_data))

        # Start saver
        saver.start()
        time.sleep(0.5)
        stop_event.set()
        saver.join(timeout=5)

        # Verify data structure
        assert os.path.exists(temp_h5_file)
        with h5py.File(temp_h5_file, 'r') as f:
            assert 'synchronous_mode' in f
            sync_group = f['synchronous_mode']

            # Check ERP data
            assert 'eeg_data' in sync_group
            assert sync_group['eeg_data'].shape == (1, 250, 60)

            # Check performance data
            assert 'accuracy' in sync_group
            assert 'confidence' in sync_group
            assert 'chance_level' in sync_group

            # Check flattened nested metrics
            assert 'detailed_metrics_precision' in sync_group
            assert 'detailed_metrics_recall' in sync_group

    def test_none_then_float_metric_handling(self, stop_event, temp_h5_file):
        """Test metric that starts as None then becomes float (regression test).

        This tests the edge case where auc_roc returns None first (multiclass or
        insufficient data), then returns a float later. Previously this would fail
        because None was stored as 'null' string, creating a string-typed dataset
        that couldn't accept float values later.
        """
        script_metadata = {'script_version': '1.0.0'}

        queue1 = multiprocessing.Queue()
        mode_queues = {'test_mode': queue1}

        saver = MetricsSaver(
            filename=temp_h5_file,
            stop_event=stop_event,
            script_metadata=script_metadata,
            mode_metrics_queues=mode_queues
        )

        # First packet: auc_roc=None (multiclass case)
        packet1 = ModeOutputPacket(
            type='performance',
            mode_name='test_mode',
            mode_type='synchronous',
            data={
                'accuracy': 0.75,
                'auc_roc': None,  # Returns None for multiclass
            },
            data_timestamp=time.time()
        )

        # Second packet: auc_roc=0.85 (binary case with enough samples)
        packet2 = ModeOutputPacket(
            type='performance',
            mode_name='test_mode',
            mode_type='synchronous',
            data={
                'accuracy': 0.80,
                'auc_roc': 0.85,  # Returns float for binary
            },
            data_timestamp=time.time() + 0.1
        )

        queue1.put(asdict(packet1))
        queue1.put(asdict(packet2))

        # Start saver - should NOT crash
        saver.start()
        time.sleep(0.5)
        stop_event.set()
        saver.join(timeout=5)

        # Verify both values were saved
        assert os.path.exists(temp_h5_file)
        with h5py.File(temp_h5_file, 'r') as f:
            assert 'test_mode' in f
            group = f['test_mode']

            # auc_roc should exist and have 2 entries
            assert 'auc_roc' in group
            auc_data = group['auc_roc'][:]
            assert len(auc_data) == 2

            # First should be NaN (from None), second should be 0.85
            assert np.isnan(auc_data[0])
            assert auc_data[1] == pytest.approx(0.85, rel=0.01)

    def test_mode_output_packet_format(self, stop_event, temp_h5_file):
        """Test exact ModeOutputPacket format from BMI modes.

        Verifies MetricsSaver correctly handles the actual packet structure
        used by BMI modes (ModeOutputPacket dataclass from mode_utils.py).
        """
        script_metadata = {'script_version': '1.0.0'}

        queue1 = multiprocessing.Queue()
        mode_queues = {'sync_mode': queue1}

        saver = MetricsSaver(
            filename=temp_h5_file,
            stop_event=stop_event,
            script_metadata=script_metadata,
            mode_metrics_queues=mode_queues
        )

        # Use actual ModeOutputPacket dataclass (from dendrite/processing/modes/base_mode.py)
        known_timestamp = time.time()
        test_packet = ModeOutputPacket(
            type='prediction',
            mode_name='sync_mode',
            mode_type='synchronous',
            data={
                'prediction': 1,
                'confidence': 0.92,
                'probabilities': [0.08, 0.92]
            },
            data_timestamp=known_timestamp
        )

        queue1.put(asdict(test_packet))

        # Start saver
        saver.start()
        time.sleep(0.5)
        stop_event.set()
        saver.join(timeout=5)

        # Verify correct field extraction
        assert os.path.exists(temp_h5_file)
        with h5py.File(temp_h5_file, 'r') as f:
            assert 'sync_mode' in f
            group = f['sync_mode']

            # Verify 'type' captured as 'packet_output_type'
            assert 'packet_output_type' in group
            assert group['packet_output_type'][0].decode('utf-8') == 'prediction'

            # Verify 'mode_name' captured as 'source_mode'
            assert 'source_mode' in group
            assert group['source_mode'][0].decode('utf-8') == 'sync_mode'

            # Verify data fields extracted
            assert 'prediction' in group
            assert group['prediction'][0] == 1
            assert 'confidence' in group
            assert group['confidence'][0] == pytest.approx(0.92, rel=0.01)


class TestTelemetrySaving:
    """Tests for telemetry metrics saving to HDF5."""

    def test_shared_state_parameter_accepted(self, stop_event, temp_h5_file, shared_state):
        """Verify MetricsSaver accepts shared_state parameter."""
        script_metadata = {'version': '1.0.0'}

        saver = MetricsSaver(
            filename=temp_h5_file,
            stop_event=stop_event,
            script_metadata=script_metadata,
            shared_state=shared_state
        )

        assert saver.shared_state is shared_state

    def test_shared_state_none_by_default(self, stop_event, temp_h5_file):
        """Verify shared_state defaults to None."""
        script_metadata = {'version': '1.0.0'}

        saver = MetricsSaver(
            filename=temp_h5_file,
            stop_event=stop_event,
            script_metadata=script_metadata
        )

        assert saver.shared_state is None

    def test_telemetry_group_created(self, stop_event, temp_h5_file, shared_state):
        """Verify telemetry group is created in HDF5 when shared_state provided."""
        script_metadata = {'version': '1.0.0'}

        saver = MetricsSaver(
            filename=temp_h5_file,
            stop_event=stop_event,
            script_metadata=script_metadata,
            shared_state=shared_state
        )

        # Start and stop
        saver.start()
        time.sleep(0.3)
        stop_event.set()
        saver.join(timeout=5)

        # Verify telemetry group exists
        with h5py.File(temp_h5_file, 'r') as f:
            assert 'telemetry' in f

    def test_stream_latencies_saved(self, stop_event, temp_h5_file, shared_state):
        """Verify stream latencies are sampled and saved."""
        from dendrite.utils.state_keys import stream_latency_key, stream_timestamp_key

        script_metadata = {'version': '1.0.0'}

        # Set latency values in shared state
        shared_state.set(stream_latency_key('EEG'), 5.5)
        shared_state.set(stream_timestamp_key('EEG'), time.time())
        shared_state.set(stream_latency_key('EMG'), 3.2)
        shared_state.set(stream_timestamp_key('EMG'), time.time())

        saver = MetricsSaver(
            filename=temp_h5_file,
            stop_event=stop_event,
            script_metadata=script_metadata,
            shared_state=shared_state
        )

        # Run long enough for at least one telemetry sample
        saver.start()
        time.sleep(1.5)  # Telemetry interval is 1s
        stop_event.set()
        saver.join(timeout=5)

        # Verify latencies were saved
        with h5py.File(temp_h5_file, 'r') as f:
            assert 'telemetry' in f
            telemetry = f['telemetry']

            # Check EEG latency saved
            assert 'eeg_latency_ms' in telemetry
            eeg_latency = telemetry['eeg_latency_ms'][:]
            assert len(eeg_latency) >= 1
            assert eeg_latency[0] == pytest.approx(5.5, rel=0.1)

            # Check EMG latency saved
            assert 'emg_latency_ms' in telemetry
            emg_latency = telemetry['emg_latency_ms'][:]
            assert len(emg_latency) >= 1
            assert emg_latency[0] == pytest.approx(3.2, rel=0.1)

    def test_e2e_latency_saved(self, stop_event, temp_h5_file, shared_state):
        """Verify E2E latency from task is saved."""
        script_metadata = {'version': '1.0.0'}

        # Set E2E latency in shared state
        shared_state.set('e2e_latency_ms', 45.0)

        saver = MetricsSaver(
            filename=temp_h5_file,
            stop_event=stop_event,
            script_metadata=script_metadata,
            shared_state=shared_state
        )

        saver.start()
        time.sleep(1.5)
        stop_event.set()
        saver.join(timeout=5)

        with h5py.File(temp_h5_file, 'r') as f:
            assert 'telemetry' in f
            telemetry = f['telemetry']
            assert 'e2e_latency_ms' in telemetry
            assert telemetry['e2e_latency_ms'][0] == pytest.approx(45.0, rel=0.1)

    def test_telemetry_no_group_without_shared_state(self, stop_event, temp_h5_file):
        """Verify no telemetry group when shared_state is None."""
        script_metadata = {'version': '1.0.0'}

        saver = MetricsSaver(
            filename=temp_h5_file,
            stop_event=stop_event,
            script_metadata=script_metadata,
            shared_state=None
        )

        saver.start()
        time.sleep(0.3)
        stop_event.set()
        saver.join(timeout=5)

        with h5py.File(temp_h5_file, 'r') as f:
            assert 'telemetry' not in f

    def test_telemetry_multiple_samples(self, stop_event, temp_h5_file, shared_state):
        """Verify multiple telemetry samples are collected over time."""
        from dendrite.utils.state_keys import stream_latency_key

        script_metadata = {'version': '1.0.0'}

        # Set initial latency
        shared_state.set(stream_latency_key('EEG'), 5.0)

        saver = MetricsSaver(
            filename=temp_h5_file,
            stop_event=stop_event,
            script_metadata=script_metadata,
            shared_state=shared_state
        )

        saver.start()
        time.sleep(0.5)

        # Update latency mid-session
        shared_state.set(stream_latency_key('EEG'), 8.0)
        time.sleep(1.5)

        # Update again
        shared_state.set(stream_latency_key('EEG'), 6.0)
        time.sleep(1.5)

        stop_event.set()
        saver.join(timeout=5)

        with h5py.File(temp_h5_file, 'r') as f:
            assert 'telemetry' in f
            telemetry = f['telemetry']
            assert 'eeg_latency_ms' in telemetry
            # Should have multiple samples (at least 2 given timing)
            samples = telemetry['eeg_latency_ms'][:]
            assert len(samples) >= 2
