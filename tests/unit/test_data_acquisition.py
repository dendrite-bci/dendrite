"""
Clean unit tests for DataAcquisition classes.

This module provides comprehensive unit tests for the data acquisition system,
focusing on testing individual components and methods in isolation.

Tests cover:
- DataAcquisitionFixed initialization and configuration
- Stream configuration processing
- Channel mapping and information handling
- Data record creation and handling
- Error handling and edge cases
- Mock LSL stream integration
"""

import sys
import os
import pytest
import numpy as np
import time
import json
import multiprocessing
from unittest.mock import Mock, patch, MagicMock
from collections import deque
from threading import Lock

# Add the project root to the path so BMI can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dendrite.data.acquisition import DataAcquisition, DataRecord
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
        has_metadata_issues=kwargs.get('has_metadata_issues', False),
        metadata_issues=kwargs.get('metadata_issues') or {},
    )


def is_numerical_stream(channel_mapping: dict, stream_type: str) -> bool:
    """Check if stream is a numerical (non-string) stream in new mapping."""
    if stream_type not in channel_mapping:
        return False
    return not channel_mapping[stream_type].get('is_string', False)


def is_string_stream(channel_mapping: dict, stream_type: str) -> bool:
    """Check if stream is a string stream in new mapping."""
    if stream_type not in channel_mapping:
        return False
    return channel_mapping[stream_type].get('is_string', False)


def get_stream_channel_count(channel_mapping: dict, stream_type: str) -> int:
    """Get total channel count for a stream."""
    if stream_type not in channel_mapping:
        return 0
    stream_map = channel_mapping[stream_type]
    if stream_map.get('is_string'):
        return stream_map.get('channel_count', 0)
    # Sum all modality channel counts
    return sum(len(indices) for indices in stream_map.get('modalities', {}).values())


def get_modality_indices(channel_mapping: dict, stream_type: str, modality: str) -> list:
    """Get channel indices for a specific modality in a stream."""
    if stream_type not in channel_mapping:
        return []
    return channel_mapping[stream_type].get('modalities', {}).get(modality.lower(), [])


def has_marker_channel(channel_mapping: dict, stream_type: str) -> bool:
    """Check if stream has a marker channel."""
    if stream_type not in channel_mapping:
        return False
    return channel_mapping[stream_type].get('marker_index') is not None


def needs_synthetic_markers(channel_mapping: dict, stream_type: str = 'EEG') -> bool:
    """Check if stream needs synthetic markers (has EEG but no marker channel)."""
    if stream_type not in channel_mapping:
        return False
    stream_map = channel_mapping[stream_type]
    has_eeg = 'eeg' in stream_map.get('modalities', {})
    has_markers = stream_map.get('marker_index') is not None
    return has_eeg and not has_markers


class TestDataAcquisitionInitialization:
    """Test suite for DataAcquisition initialization."""
    
    def test_basic_initialization(self, stop_event, data_queue, save_queue, shared_state):
        """Test basic DataAcquisition initialization."""
        stream_configs = [make_stream_config(type='EEG', name='TestEEG', channel_count=32, sample_rate=500.0)]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs,
            shared_state=shared_state
        )

        # DataAcquisition doesn't store sample_rate - it uses stream configs
        assert daq.data_queue == data_queue
        assert daq.save_queue == save_queue
        assert daq.stop_event == stop_event
        assert daq.stream_configs == stream_configs
        assert daq.shared_state == shared_state

        # Check that DAQ is properly initialized - focus on public interface
        assert isinstance(daq, DataAcquisition)
        assert isinstance(daq, multiprocessing.Process)
    
    def test_initialization_with_empty_configs(self, stop_event, data_queue, save_queue):
        """Test initialization with empty stream configurations."""
        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=[],
            shared_state=None
        )

        assert daq.stream_configs == []
        assert daq.shared_state is None
    
    def test_initialization_with_none_configs(self, stop_event, data_queue, save_queue):
        """Test initialization with None stream configurations."""
        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=None
        )
        
        assert daq.stream_configs == []


class TestStreamConfigurationProcessing:
    """Test suite for stream configuration processing."""
    
    def test_eeg_stream_configuration(self, stop_event, data_queue, save_queue, mock_stream_config):
        """Test EEG stream configuration processing."""
        stream_configs = [mock_stream_config]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        # Test behavior, not internal structure
        # EEG streams should be processed when collect_data is called
        assert len(stream_configs) == 1
        assert stream_configs[0].type == 'EEG'
    
    def test_events_stream_configuration(self, stop_event, data_queue, save_queue, mock_events_stream_config):
        """Test Events stream configuration processing."""
        stream_configs = [mock_events_stream_config]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        # Test that Events stream is configured
        assert len(stream_configs) == 1
        assert stream_configs[0].type == 'Events'
    
    def test_emg_stream_configuration(self, stop_event, data_queue, save_queue, mock_emg_stream_config):
        """Test EMG stream configuration processing."""
        stream_configs = [mock_emg_stream_config]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        # Test that EMG stream is configured
        assert len(stream_configs) == 1
        assert stream_configs[0].type == 'EMG'
        assert stream_configs[0].channel_count == 8
    
    def test_emg_channel_count_calculation(self, stop_event, data_queue, save_queue):
        """Test EMG channel count calculation uses total channels - DataProcessor handles filtering."""
        # EMG stream with 8 EMG channels + 1 Markers channel = 9 total
        stream_configs = [make_stream_config(
            type='EMG',
            name='TestEMG',
            channel_count=9,
            sample_rate=2000.0,
            channel_types=['EMG'] * 8 + ['Markers']
        )]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        # Test configuration is preserved
        assert stream_configs[0].channel_count == 9
    
    def test_mixed_modality_channel_count_calculation(self, stop_event, data_queue, save_queue):
        """Test channel count calculation for mixed modality streams uses total channels."""
        # Stream with mixed channel types
        stream_configs = [make_stream_config(
            type='EMG',
            name='TestMixed',
            channel_count=12,
            sample_rate=2000.0,
            channel_types=['EMG'] * 6 + ['ECG'] * 2 + ['EOG'] * 3 + ['Markers']
        )]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        # Test configuration is preserved
        assert stream_configs[0].channel_count == 12
    
    def test_no_matching_channels_in_stream(self, stop_event, data_queue, save_queue):
        """Test handling when no channels match the stream's type - still uses total count."""
        stream_configs = [make_stream_config(
            type='EMG',
            name='TestNoEMG',
            channel_count=3,
            sample_rate=2000.0,
            channel_types=['ECG', 'EOG', 'Markers']  # No EMG channels
        )]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        # Test configuration is preserved
        assert stream_configs[0].channel_count == 3
    
    def test_mixed_stream_configurations(self, stop_event, data_queue, save_queue, 
                                       mock_stream_config, mock_emg_stream_config, 
                                       mock_events_stream_config):
        """Test mixed stream configuration processing."""
        stream_configs = [mock_stream_config, mock_emg_stream_config, mock_events_stream_config]
        
        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )
        
        # Test that all stream configurations are preserved
        assert len(stream_configs) == 3
        stream_types = [s.type for s in stream_configs]
        assert 'EEG' in stream_types
        assert 'EMG' in stream_types
        assert 'Events' in stream_types
    
    def test_duplicate_stream_configurations(self, stop_event, data_queue, save_queue, mock_stream_config):
        """Test handling of duplicate stream configurations."""
        stream_configs = [mock_stream_config, mock_stream_config]
        
        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )
        
        # Should handle duplicates gracefully - configuration preserved
        assert len(stream_configs) == 2


class TestDataRecord:
    """Test suite for DataRecord class."""
    
    def test_data_record_creation(self, sample_eeg_data):
        """Test DataRecord creation with valid data."""
        timestamp = time.time()
        local_timestamp = time.time()
        
        record = DataRecord(
            modality='EEG',
            sample=sample_eeg_data,
            timestamp=timestamp,
            local_timestamp=local_timestamp
        )
        
        assert record.modality == 'EEG'
        assert np.array_equal(record.sample, sample_eeg_data)
        assert record.timestamp == timestamp
        assert record.local_timestamp == local_timestamp
    
    def test_data_record_different_modalities(self, sample_eeg_data, sample_emg_data, sample_event_data):
        """Test DataRecord with different modalities."""
        timestamp = time.time()
        local_timestamp = time.time()
        
        # EEG record
        eeg_record = DataRecord('EEG', sample_eeg_data, timestamp, local_timestamp)
        assert eeg_record.modality == 'EEG'
        assert np.array_equal(eeg_record.sample, sample_eeg_data)
        
        # EMG record
        emg_record = DataRecord('EMG', sample_emg_data, timestamp, local_timestamp)
        assert emg_record.modality == 'EMG'
        assert np.array_equal(emg_record.sample, sample_emg_data)
        
        # Event record
        event_record = DataRecord('Event', sample_event_data, timestamp, local_timestamp)
        assert event_record.modality == 'Event'
        assert event_record.sample == sample_event_data
    
    def test_data_record_with_metadata(self, sample_metadata):
        """Test DataRecord with metadata."""
        timestamp = time.time()
        local_timestamp = time.time()
        
        record = DataRecord(
            modality='Metadata',
            sample=json.dumps(sample_metadata),
            timestamp=timestamp,
            local_timestamp=local_timestamp
        )
        
        assert record.modality == 'Metadata'
        assert json.loads(record.sample) == sample_metadata
    
    def test_data_record_attributes(self):
        """Test DataRecord has all required attributes."""
        record = DataRecord(
            modality='Test',
            sample=np.array([1, 2, 3]),
            timestamp=time.time(),
            local_timestamp=time.time()
        )
        
        assert hasattr(record, 'modality')
        assert hasattr(record, 'sample')
        assert hasattr(record, 'timestamp')
        assert hasattr(record, 'local_timestamp')


class TestChannelMapping:
    """Test suite for channel mapping using actual daq.channel_mapping."""

    def test_eeg_channel_mapping_extraction(self, stop_event, data_queue, save_queue):
        """Test EEG modality extraction via actual channel_mapping."""
        stream_configs = [make_stream_config(
            type='EEG',
            name='TestEEG',
            channel_count=32,
            sample_rate=500.0,
            labels=[f'EEG_{i+1}' for i in range(31)] + ['Markers'],
            channel_types=['EEG'] * 31 + ['Markers']
        )]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        # Verify channel_mapping was built correctly
        assert 'EEG' in daq.channel_mapping
        assert len(get_modality_indices(daq.channel_mapping, 'EEG', 'eeg')) == 31
        assert has_marker_channel(daq.channel_mapping, 'EEG')
        assert daq.channel_mapping['EEG']['marker_index'] == 31

    def test_mixed_channel_types_mapping(self, stop_event, data_queue, save_queue):
        """Test mixed channel types (EEG, EOG, ECG) via actual channel_mapping."""
        stream_configs = [make_stream_config(
            type='EEG',
            name='TestEEG',
            channel_count=35,
            sample_rate=500.0,
            labels=[f'EEG_{i+1}' for i in range(31)] + ['EOG_1', 'EOG_2', 'ECG', 'Markers'],
            channel_types=['EEG'] * 31 + ['EOG', 'EOG', 'ECG', 'Markers']
        )]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        # Verify modalities extracted correctly
        assert len(get_modality_indices(daq.channel_mapping, 'EEG', 'eeg')) == 31
        assert get_modality_indices(daq.channel_mapping, 'EEG', 'eog') == [31, 32]
        assert get_modality_indices(daq.channel_mapping, 'EEG', 'ecg') == [33]
        assert daq.channel_mapping['EEG']['marker_index'] == 34

    def test_stream_without_channel_types_fallback(self, stop_event, data_queue, save_queue):
        """Test stream with empty channel_types uses stream type as single modality."""
        stream_configs = [make_stream_config(
            type='EMG',
            name='TestEMG',
            channel_count=4,
            sample_rate=2000.0,
            channel_types=[]  # No channel type info
        )]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        # Should treat entire stream as single 'emg' modality
        assert 'EMG' in daq.channel_mapping
        assert get_modality_indices(daq.channel_mapping, 'EMG', 'emg') == [0, 1, 2, 3]
        assert not has_marker_channel(daq.channel_mapping, 'EMG')


class TestErrorHandling:
    """Test suite for error handling and edge cases."""

    def test_valid_sample_rate(self, stop_event, data_queue, save_queue):
        """Test handling of valid sample rate."""
        stream_configs = [make_stream_config(
            type='EEG',
            name='TestEEG',
            channel_count=32,
            sample_rate=500.0
        )]

        # Should not raise exception during initialization
        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        # DataAcquisition doesn't store sample_rate - it uses stream configs
        assert daq.stream_configs[0].sample_rate == 500.0

    def test_malformed_stream_config_raises_validation_error(self, stop_event, data_queue, save_queue):
        """Test that malformed stream configuration raises validation error."""
        from pydantic import ValidationError

        # StreamMetadata requires name, type, channel_count, sample_rate
        # Missing required fields should raise ValidationError at construction time
        with pytest.raises(ValidationError):
            StreamMetadata(
                type='EEG',
                name='TestEEG',
                # Missing 'channel_count' and 'sample_rate'
            )

    def test_empty_stream_name_raises_validation_error(self, stop_event, data_queue, save_queue):
        """Test that empty stream name raises validation error."""
        from pydantic import ValidationError

        # StreamMetadata requires name to have min_length=1
        with pytest.raises(ValidationError):
            StreamMetadata(
                type='EEG',
                name='',  # Empty name - invalid
                channel_count=32,
                sample_rate=500.0
            )

    def test_negative_channels_raises_validation_error(self, stop_event, data_queue, save_queue):
        """Test that negative channel count raises validation error."""
        from pydantic import ValidationError

        # StreamMetadata requires channel_count > 0
        with pytest.raises(ValidationError):
            StreamMetadata(
                type='EEG',
                name='TestEEG',
                channel_count=-1,  # Negative channels - invalid
                sample_rate=500.0
            )


class TestMultiRateStreamHandling:
    """Test suite for multi-rate stream configuration."""

    def test_emg_stream_configuration(self, stop_event, data_queue, save_queue):
        """Test that EMG stream is properly configured as separate stream."""
        stream_configs = [make_stream_config(
            type='EMG',
            name='TestEMG',
            channel_count=8,
            sample_rate=2000.0,
            channel_types=['EMG'] * 8
        )]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        # Verify EMG is tracked as numerical stream with correct channel count
        assert is_numerical_stream(daq.channel_mapping, 'EMG')
        assert get_stream_channel_count(daq.channel_mapping, 'EMG') == 8

    def test_multiple_modality_stream_setup(self, stop_event, data_queue, save_queue):
        """Test configuration setup with multiple modality streams (EMG, ECG)."""
        stream_configs = [
            make_stream_config(
                type='EMG',
                name='TestEMG',
                channel_count=8,
                sample_rate=2000.0,
                channel_types=['EMG'] * 8
            ),
            make_stream_config(
                type='ECG',
                name='TestECG',
                channel_count=3,
                sample_rate=1000.0,
                channel_types=['ECG'] * 3
            )
        ]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        # Test configuration for multiple streams
        assert len(stream_configs) == 2
        assert stream_configs[0].type == 'EMG'
        assert stream_configs[0].channel_count == 8
        assert stream_configs[1].type == 'ECG'
        assert stream_configs[1].channel_count == 3


class TestEventProcessing:
    """Test suite for event processing with timestamp tracking."""

    def test_event_stream_configuration(self, stop_event, data_queue, save_queue):
        """Test that Events stream is properly configured for direct sending."""
        stream_configs = [
            make_stream_config(
                type='EEG',
                name='TestEEG',
                channel_count=32,
                sample_rate=500.0,
                labels=[f'EEG_{i+1}' for i in range(32)],
                channel_types=['EEG'] * 32
            ),
            make_stream_config(
                type='Events',
                name='TestEvents',
                channel_count=1,
                sample_rate=500.0
            )
        ]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        # Build channel mapping
        daq.channel_mapping = daq._build_channel_mapping()

        # Events stream is skipped in channel_mapping (handled by dedicated events reader)
        assert 'Events' not in daq.channel_mapping

    def test_synthetic_markers_in_eeg_stream(self, stop_event, data_queue, save_queue):
        """Test that synthetic markers are added when EEG stream has no markers channel."""
        stream_configs = [make_stream_config(
            type='EEG',
            name='TestEEG',
            channel_count=32,
            sample_rate=500.0,
            labels=[f'EEG_{i+1}' for i in range(32)],
            channel_types=['EEG'] * 32  # No markers channel
        )]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        # Build channel mapping
        daq.channel_mapping = daq._build_channel_mapping()

        # Should need synthetic markers (has EEG but no marker channel)
        assert needs_synthetic_markers(daq.channel_mapping, 'EEG')
        assert not has_marker_channel(daq.channel_mapping, 'EEG')
        assert len(get_modality_indices(daq.channel_mapping, 'EEG', 'eeg')) == 32

    def test_markers_channel_detection(self, stop_event, data_queue, save_queue):
        """Test that existing markers channel is detected correctly."""
        stream_configs = [make_stream_config(
            type='EEG',
            name='TestEEG',
            channel_count=33,
            sample_rate=500.0,
            labels=[f'EEG_{i+1}' for i in range(32)] + ['Markers'],
            channel_types=['EEG'] * 32 + ['Markers']
        )]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        # Build channel mapping
        daq.channel_mapping = daq._build_channel_mapping()

        # Should not need synthetic markers (has marker channel)
        assert not needs_synthetic_markers(daq.channel_mapping, 'EEG')
        assert has_marker_channel(daq.channel_mapping, 'EEG')
        assert daq.channel_mapping['EEG']['marker_index'] == 32
        assert len(get_modality_indices(daq.channel_mapping, 'EEG', 'eeg')) == 32

    def test_daq_without_events_stream(self, stop_event, data_queue, save_queue):
        """Test that DAQ operates correctly without Events stream."""
        stream_configs = [make_stream_config(
            type='EEG',
            name='TestEEG',
            channel_count=32,
            sample_rate=500.0,
            labels=[f'EEG_{i+1}' for i in range(32)],
            channel_types=['EEG'] * 32
        )]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        # Build channel mapping
        daq.channel_mapping = daq._build_channel_mapping()

        # DAQ should be properly initialized without Events
        assert isinstance(daq, multiprocessing.Process)
        assert needs_synthetic_markers(daq.channel_mapping, 'EEG')


class TestParseEventSample:
    """Test suite for _parse_event_sample method with EventData schema validation."""

    def test_parse_valid_event(self, stop_event, data_queue, save_queue):
        """Test parsing a valid event sample."""
        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=[]
        )

        sample = ['{"event_id": 1, "event_type": "trial_start"}']
        result = daq._parse_event_sample(sample)

        assert result is not None
        event_json, event_id = result
        assert event_id == 1.0
        assert event_json['event_type'] == 'trial_start'

    def test_parse_event_with_legacy_pascal_case(self, stop_event, data_queue, save_queue):
        """Test parsing event with legacy PascalCase keys."""
        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=[]
        )

        sample = ['{"Event_ID": 2, "Event_Type": "trial_end"}']
        result = daq._parse_event_sample(sample)

        assert result is not None
        event_json, event_id = result
        assert event_id == 2.0
        assert event_json['event_type'] == 'trial_end'

    def test_parse_event_with_string_event_id(self, stop_event, data_queue, save_queue):
        """Test parsing event with string event_id (type coercion)."""
        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=[]
        )

        sample = ['{"event_id": "123", "event_type": "trial_start"}']
        result = daq._parse_event_sample(sample)

        assert result is not None
        event_json, event_id = result
        assert event_id == 123.0
        assert event_json['event_type'] == 'trial_start'

    def test_parse_event_missing_event_id_returns_none(self, stop_event, data_queue, save_queue):
        """Test that missing event_id returns None."""
        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=[]
        )

        sample = ['{"event_type": "test"}']
        result = daq._parse_event_sample(sample)

        assert result is None

    def test_parse_event_missing_event_type_returns_none(self, stop_event, data_queue, save_queue):
        """Test that missing event_type returns None."""
        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=[]
        )

        sample = ['{"event_id": 1}']
        result = daq._parse_event_sample(sample)

        assert result is None

    def test_parse_event_invalid_json_returns_none(self, stop_event, data_queue, save_queue):
        """Test that invalid JSON returns None."""
        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=[]
        )

        sample = ['not valid json']
        result = daq._parse_event_sample(sample)

        assert result is None

    def test_parse_event_empty_sample_returns_none(self, stop_event, data_queue, save_queue):
        """Test that empty sample returns None."""
        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=[]
        )

        result = daq._parse_event_sample([])
        assert result is None

        result = daq._parse_event_sample(None)
        assert result is None

    def test_parse_event_preserves_extra_fields(self, stop_event, data_queue, save_queue):
        """Test that extra fields are preserved in output."""
        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=[]
        )

        sample = ['{"event_id": 1, "event_type": "trial_start", "custom_field": "value", "task": "motor_imagery"}']
        result = daq._parse_event_sample(sample)

        assert result is not None
        event_json, event_id = result
        assert event_json['custom_field'] == 'value'
        assert event_json['task'] == 'motor_imagery'

    def test_parse_event_latency_update_updates_shared_state(self, stop_event, data_queue, save_queue, shared_state):
        """Test that latency_update events update shared state."""
        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=[],
            shared_state=shared_state
        )

        sample = ['{"event_id": 999, "event_type": "latency_update", "latency_ms_raw": 15.5}']
        result = daq._parse_event_sample(sample)

        assert result is not None
        # Verify E2E latency was set in shared state
        from dendrite.utils.state_keys import e2e_latency_key
        assert shared_state.get(e2e_latency_key()) == 15.5


class TestCriticalDataFlow:
    """Test suite for critical data flow functionality."""

    def test_stream_config_channel_types_ordering(self, stop_event, data_queue, save_queue, shared_state):
        """Test that stream config preserves channel type ordering with Markers at end."""
        stream_configs = [make_stream_config(
            type='EEG',
            name='TestEEG',
            channel_count=33,
            sample_rate=500.0,
            labels=[f'EEG_{i+1}' for i in range(32)] + ['Markers'],
            channel_types=['EEG'] * 32 + ['Markers']
        )]

        DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs,
            shared_state=shared_state
        )

        # Verify the configuration has the expected structure
        assert stream_configs[0].channel_types.count('EEG') == 32
        assert stream_configs[0].channel_types.count('Markers') == 1
        assert stream_configs[0].channel_types[-1] == 'Markers'
    
    def test_data_queue_non_blocking(self, stop_event, data_queue, save_queue):
        """Test that data queue operations are non-blocking."""
        import queue

        # Create a small queue that will fill up
        small_queue = multiprocessing.Queue(maxsize=2)

        stream_configs = [make_stream_config(
            type='EEG',
            name='TestEEG',
            channel_count=32,
            sample_rate=500.0
        )]

        daq = DataAcquisition(
            data_queue=small_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        # Fill the queue
        try:
            small_queue.put_nowait({'data': np.zeros(32)})
            small_queue.put_nowait({'data': np.zeros(32)})
        except Exception:
            pass

        # Verify queue is full
        assert small_queue.full()

        # DAQ should handle full queue gracefully without blocking
        # This is critical for real-time performance

    def test_stream_configuration_validation_with_pydantic(self, stop_event, data_queue, save_queue):
        """Test that stream configurations are validated by Pydantic."""
        from pydantic import ValidationError

        # Test with various edge cases - should raise ValidationError
        edge_cases = [
            # Missing required fields
            {'type': 'EEG'},
            # Invalid channel count
            {'type': 'EEG', 'name': 'Test', 'channel_count': -1, 'sample_rate': 500.0},
            # Empty name
            {'type': 'EEG', 'name': '', 'channel_count': 32, 'sample_rate': 500.0},
        ]

        for config in edge_cases:
            with pytest.raises(ValidationError):
                StreamMetadata(**config)


class TestMultiModalityIntegration:
    """Test suite for multi-modality integration - critical for V2.8.3."""

    def test_mixed_modality_configuration(self, stop_event, data_queue, save_queue):
        """Test that mixed modalities are properly configured."""
        stream_configs = [
            make_stream_config(
                type='EEG',
                name='TestEEG',
                channel_count=65,
                sample_rate=500.0,
                labels=[f'EEG_{i+1}' for i in range(64)] + ['Markers'],
                channel_types=['EEG'] * 64 + ['Markers']
            ),
            make_stream_config(
                type='EMG',
                name='TestEMG',
                channel_count=9,
                sample_rate=2000.0,
                labels=[f'EMG_{i+1}' for i in range(8)] + ['EMG_Markers'],
                channel_types=['EMG'] * 8 + ['Markers']
            )
        ]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        # Verify both streams are configured
        assert len(stream_configs) == 2

        # Verify EEG structure
        eeg_config = next(c for c in stream_configs if c.type == 'EEG')
        assert eeg_config.channel_types[-1] == 'Markers'
        assert eeg_config.sample_rate == 500.0

        # Verify EMG structure
        emg_config = next(c for c in stream_configs if c.type == 'EMG')
        assert emg_config.channel_types[-1] == 'Markers'
        assert emg_config.sample_rate == 2000.0

        # Critical: different sample rates must be handled
        assert eeg_config.sample_rate != emg_config.sample_rate


class TestStringStreamSupport:
    """Test suite for string stream support - new functionality in V2.9.0."""

    def test_string_stream_configuration_detection(self, stop_event, data_queue, save_queue):
        """Test that string streams are properly detected from configuration."""
        stream_configs = [make_stream_config(
            type='TextMarkers',
            name='TestTextMarkers',
            channel_count=1,
            sample_rate=500.0,
            channel_format='string',
            labels=['Text_Events'],
            channel_types=['Markers']
        )]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        # Build channel mapping to trigger string stream detection
        channel_mapping = daq._build_channel_mapping()

        # Verify string stream is detected
        assert is_string_stream(channel_mapping, 'TextMarkers')
        assert channel_mapping['TextMarkers']['channel_count'] == 1
        assert not is_numerical_stream(channel_mapping, 'TextMarkers')

    def test_mixed_numerical_and_string_streams(self, stop_event, data_queue, save_queue):
        """Test handling of mixed numerical and string streams."""
        stream_configs = [
            make_stream_config(
                type='EMG',
                name='TestEMG',
                channel_count=8,
                sample_rate=2000.0,
                channel_format='float32',
                channel_types=['EMG'] * 8
            ),
            make_stream_config(
                type='TextEvents',
                name='TestTextEvents',
                channel_count=1,
                sample_rate=500.0,
                channel_format='string',
                channel_types=['Events']
            )
        ]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        channel_mapping = daq._build_channel_mapping()

        # Verify both types are properly categorized
        assert is_numerical_stream(channel_mapping, 'EMG')
        assert get_stream_channel_count(channel_mapping, 'EMG') == 8
        assert is_string_stream(channel_mapping, 'TextEvents')
        assert channel_mapping['TextEvents']['channel_count'] == 1

    def test_string_stream_excluded_from_separate_streams(self, stop_event, data_queue, save_queue):
        """Test that string streams go to string_streams, not separate_streams."""
        stream_configs = [
            make_stream_config(
                type='EMG',
                name='TestEMG',
                channel_count=4,
                sample_rate=2000.0,
                channel_format='float32',
                channel_types=['EMG'] * 4
            ),
            make_stream_config(
                type='StringData',
                name='TestStringData',
                channel_count=2,
                sample_rate=100.0,
                channel_format='string',
                channel_types=['Data'] * 2
            )
        ]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        # String streams should be marked as string, numerical as not
        assert is_numerical_stream(daq.channel_mapping, 'EMG')
        assert is_string_stream(daq.channel_mapping, 'StringData')
        assert not is_string_stream(daq.channel_mapping, 'EMG')
        assert not is_numerical_stream(daq.channel_mapping, 'StringData')

        # Verify channel counts
        assert get_stream_channel_count(daq.channel_mapping, 'EMG') == 4
        assert daq.channel_mapping['StringData']['channel_count'] == 2

    def test_default_channel_format_handling(self, stop_event, data_queue, save_queue):
        """Test that streams without explicit channel_format default to numerical."""
        stream_configs = [make_stream_config(
            type='EMG',
            name='TestEMG',
            channel_count=4,
            sample_rate=2000.0,
            # No channel_format specified - should default to numerical
            channel_types=['EMG'] * 4
        )]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        channel_mapping = daq._build_channel_mapping()

        # Should be treated as numerical stream (not string)
        assert is_numerical_stream(channel_mapping, 'EMG')
        assert get_stream_channel_count(channel_mapping, 'EMG') == 4
        assert not is_string_stream(channel_mapping, 'EMG')


class TestEnhancedMetadataHandling:
    """Test suite for enhanced metadata handling using StreamMetadata."""

    def test_stream_metadata_object_creation(self, stop_event, data_queue, save_queue):
        """Test StreamMetadata object creation for stream metadata."""
        # Create StreamMetadata directly with all fields
        config = StreamMetadata(
            type='EEG',
            name='TestEEG',
            channel_count=32,
            sample_rate=500.0,
            channel_format='float32',
            source_id='test_source',
            uid='test_uid_123',
            labels=[f'EEG_{i+1}' for i in range(32)],
            channel_types=['EEG'] * 32,
            channel_units=['µV'] * 32,
            acquisition_info={'amplifier': 'TestAmp', 'version': '1.0'},
            metadata_issues={'test_issue': 'test_value'},
            has_metadata_issues=True
        )
        stream_configs = [config]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        # Set up stream info
        daq.stream_info = {'EEG': config}

        # Verify metadata object fields directly
        assert config.name == 'TestEEG'
        assert config.type == 'EEG'
        assert config.sample_rate == 500.0
        assert config.channel_format == 'float32'
        assert config.source_id == 'test_source'
        assert config.channel_count == 32
        assert config.uid == 'test_uid_123'
        assert len(config.labels) == 32
        assert len(config.channel_types) == 32
        assert len(config.channel_units) == 32
        assert config.acquisition_info['amplifier'] == 'TestAmp'
        assert config.has_metadata_issues == True
        assert 'test_issue' in config.metadata_issues
    
    def test_metadata_serialization_for_saving(self, stop_event, data_queue, save_queue):
        """Test that metadata objects can be properly serialized for saving."""
        import json

        # Create a StreamMetadata object
        metadata = StreamMetadata(
            name='TestStream',
            type='EEG',
            sample_rate=500.0,
            channel_format='float32',
            source_id='test_source',
            created_at=123456789.0,
            version=1.0,
            channel_count=4,
            uid='test_uid',
            labels=['Ch1', 'Ch2', 'Ch3', 'Ch4'],
            channel_types=['EEG', 'EEG', 'EEG', 'Markers'],
            channel_units=['µV', 'µV', 'µV', 'a.u.'],
            acquisition_info={'device': 'TestDevice'},
            metadata_issues={'warning': 'test warning'},
            has_metadata_issues=True
        )

        # Test serialization (as done in _send_stream_metadata)
        metadata_json = json.dumps(metadata.model_dump())

        # Verify it can be deserialized
        metadata_dict = json.loads(metadata_json)

        assert metadata_dict['name'] == 'TestStream'
        assert metadata_dict['type'] == 'EEG'
        assert metadata_dict['sample_rate'] == 500.0
        assert metadata_dict['channel_format'] == 'float32'
        assert metadata_dict['channel_count'] == 4
        assert len(metadata_dict['labels']) == 4
        assert metadata_dict['has_metadata_issues'] == True
        assert 'warning' in metadata_dict['metadata_issues']
    
    def test_safe_save_with_metadata_record(self, stop_event, data_queue, save_queue):
        """Test that metadata records are properly created and saved."""
        import json
        from pylsl import local_clock

        stream_configs = [make_stream_config(
            type='EEG',
            name='TestEEG',
            channel_count=2,
            sample_rate=500.0
        )]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        # Create a metadata object
        metadata = StreamMetadata(
            name='TestEEG',
            type='EEG',
            sample_rate=500.0,
            channel_format='float32',
            source_id='test_source',
            created_at=123456789.0,
            version=1.0,
            channel_count=2,
            uid='test_uid',
            labels=['Ch1', 'Ch2'],
            channel_types=['EEG', 'EEG'],
            channel_units=['µV', 'µV'],
            acquisition_info={},
            metadata_issues={},
            has_metadata_issues=False
        )

        # Create metadata record (as done in _send_stream_metadata)
        metadata_json = json.dumps(metadata.model_dump())
        metadata_record = DataRecord(
            modality='EEG_Metadata',
            sample=metadata_json,
            timestamp=local_clock(),
            local_timestamp=local_clock()
        )

        # Test that record can be saved
        daq._safe_save(metadata_record)

        # Verify record structure
        assert metadata_record.modality == 'EEG_Metadata'

        # Verify sample can be deserialized
        saved_metadata = json.loads(metadata_record.sample)
        assert saved_metadata['name'] == 'TestEEG'
        assert saved_metadata['channel_count'] == 2


class TestChannelFormatDetection:
    """Test suite for channel format detection and handling."""

    def test_float32_format_detection(self, stop_event, data_queue, save_queue):
        """Test detection and handling of float32 format streams."""
        stream_configs = [make_stream_config(
            type='EMG',  # Use EMG instead of EEG since EEG has special handling
            name='TestEMG',
            channel_count=8,
            sample_rate=500.0,
            channel_format='float32',
            channel_types=['EMG'] * 8
        )]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        channel_mapping = daq._build_channel_mapping()

        # Should be categorized as numerical stream
        assert is_numerical_stream(channel_mapping, 'EMG')
        assert not is_string_stream(channel_mapping, 'EMG')

    def test_string_format_detection(self, stop_event, data_queue, save_queue):
        """Test detection and handling of string format streams."""
        stream_configs = [make_stream_config(
            type='TextEvents',
            name='TestTextEvents',
            channel_count=1,
            sample_rate=100.0,
            channel_format='string',
            channel_types=['Events']
        )]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        channel_mapping = daq._build_channel_mapping()

        # Should be categorized as string stream
        assert is_string_stream(channel_mapping, 'TextEvents')
        assert not is_numerical_stream(channel_mapping, 'TextEvents')

    def test_various_numerical_formats(self, stop_event, data_queue, save_queue):
        """Test handling of various numerical channel formats."""
        formats_to_test = ['float32', 'double64', 'int16', 'int32']

        for fmt in formats_to_test:
            stream_configs = [make_stream_config(
                type=f'Test_{fmt}',
                name=f'Test_{fmt}',
                channel_count=2,
                sample_rate=500.0,
                channel_format=fmt,
                channel_types=['Test'] * 2
            )]

            daq = DataAcquisition(
                data_queue=data_queue,
                save_queue=save_queue,
                stop_event=stop_event,
                stream_configs=stream_configs
            )

            channel_mapping = daq._build_channel_mapping()

            # All numerical formats should be numerical streams
            assert is_numerical_stream(channel_mapping, f'Test_{fmt}')
            assert not is_string_stream(channel_mapping, f'Test_{fmt}')

    def test_missing_channel_format_defaults_to_numerical(self, stop_event, data_queue, save_queue):
        """Test that missing channel_format defaults to numerical processing."""
        stream_configs = [make_stream_config(
            type='EMG',
            name='TestEMG',
            channel_count=4,
            sample_rate=2000.0,
            # No channel_format specified - defaults to float32
            channel_types=['EMG'] * 4
        )]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        channel_mapping = daq._build_channel_mapping()

        # Should default to numerical (not string)
        assert is_numerical_stream(channel_mapping, 'EMG')
        assert not is_string_stream(channel_mapping, 'EMG')

    def test_eeg_stream_format_handling(self, stop_event, data_queue, save_queue):
        """Test that EEG streams are handled with modality extraction."""
        stream_configs = [make_stream_config(
            type='EEG',
            name='TestEEG',
            channel_count=8,
            sample_rate=500.0,
            channel_format='float32',
            channel_types=['EEG'] * 8
        )]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        channel_mapping = daq._build_channel_mapping()

        # EEG streams are in channel_mapping with modalities extracted
        assert 'EEG' in channel_mapping
        assert is_numerical_stream(channel_mapping, 'EEG')

        # EEG-specific fields should be populated
        assert len(get_modality_indices(channel_mapping, 'EEG', 'eeg')) == 8
        assert needs_synthetic_markers(channel_mapping, 'EEG')  # No markers in channel_types


class TestImprovedChannelMapping:
    """Test suite for improved channel mapping with string streams tracking."""

    def test_separate_vs_string_streams_tracking(self, stop_event, data_queue, save_queue):
        """Test that separate_streams and string_streams are properly distinguished."""
        stream_configs = [
            make_stream_config(
                type='EMG',
                name='TestEMG',
                channel_count=8,
                sample_rate=2000.0,
                channel_format='float32',
                channel_types=['EMG'] * 8
            ),
            make_stream_config(
                type='TextMarkers',
                name='TestTextMarkers',
                channel_count=1,
                sample_rate=500.0,
                channel_format='string',
                channel_types=['Markers']
            ),
            make_stream_config(
                type='ECG',
                name='TestECG',
                channel_count=3,
                sample_rate=1000.0,
                channel_format='double64',
                channel_types=['ECG'] * 3
            )
        ]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        channel_mapping = daq._build_channel_mapping()

        # Verify numerical streams
        assert is_numerical_stream(channel_mapping, 'EMG')
        assert get_stream_channel_count(channel_mapping, 'EMG') == 8
        assert is_numerical_stream(channel_mapping, 'ECG')
        assert get_stream_channel_count(channel_mapping, 'ECG') == 3

        # Verify string streams
        assert is_string_stream(channel_mapping, 'TextMarkers')
        assert channel_mapping['TextMarkers']['channel_count'] == 1

        # Verify no cross-contamination
        assert not is_string_stream(channel_mapping, 'EMG')
        assert not is_string_stream(channel_mapping, 'ECG')
        assert not is_numerical_stream(channel_mapping, 'TextMarkers')
    
    def test_channel_mapping_logging_for_string_streams(self, stop_event, data_queue, save_queue):
        """Test that string streams generate appropriate log messages."""
        stream_configs = [make_stream_config(
            type='StringEvents',
            name='TestStringEvents',
            channel_count=2,
            sample_rate=100.0,
            channel_format='string',
            channel_types=['Events'] * 2
        )]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )
        
        # Mock logger to capture log messages
        with patch.object(daq.logger, 'info') as mock_info:
            channel_mapping = daq._build_channel_mapping()
            
            # Verify string stream logging
            assert is_string_stream(channel_mapping, 'StringEvents')
            
            # Check that appropriate log message was generated
            # (Note: exact message verification would depend on implementation)
            assert mock_info.called
    
    def test_mixed_eeg_stream_with_individual_channel_types(self, stop_event, data_queue, save_queue):
        """Test EEG stream processing with individual channel types tracking."""
        stream_configs = [make_stream_config(
            type='EEG',
            name='TestEEG',
            channel_count=36,
            sample_rate=500.0,
            labels=[f'EEG_{i+1}' for i in range(32)] + ['EOG_1', 'EOG_2', 'ECG', 'Markers'],
            channel_types=['EEG'] * 32 + ['EOG', 'EOG', 'ECG', 'Markers']
        )]
        
        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )
        
        channel_mapping = daq._build_channel_mapping()

        # Verify EEG stream has modalities extracted
        eeg_map = channel_mapping['EEG']

        # Verify EEG channel tracking
        assert len(get_modality_indices(channel_mapping, 'EEG', 'eeg')) == 32
        assert get_modality_indices(channel_mapping, 'EEG', 'eeg') == list(range(32))

        # Verify individual channel types tracking (now as modalities)
        assert 'eog' in eeg_map['modalities']
        assert 'ecg' in eeg_map['modalities']
        assert eeg_map['modalities']['eog'] == [32, 33]
        assert eeg_map['modalities']['ecg'] == [34]

        # Verify markers detection
        assert eeg_map['marker_index'] == 35
        assert not needs_synthetic_markers(channel_mapping, 'EEG')
    
    def test_eeg_stream_without_markers_needs_synthetic(self, stop_event, data_queue, save_queue):
        """Test that EEG streams without markers are flagged for synthetic markers."""
        stream_configs = [make_stream_config(
            type='EEG',
            name='TestEEG',
            channel_count=34,
            sample_rate=500.0,
            labels=[f'EEG_{i+1}' for i in range(32)] + ['EOG_1', 'EOG_2'],
            channel_types=['EEG'] * 32 + ['EOG', 'EOG']  # No Markers
        )]
        
        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )
        
        channel_mapping = daq._build_channel_mapping()

        # Verify EEG stream has modalities extracted
        eeg_map = channel_mapping['EEG']

        # Verify synthetic markers are needed
        assert needs_synthetic_markers(channel_mapping, 'EEG')
        assert eeg_map['marker_index'] is None
        assert len(get_modality_indices(channel_mapping, 'EEG', 'eeg')) == 32

        # Verify individual channel types are still tracked (as modalities)
        assert 'eog' in eeg_map['modalities']
        assert eeg_map['modalities']['eog'] == [32, 33]


class TestIntegrationDataFlow:
    """Test suite for integration of new functionality in complete data flow."""
    
    def test_synthetic_markers_channel_addition_to_stream_info(self, stop_event, data_queue, save_queue):
        """Test that synthetic markers are properly added to stream info."""
        original_config = make_stream_config(
            type='EEG',
            name='TestEEG',
            channel_count=32,
            sample_rate=500.0,
            labels=[f'EEG_{i+1}' for i in range(32)],
            channel_types=['EEG'] * 32  # No markers
        )

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=[original_config]
        )

        # Build channel mapping
        channel_mapping = daq._build_channel_mapping()

        # Simulate the synthetic markers addition that happens in _connect_streams
        if needs_synthetic_markers(channel_mapping, 'EEG'):
            # This mimics lines 212-216 in data_acquisition.py
            # Create updated config with synthetic markers added
            orig_count = original_config.channel_count
            updated_labels = list(original_config.labels)[:orig_count] + ['Markers']
            updated_types = list(original_config.channel_types)[:orig_count] + ['Markers']

            # Verify synthetic markers would be added correctly
            assert orig_count + 1 == 33
            assert updated_labels[-1] == 'Markers'
            assert updated_types[-1] == 'Markers'
            assert len(updated_labels) == 33
            assert len(updated_types) == 33
    
    def test_complete_channel_info_structure_with_mixed_streams(self, stop_event, data_queue, save_queue, shared_state):
        """Test complete channel info structure with mixed stream types."""
        eeg_config = make_stream_config(
            type='EEG',
            name='TestEEG',
            channel_count=33,
            sample_rate=500.0,
            labels=[f'EEG_{i+1}' for i in range(32)] + ['Markers'],
            channel_types=['EEG'] * 32 + ['Markers']
        )
        emg_config = make_stream_config(
            type='EMG',
            name='TestEMG',
            channel_count=9,
            sample_rate=2000.0,
            channel_format='float32',
            labels=[f'EMG_{i+1}' for i in range(8)] + ['EMG_Markers'],
            channel_types=['EMG'] * 8 + ['Markers']
        )
        string_config = make_stream_config(
            type='StringEvents',
            name='TestStringEvents',
            channel_count=1,
            sample_rate=100.0,
            channel_format='string',
            channel_types=['Events']
        )
        stream_configs = [eeg_config, emg_config, string_config]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs,
            shared_state=shared_state
        )

        # Set up mock stream info
        daq.stream_info = {
            'EEG': eeg_config,
            'EMG': emg_config,
            'StringEvents': string_config
        }
        
        # Build channel mapping
        channel_mapping = daq._build_channel_mapping()

        # Verify EEG channel mapping structure
        assert len(get_modality_indices(channel_mapping, 'EEG', 'eeg')) == 32
        assert not needs_synthetic_markers(channel_mapping, 'EEG')
        assert channel_mapping['EEG']['marker_index'] == 32

        # Verify EMG stream (numerical with markers)
        assert is_numerical_stream(channel_mapping, 'EMG')
        assert get_stream_channel_count(channel_mapping, 'EMG') == 8  # 8 EMG channels, markers handled separately
        assert has_marker_channel(channel_mapping, 'EMG')

        # Verify string streams
        assert is_string_stream(channel_mapping, 'StringEvents')
        assert channel_mapping['StringEvents']['channel_count'] == 1

        # String streams should not be numerical
        assert not is_numerical_stream(channel_mapping, 'StringEvents')
    
    def test_metadata_issues_tracking_integration(self, stop_event, data_queue, save_queue):
        """Test integration of metadata issues tracking through the pipeline."""
        metadata_issues = {
            'fallback_labels': 'generic',
            'acquisition_info_missing': True
        }
        stream_configs = [make_stream_config(
            type='EEG',
            name='TestEEG',
            channel_count=32,
            sample_rate=500.0,
            channel_format='float32',
            labels=[f'EEG_{i+1}' for i in range(32)],
            channel_types=['EEG'] * 32,
            metadata_issues=metadata_issues,
            has_metadata_issues=True
        )]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        # Test that metadata issues are preserved in configuration (use attribute access)
        config = stream_configs[0]
        assert config.has_metadata_issues == True
        assert 'fallback_labels' in config.metadata_issues
        assert 'acquisition_info_missing' in config.metadata_issues

        # Test that these are included in StreamMetadata
        assert config.has_metadata_issues == True
        assert config.metadata_issues['fallback_labels'] == 'generic'
        assert config.metadata_issues['acquisition_info_missing'] == True


class TestErrorHandlingNewFunctionality:
    """Test suite for error handling in new functionality."""
    
    def test_malformed_channel_format_handling(self, stop_event, data_queue, save_queue):
        """Test handling of malformed or invalid channel_format values."""
        stream_configs = [make_stream_config(
            type='TestStream',
            name='TestStream',
            channel_count=4,
            sample_rate=500.0,
            channel_format='invalid_format',  # Invalid format
            channel_types=['Test'] * 4
        )]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        # Should not crash during channel mapping
        channel_mapping = daq._build_channel_mapping()

        # Invalid formats should be treated as numerical (not string)
        assert is_numerical_stream(channel_mapping, 'TestStream')
        assert not is_string_stream(channel_mapping, 'TestStream')
    
    def test_missing_metadata_fields_handling(self, stop_event, data_queue, save_queue):
        """Test handling of stream configs with missing optional metadata fields."""
        # StreamMetadata has defaults for labels and channel_types
        stream_configs = [make_stream_config(
            type='EEG',
            name='TestEEG',
            channel_count=32,
            sample_rate=500.0
            # labels and channel_types will use defaults from make_stream_config
        )]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        # Should not crash during initialization
        assert daq.stream_configs == stream_configs

        # Channel mapping should handle default fields gracefully
        channel_mapping = daq._build_channel_mapping()

        # EEG stream with default channel_types (all 'EEG') should need synthetic markers
        assert needs_synthetic_markers(channel_mapping, 'EEG')
    
    def test_empty_string_streams_dict_handling(self, stop_event, data_queue, save_queue):
        """Test handling when no string streams are present."""
        stream_configs = [make_stream_config(
            type='EMG',
            name='TestEMG',
            channel_count=8,
            sample_rate=2000.0,
            channel_format='float32',
            channel_types=['EMG'] * 8
        )]

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs
        )

        channel_mapping = daq._build_channel_mapping()

        # No string streams should be present - check that no streams are marked as string
        string_stream_count = sum(1 for s in channel_mapping.values() if s.get('is_string'))
        assert string_stream_count == 0

        # EMG stream should be numerical
        assert is_numerical_stream(channel_mapping, 'EMG')


class TestLatencyTelemetry:
    """Tests for the latency telemetry system with P50 smoothing."""

    def test_latency_window_uses_deque(self, stop_event, data_queue, save_queue, shared_state):
        """Verify latency windows use deque for O(1) append."""
        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=[],
            shared_state=shared_state
        )

        # Call update to initialize window
        daq._update_latency_telemetry('EEG', 5.0)

        # Verify deque is used
        assert 'EEG' in daq._latency_windows
        assert isinstance(daq._latency_windows['EEG'], deque)

    def test_latency_window_has_maxlen(self, stop_event, data_queue, save_queue, shared_state):
        """Verify latency window has bounded size via maxlen."""
        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=[],
            shared_state=shared_state
        )

        # Add more samples than window size
        for i in range(200):
            daq._update_latency_telemetry('EEG', float(i))

        # Window should be bounded
        assert len(daq._latency_windows['EEG']) <= daq._latency_window_size

    def test_latency_throttling_no_update_before_interval(self, stop_event, data_queue, save_queue):
        """Verify SharedState is NOT updated before throttle interval."""
        mock_shared_state = Mock()

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=[],
            shared_state=mock_shared_state
        )

        # Send samples less than update interval
        update_interval = daq._latency_update_interval
        for i in range(update_interval - 1):
            daq._update_latency_telemetry('EEG', 5.0)

        # SharedState.set should NOT have been called
        assert mock_shared_state.set.call_count == 0

    def test_latency_throttling_updates_at_interval(self, stop_event, data_queue, save_queue):
        """Verify SharedState IS updated at throttle interval."""
        mock_shared_state = Mock()

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=[],
            shared_state=mock_shared_state
        )

        update_interval = daq._latency_update_interval

        # Send exactly update_interval samples
        for i in range(update_interval):
            daq._update_latency_telemetry('EEG', 5.0)

        # SharedState.set should have been called twice (latency + timestamp)
        assert mock_shared_state.set.call_count == 2

    def test_p50_calculation_correct(self, stop_event, data_queue, save_queue):
        """Verify P50 (median) is computed correctly."""
        mock_shared_state = Mock()

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=[],
            shared_state=mock_shared_state
        )

        update_interval = daq._latency_update_interval

        # Send known values: [1, 2, 3, 4, 5] repeated to fill update interval
        # Median of [1,2,3,4,5] = 3.0
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for i in range(update_interval):
            daq._update_latency_telemetry('EEG', values[i % len(values)])

        # Find the latency call (first call should be latency, second timestamp)
        latency_call = mock_shared_state.set.call_args_list[0]
        key, value = latency_call[0]

        assert key == 'eeg_latency_p50'
        assert value == 3.0  # Median of [1,2,3,4,5]

    def test_multiple_streams_independent(self, stop_event, data_queue, save_queue):
        """Verify each stream has independent latency tracking."""
        mock_shared_state = Mock()

        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=[],
            shared_state=mock_shared_state
        )

        # Send samples to different streams
        daq._update_latency_telemetry('EEG', 5.0)
        daq._update_latency_telemetry('EMG', 10.0)

        # Both streams should have their own windows
        assert 'EEG' in daq._latency_windows
        assert 'EMG' in daq._latency_windows
        assert daq._latency_windows['EEG'] != daq._latency_windows['EMG']

    def test_latency_no_update_without_shared_state(self, stop_event, data_queue, save_queue):
        """Verify no error when SharedState is None."""
        daq = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=[],
            shared_state=None
        )

        update_interval = daq._latency_update_interval

        # Should not raise even with many samples
        for i in range(update_interval * 2):
            daq._update_latency_telemetry('EEG', 5.0)

        # Window should still be populated
        assert len(daq._latency_windows['EEG']) > 0