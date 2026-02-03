"""
Tests for StreamConfigManager modality filtering behavior.

Tests ensure that marker/event channels and streams with invalid sample rates
are properly excluded from modality data used by preprocessing and mode configs.
"""

import pytest
from dendrite.gui.config.stream_config_manager import StreamConfigManager
from dendrite.data.stream_schemas import StreamMetadata


def create_stream(
    name: str,
    stream_type: str,
    sample_rate: float,
    labels: list[str],
    channel_types: list[str],
    uid: str = None
) -> StreamMetadata:
    """Helper to create StreamMetadata for testing."""
    return StreamMetadata(
        name=name,
        type=stream_type,
        channel_count=len(labels),
        sample_rate=sample_rate,
        labels=labels,
        channel_types=channel_types,
        uid=uid or f"test-{name}"
    )


class TestModalityFiltering:
    """Test modality filtering in StreamConfigManager."""

    def test_metadata_channel_types_filtered_from_modalities(self):
        """Marker/event channel types should be excluded from modalities."""
        manager = StreamConfigManager()

        # Stream with mixed channel types including markers
        stream = create_stream(
            name="TestEEG",
            stream_type="EEG",
            sample_rate=500,
            labels=["Fp1", "Fp2", "STI001"],
            channel_types=["EEG", "EEG", "Markers"]
        )
        manager._streams = {stream.uid: stream}

        # get_modalities_by_stream should exclude markers
        result = manager.get_modalities_by_stream()
        assert stream.uid in result
        modalities = result[stream.uid]['modalities']
        assert 'eeg' in modalities
        assert 'markers' not in modalities
        assert len(modalities['eeg']) == 2  # Only Fp1, Fp2

    def test_singular_marker_filtered(self):
        """Both 'marker' (singular) and 'markers' (plural) should be filtered."""
        manager = StreamConfigManager()

        # Test singular 'marker'
        stream = create_stream(
            name="TestEEG",
            stream_type="EEG",
            sample_rate=500,
            labels=["Fp1", "STI001"],
            channel_types=["EEG", "marker"]  # Singular
        )
        manager._streams = {stream.uid: stream}

        result = manager.get_modalities_by_stream()
        modalities = result[stream.uid]['modalities']
        assert 'marker' not in modalities
        assert 'eeg' in modalities

    def test_events_channel_type_filtered(self):
        """'events' channel type should be filtered."""
        manager = StreamConfigManager()

        stream = create_stream(
            name="TestEEG",
            stream_type="EEG",
            sample_rate=500,
            labels=["Fp1", "Event"],
            channel_types=["EEG", "events"]
        )
        manager._streams = {stream.uid: stream}

        result = manager.get_modalities_by_stream()
        modalities = result[stream.uid]['modalities']
        assert 'events' not in modalities
        assert 'eeg' in modalities

    def test_streams_with_zero_sample_rate_excluded(self):
        """Streams with sample_rate=0 should be excluded from get_modalities_by_stream()."""
        manager = StreamConfigManager()

        # Events stream with sample_rate=0
        events_stream = create_stream(
            name="Events",
            stream_type="Events",
            sample_rate=0,  # Irregular rate
            labels=["marker"],
            channel_types=["misc"],  # Would pass filter if not for sample_rate check
            uid="events-uid"
        )
        # Normal EEG stream
        eeg_stream = create_stream(
            name="EEG",
            stream_type="EEG",
            sample_rate=500,
            labels=["Fp1", "Fp2"],
            channel_types=["EEG", "EEG"],
            uid="eeg-uid"
        )
        manager._streams = {
            events_stream.uid: events_stream,
            eeg_stream.uid: eeg_stream
        }

        result = manager.get_modalities_by_stream()

        # Events stream should be excluded
        assert "events-uid" not in result
        # EEG stream should be included
        assert "eeg-uid" in result

    def test_has_modalities_with_only_metadata_channels(self):
        """has_modalities() returns False when only marker channels exist."""
        manager = StreamConfigManager()

        stream = create_stream(
            name="Events",
            stream_type="Events",
            sample_rate=0,
            labels=["marker1", "marker2"],
            channel_types=["Markers", "Markers"]
        )
        manager._streams = {stream.uid: stream}

        assert not manager.has_modalities()

    def test_has_modalities_with_mixed_channels(self):
        """has_modalities() returns True when usable modalities exist."""
        manager = StreamConfigManager()

        stream = create_stream(
            name="EEG",
            stream_type="EEG",
            sample_rate=500,
            labels=["Fp1", "STI001"],
            channel_types=["EEG", "Markers"]
        )
        manager._streams = {stream.uid: stream}

        assert manager.has_modalities()

    def test_get_modality_data_excludes_metadata_types(self):
        """get_modality_data() should exclude marker/event channel types."""
        manager = StreamConfigManager()

        stream = create_stream(
            name="TestEEG",
            stream_type="EEG",
            sample_rate=500,
            labels=["Fp1", "Fp2", "STI001", "Event1"],
            channel_types=["EEG", "EEG", "Markers", "events"]
        )
        manager._streams = {stream.uid: stream}

        result = manager.get_modality_data()

        assert 'eeg' in result
        assert 'markers' not in result
        assert 'events' not in result
        assert result['eeg']['total_count'] == 2

    def test_get_modality_data_excludes_zero_sample_rate_streams(self):
        """get_modality_data() should exclude streams with sample_rate=0 (e.g., Events)."""
        manager = StreamConfigManager()

        # Events stream with sample_rate=0 and channel_types=['EEG'] (the bug scenario)
        events_stream = create_stream(
            name="Events",
            stream_type="Events",
            sample_rate=0,
            labels=["event_marker"],
            channel_types=["EEG"],  # Misleading type that would cause label mismatch
            uid="events-uid"
        )
        # Normal EEG stream
        eeg_stream = create_stream(
            name="EEG",
            stream_type="EEG",
            sample_rate=500,
            labels=["Fp1", "Fp2"],
            channel_types=["EEG", "EEG"],
            uid="eeg-uid"
        )
        manager._streams = {
            events_stream.uid: events_stream,
            eeg_stream.uid: eeg_stream
        }

        result = manager.get_modality_data()

        # Should only count EEG channels from the actual EEG stream, not from Events
        assert 'eeg' in result
        assert result['eeg']['total_count'] == 2
        assert result['eeg']['channel_labels'] == ["Fp1", "Fp2"]


class TestMetadataChannelTypesConstant:
    """Test the METADATA_CHANNEL_TYPES constant covers expected values."""

    def test_metadata_channel_types_includes_variants(self):
        """Verify METADATA_CHANNEL_TYPES includes all expected variants."""
        expected = {'markers', 'marker', 'events', 'triggers', 'annotations'}
        assert StreamConfigManager.METADATA_CHANNEL_TYPES == expected

    def test_metadata_channel_types_case_insensitive_via_get_modality_data(self):
        """Verify case-insensitivity through public API."""
        manager = StreamConfigManager()

        # Mixed-case channel types should be normalized and excluded
        stream = create_stream(
            name="TestStream",
            stream_type="EEG",
            sample_rate=500,
            labels=["Ch1", "Ch2", "Evt1"],
            channel_types=["eeg", "EEG", "MARKERS"]  # Mixed case
        )
        manager._streams = {stream.uid: stream}

        result = manager.get_modality_data()

        # Both lowercase and uppercase EEG should be combined
        assert 'eeg' in result
        assert result['eeg']['total_count'] == 2

        # MARKERS should be excluded (normalized to lowercase)
        assert 'markers' not in result
