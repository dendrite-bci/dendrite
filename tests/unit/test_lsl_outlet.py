"""Tests for LSLOutlet class.

Unit tests for the LSL outlet wrapper that handles stream creation and data pushing.
"""
import pytest
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

from dendrite.data.lsl_helpers import LSLOutlet
from dendrite.data.stream_schemas import StreamConfig


class TestLSLOutletInit:
    """Test LSLOutlet initialization."""

    def test_init_with_valid_config(self):
        """LSLOutlet initializes with valid StreamConfig."""
        config = StreamConfig(
            name='TestStream',
            type='EEG',
            channel_count=4,
            sample_rate=250.0
        )
        outlet = LSLOutlet(config)
        assert outlet.name == 'TestStream'
        assert outlet.channel_count == 4
        assert outlet.outlet is None  # Lazy init

    def test_init_empty_name_raises_at_schema_level(self):
        """Empty stream name raises ValidationError at StreamConfig level."""
        # StreamConfig validates name has min_length=1
        with pytest.raises(ValidationError):
            StreamConfig(name='', type='EEG', channel_count=4, sample_rate=250.0)

    def test_init_zero_channels_raises_at_schema_level(self):
        """Zero channel count raises ValidationError at StreamConfig level."""
        # StreamConfig validates channel_count > 0
        with pytest.raises(ValidationError):
            StreamConfig(name='Test', type='EEG', channel_count=0, sample_rate=250.0)

    def test_init_negative_channels_raises_at_schema_level(self):
        """Negative channel count raises ValidationError at StreamConfig level."""
        with pytest.raises(ValidationError):
            StreamConfig(name='Test', type='EEG', channel_count=-1, sample_rate=250.0)

    def test_init_mismatched_labels_raises_at_schema_level(self):
        """Mismatched labels count raises ValidationError at StreamConfig level."""
        # StreamConfig has validator that checks labels count vs channel_count
        with pytest.raises(ValidationError):
            StreamConfig(
                name='Test', type='EEG', channel_count=4, sample_rate=250.0,
                labels=['Ch1', 'Ch2']  # Only 2 labels for 4 channels
            )

    def test_init_generates_default_labels(self):
        """Default labels generated when not provided."""
        config = StreamConfig(name='Test', type='EEG', channel_count=3, sample_rate=250.0)
        outlet = LSLOutlet(config)
        assert outlet.info is not None
        assert outlet.channel_count == 3

    def test_init_with_custom_labels(self):
        """Custom labels are accepted when count matches."""
        labels = ['Fp1', 'Fp2', 'Fz', 'Cz']
        config = StreamConfig(
            name='Test', type='EEG', channel_count=4, sample_rate=250.0,
            labels=labels
        )
        outlet = LSLOutlet(config)
        assert outlet.channel_count == 4

    def test_init_stores_config_attributes(self):
        """Init stores commonly used attributes for easy access."""
        config = StreamConfig(
            name='MyStream',
            type='EMG',
            channel_count=8,
            sample_rate=2000.0,
            channel_format='float32'
        )
        outlet = LSLOutlet(config)
        assert outlet.name == 'MyStream'
        assert outlet.stream_type == 'EMG'
        assert outlet.channel_count == 8
        assert outlet.sample_rate == 2000.0
        assert outlet.channel_format == 'float32'


class TestLSLOutletPushSample:
    """Test push_sample functionality."""

    @patch('dendrite.data.lsl_helpers.StreamOutlet')
    def test_push_sample_creates_outlet_lazily(self, mock_outlet_class):
        """Outlet created on first push."""
        config = StreamConfig(name='Test', type='EEG', channel_count=2, sample_rate=250.0)
        outlet = LSLOutlet(config)
        assert outlet.outlet is None

        outlet.push_sample([1.0, 2.0])
        assert outlet.outlet is not None
        mock_outlet_class.assert_called_once()

    @patch('dendrite.data.lsl_helpers.StreamOutlet')
    def test_push_sample_wrong_length_raises(self, mock_outlet_class):
        """Wrong sample length raises ValueError."""
        config = StreamConfig(name='Test', type='EEG', channel_count=2, sample_rate=250.0)
        outlet = LSLOutlet(config)

        with pytest.raises(ValueError, match="must match channel count"):
            outlet.push_sample([1.0, 2.0, 3.0])  # 3 values for 2 channels

    @patch('dendrite.data.lsl_helpers.StreamOutlet')
    def test_push_sample_with_timestamp(self, mock_outlet_class):
        """Push sample with explicit timestamp."""
        mock_outlet = MagicMock()
        mock_outlet_class.return_value = mock_outlet

        config = StreamConfig(name='Test', type='EEG', channel_count=2, sample_rate=250.0)
        outlet = LSLOutlet(config)
        outlet.push_sample([1.0, 2.0], timestamp=123.456)

        mock_outlet.push_sample.assert_called_once_with([1.0, 2.0], 123.456)

    @patch('dendrite.data.lsl_helpers.StreamOutlet')
    def test_push_sample_without_timestamp(self, mock_outlet_class):
        """Push sample without timestamp uses LSL local_clock."""
        mock_outlet = MagicMock()
        mock_outlet_class.return_value = mock_outlet

        config = StreamConfig(name='Test', type='EEG', channel_count=2, sample_rate=250.0)
        outlet = LSLOutlet(config)
        outlet.push_sample([1.0, 2.0])

        # Called without timestamp argument
        mock_outlet.push_sample.assert_called_once_with([1.0, 2.0])


class TestLSLOutletPushChunk:
    """Test push_chunk functionality."""

    @patch('dendrite.data.lsl_helpers.StreamOutlet')
    def test_push_chunk_creates_outlet_lazily(self, mock_outlet_class):
        """Outlet created on first chunk push."""
        config = StreamConfig(name='Test', type='EEG', channel_count=2, sample_rate=250.0)
        outlet = LSLOutlet(config)
        assert outlet.outlet is None

        outlet.push_chunk([[1.0, 2.0], [3.0, 4.0]])
        assert outlet.outlet is not None

    @patch('dendrite.data.lsl_helpers.StreamOutlet')
    def test_push_chunk_validates_channel_count(self, mock_outlet_class):
        """Chunk must have correct channel count."""
        config = StreamConfig(name='Test', type='EEG', channel_count=2, sample_rate=250.0)
        outlet = LSLOutlet(config)

        with pytest.raises(ValueError, match="must match channel count"):
            outlet.push_chunk([[1, 2, 3], [4, 5, 6]])  # 3 channels instead of 2

    @patch('dendrite.data.lsl_helpers.StreamOutlet')
    def test_push_chunk_validates_2d_array(self, mock_outlet_class):
        """Chunk must be 2D array."""
        config = StreamConfig(name='Test', type='EEG', channel_count=2, sample_rate=250.0)
        outlet = LSLOutlet(config)

        with pytest.raises(ValueError, match="must be 2D"):
            outlet.push_chunk([1.0, 2.0])  # 1D instead of 2D

    @patch('dendrite.data.lsl_helpers.StreamOutlet')
    def test_push_chunk_with_timestamps(self, mock_outlet_class):
        """Push chunk with explicit timestamps."""
        mock_outlet = MagicMock()
        mock_outlet_class.return_value = mock_outlet

        config = StreamConfig(name='Test', type='EEG', channel_count=2, sample_rate=250.0)
        outlet = LSLOutlet(config)

        chunk = [[1.0, 2.0], [3.0, 4.0]]
        timestamps = [100.0, 100.004]
        outlet.push_chunk(chunk, timestamps)

        # Verify push_chunk was called with timestamps
        assert mock_outlet.push_chunk.called

    @patch('dendrite.data.lsl_helpers.StreamOutlet')
    def test_push_chunk_mismatched_timestamps_raises(self, mock_outlet_class):
        """Mismatched timestamp count raises ValueError."""
        config = StreamConfig(name='Test', type='EEG', channel_count=2, sample_rate=250.0)
        outlet = LSLOutlet(config)

        chunk = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]  # 3 samples
        timestamps = [100.0, 100.004]  # Only 2 timestamps

        with pytest.raises(ValueError, match="must match samples"):
            outlet.push_chunk(chunk, timestamps)


class TestLSLOutletHaveConsumers:
    """Test have_consumers functionality."""

    def test_have_consumers_returns_false_before_outlet_created(self):
        """Returns False when outlet not yet created."""
        config = StreamConfig(name='Test', type='EEG', channel_count=2, sample_rate=250.0)
        outlet = LSLOutlet(config)
        assert outlet.have_consumers() is False

    @patch('dendrite.data.lsl_helpers.StreamOutlet')
    def test_have_consumers_delegates_to_outlet(self, mock_outlet_class):
        """Delegates to underlying outlet when created."""
        mock_outlet = MagicMock()
        mock_outlet.have_consumers.return_value = True
        mock_outlet_class.return_value = mock_outlet

        config = StreamConfig(name='Test', type='EEG', channel_count=2, sample_rate=250.0)
        outlet = LSLOutlet(config)
        outlet.create_outlet()

        assert outlet.have_consumers() is True
        mock_outlet.have_consumers.assert_called_once()

    @patch('dendrite.data.lsl_helpers.StreamOutlet')
    def test_have_consumers_returns_false_when_no_consumers(self, mock_outlet_class):
        """Returns False when outlet exists but no consumers."""
        mock_outlet = MagicMock()
        mock_outlet.have_consumers.return_value = False
        mock_outlet_class.return_value = mock_outlet

        config = StreamConfig(name='Test', type='EEG', channel_count=2, sample_rate=250.0)
        outlet = LSLOutlet(config)
        outlet.create_outlet()

        assert outlet.have_consumers() is False


class TestLSLOutletClose:
    """Test close functionality."""

    @patch('dendrite.data.lsl_helpers.StreamOutlet')
    def test_close_clears_outlet(self, mock_outlet_class):
        """Close sets outlet to None."""
        config = StreamConfig(name='Test', type='EEG', channel_count=2, sample_rate=250.0)
        outlet = LSLOutlet(config)
        outlet.create_outlet()
        assert outlet.outlet is not None

        outlet.close()
        assert outlet.outlet is None

    def test_close_idempotent(self):
        """Multiple close calls don't raise."""
        config = StreamConfig(name='Test', type='EEG', channel_count=2, sample_rate=250.0)
        outlet = LSLOutlet(config)
        outlet.close()  # First close (no outlet created)
        outlet.close()  # Second close - should not raise

    @patch('dendrite.data.lsl_helpers.StreamOutlet')
    def test_close_after_push(self, mock_outlet_class):
        """Close works after pushing data."""
        config = StreamConfig(name='Test', type='EEG', channel_count=2, sample_rate=250.0)
        outlet = LSLOutlet(config)
        outlet.push_sample([1.0, 2.0])
        assert outlet.outlet is not None

        outlet.close()
        assert outlet.outlet is None


class TestLSLOutletCreateOutlet:
    """Test create_outlet functionality."""

    @patch('dendrite.data.lsl_helpers.StreamOutlet')
    def test_create_outlet_creates_stream_outlet(self, mock_outlet_class):
        """create_outlet creates StreamOutlet."""
        config = StreamConfig(name='Test', type='EEG', channel_count=2, sample_rate=250.0)
        outlet = LSLOutlet(config)
        assert outlet.outlet is None

        outlet.create_outlet()
        assert outlet.outlet is not None
        mock_outlet_class.assert_called_once_with(outlet.info)

    @patch('dendrite.data.lsl_helpers.StreamOutlet')
    def test_create_outlet_failure_propagates(self, mock_outlet_class):
        """StreamOutlet creation failure propagates exception."""
        mock_outlet_class.side_effect = Exception("LSL error")

        config = StreamConfig(name='Test', type='EEG', channel_count=2, sample_rate=250.0)
        outlet = LSLOutlet(config)

        with pytest.raises(Exception, match="LSL error"):
            outlet.create_outlet()
