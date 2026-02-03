"""Tests for LSLStreamer."""

import pytest
from unittest.mock import Mock, patch

from dendrite.data.streaming.lsl import LSLStreamer
from dendrite.data.streaming.base import LSLBaseStreamer


class TestLSLStreamer:
    """Test suite for LSLStreamer class."""

    @patch('dendrite.data.streaming.base.LSLOutlet')
    def test_initialization_with_default_config(self, mock_lsl_outlet, input_queue, mock_lsl_stream_config, stop_event):
        """Test LSLStreamer initialization with default config."""
        streamer = LSLStreamer(input_queue, mock_lsl_stream_config, stop_event)

        assert streamer.stream_info == mock_lsl_stream_config
        assert streamer.stream_info.name == "TestLSLStream"
        assert streamer.stream_info.type == "Predictions"
        assert streamer.stream_info.source_id == "test_lsl_source"

    @patch('dendrite.data.streaming.base.LSLOutlet')
    def test_initialization_with_lsl_config_override(self, mock_lsl_outlet, input_queue, mock_lsl_stream_config, stop_event):
        """Test LSLStreamer initialization with config override."""
        lsl_config = {
            'stream_name': 'CustomStreamName',
            'stream_type': 'CustomType',
            'source_id': 'custom_source_456'
        }

        streamer = LSLStreamer(input_queue, mock_lsl_stream_config, stop_event, lsl_config)

        assert streamer.stream_info.name == 'CustomStreamName'
        assert streamer.stream_info.type == 'CustomType'
        assert streamer.stream_info.source_id == 'custom_source_456'

    @patch('dendrite.data.streaming.base.LSLOutlet')
    def test_initialization_with_partial_lsl_config(self, mock_lsl_outlet, input_queue, mock_lsl_stream_config, stop_event):
        """Test LSLStreamer initialization with partial config override."""
        lsl_config = {
            'stream_name': 'PartialCustomName'
        }

        streamer = LSLStreamer(input_queue, mock_lsl_stream_config, stop_event, lsl_config)

        assert streamer.stream_info.name == 'PartialCustomName'
        assert streamer.stream_info.type == "Predictions"
        assert streamer.stream_info.source_id == "test_lsl_source"

    @patch('dendrite.data.streaming.base.LSLOutlet')
    def test_inherits_lsl_base_functionality(self, mock_lsl_outlet, input_queue, mock_lsl_stream_config, stop_event):
        """Test that LSLStreamer inherits LSLBaseStreamer functionality."""
        streamer = LSLStreamer(input_queue, mock_lsl_stream_config, stop_event)

        assert hasattr(streamer, '_initialize_output')
        assert hasattr(streamer, '_send_data')
        assert hasattr(streamer, '_cleanup')
        assert hasattr(streamer, '_report_bandwidth')
        assert hasattr(streamer, '_format_bandwidth')

        assert isinstance(streamer, LSLBaseStreamer)
