"""Tests for utility functions (detection functions)."""

import pytest
from unittest.mock import Mock, patch

from dendrite.data.streaming.ros2 import _detect_ros2
from dendrite.data.streaming.zmq import _detect_zmq


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_detect_ros2_with_available_ros2(self):
        """Test ROS2 detection when ROS2 is available."""
        with patch('builtins.__import__') as mock_import:
            def mock_import_func(name, *args, **kwargs):
                if name.startswith('rclpy') or name.startswith('std_msgs'):
                    return Mock()
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = mock_import_func

            result = _detect_ros2()
            assert result is True

    def test_detect_ros2_with_missing_rclpy(self):
        """Test ROS2 detection when rclpy is missing."""
        with patch('builtins.__import__') as mock_import:
            def mock_import_func(name, *args, **kwargs):
                if name == 'rclpy':
                    raise ImportError("No module named 'rclpy'")
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = mock_import_func

            result = _detect_ros2()
            assert result is False

    def test_detect_ros2_with_other_errors(self):
        """Test ROS2 detection with other errors."""
        with patch('builtins.__import__') as mock_import:
            def mock_import_func(name, *args, **kwargs):
                if name == 'rclpy':
                    raise RuntimeError("ROS2 not initialized")
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = mock_import_func

            result = _detect_ros2()
            assert result is False

    def test_detect_zmq_with_available_zmq(self):
        """Test ZMQ detection when ZMQ is available."""
        with patch('builtins.__import__') as mock_import:
            mock_zmq = Mock()
            mock_context = Mock()
            mock_zmq.Context.return_value = mock_context
            mock_context.term.return_value = None

            def mock_import_func(name, *args, **kwargs):
                if name == 'zmq':
                    return mock_zmq
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = mock_import_func

            result = _detect_zmq()
            assert result is True
            mock_zmq.Context.assert_called_once()
            mock_context.term.assert_called_once()

    def test_detect_zmq_with_missing_zmq(self):
        """Test ZMQ detection when pyzmq is missing."""
        with patch('builtins.__import__') as mock_import:
            def mock_import_func(name, *args, **kwargs):
                if name == 'zmq':
                    raise ImportError("No module named 'zmq'")
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = mock_import_func

            result = _detect_zmq()
            assert result is False

    def test_detect_zmq_with_context_error(self):
        """Test ZMQ detection with context creation error."""
        with patch('builtins.__import__') as mock_import:
            mock_zmq = Mock()
            mock_zmq.Context.side_effect = RuntimeError("ZMQ error")

            def mock_import_func(name, *args, **kwargs):
                if name == 'zmq':
                    return mock_zmq
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = mock_import_func

            result = _detect_zmq()
            assert result is False
