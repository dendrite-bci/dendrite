"""Tests for ROS2Streamer."""

import pytest
import json
from unittest.mock import Mock, patch

from dendrite.data.streaming import ROS2Streamer


class TestROS2Streamer:
    """Test suite for ROS2Streamer class."""

    def test_initialization_with_defaults(self, input_queue, stop_event):
        """Test ROS2Streamer initialization with defaults."""
        streamer = ROS2Streamer(input_queue, stop_event)

        assert streamer.topic_name == 'bmi_predictions'
        assert streamer.node_name == 'bmi_prediction_node'
        assert streamer.classifier_names == []
        assert streamer.stream_name == "ROS2_BMI_Predictions"
        assert streamer.ros2_context is None
        assert streamer.ros2_node is None
        assert streamer.ros2_publisher is None
        assert streamer.ros2_executor is None

    def test_initialization_with_ros2_config(self, input_queue, stop_event):
        """Test ROS2Streamer initialization with custom config."""
        ros2_config = {
            'topic_name': 'custom_predictions',
            'node_name': 'custom_prediction_node'
        }
        classifier_names = ['LDA', 'SVM']

        streamer = ROS2Streamer(
            input_queue,
            stop_event,
            ros2_config,
            "CustomStream",
            classifier_names
        )

        assert streamer.topic_name == 'custom_predictions'
        assert streamer.node_name == 'custom_prediction_node'
        assert streamer.classifier_names == ['LDA', 'SVM']
        assert streamer.stream_name == "ROS2_CustomStream"

    @patch('dendrite.data.streaming.ros2.HAS_ROS2', True)
    def test_initialize_output_with_ros2_available(self, input_queue, stop_event):
        """Test ROS2 initialization when ROS2 is available."""
        with patch('builtins.__import__') as mock_import:
            mock_rclpy = Mock()
            mock_node_class = Mock()
            mock_node = Mock()
            mock_publisher = Mock()
            mock_executor_class = Mock()
            mock_executor = Mock()
            mock_string_class = Mock()

            mock_rclpy.init.return_value = None
            mock_node_class.return_value = mock_node
            mock_node.create_publisher.return_value = mock_publisher
            mock_executor_class.return_value = mock_executor

            def mock_import_func(name, *args, **kwargs):
                if name == 'rclpy':
                    return mock_rclpy
                elif name == 'rclpy.node':
                    mock_node_module = Mock()
                    mock_node_module.Node = mock_node_class
                    return mock_node_module
                elif name == 'std_msgs.msg':
                    mock_string_module = Mock()
                    mock_string_module.String = mock_string_class
                    return mock_string_module
                elif name == 'rclpy.executors':
                    mock_executor_module = Mock()
                    mock_executor_module.SingleThreadedExecutor = mock_executor_class
                    return mock_executor_module
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = mock_import_func

            streamer = ROS2Streamer(input_queue, stop_event)
            streamer._initialize_output()

            mock_rclpy.init.assert_called_once()
            mock_node_class.assert_called_once_with('bmi_prediction_node')
            mock_node.create_publisher.assert_called_once_with(mock_string_class, 'bmi_predictions', 10)
            mock_executor.add_node.assert_called_once_with(mock_node)

            assert streamer.ros2_node == mock_node
            assert streamer.ros2_publisher == mock_publisher
            assert streamer.ros2_executor == mock_executor

    @patch('dendrite.data.streaming.ros2.HAS_ROS2', False)
    def test_initialize_output_without_ros2(self, input_queue, stop_event):
        """Test ROS2 initialization when ROS2 is not available."""
        streamer = ROS2Streamer(input_queue, stop_event)

        with pytest.raises(RuntimeError, match="ROS2 not available"):
            streamer._initialize_output()

    @patch('dendrite.data.streaming.ros2.HAS_ROS2', True)
    def test_send_data(self, input_queue, stop_event):
        """Test sending data via ROS2."""
        with patch('builtins.__import__') as mock_import:
            mock_publisher = Mock()
            mock_string_class = Mock()
            mock_msg = Mock()
            mock_string_class.return_value = mock_msg

            def mock_import_func(name, *args, **kwargs):
                if name == 'std_msgs.msg':
                    mock_string_module = Mock()
                    mock_string_module.String = mock_string_class
                    return mock_string_module
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = mock_import_func

            streamer = ROS2Streamer(input_queue, stop_event)
            streamer.ros2_publisher = mock_publisher

            test_data = {'prediction': 'forward', 'confidence': 0.95}
            streamer._send_data(test_data)

            mock_string_class.assert_called_once()
            mock_publisher.publish.assert_called_once_with(mock_msg)

            expected_json = json.dumps(test_data)
            assert mock_msg.data == expected_json

    @patch('dendrite.data.streaming.ros2.HAS_ROS2', True)
    def test_cleanup(self, input_queue, stop_event):
        """Test ROS2 resource cleanup."""
        with patch('builtins.__import__') as mock_import:
            mock_rclpy = Mock()
            mock_executor = Mock()
            mock_node = Mock()

            def mock_import_func(name, *args, **kwargs):
                if name == 'rclpy':
                    return mock_rclpy
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = mock_import_func

            streamer = ROS2Streamer(input_queue, stop_event)
            streamer.ros2_executor = mock_executor
            streamer.ros2_node = mock_node

            streamer._cleanup()

            mock_executor.shutdown.assert_called_once()
            mock_node.destroy_node.assert_called_once()
            mock_rclpy.shutdown.assert_called_once()

    @patch('dendrite.data.streaming.ros2.HAS_ROS2', False)
    def test_cleanup_without_ros2(self, input_queue, stop_event):
        """Test cleanup when ROS2 is not available."""
        streamer = ROS2Streamer(input_queue, stop_event)
        streamer._cleanup()

    @patch('dendrite.data.streaming.ros2.HAS_ROS2', True)
    def test_cleanup_on_partial_init_failure(self, input_queue, stop_event):
        """Test that rclpy.shutdown() is called when node creation fails.

        Bug: If rclpy.init() succeeds but Node() creation fails,
        rclpy.shutdown() is never called, leaving ROS2 in an inconsistent state.
        """
        with patch('builtins.__import__') as mock_import:
            mock_rclpy = Mock()
            mock_node_class = Mock()

            # rclpy.init() succeeds
            mock_rclpy.init.return_value = None
            # But Node() creation fails
            mock_node_class.side_effect = RuntimeError("Node creation failed")

            def mock_import_func(name, *args, **kwargs):
                if name == 'rclpy':
                    return mock_rclpy
                elif name == 'rclpy.node':
                    mock_node_module = Mock()
                    mock_node_module.Node = mock_node_class
                    return mock_node_module
                elif name == 'std_msgs.msg':
                    return Mock()
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = mock_import_func

            streamer = ROS2Streamer(input_queue, stop_event)

            # _initialize_output should raise the error
            with pytest.raises(RuntimeError, match="Node creation failed"):
                streamer._initialize_output()

            # CRITICAL: rclpy.shutdown() should have been called to clean up
            # after rclpy.init() succeeded but subsequent steps failed
            mock_rclpy.shutdown.assert_called_once()
