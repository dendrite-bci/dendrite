"""
ROS2-based streamer for robotics framework integration.
"""

import json
import logging
import multiprocessing
import queue
from multiprocessing.synchronize import Event
from typing import Any

from .base import BaseOutputStreamer

logger = logging.getLogger(__name__)


def _detect_ros2():
    """Detect if ROS2 is available and properly configured."""
    try:
        import rclpy

        # Try to create a basic context to test if ROS2 is fully functional
        import rclpy.executors  # noqa: F401
        from std_msgs.msg import String  # noqa: F401

        return True
    except ImportError as e:
        # Log which specific ROS2 component is missing
        missing_component = str(e).split("'")[-2] if "'" in str(e) else "unknown"
        logger.info(f"ROS2 not available: Missing {missing_component}")
        return False
    except Exception as e:
        logger.info(f"ROS2 not available: {e}")
        return False


HAS_ROS2 = _detect_ros2()


class ROS2Streamer(BaseOutputStreamer):
    """
    Pure ROS2-only streamer.

    Provides ROS2 output functionality without LSL dependency.
    """

    def __init__(
        self,
        input_queue: multiprocessing.Queue,
        stop_event: Event | None = None,
        ros2_config: dict[str, Any] | None = None,
        stream_name: str = "BMI_Predictions",
        classifier_names: list[str] | None = None,
        shared_state: Any | None = None,
    ) -> None:
        """
        Initialize ROS2 streamer.

        Args:
            input_queue: Queue containing data to be streamed
            stop_event: Event to signal when to stop streaming
            ros2_config: Configuration with topic_name, node_name
            stream_name: Name for the stream
            classifier_names: List of classifier names for ROS2 messages
            shared_state: SharedState instance for exposing streaming metrics
        """
        super().__init__(input_queue, f"ROS2_{stream_name}", stop_event, shared_state)

        # Parse configuration
        config = ros2_config or {}
        self.topic_name = config.get("topic_name", "bmi_predictions")
        self.node_name = config.get("node_name", "bmi_prediction_node")
        self.classifier_names = classifier_names or []

        # ROS2 state
        self.ros2_context = None
        self.ros2_node = None
        self.ros2_publisher = None
        self.ros2_executor = None

        # Check ROS2 availability
        if not HAS_ROS2:
            self.logger.warning("ROS2 not available - ROS2Streamer will not function")

    def _initialize_output(self) -> None:
        """Initialize ROS2 node and publisher."""
        if not HAS_ROS2:
            self.logger.error("ROS2 not available - cannot initialize")
            raise RuntimeError("ROS2 not available")

        rclpy_initialized = False
        try:
            import rclpy
            from rclpy.node import Node
            from std_msgs.msg import String

            rclpy.init()
            rclpy_initialized = True

            self.ros2_node = Node(self.node_name)
            self.ros2_publisher = self.ros2_node.create_publisher(
                String,
                self.topic_name,
                10,  # QoS depth
            )

            from rclpy.executors import SingleThreadedExecutor

            self.ros2_executor = SingleThreadedExecutor()
            self.ros2_executor.add_node(self.ros2_node)

            self.logger.info(f"ROS2 node '{self.node_name}' publishing to '{self.topic_name}'")

        except Exception as e:
            self.logger.error(f"Failed to initialize ROS2: {e}")
            # Clean up ROS2 if it was initialized before the failure
            if rclpy_initialized:
                try:
                    import rclpy

                    rclpy.shutdown()
                except RuntimeError as shutdown_error:
                    self.logger.debug(f"ROS2 shutdown during cleanup: {shutdown_error}")
            raise

    def _process_loop(self) -> None:
        """Override process loop to include ROS2 spinning."""
        self.logger.info("Starting ROS2 processing loop")

        import rclpy

        while not (self.stop_event and self.stop_event.is_set()) and rclpy.ok():
            try:
                # Spin ROS2 callbacks
                if self.ros2_executor:
                    self.ros2_executor.spin_once(timeout_sec=0)

                # Process queue data
                try:
                    data = self.input_queue.get(timeout=0.1)
                    self.messages_sent += 1
                    self._send_data(data)

                    if self.messages_sent % 100 == 0:
                        self._log_statistics()

                except queue.Empty:
                    continue

            except Exception as e:
                self.logger.error(f"Error in ROS2 processing loop: {e}", exc_info=True)

        self.logger.info("ROS2 processing loop stopped")

    def _send_data(self, data: Any) -> None:
        """Send data via ROS2."""
        if self.ros2_publisher:
            try:
                from std_msgs.msg import String

                # Convert to JSON string
                serializable_data = self._make_json_serializable(data)
                json_str = json.dumps(serializable_data)

                # Create and publish ROS2 message
                msg = String()
                msg.data = json_str
                self.ros2_publisher.publish(msg)

                self.bytes_sent += len(json_str.encode("utf-8"))

            except Exception as e:
                self.logger.error(f"Error sending ROS2 data: {e}")

    def _cleanup(self) -> None:
        """Clean up ROS2 resources."""
        if not HAS_ROS2:
            return

        try:
            import rclpy

            if self.ros2_executor:
                self.ros2_executor.shutdown()

            if self.ros2_node:
                self.ros2_node.destroy_node()

            rclpy.shutdown()

            self.logger.info("ROS2 resources cleaned up")

        except Exception as e:
            self.logger.error(f"Error cleaning up ROS2: {e}")
