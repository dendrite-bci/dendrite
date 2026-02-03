"""
Output Configuration Widget

A widget for configuring output protocols and displaying output formats.
Supports LSL, Socket (TCP/UDP), ZMQ, and ROS2 output protocols.
"""

import socket
import time
from typing import Any

from PyQt6 import QtWidgets

from dendrite.data.streaming.output_schemas import (
    DEFAULT_LSL_CONFIG,
    DEFAULT_ROS2_CONFIG,
    DEFAULT_SOCKET_CONFIG,
    DEFAULT_ZMQ_CONFIG,
    LSLConfig,
    ROS2Config,
    SocketConfig,
    ZMQConfig,
)
from dendrite.gui.styles.widget_styles import LAYOUT, WidgetStyles
from dendrite.gui.widgets.common import FieldConfig, ProtocolConfigCard
from dendrite.utils.logger_central import get_logger


def _check_ros2_availability():
    """Check if ROS2 is available and properly configured."""
    try:
        import rclpy
        import rclpy.executors  # noqa: F401
        from std_msgs.msg import String  # noqa: F401

        return True, "Available"
    except ImportError as e:
        missing = str(e).split("'")[-2] if "'" in str(e) else "module"
        return False, f"Missing {missing}"
    except RuntimeError as e:
        return False, f"Error: {str(e)[:30]}"


def _check_zmq_availability():
    """Check if ZeroMQ (pyzmq) is available."""
    try:
        import zmq

        context = zmq.Context()
        context.term()
        return True, "Available"
    except ImportError:
        return False, "pyzmq not installed"
    except RuntimeError as e:
        return False, f"Error: {str(e)[:30]}"


ROS2_AVAILABLE, ROS2_STATUS = _check_ros2_availability()
ZMQ_AVAILABLE, ZMQ_STATUS = _check_zmq_availability()


class OutputWidget(QtWidgets.QWidget):
    """Widget for configuring output protocols using ProtocolConfigCard components."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.logger = get_logger("OutputWidget")

        # Protocol cards
        self._cards: dict[str, ProtocolConfigCard] = {}

        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(
            LAYOUT["margin"], LAYOUT["margin"], LAYOUT["margin"], LAYOUT["margin"]
        )
        main_layout.setSpacing(LAYOUT["spacing_lg"])

        # Section header (muted_header style like other sections)
        header = QtWidgets.QLabel("OUTPUT PROTOCOLS")
        header.setStyleSheet(WidgetStyles.label("muted_header", uppercase=True, letter_spacing=1))
        main_layout.addWidget(header)
        main_layout.addSpacing(LAYOUT["spacing_sm"])

        # Generate unique source ID for LSL
        hostname = socket.gethostname().replace("-", "_").replace(".", "_")[:12]
        unique_id = f"dendrite_{hostname}_{int(time.time())}"

        # LSL Output Card
        self._cards["lsl"] = ProtocolConfigCard(
            protocol_name="LSL Output",
            fields=[
                FieldConfig(
                    "stream_name", "Stream Name:", "text", DEFAULT_LSL_CONFIG["stream_name"]
                ),
                FieldConfig(
                    "stream_type", "Stream Type:", "text", DEFAULT_LSL_CONFIG["stream_type"]
                ),
                FieldConfig(
                    "source_id",
                    "Source ID:",
                    "text",
                    unique_id,
                    tooltip="Unique identifier for this LSL stream source",
                ),
            ],
            schema_class=LSLConfig,
            default_config=DEFAULT_LSL_CONFIG,
            default_enabled=True,
        )
        main_layout.addWidget(self._cards["lsl"])

        # Socket Output Card
        self._cards["socket"] = ProtocolConfigCard(
            protocol_name="Socket Output",
            fields=[
                FieldConfig("protocol", "Protocol:", "pill", "TCP", options=["TCP", "UDP"]),
                FieldConfig(
                    "ip",
                    "Bind Address:",
                    "text",
                    DEFAULT_SOCKET_CONFIG["ip"],
                    tooltip="Address to bind for incoming connections\n"
                    "• 127.0.0.1 = localhost only (secure, default)\n"
                    "• 0.0.0.0 = all network interfaces (allows remote access)",
                ),
                FieldConfig(
                    "port",
                    "Port:",
                    "spinbox",
                    DEFAULT_SOCKET_CONFIG["port"],
                    min_val=1,
                    max_val=65535,
                ),
            ],
            schema_class=SocketConfig,
            default_config=DEFAULT_SOCKET_CONFIG,
            default_enabled=False,
        )
        main_layout.addWidget(self._cards["socket"])

        # ZMQ Output Card
        self._cards["zmq"] = ProtocolConfigCard(
            protocol_name="ZeroMQ Output",
            fields=[
                FieldConfig(
                    "ip",
                    "Bind Address:",
                    "text",
                    DEFAULT_ZMQ_CONFIG["ip"],
                    tooltip="Address to bind for publishing\n"
                    "• 127.0.0.1 = localhost only (secure, default)\n"
                    "• * or 0.0.0.0 = all network interfaces",
                ),
                FieldConfig(
                    "port", "Port:", "spinbox", DEFAULT_ZMQ_CONFIG["port"], min_val=1, max_val=65535
                ),
            ],
            schema_class=ZMQConfig,
            default_config=DEFAULT_ZMQ_CONFIG,
            available=ZMQ_AVAILABLE,
            availability_msg=ZMQ_STATUS if not ZMQ_AVAILABLE else "",
            default_enabled=False,
        )
        main_layout.addWidget(self._cards["zmq"])

        # ROS2 Output Card
        self._cards["ros2"] = ProtocolConfigCard(
            protocol_name="ROS2 Output",
            fields=[
                FieldConfig("topic_name", "Topic Name:", "text", DEFAULT_ROS2_CONFIG["topic_name"]),
                FieldConfig("node_name", "Node Name:", "text", DEFAULT_ROS2_CONFIG["node_name"]),
            ],
            schema_class=ROS2Config,
            default_config=DEFAULT_ROS2_CONFIG,
            available=ROS2_AVAILABLE,
            availability_msg=ROS2_STATUS if not ROS2_AVAILABLE else "",
            default_enabled=False,
        )
        main_layout.addWidget(self._cards["ros2"])

        main_layout.addStretch()

    # Backward compatibility properties for direct widget access
    @property
    def lsl_checkbox(self):
        return self._cards["lsl"]._toggle

    @property
    def socket_checkbox(self):
        return self._cards["socket"]._toggle

    @property
    def zmq_checkbox(self):
        return self._cards["zmq"]._toggle

    @property
    def ros2_checkbox(self):
        return self._cards["ros2"]._toggle

    @property
    def lsl_stream_name_edit(self):
        return self._cards["lsl"].get_widget("stream_name")

    @property
    def lsl_stream_type_edit(self):
        return self._cards["lsl"].get_widget("stream_type")

    @property
    def lsl_source_id_edit(self):
        return self._cards["lsl"].get_widget("source_id")

    @property
    def socket_protocol_combo(self):
        return self._cards["socket"].get_widget("protocol")

    @property
    def socket_ip_edit(self):
        return self._cards["socket"].get_widget("ip")

    @property
    def socket_port_spinbox(self):
        return self._cards["socket"].get_widget("port")

    @property
    def zmq_ip_edit(self):
        return self._cards["zmq"].get_widget("ip")

    @property
    def zmq_port_spinbox(self):
        return self._cards["zmq"].get_widget("port")

    @property
    def ros2_topic_name_edit(self):
        return self._cards["ros2"].get_widget("topic_name")

    @property
    def ros2_node_name_edit(self):
        return self._cards["ros2"].get_widget("node_name")

    def set_protocol_connected(self, protocol: str, connected: bool):
        """Set the connection status for a protocol."""
        if protocol in self._cards:
            self._cards[protocol].set_connected(connected)

    def validate_all_configurations(self) -> tuple[bool, list[str]]:
        """Validate all enabled protocol configurations."""
        errors = []

        for protocol, card in self._cards.items():
            if card.is_enabled():
                is_valid, card_errors = card.validate()
                if not is_valid:
                    for error in card_errors:
                        errors.append(f"{protocol.upper()} {error}")

        return len(errors) == 0, errors

    def get_output_configuration(self) -> dict[str, Any]:
        """Get the current output configuration."""
        protocols = {}

        lsl_config = self._cards["lsl"].get_config()
        protocols["lsl"] = {"enabled": self._cards["lsl"].is_enabled(), "config": lsl_config}

        socket_config = self._cards["socket"].get_config()
        protocols["socket"] = {
            "enabled": self._cards["socket"].is_enabled(),
            "config": socket_config,
        }

        zmq_config = self._cards["zmq"].get_config()
        zmq_config["message_format"] = "JSON"  # Always JSON
        protocols["zmq"] = {"enabled": self._cards["zmq"].is_enabled(), "config": zmq_config}

        # ROS2
        ros2_config = self._cards["ros2"].get_config()
        protocols["ros2"] = {"enabled": self._cards["ros2"].is_enabled(), "config": ros2_config}

        return {"protocols": protocols}

    def load_output_configuration(self, config: dict[str, Any]):
        """Load output configuration from saved config."""
        try:
            protocols = config.get("protocols", {})

            if "lsl" in protocols:
                lsl_cfg = protocols["lsl"]
                self._cards["lsl"].set_enabled(lsl_cfg.get("enabled", True))
                if "config" in lsl_cfg:
                    self._cards["lsl"].set_config(lsl_cfg["config"])

            if "socket" in protocols:
                socket_cfg = protocols["socket"]
                self._cards["socket"].set_enabled(socket_cfg.get("enabled", False))
                if "config" in socket_cfg:
                    self._cards["socket"].set_config(socket_cfg["config"])

            # Load ZMQ (only if available)
            if "zmq" in protocols and ZMQ_AVAILABLE:
                zmq_cfg = protocols["zmq"]
                self._cards["zmq"].set_enabled(zmq_cfg.get("enabled", False))
                if "config" in zmq_cfg:
                    self._cards["zmq"].set_config(zmq_cfg["config"])

            # Load ROS2 (only if available)
            if "ros2" in protocols and ROS2_AVAILABLE:
                ros2_cfg = protocols["ros2"]
                self._cards["ros2"].set_enabled(ros2_cfg.get("enabled", False))
                if "config" in ros2_cfg:
                    self._cards["ros2"].set_config(ros2_cfg["config"])

        except (KeyError, TypeError, AttributeError) as e:
            self.logger.error(f"Error loading output configuration: {e}")
