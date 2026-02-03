"""
Pydantic configuration schemas for output protocols.

Defines validated configuration models for LSL, Socket, ZMQ, and ROS2 output protocols.
"""

import ipaddress
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

DEFAULT_LSL_CONFIG = {
    "stream_name": "PredictionStream",
    "stream_type": "PredictionStream",
    "source_id": "dendrite_default",
}

DEFAULT_SOCKET_CONFIG = {"protocol": "TCP", "ip": "127.0.0.1", "port": 8080}

DEFAULT_ZMQ_CONFIG = {"ip": "127.0.0.1", "port": 5556, "message_format": "JSON"}

DEFAULT_ROS2_CONFIG = {"topic_name": "bmi_predictions", "node_name": "bmi_prediction_node"}


def _validate_ipv4_address(v: str) -> str:
    """Validate IPv4 address format. Allows special values: 'localhost', '*', '0.0.0.0'."""
    if not v:
        raise ValueError("IP address cannot be empty")

    if len(v) > 15:  # Max IPv4 length is 15 chars (xxx.xxx.xxx.xxx)
        raise ValueError("IP address too long")

    if v in ["localhost", "*", "0.0.0.0"]:
        return v

    if not re.match(r"^[0-9.]+$", v):
        raise ValueError("IP address contains invalid characters")

    try:
        ipaddress.IPv4Address(v)
        return v
    except ipaddress.AddressValueError:
        raise ValueError("Invalid IP address format") from None


class LSLConfig(BaseModel):
    """Configuration for LSL (Lab Streaming Layer) output."""

    stream_name: str = Field(..., min_length=1, description="LSL stream name")
    stream_type: str = Field(..., min_length=1, description="LSL stream type")
    source_id: str = Field(..., min_length=1, description="Unique source identifier")

    @field_validator("stream_name", "stream_type", "source_id")
    @classmethod
    def validate_lsl_name(cls, v: str) -> str:
        """Validate LSL stream names - alphanumeric, underscore, hyphen only."""
        if not v:
            raise ValueError("Cannot be empty")
        if not re.match(r"^[a-zA-Z0-9_\-]+$", v):
            raise ValueError("Should only contain letters, numbers, underscore, and hyphen")
        return v


class SocketConfig(BaseModel):
    """Configuration for Socket (TCP/UDP) output."""

    protocol: Literal["TCP", "UDP"] = Field("TCP", description="Socket protocol")
    ip: str = Field(..., min_length=1, description="Bind IP address")
    port: int = Field(..., ge=1, le=65535, description="Port number")

    @field_validator("ip")
    @classmethod
    def validate_ip(cls, v: str) -> str:
        """Validate IP address format."""
        return _validate_ipv4_address(v)


class ZMQConfig(BaseModel):
    """Configuration for ZeroMQ (ZMQ) output."""

    ip: str = Field(..., min_length=1, description="Bind IP address")
    port: int = Field(..., ge=1, le=65535, description="Port number")
    message_format: Literal["JSON"] = Field("JSON", description="Message format (JSON only)")

    @field_validator("ip")
    @classmethod
    def validate_ip(cls, v: str) -> str:
        """Validate IP address format (allows ZMQ wildcard '*')."""
        return _validate_ipv4_address(v)


class ROS2Config(BaseModel):
    """Configuration for ROS2 output."""

    topic_name: str = Field(..., min_length=1, description="ROS2 topic name")
    node_name: str = Field(..., min_length=1, description="ROS2 node name")

    @field_validator("topic_name")
    @classmethod
    def validate_topic_name(cls, v: str) -> str:
        """Validate ROS2 topic name - lowercase, alphanumeric, underscores, slashes."""
        if not v:
            raise ValueError("Topic name cannot be empty")
        if not re.match(r"^[a-z0-9_/]+$", v):
            raise ValueError(
                "Topic name should only contain lowercase letters, numbers, underscores, and forward slashes"
            )
        return v

    @field_validator("node_name")
    @classmethod
    def validate_node_name(cls, v: str) -> str:
        """Validate ROS2 node name - alphanumeric and underscores, must start with letter."""
        if not v:
            raise ValueError("Node name cannot be empty")
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", v):
            raise ValueError(
                "Node name must start with a letter and contain only letters, numbers, and underscores"
            )
        return v


class ProtocolInstanceConfig(BaseModel):
    """Wrapper for protocol config with enabled flag."""

    enabled: bool = Field(False, description="Protocol enabled")
    config: dict = Field(default_factory=dict, description="Protocol-specific configuration")
