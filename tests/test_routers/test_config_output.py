"""Tests for output protocol configuration endpoints."""


async def test_get_availability(client):
    res = await client.get("/api/config/output/availability")
    assert res.status_code == 200
    data = res.json()
    assert data["lsl"] is True
    assert data["socket"] is True
    assert isinstance(data["zmq"], bool)
    assert isinstance(data["ros2"], bool)


async def test_get_defaults(client):
    res = await client.get("/api/config/output/defaults")
    assert res.status_code == 200
    data = res.json()
    for proto in ("lsl", "socket", "zmq", "ros2"):
        assert proto in data
        assert isinstance(data[proto], dict)
        assert len(data[proto]) > 0
    # Spot-check known defaults
    assert data["lsl"]["stream_name"] == "PredictionStream"
    assert data["socket"]["port"] == 8080
    assert data["zmq"]["port"] == 5556
    assert data["ros2"]["topic_name"] == "bmi_predictions"


async def test_update_valid_lsl(client):
    res = await client.put("/api/config/output", json={
        "protocols": {
            "lsl": {
                "enabled": True,
                "config": {
                    "stream_name": "MyStream",
                    "stream_type": "EEG",
                    "source_id": "test_source",
                },
            }
        }
    })
    assert res.status_code == 200
    assert res.json()["output"]["lsl"]["enabled"] is True


async def test_update_invalid_port(client):
    res = await client.put("/api/config/output", json={
        "protocols": {
            "socket": {
                "enabled": True,
                "config": {"protocol": "TCP", "ip": "127.0.0.1", "port": 99999},
            }
        }
    })
    assert res.status_code == 422
    detail = res.json()["detail"]
    assert "protocol_errors" in detail
    assert "socket" in detail["protocol_errors"]


async def test_disabled_skips_validation(client):
    """Disabled protocols with invalid config should not cause validation errors."""
    res = await client.put("/api/config/output", json={
        "protocols": {
            "socket": {
                "enabled": False,
                "config": {"protocol": "TCP", "ip": "invalid", "port": 99999},
            }
        }
    })
    assert res.status_code == 200


async def test_invalid_ros2_topic(client):
    res = await client.put("/api/config/output", json={
        "protocols": {
            "ros2": {
                "enabled": True,
                "config": {"topic_name": "UPPERCASE", "node_name": "valid_node"},
            }
        }
    })
    assert res.status_code == 422
    errors = res.json()["detail"]["protocol_errors"]["ros2"]
    assert any("topic_name" in e["field"] for e in errors)
