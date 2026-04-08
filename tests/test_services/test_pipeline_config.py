"""Tests for PipelineConfig — typed pipeline configuration contract."""

import pytest
from pydantic import ValidationError

from dendrite.data.stream_schemas import StreamMetadata
from dendrite.processing.pipeline_schemas import PipelineConfig


def test_minimal_config():
    """PipelineConfig with all defaults should be valid."""
    cfg = PipelineConfig()
    assert cfg.mode_instances == {}
    assert cfg.stream_configs == []


def test_full_config():
    """PipelineConfig with all fields populated."""
    stream = StreamMetadata(
        name="TestEEG", type="EEG", channel_count=8,
        sample_rate=250.0, channel_format="float32",
    )
    cfg = PipelineConfig(
        study_name="test_study",
        subject_id="001",
        session_id="01",
        recording_name="task",
        stream_configs=[stream],
        modalities_by_stream={},
        mode_instances={"P300": {"name": "P300", "mode": "synchronous"}},
        output={"protocols": {"lsl": {"enabled": True}}},
    )
    assert len(cfg.stream_configs) == 1
    assert cfg.stream_configs[0].name == "TestEEG"


def test_extra_fields_rejected():
    """PipelineConfig should reject unknown fields."""
    with pytest.raises(ValidationError):
        PipelineConfig(unknown_field="oops")


def test_model_dump_roundtrip():
    """model_dump() should produce a dict that can reconstruct the model."""
    cfg = PipelineConfig(study_name="roundtrip")
    dumped = cfg.model_dump()
    restored = PipelineConfig(**dumped)
    assert restored.study_name == "roundtrip"


def test_json_serialization():
    """model_dump(mode='json') should produce JSON-safe output."""
    stream = StreamMetadata(
        name="TestEEG", type="EEG", channel_count=4,
        sample_rate=250.0, channel_format="float32",
    )
    cfg = PipelineConfig(stream_configs=[stream])
    json_dict = cfg.model_dump(mode="json")
    assert isinstance(json_dict["stream_configs"][0], dict)
    assert json_dict["stream_configs"][0]["name"] == "TestEEG"
