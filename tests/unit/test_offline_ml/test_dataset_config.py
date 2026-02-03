"""Tests for DatasetConfig dataclass - TDD approach.

Tests validation, serialization, and computed properties.
"""

import pytest
import json


class TestDatasetConfigDefaults:
    """Test default values and basic creation."""

    def test_default_values(self):
        """DatasetConfig should have sensible defaults."""
        from dendrite.auxiliary.ml_workbench.datasets.config import DatasetConfig

        config = DatasetConfig(name="test_dataset")

        assert config.name == "test_dataset"
        assert config.description == ""
        assert config.data_root == ""
        assert config.file_pattern == "*.fif"
        assert config.source_type == "fif"
        assert config.subjects == []
        assert config.events == {}
        assert config.blocks is None
        assert config.sample_rate == 500.0
        assert config.channels is None
        assert config.epoch_tmin == 0.0
        assert config.epoch_tmax == 0.5
        assert config.preproc_lowcut is None
        assert config.preproc_highcut is None
        assert config.preproc_rereference is False
        assert config.moabb_dataset is None
        assert config.moabb_paradigm is None

    def test_custom_values(self):
        """DatasetConfig should accept custom values."""
        from dendrite.auxiliary.ml_workbench.datasets.config import DatasetConfig

        config = DatasetConfig(
            name="custom_dataset",
            description="A custom dataset",
            data_root="/path/to/data",
            source_type="hdf5",
            subjects=[1, 2, 3],
            events={"error": 1, "correct": 0},
            sample_rate=256.0,
            epoch_tmin=-0.2,
            epoch_tmax=0.8,
        )

        assert config.name == "custom_dataset"
        assert config.description == "A custom dataset"
        assert config.data_root == "/path/to/data"
        assert config.source_type == "hdf5"
        assert config.subjects == [1, 2, 3]
        assert config.events == {"error": 1, "correct": 0}
        assert config.sample_rate == 256.0
        assert config.epoch_tmin == -0.2
        assert config.epoch_tmax == 0.8


class TestDatasetConfigValidation:
    """Test validate() method catches issues."""

    def test_validate_valid_fif_config(self):
        """Valid FIF config should pass validation."""
        from dendrite.auxiliary.ml_workbench.datasets.config import DatasetConfig

        config = DatasetConfig(
            name="valid_fif",
            data_root="/path/to/data",
            subjects=[1, 2],
            events={"error": 1, "correct": 0},
        )

        issues = config.validate()
        assert issues == []

    def test_validate_valid_moabb_config(self):
        """Valid MOABB config should pass validation."""
        from dendrite.auxiliary.ml_workbench.datasets.config import DatasetConfig

        config = DatasetConfig(
            name="valid_moabb",
            source_type="moabb",
            subjects=[1, 2, 3],
            moabb_dataset="BNCI2014_001",
            moabb_paradigm="MotorImagery",
        )

        issues = config.validate()
        assert issues == []

    def test_validate_empty_name(self):
        """Empty name should fail validation."""
        from dendrite.auxiliary.ml_workbench.datasets.config import DatasetConfig

        config = DatasetConfig(
            name="",
            data_root="/path/to/data",
            subjects=[1],
            events={"event": 1},
        )

        issues = config.validate()
        assert any("name" in issue.lower() for issue in issues)

    def test_validate_moabb_requires_dataset(self):
        """MOABB source type requires moabb_dataset."""
        from dendrite.auxiliary.ml_workbench.datasets.config import DatasetConfig

        config = DatasetConfig(
            name="missing_moabb_dataset",
            source_type="moabb",
            subjects=[1],
            moabb_paradigm="MotorImagery",
        )

        issues = config.validate()
        assert any("moabb_dataset" in issue.lower() for issue in issues)

    def test_validate_moabb_requires_paradigm(self):
        """MOABB source type requires moabb_paradigm."""
        from dendrite.auxiliary.ml_workbench.datasets.config import DatasetConfig

        config = DatasetConfig(
            name="missing_moabb_paradigm",
            source_type="moabb",
            subjects=[1],
            moabb_dataset="BNCI2014_001",
        )

        issues = config.validate()
        assert any("moabb_paradigm" in issue.lower() for issue in issues)

    def test_validate_non_moabb_requires_data_root(self):
        """Non-MOABB source type requires data_root."""
        from dendrite.auxiliary.ml_workbench.datasets.config import DatasetConfig

        config = DatasetConfig(
            name="missing_data_root",
            source_type="fif",
            subjects=[1],
            events={"event": 1},
            data_root="",  # empty
        )

        issues = config.validate()
        assert any("data_root" in issue.lower() for issue in issues)

    def test_validate_empty_subjects(self):
        """Empty subjects list should fail validation."""
        from dendrite.auxiliary.ml_workbench.datasets.config import DatasetConfig

        config = DatasetConfig(
            name="no_subjects",
            data_root="/path/to/data",
            subjects=[],  # empty
            events={"event": 1},
        )

        issues = config.validate()
        assert any("subjects" in issue.lower() for issue in issues)

    def test_validate_empty_events_fif(self):
        """Empty events for non-MOABB should fail validation."""
        from dendrite.auxiliary.ml_workbench.datasets.config import DatasetConfig

        config = DatasetConfig(
            name="no_events",
            data_root="/path/to/data",
            subjects=[1],
            events={},  # empty
        )

        issues = config.validate()
        assert any("events" in issue.lower() for issue in issues)

    def test_validate_empty_events_moabb_allowed(self):
        """Empty events for MOABB is allowed (auto-generated)."""
        from dendrite.auxiliary.ml_workbench.datasets.config import DatasetConfig

        config = DatasetConfig(
            name="moabb_no_events",
            source_type="moabb",
            subjects=[1],
            events={},
            moabb_dataset="BNCI2014_001",
            moabb_paradigm="MotorImagery",
        )

        issues = config.validate()
        assert not any("events" in issue.lower() for issue in issues)

    def test_validate_invalid_sample_rate(self):
        """Zero or negative sampling rate should fail."""
        from dendrite.auxiliary.ml_workbench.datasets.config import DatasetConfig

        config = DatasetConfig(
            name="bad_sampling",
            data_root="/path",
            subjects=[1],
            events={"e": 1},
            sample_rate=0,
        )

        issues = config.validate()
        assert any("sample_rate" in issue.lower() for issue in issues)

    def test_validate_negative_sample_rate(self):
        """Negative sampling rate should fail."""
        from dendrite.auxiliary.ml_workbench.datasets.config import DatasetConfig

        config = DatasetConfig(
            name="negative_sampling",
            data_root="/path",
            subjects=[1],
            events={"e": 1},
            sample_rate=-100,
        )

        issues = config.validate()
        assert any("sample_rate" in issue.lower() for issue in issues)

    def test_validate_epoch_bounds(self):
        """tmax must be greater than tmin."""
        from dendrite.auxiliary.ml_workbench.datasets.config import DatasetConfig

        config = DatasetConfig(
            name="bad_epoch",
            data_root="/path",
            subjects=[1],
            events={"e": 1},
            epoch_tmin=0.5,
            epoch_tmax=0.2,  # tmax < tmin
        )

        issues = config.validate()
        assert any("epoch" in issue.lower() or "tmax" in issue.lower() for issue in issues)

    def test_validate_epoch_bounds_equal(self):
        """tmax equal to tmin should fail."""
        from dendrite.auxiliary.ml_workbench.datasets.config import DatasetConfig

        config = DatasetConfig(
            name="equal_epoch",
            data_root="/path",
            subjects=[1],
            events={"e": 1},
            epoch_tmin=0.5,
            epoch_tmax=0.5,  # equal
        )

        issues = config.validate()
        assert any("epoch" in issue.lower() or "tmax" in issue.lower() for issue in issues)


class TestDatasetConfigWindowSamples:
    """Test window_samples computed property."""

    def test_window_samples_default(self):
        """Default epoch (0.0 to 0.5s at 500Hz) should be 251 samples."""
        from dendrite.auxiliary.ml_workbench.datasets.config import DatasetConfig

        config = DatasetConfig(name="test")

        # (0.5 - 0.0) * 500 + 1 = 251
        assert config.window_samples == 251

    def test_window_samples_custom_epoch(self):
        """Custom epoch bounds should calculate correctly."""
        from dendrite.auxiliary.ml_workbench.datasets.config import DatasetConfig

        config = DatasetConfig(
            name="test",
            epoch_tmin=-0.2,
            epoch_tmax=0.8,
            sample_rate=250.0,
        )

        # (0.8 - (-0.2)) * 250 + 1 = 1.0 * 250 + 1 = 251
        assert config.window_samples == 251

    def test_window_samples_high_rate(self):
        """Higher sampling rate should increase samples."""
        from dendrite.auxiliary.ml_workbench.datasets.config import DatasetConfig

        config = DatasetConfig(
            name="test",
            epoch_tmin=0.0,
            epoch_tmax=1.0,
            sample_rate=1000.0,
        )

        # (1.0 - 0.0) * 1000 + 1 = 1001
        assert config.window_samples == 1001

    def test_window_samples_negative_tmin(self):
        """Negative tmin should work correctly."""
        from dendrite.auxiliary.ml_workbench.datasets.config import DatasetConfig

        config = DatasetConfig(
            name="test",
            epoch_tmin=-0.5,
            epoch_tmax=0.5,
            sample_rate=500.0,
        )

        # (0.5 - (-0.5)) * 500 + 1 = 1.0 * 500 + 1 = 501
        assert config.window_samples == 501


class TestDatasetConfigSerialization:
    """Test serialization to/from dict and JSON."""

    def test_to_dict(self):
        """to_dict should return complete dictionary."""
        from dendrite.auxiliary.ml_workbench.datasets.config import DatasetConfig

        config = DatasetConfig(
            name="test",
            data_root="/path",
            subjects=[1, 2, 3],
            events={"a": 0, "b": 1},
            sample_rate=256.0,
        )

        d = config.to_dict()

        assert isinstance(d, dict)
        assert d["name"] == "test"
        assert d["data_root"] == "/path"
        assert d["subjects"] == [1, 2, 3]
        assert d["events"] == {"a": 0, "b": 1}
        assert d["sample_rate"] == 256.0
        # All fields should be present
        assert "epoch_tmin" in d
        assert "epoch_tmax" in d
        assert "source_type" in d

    def test_from_dict(self):
        """from_dict should restore config correctly."""
        from dendrite.auxiliary.ml_workbench.datasets.config import DatasetConfig

        data = {
            "name": "restored",
            "data_root": "/restored/path",
            "subjects": [5, 6],
            "events": {"x": 0, "y": 1},
            "sample_rate": 128.0,
            "epoch_tmin": -0.1,
            "epoch_tmax": 0.9,
            "source_type": "hdf5",
            "file_pattern": "*.fif",
            "description": "",
            "blocks": None,
            "channels": None,
            "preproc_lowcut": None,
            "preproc_highcut": None,
            "preproc_rereference": False,
            "event_patterns": None,
            "moabb_dataset": None,
            "moabb_paradigm": None,
            "moabb_n_classes": None,
            "moabb_events": None,
        }

        config = DatasetConfig.from_dict(data)

        assert config.name == "restored"
        assert config.data_root == "/restored/path"
        assert config.subjects == [5, 6]
        assert config.events == {"x": 0, "y": 1}
        assert config.sample_rate == 128.0
        assert config.epoch_tmin == -0.1
        assert config.epoch_tmax == 0.9
        assert config.source_type == "hdf5"

    def test_to_dict_from_dict_roundtrip(self):
        """Config should survive to_dict -> from_dict roundtrip."""
        from dendrite.auxiliary.ml_workbench.datasets.config import DatasetConfig

        original = DatasetConfig(
            name="roundtrip_test",
            description="Test roundtrip",
            data_root="/test/path",
            source_type="edf",
            subjects=[10, 20, 30],
            events={"event1": 0, "event2": 1, "event3": 2},
            blocks={"train": [1, 2], "test": [3]},
            sample_rate=512.0,
            channels=["Cz", "Pz", "Fz"],
            epoch_tmin=-0.3,
            epoch_tmax=0.7,
            preproc_lowcut=1.0,
            preproc_highcut=40.0,
            preproc_rereference=True,
        )

        restored = DatasetConfig.from_dict(original.to_dict())

        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.data_root == original.data_root
        assert restored.source_type == original.source_type
        assert restored.subjects == original.subjects
        assert restored.events == original.events
        assert restored.blocks == original.blocks
        assert restored.sample_rate == original.sample_rate
        assert restored.channels == original.channels
        assert restored.epoch_tmin == original.epoch_tmin
        assert restored.epoch_tmax == original.epoch_tmax
        assert restored.preproc_lowcut == original.preproc_lowcut
        assert restored.preproc_highcut == original.preproc_highcut
        assert restored.preproc_rereference == original.preproc_rereference

    def test_to_json(self):
        """to_json should return valid JSON string."""
        from dendrite.auxiliary.ml_workbench.datasets.config import DatasetConfig

        config = DatasetConfig(
            name="json_test",
            subjects=[1, 2],
            events={"a": 1},
        )

        json_str = config.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["name"] == "json_test"
        assert parsed["subjects"] == [1, 2]

    def test_from_json(self):
        """from_json should parse JSON correctly."""
        from dendrite.auxiliary.ml_workbench.datasets.config import DatasetConfig

        json_str = json.dumps({
            "name": "from_json_test",
            "data_root": "/json/path",
            "subjects": [3, 4],
            "events": {"b": 0},
            "sample_rate": 300.0,
            "description": "",
            "file_pattern": "*.fif",
            "source_type": "fif",
            "blocks": None,
            "channels": None,
            "epoch_tmin": 0.0,
            "epoch_tmax": 0.5,
            "preproc_lowcut": None,
            "preproc_highcut": None,
            "preproc_rereference": False,
            "event_patterns": None,
            "moabb_dataset": None,
            "moabb_paradigm": None,
            "moabb_n_classes": None,
            "moabb_events": None,
        })

        config = DatasetConfig.from_json(json_str)

        assert config.name == "from_json_test"
        assert config.data_root == "/json/path"
        assert config.subjects == [3, 4]
        assert config.sample_rate == 300.0

    def test_to_json_from_json_roundtrip(self):
        """Config should survive to_json -> from_json roundtrip."""
        from dendrite.auxiliary.ml_workbench.datasets.config import DatasetConfig

        original = DatasetConfig(
            name="json_roundtrip",
            data_root="/json/roundtrip",
            subjects=[7, 8, 9],
            events={"left": 0, "right": 1},
            sample_rate=1000.0,
            epoch_tmin=-0.5,
            epoch_tmax=1.5,
        )

        json_str = original.to_json()
        restored = DatasetConfig.from_json(json_str)

        assert restored.name == original.name
        assert restored.data_root == original.data_root
        assert restored.subjects == original.subjects
        assert restored.events == original.events
        assert restored.sample_rate == original.sample_rate
        assert restored.epoch_tmin == original.epoch_tmin
        assert restored.epoch_tmax == original.epoch_tmax
