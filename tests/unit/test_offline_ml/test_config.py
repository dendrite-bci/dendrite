"""Tests for offline ML config module - TDD approach."""

import pytest
import numpy as np
from typing import Dict, Any


class TestDecoderConfig:
    """Tests for DecoderConfig (used by ml_workbench)."""

    def test_decoder_config_defaults(self):
        """Config should have sensible defaults."""
        from dendrite.ml.decoders.decoder_schemas import DecoderConfig

        config = DecoderConfig()

        assert config.model_type == "EEGNet"
        assert config.epochs == 100
        assert config.batch_size == 32
        assert config.learning_rate == 0.001
        assert config.validation_split == 0.2

    def test_decoder_config_custom_values(self):
        """Config should accept custom values."""
        from dendrite.ml.decoders.decoder_schemas import DecoderConfig

        config = DecoderConfig(
            model_type="CSP+LDA",
            epochs=50,
            batch_size=16,
            learning_rate=0.01,
        )

        assert config.model_type == "CSP+LDA"
        assert config.epochs == 50
        assert config.batch_size == 16
        assert config.learning_rate == 0.01

    def test_decoder_config_model_dump(self):
        """Config should serialize to dictionary."""
        from dendrite.ml.decoders.decoder_schemas import DecoderConfig

        config = DecoderConfig(model_type="EEGNet", epochs=25)
        d = config.model_dump()

        assert isinstance(d, dict)
        assert d["model_type"] == "EEGNet"
        assert d["epochs"] == 25
        assert "batch_size" in d  # defaults included

    def test_decoder_config_from_dict(self):
        """Config should deserialize from dictionary."""
        from dendrite.ml.decoders.decoder_schemas import DecoderConfig

        data = {
            "model_type": "EEGNet",
            "epochs": 200,
            "batch_size": 64,
            "learning_rate": 0.0001,
            "validation_split": 0.3,
            "modality": "emg",
        }
        config = DecoderConfig(**data)

        assert config.model_type == "EEGNet"
        assert config.epochs == 200
        assert config.batch_size == 64
        assert config.modality == "emg"

    def test_decoder_config_roundtrip(self):
        """Config should survive model_dump -> reconstruction roundtrip."""
        from dendrite.ml.decoders.decoder_schemas import DecoderConfig

        original = DecoderConfig(
            model_type="TransformerEEG",
            epochs=150,
            batch_size=64,
            learning_rate=0.0005,
        )
        restored = DecoderConfig(**original.model_dump())

        assert restored.model_type == original.model_type
        assert restored.epochs == original.epochs
        assert restored.batch_size == original.batch_size
        assert restored.learning_rate == original.learning_rate

    def test_decoder_config_pydantic_validation(self):
        """Config should validate via pydantic."""
        from dendrite.ml.decoders.decoder_schemas import DecoderConfig
        from pydantic import ValidationError

        # Invalid epochs (less than min)
        with pytest.raises(ValidationError):
            DecoderConfig(epochs=0)

        # Invalid learning_rate (greater than max)
        with pytest.raises(ValidationError):
            DecoderConfig(learning_rate=2.0)


class TestTrainResult:
    """Tests for TrainResult dataclass."""

    def test_train_result_creation(self):
        """TrainResult should store training outcomes."""
        from dendrite.auxiliary.ml_workbench import TrainResult

        # Mock decoder and data
        mock_decoder = object()
        confusion = np.array([[10, 2], [1, 12]])
        history = {"loss": [1.0, 0.5, 0.2], "accuracy": [0.5, 0.8, 0.9]}

        result = TrainResult(
            decoder=mock_decoder,
            accuracy=0.88,
            val_accuracy=0.85,
            confusion_matrix=confusion,
            train_history=history,
        )

        assert result.decoder is mock_decoder
        assert result.accuracy == 0.88
        assert result.val_accuracy == 0.85
        assert result.confusion_matrix.shape == (2, 2)
        assert len(result.train_history["loss"]) == 3

    def test_train_result_optional_cv(self):
        """TrainResult cv_results should be optional."""
        from dendrite.auxiliary.ml_workbench import TrainResult

        result = TrainResult(
            decoder=None,
            accuracy=0.9,
            val_accuracy=0.85,
            confusion_matrix=np.eye(2),
            train_history={},
        )

        assert result.cv_results is None

    def test_train_result_with_cv(self):
        """TrainResult should accept cv_results."""
        from dendrite.auxiliary.ml_workbench import TrainResult

        cv_results = {
            "fold_scores": [0.85, 0.87, 0.89, 0.86, 0.88],
            "mean_accuracy": 0.87,
            "std_accuracy": 0.015,
        }

        result = TrainResult(
            decoder=None,
            accuracy=0.87,
            val_accuracy=0.85,
            confusion_matrix=np.eye(2),
            train_history={},
            cv_results=cv_results,
        )

        assert result.cv_results["mean_accuracy"] == 0.87
        assert len(result.cv_results["fold_scores"]) == 5
