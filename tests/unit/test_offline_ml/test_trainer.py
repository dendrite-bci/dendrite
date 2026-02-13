"""Tests for offline ML trainer module - TDD approach."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from dendrite.auxiliary.ml_workbench import TrainResult
from dendrite.ml.decoders.decoder_schemas import DecoderConfig


@pytest.fixture
def sample_data():
    """Create sample EEG data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_channels = 8
    n_times = 250
    X = np.random.randn(n_samples, n_channels, n_times).astype(np.float32)
    y = np.random.randint(0, 2, n_samples)
    return X, y


class TestOfflineTrainer:
    """Tests for OfflineTrainer class."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return DecoderConfig(
            model_type="EEGNet",
            epochs=2,  # Fast for tests
            batch_size=16,
            learning_rate=0.001,
            modality="eeg",
        )

    def test_trainer_creation(self):
        """Trainer should initialize without errors."""
        from dendrite.auxiliary.ml_workbench import OfflineTrainer

        trainer = OfflineTrainer()
        assert trainer is not None

    def test_train_returns_result(self, sample_data, config):
        """Train should return a TrainResult."""
        from dendrite.auxiliary.ml_workbench import OfflineTrainer

        X, y = sample_data
        trainer = OfflineTrainer()

        result = trainer.train(X, y, config)

        assert isinstance(result, TrainResult)
        assert result.decoder is not None
        assert 0 <= result.accuracy <= 1
        assert 0 <= result.val_accuracy <= 1

    def test_train_creates_decoder(self, sample_data, config):
        """Train should create a usable decoder."""
        from dendrite.auxiliary.ml_workbench import OfflineTrainer

        X, y = sample_data
        trainer = OfflineTrainer()

        result = trainer.train(X, y, config)

        # Decoder should be able to predict
        assert hasattr(result.decoder, "predict")

    def test_train_confusion_matrix(self, sample_data, config):
        """Train should produce confusion matrix."""
        from dendrite.auxiliary.ml_workbench import OfflineTrainer

        X, y = sample_data
        trainer = OfflineTrainer()

        result = trainer.train(X, y, config)

        assert result.confusion_matrix is not None
        assert result.confusion_matrix.shape[0] == result.confusion_matrix.shape[1]

    def test_train_history(self, sample_data, config):
        """Train should record training history."""
        from dendrite.auxiliary.ml_workbench import OfflineTrainer

        X, y = sample_data
        trainer = OfflineTrainer()

        result = trainer.train(X, y, config)

        assert result.train_history is not None
        assert isinstance(result.train_history, dict)

    def test_train_progress_callback(self, sample_data, config):
        """Train should call progress callback."""
        from dendrite.auxiliary.ml_workbench import OfflineTrainer

        X, y = sample_data
        trainer = OfflineTrainer()

        progress_calls = []
        def on_progress(msg):
            progress_calls.append(msg)

        result = trainer.train(X, y, config, progress_callback=on_progress)

        assert len(progress_calls) > 0

    def test_train_with_cv(self, sample_data, config):
        """Train with cross-validation should include CV results."""
        from dendrite.auxiliary.ml_workbench import OfflineTrainer

        X, y = sample_data
        trainer = OfflineTrainer()

        result = trainer.train(X, y, config, n_folds=3)

        assert result.cv_results is not None
        assert "fold_scores" in result.cv_results
        assert "mean_accuracy" in result.cv_results

    def test_train_invalid_data_raises(self, config):
        """Train with invalid data should raise."""
        from dendrite.auxiliary.ml_workbench import OfflineTrainer

        trainer = OfflineTrainer()

        # Empty data
        with pytest.raises(ValueError):
            trainer.train(np.array([]), np.array([]), config)

    def test_train_mismatched_labels_raises(self, config):
        """Train with mismatched X/y should raise."""
        from dendrite.auxiliary.ml_workbench import OfflineTrainer

        trainer = OfflineTrainer()
        X = np.random.randn(100, 8, 250)
        y = np.random.randint(0, 2, 50)  # Wrong length

        with pytest.raises(ValueError):
            trainer.train(X, y, config)

    def test_train_eegnet_default(self, sample_data, config):
        """Train should work with default EEGNet model."""
        from dendrite.auxiliary.ml_workbench import OfflineTrainer

        X, y = sample_data
        trainer = OfflineTrainer()

        # EEGNet is the default and known to work
        result = trainer.train(X, y, config)
        assert result.decoder is not None
        assert result.accuracy >= 0


class TestConfusionMatrixOnValidationData:
    """Tests that confusion matrix is computed on held-out data, not training data."""

    @pytest.fixture
    def config(self):
        """Create test config with known validation split."""
        return DecoderConfig(
            model_type="EEGNet",
            epochs=2,
            batch_size=16,
            learning_rate=0.001,
            modality="eeg",
            validation_split=0.2,
        )

    def test_simple_train_confusion_matrix_uses_validation_data(self, sample_data, config):
        """Confusion matrix from simple training should use validation set, not full data."""
        from dendrite.auxiliary.ml_workbench import OfflineTrainer

        X, y = sample_data
        trainer = OfflineTrainer()

        result = trainer.train(X, y, config)

        # Confusion matrix sum should equal validation set size, not total samples
        n_val = int(len(y) * config.validation_split)
        conf_total = result.confusion_matrix.sum()
        assert conf_total == n_val, (
            f"Confusion matrix has {conf_total} samples but validation set has {n_val}. "
            f"Matrix should be computed on validation data, not training data."
        )

    def test_cv_train_confusion_matrix_uses_oof_predictions(self, sample_data, config):
        """Confusion matrix from CV training should use out-of-fold predictions."""
        from dendrite.auxiliary.ml_workbench import OfflineTrainer

        X, y = sample_data
        trainer = OfflineTrainer()

        result = trainer.train(X, y, config, n_folds=3)

        # OOF predictions cover all samples
        conf_total = result.confusion_matrix.sum()
        assert conf_total == len(y), (
            f"CV confusion matrix has {conf_total} samples but should have {len(y)} "
            f"(all samples via out-of-fold predictions)."
        )

    def test_confusion_matrix_shape_matches_classes(self, sample_data, config):
        """Confusion matrix dimensions should match number of classes."""
        from dendrite.auxiliary.ml_workbench import OfflineTrainer

        X, y = sample_data
        trainer = OfflineTrainer()

        result = trainer.train(X, y, config)

        n_classes = len(np.unique(y))
        assert result.confusion_matrix.shape == (n_classes, n_classes)

    def test_simple_train_no_validation_split_uses_training_data(self):
        """When validation_split=0, confusion matrix falls back to training data."""
        from dendrite.auxiliary.ml_workbench import OfflineTrainer

        np.random.seed(42)
        X = np.random.randn(50, 8, 250).astype(np.float32)
        y = np.random.randint(0, 2, 50)

        config = DecoderConfig(
            model_type="EEGNet",
            epochs=2,
            batch_size=16,
            modality="eeg",
            validation_split=0.0,
        )

        trainer = OfflineTrainer()
        result = trainer.train(X, y, config)

        # With no validation split, falls back to training data
        conf_total = result.confusion_matrix.sum()
        assert conf_total == len(y)


class TestTrainerIntegration:
    """Integration tests that use real decoder factory."""

    @pytest.fixture
    def real_data(self):
        """Create realistic EEG data."""
        np.random.seed(42)
        n_samples = 50
        n_channels = 8
        n_times = 125  # 0.5 seconds at 250 Hz
        X = np.random.randn(n_samples, n_channels, n_times).astype(np.float32)
        y = np.random.randint(0, 2, n_samples)
        return X, y

    @pytest.mark.slow
    def test_full_training_pipeline(self, real_data):
        """Test complete training pipeline with real decoder."""
        from dendrite.auxiliary.ml_workbench import OfflineTrainer
        from dendrite.ml.decoders.decoder_schemas import DecoderConfig

        X, y = real_data
        config = DecoderConfig(
            model_type="EEGNet",
            epochs=3,
            batch_size=8,
            modality="eeg",
        )

        trainer = OfflineTrainer()
        result = trainer.train(X, y, config)

        # Verify result completeness
        assert result.decoder is not None
        assert result.accuracy >= 0
        assert result.confusion_matrix is not None
        assert result.train_history is not None
