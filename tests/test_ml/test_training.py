"""Tests for training loop behaviors with tiny data."""

import numpy as np
import pytest
import torch

from dendrite.ml import create_decoder
from dendrite.ml.training.augmentation import apply_cutmix, apply_mixup
from dendrite.ml.training.losses import FocalLoss


# ---------------------------------------------------------------------------
# Basic Training
# ---------------------------------------------------------------------------


class TestTrainingBasic:
    def test_training_loop_fit_returns_history(self, eeg_data_2class, fast_decoder_config):
        X, y = eeg_data_2class
        cfg = {**fast_decoder_config, "input_shapes": {"eeg": [8, 64]}}
        decoder = create_decoder(**cfg)
        decoder.fit(X, y)
        metrics = decoder.get_training_metrics()
        assert metrics is not None
        # Metrics are nested under classifier name
        for component_metrics in metrics.values():
            assert "history" in component_metrics

    def test_training_history_has_expected_keys(self, eeg_data_2class, fast_decoder_config):
        X, y = eeg_data_2class
        cfg = {**fast_decoder_config, "input_shapes": {"eeg": [8, 64]}}
        decoder = create_decoder(**cfg)
        decoder.fit(X, y)
        history = decoder.get_training_history()
        assert history is not None
        assert "loss" in history
        assert "accuracy" in history

    def test_training_respects_epoch_count(self, eeg_data_2class, fast_decoder_config):
        X, y = eeg_data_2class
        cfg = {**fast_decoder_config, "input_shapes": {"eeg": [8, 64]}, "epochs": 3}
        decoder = create_decoder(**cfg)
        decoder.fit(X, y)
        history = decoder.get_training_history()
        assert history is not None
        assert len(history["loss"]) == 3

    def test_training_without_validation(self, eeg_data_2class, fast_decoder_config):
        X, y = eeg_data_2class
        cfg = {
            **fast_decoder_config,
            "input_shapes": {"eeg": [8, 64]},
            "validation_split": 0.0,
        }
        decoder = create_decoder(**cfg)
        decoder.fit(X, y)
        assert decoder.is_fitted
        history = decoder.get_training_history()
        assert history is not None
        assert "loss" in history


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    def test_early_stopping_stops_early(self, eeg_data_2class):
        X, y = eeg_data_2class
        decoder = create_decoder(
            model_type="EEGNet",
            num_classes=2,
            input_shapes={"eeg": [8, 64]},
            epochs=100,
            batch_size=10,
            device="cpu",
            use_early_stopping=True,
            early_stopping_patience=3,
            validation_split=0.2,
            use_lr_scheduler=False,
            use_lr_warmup=False,
            use_augmentation=False,
            use_swa=False,
        )
        decoder.fit(X, y)
        history = decoder.get_training_history()
        # With patience=3 on tiny random data, it should stop well before 100
        assert history is not None
        assert len(history["loss"]) < 100

    def test_early_stopping_patience(self, eeg_data_2class):
        X, y = eeg_data_2class
        decoder = create_decoder(
            model_type="EEGNet",
            num_classes=2,
            input_shapes={"eeg": [8, 64]},
            epochs=50,
            batch_size=10,
            device="cpu",
            use_early_stopping=True,
            early_stopping_patience=5,
            validation_split=0.2,
            use_lr_scheduler=False,
            use_lr_warmup=False,
            use_augmentation=False,
            use_swa=False,
        )
        decoder.fit(X, y)
        history = decoder.get_training_history()
        assert history is not None
        # Should train for at least patience epochs
        assert len(history["loss"]) >= 5


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


class TestAugmentation:
    def test_mixup_shape_preserved(self):
        X = np.random.randn(10, 8, 64).astype(np.float32)
        y = np.array([0, 1] * 5)
        x_mixed, y1, y2, lam = apply_mixup(X, y, alpha=0.2)
        assert x_mixed.shape == X.shape
        assert y1.shape == y.shape
        assert y2.shape == y.shape
        assert 0.0 <= lam <= 1.0

    def test_cutmix_shape_preserved(self):
        X = np.random.randn(10, 8, 64).astype(np.float32)
        y = np.array([0, 1] * 5)
        x_mixed, y1, y2, lam = apply_cutmix(X, y, alpha=0.2)
        assert x_mixed.shape == X.shape
        assert y1.shape == y.shape
        assert y2.shape == y.shape
        assert 0.0 <= lam <= 1.0

    def test_focal_loss_shape(self):
        loss_fn = FocalLoss(gamma=2.0)
        inputs = torch.randn(10, 3)  # batch=10, classes=3
        targets = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        loss = loss_fn(inputs, targets)
        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0


# ---------------------------------------------------------------------------
# Integration (slow)
# ---------------------------------------------------------------------------


class TestTrainingIntegration:
    @pytest.mark.slow
    def test_training_improves_accuracy(self):
        """Accuracy > chance after 50 epochs on easy separable synthetic data."""
        rng = np.random.RandomState(42)
        n_samples = 40
        X = rng.randn(n_samples, 8, 64).astype(np.float32)
        # Make class 0 have positive mean, class 1 negative
        y = np.array([0] * 20 + [1] * 20)
        X[:20] += 0.5
        X[20:] -= 0.5

        decoder = create_decoder(
            model_type="EEGNet",
            num_classes=2,
            input_shapes={"eeg": [8, 64]},
            epochs=50,
            batch_size=10,
            device="cpu",
            validation_split=0.0,
            use_early_stopping=False,
            use_augmentation=False,
            use_swa=False,
            use_lr_scheduler=True,
            lr_scheduler_type="OneCycleLR",
            use_lr_warmup=False,
        )
        decoder.fit(X, y)
        score = decoder.score(X, y)
        assert score > 0.5  # Better than chance

    @pytest.mark.slow
    def test_training_with_all_features(self, eeg_data_2class):
        """Training with early_stopping + augmentation + SWA together."""
        X, y = eeg_data_2class
        decoder = create_decoder(
            model_type="EEGNet",
            num_classes=2,
            input_shapes={"eeg": [8, 64]},
            epochs=10,
            batch_size=10,
            device="cpu",
            use_early_stopping=True,
            early_stopping_patience=5,
            use_augmentation=True,
            aug_strategy="light",
            use_swa=True,
            swa_start_epoch=0.5,
            validation_split=0.2,
            use_lr_scheduler=False,
            use_lr_warmup=False,
        )
        decoder.fit(X, y)
        assert decoder.is_fitted
        preds = decoder.predict(X)
        assert len(preds) == 20 or isinstance(preds, (int, np.integer))
