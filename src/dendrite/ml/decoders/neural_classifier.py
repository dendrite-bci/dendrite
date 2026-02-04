"""
Neural Network Classifier for BMI Decoders.

This module provides a sklearn-compatible neural network classifier that handles
model creation and prediction. Training is delegated to the TrainingLoop class which
uses config-driven behaviors (early stopping, SWA, etc.).
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin

from dendrite.ml.decoders.decoder_schemas import NeuralNetConfig
from dendrite.ml.models import MODEL_REGISTRY, create_model
from dendrite.ml.training import TrainingLoop
from dendrite.utils.logger_central import get_logger


class NeuralNetClassifier(BaseEstimator, ClassifierMixin):
    """
    Neural network classifier using config-driven TrainingLoop.

    sklearn-compatible classifier that:
    - Creates neural network models
    - Delegates training to TrainingLoop
    - Makes predictions

    Training behaviors (early stopping, SWA, scheduling) are handled by
    TrainingLoop based on config flags.
    """

    @property
    def name(self) -> str:
        """Component name for pipeline compatibility."""
        return f"NeuralNetClassifier_{self.model_type}"

    def __init__(self, config: NeuralNetConfig, epoch_callback=None):
        """
        Initialize neural network classifier.

        Args:
            config: Validated NeuralNetConfig with all training parameters
            epoch_callback: Optional callback for per-epoch progress updates.
                           Signature: (epoch, total_epochs, train_loss, train_acc, val_loss, val_acc)
                           Can be set via sklearn's set_params(epoch_callback=...) for pipeline use.
        """
        self.config = config
        self.epoch_callback = epoch_callback
        self.model_type = config.model_type
        self.num_classes = config.num_classes
        self.device = config.get_device()
        self.learning_rate = config.learning_rate

        # Model state
        self.model: nn.Module | None = None
        self.input_shape: tuple | None = None
        self.training_results: dict[str, Any] | None = None
        self.is_fitted = False
        self.classes_ = None

        self.logger = get_logger(__name__)
        self.logger.info(
            f"NeuralNetClassifier initialized: {self.model_type}, device: {self.device}"
        )

    def _create_model(self, input_shape: tuple) -> nn.Module:
        """Create neural network model based on configured architecture."""
        entry = MODEL_REGISTRY.get(self.model_type)
        if entry:
            model_info = entry["class"].get_model_info()
            input_domain = model_info.get("input_domain", "time-series")
        else:
            input_domain = "time-series"

        model_params = self.config.get_model_specific_params()

        if input_domain == "time-frequency":
            n_channels, n_frequencies, n_times = input_shape
            self.logger.info(
                f"Creating TFR model: channels={n_channels}, freq={n_frequencies}, time={n_times}"
            )
            model_params["n_frequencies"] = n_frequencies
            model = create_model(
                model_type=self.model_type,
                num_classes=self.num_classes,
                input_shape=(n_channels, n_times),
                **model_params,
            )
        else:
            model = create_model(
                model_type=self.model_type,
                num_classes=self.num_classes,
                input_shape=input_shape,
                **model_params,
            )

        return model.to(self.device)

    def _validate_training_inputs(self, X: np.ndarray, y: np.ndarray) -> str:
        """Validate training inputs and determine input shapes."""
        entry = MODEL_REGISTRY.get(self.model_type)
        if not entry:
            raise ValueError(f"Unknown model type: {self.model_type}")

        model_info = entry["class"].get_model_info()
        input_domain = model_info.get("input_domain", "time-series")

        # Validate input shape
        if input_domain == "time-frequency":
            if X.ndim != 4:
                raise ValueError(f"TFR model expects 4D input, got {X.shape}")
            self.input_shape = X.shape[1:]
        else:
            if X.ndim != 3:
                raise ValueError(
                    f"Model expects 3D input (n_samples, n_channels, n_times), got {X.shape}"
                )
            self.input_shape = X.shape[1:]

        return input_domain

    def _split_data(self, X: np.ndarray, y: np.ndarray, validation_split: float) -> tuple:
        """Split data for training and validation."""
        n_samples = len(X)

        if validation_split > 0.0:
            n_val = int(n_samples * validation_split)
            rng = np.random.RandomState(self.config.seed)
            indices = rng.permutation(n_samples)
            train_indices, val_indices = indices[n_val:], indices[:n_val]

            X_train = X[train_indices]
            X_val = X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]
        else:
            X_train, X_val = X, None
            y_train, y_val = y, None

        return X_train, y_train, X_val, y_val

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralNetClassifier":
        """
        Train the neural network on provided data.

        Args:
            X: Training data with shape (n_samples, n_channels, n_times)
            y: Class labels

        Returns:
            Self for method chaining
        """
        self._validate_training_inputs(X, y)

        self.model = self._create_model(self.input_shape)
        X_train, y_train, X_val, y_val = self._split_data(X, y, self.config.validation_split)
        self._log_training_setup()

        # Train using Trainer (epoch_callback set via constructor or set_params)
        results = self._fit_with_trainer(X_train, y_train, X_val, y_val, self.epoch_callback)

        self.training_results = results
        self.is_fitted = True
        self.classes_ = np.unique(y)  # sklearn compatibility

        # Log results
        if "final_val_acc" in results:
            self.logger.info(
                f"Training completed: Train Acc: {results['final_train_acc']:.4f}, "
                f"Val Acc: {results['final_val_acc']:.4f}, "
                f"Epochs: {results['epochs_completed']}"
            )
        else:
            self.logger.info(
                f"Training completed: Train Acc: {results['final_train_acc']:.4f}, "
                f"Loss: {results['final_train_loss']:.4f}, "
                f"Epochs: {results['epochs_completed']}"
            )

        return self

    def _log_training_setup(self) -> None:
        """Log training configuration."""
        if self.config.loss_type == "focal":
            self.logger.info(f"Using FocalLoss (gamma={self.config.focal_gamma})")
        if self.config.label_smoothing_factor > 0:
            self.logger.info(f"Label smoothing: {self.config.label_smoothing_factor}")
        if self.config.optimizer_type == "AdamW":
            self.logger.info(f"Using AdamW with weight_decay={self.config.weight_decay}")
        if self.config.use_lr_warmup:
            self.logger.info(f"LR warmup: {self.config.warmup_epochs} epochs")
        if self.config.use_augmentation:
            self.logger.info(f"Augmentation enabled: {self.config.aug_strategy}")

        self.logger.info(
            f"Training: epochs={self.config.epochs}, batch_size={self.config.batch_size}, "
            f"val_split={self.config.validation_split}"
        )

    def _fit_with_trainer(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
        epoch_callback=None,
    ) -> dict[str, Any]:
        """Train using TrainingLoop."""
        training_loop = TrainingLoop(
            model=self.model,
            config=self.config,
            prepare_input_fn=self._prepare_input_tensor,
        )

        results = training_loop.fit(X_train, y_train, X_val, y_val, epoch_callback=epoch_callback)

        # TrainingLoop may have modified model (SWA, best checkpoint restore)
        self.model = training_loop.model

        return results

    def _prepare_input_tensor(self, X: np.ndarray) -> torch.Tensor:
        """Convert input data to PyTorch tensor with proper shape and device."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        return self._adapt_tensor_shape(X_tensor)

    def _adapt_tensor_shape(self, X_tensor: torch.Tensor) -> torch.Tensor:
        """Adapt tensor shape based on model requirements."""
        if callable(getattr(self.model, "get_model_info", None)):
            capabilities = self.model.get_model_info()
            expected_format = capabilities.get("input_format", "3D")

            if expected_format == "4D" and len(X_tensor.shape) == 3:
                X_tensor = X_tensor.unsqueeze(1)
        else:
            if len(X_tensor.shape) == 3:
                X_tensor = X_tensor.unsqueeze(1)

        return X_tensor

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make class predictions."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")

        self.model.eval()
        with torch.no_grad():
            X_tensor = self._prepare_input_tensor(X)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get class prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")

        self.model.eval()
        with torch.no_grad():
            X_tensor = self._prepare_input_tensor(X)
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()

    def get_training_results(self) -> dict[str, Any] | None:
        """Get training results from the last training session."""
        return self.training_results
