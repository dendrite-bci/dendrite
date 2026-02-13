"""Offline trainer module - clean training logic.

Ported from v1 trainer/backend.py with simplified API.
"""

import time
from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_val_score

from dendrite.ml.decoders.decoder_schemas import DecoderConfig
from dendrite.utils.logger_central import get_logger

from .types import TrainResult

logger = get_logger(__name__)


class OfflineTrainer:
    """Trains decoders offline with optional cross-validation.

    This is the public API for offline training. It orchestrates:
    1. Data validation and label normalization
    2. Decoder creation via dendrite.decoders.create_decoder()
    3. Training with sklearn-compatible .fit() interface
    4. Metrics extraction (accuracy, confusion matrix, history)

    Architecture Note:
        OfflineTrainer -> Decoder.fit() -> NeuralNetClassifier -> Trainer

        The internal dendrite.ml.training.trainer.Trainer handles the PyTorch
        training loop. Users should only interact with OfflineTrainer.
    """

    def __init__(self):
        """Initialize trainer."""
        self._available_models = None

    @property
    def available_models(self) -> list[str]:
        """Get list of available model types."""
        if self._available_models is None:
            from dendrite.ml.models import get_available_models

            self._available_models = get_available_models()
        return self._available_models

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: DecoderConfig,
        n_folds: int | None = None,
        progress_callback: Callable[[str], None] | None = None,
        epoch_callback: Callable[[int, int, float, float, float | None, float | None], None]
        | None = None,
    ) -> TrainResult:
        """Train a decoder on the provided data.

        Args:
            X: Training data, shape (n_samples, n_channels, n_times)
            y: Labels, shape (n_samples,)
            config: DecoderConfig with training parameters
            n_folds: Number of CV folds. If None, no CV (simple train/val split).
            progress_callback: Optional callback for progress updates
            epoch_callback: Optional callback for per-epoch updates (epoch, total, train_loss, train_acc, val_loss, val_acc)

        Returns:
            TrainResult with trained decoder and metrics

        Raises:
            ValueError: If data is invalid
        """
        self._validate_data(X, y)

        # Normalize labels to 0-indexed contiguous range
        y = self._normalize_labels(y)

        def emit(msg: str):
            if progress_callback:
                progress_callback(msg)
            logger.info(msg)

        emit(f"Starting training: {config.model_type}, {len(y)} samples")

        start_time = time.time()

        emit(f"Creating {config.model_type} decoder...")
        decoder = self._create_decoder(X, y, config)

        # Train with or without CV
        if n_folds and n_folds > 1:
            emit(f"Training with {n_folds}-fold cross-validation...")
            result = self._train_with_cv(decoder, X, y, config, n_folds, emit, epoch_callback)
        else:
            emit("Training with train/val split...")
            result = self._train_simple(decoder, X, y, config, emit, epoch_callback)

        training_time = time.time() - start_time
        emit(f"Training completed in {training_time:.1f}s")

        return result

    def _validate_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validate input data."""
        if X.size == 0 or y.size == 0:
            raise ValueError("Training data cannot be empty")

        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")

        if X.ndim != 3:
            raise ValueError(f"X must be 3D (samples, channels, times), got {X.ndim}D")

    def _normalize_labels(self, y: np.ndarray) -> np.ndarray:
        """Normalize labels to 0-indexed contiguous range.

        CrossEntropyLoss requires labels in [0, n_classes-1]. MOABB datasets
        may have non-contiguous or 1-indexed labels (e.g., [1, 2] instead of [0, 1]).
        """
        unique_labels = np.unique(y)
        num_classes = len(unique_labels)

        # Check if already normalized
        if unique_labels[0] == 0 and unique_labels[-1] == num_classes - 1:
            return y

        # Remap to 0-indexed
        label_remap = {old: new for new, old in enumerate(unique_labels)}
        y_normalized = np.array([label_remap[label] for label in y], dtype=y.dtype)
        logger.info(f"Remapped labels: {label_remap}")
        return y_normalized

    def _create_decoder(self, X: np.ndarray, y: np.ndarray, config: DecoderConfig) -> Any:
        """Create decoder using factory."""
        from dendrite.ml.decoders import Decoder

        n_channels = X.shape[1]
        n_times = X.shape[2]
        num_classes = len(np.unique(y))

        # Update config with data-derived values
        modality = config.modality or "eeg"
        config.input_shapes = {modality: (n_channels, n_times)}
        config.num_classes = num_classes

        return Decoder(config)

    def _train_simple(
        self,
        decoder: Any,
        X: np.ndarray,
        y: np.ndarray,
        config: DecoderConfig,
        emit: Callable,
        epoch_callback: Callable | None = None,
    ) -> TrainResult:
        """Train without cross-validation."""
        # Fit decoder (sklearn-compatible - accepts plain arrays)
        decoder.fit(X, y, epoch_callback=epoch_callback)

        history = self._get_history(decoder)
        accuracy, val_accuracy = self._get_accuracies(decoder)

        # Compute confusion matrix on validation set (same split as NeuralNetClassifier)
        val_indices = self._get_validation_indices(X, config)
        X_eval = X[val_indices] if val_indices is not None else X
        y_eval = y[val_indices] if val_indices is not None else y
        conf_matrix = self._compute_confusion_matrix(decoder, X_eval, y_eval)

        emit(f"Accuracy: {accuracy:.3f}, Val: {val_accuracy:.3f}")

        return TrainResult(
            decoder=decoder,
            accuracy=accuracy,
            val_accuracy=val_accuracy,
            confusion_matrix=conf_matrix,
            train_history=history,
            cv_results=None,
        )

    def _train_with_cv(
        self,
        decoder: Any,
        X: np.ndarray,
        y: np.ndarray,
        config: DecoderConfig,
        n_folds: int,
        emit: Callable,
        epoch_callback: Callable | None = None,
    ) -> TrainResult:
        """Train with cross-validation using sklearn."""
        # Run CV evaluation first (sklearn-compatible)
        emit(f"Running {n_folds}-fold cross-validation...")
        cv_scores = cross_val_score(decoder, X, y, cv=n_folds, scoring="accuracy")

        # Get out-of-fold predictions for confusion matrix
        y_pred_oof = cross_val_predict(decoder, X, y, cv=n_folds)
        conf_matrix = confusion_matrix(y, y_pred_oof)

        cv_results = {
            "n_folds": n_folds,
            "fold_scores": cv_scores.tolist(),
            "mean_accuracy": float(cv_scores.mean()),
            "std_accuracy": float(cv_scores.std()),
        }

        emit(f"CV Accuracy: {cv_results['mean_accuracy']:.3f} ± {cv_results['std_accuracy']:.3f}")

        # Now train on full data for final model
        emit("Training final model on all data...")
        decoder.fit(X, y, epoch_callback=epoch_callback)

        # Get metrics from final model
        history = self._get_history(decoder)
        accuracy, val_accuracy = self._get_accuracies(decoder)

        return TrainResult(
            decoder=decoder,
            accuracy=accuracy,
            val_accuracy=val_accuracy,
            confusion_matrix=conf_matrix,
            train_history=history,
            cv_results=cv_results,
        )

    def _get_history(self, decoder: Any) -> dict[str, list[float]]:
        """Extract training history from decoder."""
        if hasattr(decoder, "get_training_history"):
            history = decoder.get_training_history()
            if history:
                return history
        return {}

    def _get_accuracies(self, decoder: Any) -> tuple:
        """Extract accuracies from decoder training metrics."""
        accuracy = 0.0
        val_accuracy = 0.0

        if hasattr(decoder, "get_training_metrics"):
            metrics = decoder.get_training_metrics()
            if metrics:
                for _component, m in metrics.items():
                    if isinstance(m, dict):
                        accuracy = m.get("final_train_acc", accuracy)
                        val_accuracy = m.get("final_val_acc", val_accuracy)
                        break

        return accuracy, val_accuracy

    def _get_validation_indices(self, X: np.ndarray, config: DecoderConfig) -> np.ndarray | None:
        """Get validation split indices matching NeuralNetClassifier._split_data.

        Uses the same seed and split logic so the confusion matrix is computed
        on the exact samples the model never trained on.

        Mirrors NeuralNetClassifier._split_data — keep in sync if split logic changes.
        """
        if config.validation_split <= 0.0:
            return None

        n_samples = len(X)
        n_val = int(n_samples * config.validation_split)
        if n_val == 0:
            return None

        rng = np.random.RandomState(config.seed)
        indices = rng.permutation(n_samples)
        return indices[:n_val]

    def _compute_confusion_matrix(self, decoder: Any, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute confusion matrix on the provided data subset."""
        try:
            if hasattr(decoder, "predict"):
                y_pred = decoder.predict(X)
                return confusion_matrix(y, y_pred)
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Could not compute confusion matrix: {e}")

        # Return empty matrix
        n_classes = len(np.unique(y))
        return np.zeros((n_classes, n_classes), dtype=int)
