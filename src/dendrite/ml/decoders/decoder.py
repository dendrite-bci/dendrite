"""
Decoder for Dendrite applications.

Provides a unified interface for neural network and classical ML classifiers
for EEG signal classification. This is the primary decoder implementation.

Note: This decoder is a pure algorithm implementation. Data management and
workflow orchestration should be handled by the mode that uses this decoder.
"""

import json
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin

from dendrite.constants import DEFAULT_STUDY_NAME, get_study_paths
from dendrite.ml.decoders.decoder_schemas import DecoderConfig
from dendrite.ml.decoders.neural_classifier import NeuralNetClassifier
from dendrite.ml.decoders.registry import get_decoder_entry
from dendrite.utils.logger_central import get_logger


class Decoder(BaseEstimator, ClassifierMixin):
    """
    Primary decoder for EEG classification.

    Wraps neural network and classical ML classifiers in an sklearn-compatible
    interface for training, prediction, and cross-validation.

    Features:
    - Neural network classifiers (EEGNet, TransformerEEG, LinearEEG, etc.)
    - Classical ML pipelines (CSP+LDA, CSP+SVM, etc.)
    - sklearn-compatible interface (works with cross_val_score, GridSearchCV, MOABB)
    """

    # sklearn classifier type (legacy, kept for older sklearn versions)
    _estimator_type = "classifier"

    def __sklearn_tags__(self):
        """Return sklearn tags for classifier detection (sklearn 1.6+)."""
        from sklearn.utils._tags import ClassifierTags

        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.classifier_tags = ClassifierTags()
        return tags

    def __init__(self, config: DecoderConfig):
        """
        Initialize decoder with configuration.

        Args:
            config: DecoderConfig with model type, input shape, and training params

        Example:
            config = DecoderConfig(
                model_type='EEGNet',
                num_classes=3,
                input_shape=(32, 250),  # (channels, times)
                epochs=200,
                event_mapping={1: 'left', 2: 'right'},
                label_mapping={'left': 0, 'right': 1}
            )
            decoder = Decoder(config)
        """
        self.num_classes = config.num_classes
        self.input_shapes = config.input_shapes
        self._is_fitted = False
        self.logger = get_logger()

        # Store the complete configuration
        self.config = config

        # Extract mappings from config
        self.event_mapping = config.event_mapping
        self.label_mapping = config.label_mapping

        # sklearn Pipeline for both classical and neural
        self.pipeline = None

        # Build pipeline if we have required info
        if self.num_classes is not None and self.input_shapes is not None:
            self._build_pipeline()

        self.logger.info(f"Decoder initialized: {self.config.model_type}")

    def _is_neural_pipeline(self) -> bool:
        """Check if pipeline contains neural classifier (needs state_dict serialization)."""
        return (
            self.pipeline is not None
            and hasattr(self.pipeline, "named_steps")
            and "classifier" in self.pipeline.named_steps
            and isinstance(self.pipeline.named_steps["classifier"], NeuralNetClassifier)
        )

    @property
    def is_fitted(self) -> bool:
        """Check if the decoder is fitted and ready."""
        return self._is_fitted and self.pipeline is not None

    def _build_pipeline(self) -> None:
        """Build sklearn Pipeline from registry."""
        if not self.input_shapes or len(self.input_shapes) == 0:
            raise ValueError("Input shapes must be provided before building pipeline")

        model_type = self.config.model_type
        entry = get_decoder_entry(model_type)

        if not entry:
            raise ValueError(f"Unknown decoder type: {model_type}")

        # Populate model info for neural decoders (has model_class)
        if "model_class" in entry:
            self._populate_model_info()

        # Unified: all decoders use pipeline_builder
        self.pipeline = entry["pipeline_builder"](self.config)
        self.logger.info(f"Built pipeline: {model_type}")

    def _populate_model_info(self) -> None:
        """Populate model info and default params from decoder registry."""
        try:
            entry = get_decoder_entry(self.config.model_type) or {}
            model_class = entry.get("model_class")
            if model_class and hasattr(model_class, "get_model_info"):
                model_info = model_class.get_model_info()
                self.config.model_info = model_info
                self._populate_model_params_from_defaults(model_info)
        except (AttributeError, TypeError, KeyError) as e:
            self.logger.debug(f"Could not get model info for {self.config.model_type}: {e}")

    def _populate_model_params_from_defaults(self, model_info: dict[str, Any]):
        """Populate model_params with default parameters, preserving user overrides."""
        default_params = model_info.get("default_parameters", {})
        if not default_params:
            return  # No defaults to merge

        # Start with model defaults
        merged_params = default_params.copy()

        # Override with any user-provided parameters (preserve user settings)
        if self.config.model_params:
            merged_params.update(self.config.model_params)

        # Store the merged parameters back to config for persistence
        self.config.model_params = merged_params

        self.logger.info(f"Model parameters populated: {merged_params}")

    def _ensure_3d(self, X: np.ndarray) -> np.ndarray:
        """Ensure input is 3D (n_samples, n_channels, n_times)."""
        if X.ndim == 2:  # Single sample: (channels, times)
            return X[np.newaxis, :, :]
        return X

    def fit(self, X: np.ndarray, y: np.ndarray, epoch_callback=None) -> "Decoder":
        """Train the decoder on provided data.

        For cross-validation, use sklearn's cross_val_score() externally:
            scores = cross_val_score(decoder, X, y, cv=5)

        Args:
            X: Training data with shape (n_samples, n_channels, n_times).
            y: Training labels with shape (n_samples,).
            epoch_callback: Optional callback invoked after each training epoch.
                Signature: callback(epoch, total_epochs, train_loss, train_acc,
                val_loss, val_acc) where val_loss and val_acc are None if no
                validation set is used.

        Returns:
            Self for method chaining (sklearn compatible). After training,
            `training_metrics` attribute is populated for neural models with:
            - 'history': Dict with 'train_loss', 'train_acc', 'val_loss', 'val_acc' lists
            - 'final_train_acc', 'final_val_acc': Final epoch accuracies
            - 'epochs_completed': Number of epochs trained
        """
        self.logger.info(f"Training with {len(y)} samples")

        X = self._ensure_3d(X)

        # Build pipeline if needed
        if self.pipeline is None:
            self._build_pipeline()

        # Set epoch_callback on neural classifier using sklearn's set_params
        if self._is_neural_pipeline() and epoch_callback is not None:
            self.pipeline.set_params(classifier__epoch_callback=epoch_callback)

        self.pipeline.fit(X, y)

        # Clear epoch_callback after training to enable pickling during save
        if self._is_neural_pipeline() and epoch_callback is not None:
            self.pipeline.set_params(classifier__epoch_callback=None)

        if self._is_neural_pipeline():
            self.training_metrics = self._extract_training_metrics()

        self._is_fitted = True
        self.classes_ = np.unique(y)

        self.logger.info("Training finished")
        return self

    def _extract_training_metrics(self) -> dict[str, Any] | None:
        """Extract training metrics from neural classifier."""
        if not self._is_neural_pipeline():
            return None

        # Get the classifier from the pipeline (it's a step, not the pipeline itself)
        classifier = self.pipeline.named_steps.get("classifier")
        if classifier is None:
            return None

        if callable(getattr(classifier, "get_training_results", None)):
            results = classifier.get_training_results()
            if results:
                return {classifier.__class__.__name__: results}
        return None

    def get_training_metrics(self) -> dict[str, Any] | None:
        """Get training metrics from the last training session.

        Returns:
            Training metrics if available, None otherwise
        """
        return getattr(self, "training_metrics", None)

    def get_training_history(self) -> dict[str, Any] | None:
        """Get detailed training history (loss/accuracy curves) if available.

        Returns:
            Training history if available, None otherwise
        """
        metrics = self.get_training_metrics()
        if not metrics:
            self.logger.debug("No training metrics available")
            return None

        # Extract history from neural network components
        for component_name, component_metrics in metrics.items():
            if isinstance(component_metrics, dict) and "history" in component_metrics:
                history = component_metrics["history"]
                self.logger.info(
                    f"Found training history from {component_name}: {list(history.keys())}"
                )
                return history

        self.logger.debug("No training history found in metrics")
        return None

    def predict(self, X: np.ndarray) -> int | np.ndarray:
        """Predict class labels.

        For Dendrite applications, consider using predict_sample() to also get confidence.

        Args:
            X: Input data with shape (n_samples, n_channels, n_times) or (n_channels, n_times)

        Returns:
            Predicted class(es). Single int for single sample, array of ints for batch.
        """
        probabilities = self.predict_proba(X)
        predictions = np.argmax(probabilities, axis=-1)

        # Return single int for single sample, array for batch
        if probabilities.shape[0] == 1:
            return int(predictions[0])
        else:
            return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Input data with shape (n_samples, n_channels, n_times) or (n_channels, n_times)

        Returns:
            Class probabilities with shape (n_samples, n_classes)

        Raises:
            RuntimeError: If model is not fitted or prediction fails.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() before predict_proba().")

        X = self._ensure_3d(X)
        return self.pipeline.predict_proba(X)

    def save(self, file_identifier: str, study_name: str | None = None) -> str:
        """Save complete decoder (pipeline + config + metadata) to disk.

        Uses format: .json (metadata) + .pt (neural state) or .joblib (classical pipeline).
        Neural models use torch.save for state_dict to handle parametrized modules.

        Args:
            file_identifier: Unique identifier for the decoder file.
            study_name: Study name for study-scoped storage. If None, uses
                global decoders directory.

        Returns:
            Path to the saved JSON metadata file.

        Raises:
            ValueError: If file_identifier is empty or decoder is not fitted.
        """
        if not file_identifier or not self.is_fitted:
            raise ValueError("Cannot save: missing identifier or unfitted decoder")

        # Update config with runtime metadata
        self.config.event_mapping = self.event_mapping
        self.config.label_mapping = self.label_mapping
        self.config.input_shapes = self.input_shapes
        if hasattr(self, "channel_labels") and self.channel_labels:
            self.config.channel_labels = self.channel_labels
        self.config.sample_rate = getattr(self, "sample_rate", self.config.sample_rate)
        self.config.target_sample_rate = getattr(
            self, "target_sample_rate", self.config.target_sample_rate
        )

        decoders_dir = get_study_paths(study_name or DEFAULT_STUDY_NAME)["decoders"]
        decoders_dir.mkdir(parents=True, exist_ok=True)
        full_path = decoders_dir / file_identifier
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Save metadata JSON
        json_path = f"{full_path}.json"
        with open(json_path, "w") as f:
            json.dump(self.config.model_dump(), f, indent=2, default=str)

        # Save pipeline
        if self._is_neural_pipeline():
            # Neural: embed pipeline (with model=None) in .pt file
            classifier = self.pipeline.named_steps["classifier"]
            model_state = classifier.model.state_dict()
            input_shape = classifier.input_shape
            classes = classifier.classes_

            model_backup = classifier.model
            classifier.model = None
            torch.save(
                {
                    "model_state_dict": model_state,
                    "input_shape": input_shape,
                    "classes_": classes,
                    "pipeline": self.pipeline,
                },
                f"{full_path}.pt",
            )
            classifier.model = model_backup
        else:
            # Classical: joblib
            joblib.dump(self.pipeline, f"{full_path}.joblib")

        self.logger.info(f"Decoder saved: {json_path}")
        return json_path

    def get_expected_sample_rate(self) -> float:
        """Get the rate at which model was trained (for online validation)."""
        return self.config.effective_sample_rate if self.config else 500.0

    def predict_sample(self, X: np.ndarray) -> tuple[int, float]:
        """Primary prediction interface for BMI applications.

        Returns both prediction and confidence for a single sample.

        Args:
            X: Single sample with shape (n_channels, n_times) or (1, n_channels, n_times)

        Returns:
            Tuple of (prediction, confidence)
        """
        probabilities = self.predict_proba(X)
        prediction = int(np.argmax(probabilities[0]))
        confidence = float(np.max(probabilities[0]))
        return prediction, confidence

    def __repr__(self) -> str:
        """String representation of the decoder."""
        return (
            f"Decoder(model={self.config.model_type}, "
            f"classes={self.num_classes}, fitted={self.is_fitted})"
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for sklearn compatibility.

        Required for cross_val_score, GridSearchCV, clone(), etc.

        Args:
            deep: If True, return nested parameters (ignored, config is atomic)

        Returns:
            Parameter dict with 'config' key containing DecoderConfig
        """
        return {"config": self.config}

    def set_params(self, **params) -> "Decoder":
        """Set parameters for sklearn compatibility.

        Args:
            **params: Parameters to set. Supports 'config' for full config replacement.

        Returns:
            Self for method chaining.
        """
        if "config" in params:
            self.config = params["config"]
            # Re-initialize with new config
            self.num_classes = self.config.num_classes
            self.input_shapes = self.config.input_shapes
            self.event_mapping = self.config.event_mapping
            self.label_mapping = self.config.label_mapping
            # Rebuild pipeline with new config
            if self.num_classes is not None and self.input_shapes is not None:
                self._build_pipeline()
        return self

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy score for sklearn compatibility.

        Args:
            X: Test samples with shape (n_samples, n_channels, n_times)
            y: True labels

        Returns:
            Accuracy score (fraction of correct predictions)
        """
        predictions = self.predict(X)
        if isinstance(predictions, int):
            predictions = np.array([predictions])
        return float(np.mean(predictions == y))
