"""Decoder — unified interface for neural and classical EEG classifiers."""

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
    """sklearn-compatible decoder wrapping neural and classical pipelines."""

    def __sklearn_tags__(self):
        from sklearn.utils._tags import ClassifierTags
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.classifier_tags = ClassifierTags()
        return tags

    def __init__(self, config: DecoderConfig):
        self.config = config
        self._is_fitted = False
        self.pipeline = None
        self.training_metrics = None
        self.logger = get_logger()

        if config.num_classes is not None and config.input_shapes is not None:
            self._build_pipeline()
        self.logger.info(f"Decoder initialized: {config.model_type}")

    # ── Config-backed properties (single source of truth) ────────────────

    @property
    def num_classes(self):
        return self.config.num_classes

    @num_classes.setter
    def num_classes(self, value):
        self.config.num_classes = value

    @property
    def input_shapes(self):
        return self.config.input_shapes

    @input_shapes.setter
    def input_shapes(self, value):
        self.config.input_shapes = value

    @property
    def event_mapping(self):
        return self.config.event_mapping

    @event_mapping.setter
    def event_mapping(self, value):
        self.config.event_mapping = value

    @property
    def label_mapping(self):
        return self.config.label_mapping

    @label_mapping.setter
    def label_mapping(self, value):
        self.config.label_mapping = value

    @property
    def sample_rate(self):
        return self.config.sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        self.config.sample_rate = value

    @property
    def channel_labels(self):
        return self.config.channel_labels

    @channel_labels.setter
    def channel_labels(self, value):
        self.config.channel_labels = value

    # ── Pipeline ─────────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted and self.pipeline is not None

    def _build_pipeline(self) -> None:
        if not self.config.input_shapes:
            raise ValueError("Input shapes must be provided before building pipeline")
        entry = get_decoder_entry(self.config.model_type)
        if not entry:
            raise ValueError(f"Unknown decoder type: {self.config.model_type}")
        self.pipeline = entry["pipeline_builder"](self.config)

    def _is_neural_pipeline(self) -> bool:
        return (
            self.pipeline is not None
            and hasattr(self.pipeline, "named_steps")
            and isinstance(self.pipeline.named_steps.get("classifier"), NeuralNetClassifier)
        )

    # ── Training ─────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray, epoch_callback=None, stop_event=None) -> "Decoder":
        self.logger.info(f"Training with {len(y)} samples")
        X = self._ensure_3d(X)

        if self.pipeline is None:
            self._build_pipeline()

        if self._is_neural_pipeline():
            if epoch_callback is not None:
                self.pipeline.set_params(classifier__epoch_callback=epoch_callback)
            if stop_event is not None:
                self.pipeline.set_params(classifier__stop_event=stop_event)

        self.pipeline.fit(X, y)

        if self._is_neural_pipeline():
            self.pipeline.set_params(
                classifier__epoch_callback=None, classifier__stop_event=None,
            )
            classifier = self.pipeline.named_steps["classifier"]
            if callable(getattr(classifier, "get_training_results", None)):
                results = classifier.get_training_results()
                if results:
                    self.training_metrics = {classifier.__class__.__name__: results}

        self._is_fitted = True
        self.classes_ = np.unique(y)
        self.logger.info("Training finished")
        return self

    def get_training_metrics(self) -> dict[str, Any] | None:
        return self.training_metrics

    def get_training_history(self) -> dict[str, Any] | None:
        if not self.training_metrics:
            return None
        for _, metrics in self.training_metrics.items():
            if isinstance(metrics, dict) and "history" in metrics:
                return metrics["history"]
        return None

    # ── Prediction ───────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> int | np.ndarray:
        proba = self.predict_proba(X)
        preds = np.argmax(proba, axis=-1)
        return int(preds[0]) if proba.shape[0] == 1 else preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.pipeline.predict_proba(self._ensure_3d(X))

    def predict_sample(self, X: np.ndarray) -> tuple[int, float]:
        """Predict single sample, returning (class, confidence)."""
        proba = self.predict_proba(X)
        return int(np.argmax(proba[0])), float(np.max(proba[0]))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        if isinstance(preds, int):
            preds = np.array([preds])
        return float(np.mean(preds == y))

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, file_identifier: str, study_name: str | None = None) -> str:
        if not file_identifier or not self.is_fitted:
            raise ValueError("Cannot save: missing identifier or unfitted decoder")

        # Auto-populate modality from input_shapes if not set
        if not self.config.modality and self.config.input_shapes:
            self.config.modality = next(iter(self.config.input_shapes))

        decoders_dir = get_study_paths(study_name or DEFAULT_STUDY_NAME)["decoders"]
        decoders_dir.mkdir(parents=True, exist_ok=True)
        full_path = decoders_dir / file_identifier

        # Metadata JSON
        json_path = f"{full_path}.json"
        with open(json_path, "w") as f:
            json.dump(self.config.model_dump(), f, indent=2, default=str)

        # Pipeline state
        if self._is_neural_pipeline():
            classifier = self.pipeline.named_steps["classifier"]
            model_state = classifier.model.state_dict()
            model_backup = classifier.model
            classifier.model = None
            torch.save({
                "model_state_dict": model_state,
                "input_shape": classifier.input_shape,
                "classes_": classifier.classes_,
                "pipeline": self.pipeline,
            }, f"{full_path}.pt")
            classifier.model = model_backup
        else:
            joblib.dump(self.pipeline, f"{full_path}.joblib")

        self.logger.info(f"Decoder saved: {json_path}")
        return json_path

    # ── sklearn compat ───────────────────────────────────────────────────

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"config": self.config}

    def set_params(self, **params) -> "Decoder":
        if "config" in params:
            self.config = params["config"]
            if self.config.num_classes is not None and self.config.input_shapes is not None:
                self._build_pipeline()
        return self

    def __repr__(self) -> str:
        return f"Decoder(model={self.config.model_type}, classes={self.num_classes}, fitted={self.is_fitted})"

    @staticmethod
    def _ensure_3d(X: np.ndarray) -> np.ndarray:
        return X[np.newaxis, :, :] if X.ndim == 2 else X
