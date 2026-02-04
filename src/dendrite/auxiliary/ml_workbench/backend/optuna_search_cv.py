"""Sklearn-compatible Optuna hyperparameter search wrapper.

Implements nested cross-validation by running Optuna search inside fit(),
ensuring HPO only sees training data when used with MOABB evaluation.
"""

import logging
from typing import Any

import numpy as np
import optuna
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from dendrite.ml.search import OptunaConfig, suggest_params

from .optuna_runner import OptunaRunner

logger = logging.getLogger(__name__)


class OptunaSearchCV(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible Optuna hyperparameter search.

    When MOABB calls `pipeline.fit(X_train, y_train)`, HPO runs only on
    training data via inner cross-validation. This prevents data leakage
    that occurs when HPO is run on full data before MOABB evaluation.

    Args:
        estimator_factory: Callable that creates a fresh estimator given params.
            Signature: (n_classes, input_shape, **params) -> estimator
        param_distributions: Dict mapping param names to search specs.
            Each spec is a dict with 'type' and range/choices:
            - {'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log': True}
            - {'type': 'int', 'low': 1, 'high': 10}
            - {'type': 'categorical', 'choices': [16, 32, 64]}
        n_trials: Number of Optuna trials (default 20).
        cv: Number of inner CV folds for HPO (default 3).
        scoring: Sklearn scoring metric (default 'accuracy').
        n_jobs: Number of parallel jobs (default 1 for reproducibility).
        refit: Whether to refit best model on full training data (default True).
        random_state: Random seed for reproducibility.
        epochs: Training epochs passed to estimator factory.
        base_params: Additional fixed params for estimator factory.

    Example:
        >>> from dendrite.ml.decoders import create_decoder
        >>> from dendrite.ml.search import BENCHMARK_SEARCH_SPACE
        >>>
        >>> def factory(n_classes, input_shape, **params):
        ...     return create_decoder(
        ...         model_type='EEGNet',
        ...         num_classes=n_classes,
        ...         input_shapes={'eeg': input_shape},
        ...         **params
        ...     )
        >>>
        >>> search = OptunaSearchCV(
        ...     estimator_factory=factory,
        ...     param_distributions=BENCHMARK_SEARCH_SPACE,
        ...     n_trials=10,
        ...     random_state=42,
        ... )
        >>> search.fit(X_train, y_train)  # HPO happens here
        >>> predictions = search.predict(X_test)
    """

    # sklearn classifier type marker (legacy, for older sklearn versions)
    _estimator_type = "classifier"

    def __sklearn_tags__(self):
        """Return sklearn tags for classifier detection (sklearn 1.6+)."""
        from sklearn.utils._tags import ClassifierTags

        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.classifier_tags = ClassifierTags()
        return tags

    def __init__(
        self,
        estimator_factory: callable,
        param_distributions: dict[str, dict[str, Any]],
        n_trials: int = 20,
        cv: int = 3,
        scoring: str = "accuracy",
        n_jobs: int = 1,
        refit: bool = True,
        random_state: int | None = None,
        epochs: int = 100,
        base_params: dict[str, Any] | None = None,
    ):
        self.estimator_factory = estimator_factory
        self.param_distributions = param_distributions
        self.n_trials = n_trials
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.random_state = random_state
        self.epochs = epochs
        self.base_params = base_params or {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Run Optuna HPO on training data, then refit best model.

        Args:
            X: Training data (n_samples, n_channels, n_times).
            y: Training labels (n_samples,).

        Returns:
            self with best_estimator_, best_params_, best_score_, study_ set.
        """
        # Encode labels if they're strings (MOABB datasets use string labels)
        self._label_encoder = None
        if y.dtype.kind in ("U", "S", "O"):
            self._label_encoder = LabelEncoder()
            y = self._label_encoder.fit_transform(y)
            self.classes_ = self._label_encoder.classes_
        else:
            self.classes_ = np.unique(y)

        # Store data shape for estimator creation
        n_channels, n_times = X.shape[1], X.shape[2]
        n_classes = len(np.unique(y))
        self._n_classes = n_classes
        self._input_shape = [n_channels, n_times]

        def objective(trial: optuna.Trial) -> float:
            params = suggest_params(trial, self.param_distributions)
            all_params = {**self.base_params, **params, "epochs": self.epochs}

            estimator = self.estimator_factory(
                n_classes=n_classes,
                input_shape=[n_channels, n_times],
                **all_params,
            )

            try:
                scores = cross_val_score(
                    estimator,
                    X,
                    y,
                    cv=StratifiedKFold(
                        n_splits=self.cv,
                        shuffle=True,
                        random_state=self.random_state,
                    ),
                    scoring=self.scoring,
                    n_jobs=1,  # Sequential for reproducibility
                )
                return float(np.mean(scores))
            except (ValueError, RuntimeError, AttributeError) as e:
                logger.warning(f"Trial {trial.number} failed: {type(e).__name__}: {e}")
                return 0.0

        # Use OptunaRunner for optimization
        config = OptunaConfig(
            n_trials=self.n_trials,
            direction="maximize",
            seed=self.random_state,
            sampler_type="tpe",
            pruner_type="none",  # No pruning for CV-based evaluation
            verbose=False,
        )
        runner = OptunaRunner(config)
        runner.optimize(objective)

        # Store study reference for compatibility
        self.study_ = runner.study

        # Store best results
        self.best_params_ = self.study_.best_params
        self.best_score_ = self.study_.best_value

        # Collect all trial results
        self.all_trials_ = []
        for trial in self.study_.trials:
            if trial.value is not None:
                self.all_trials_.append(
                    {
                        "trial": trial.number,
                        "accuracy": trial.value,
                        **trial.params,
                    }
                )

        # Refit best model on full training data
        if self.refit:
            all_params = {**self.base_params, **self.best_params_, "epochs": self.epochs}
            self.best_estimator_ = self.estimator_factory(
                n_classes=n_classes,
                input_shape=[n_channels, n_times],
                **all_params,
            )
            self.best_estimator_.fit(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the best estimator."""
        return self.best_estimator_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using the best estimator."""
        if hasattr(self.best_estimator_, "predict_proba"):
            return self.best_estimator_.predict_proba(X)
        raise AttributeError("Best estimator does not support predict_proba")

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for sklearn compatibility."""
        return {
            "estimator_factory": self.estimator_factory,
            "param_distributions": self.param_distributions,
            "n_trials": self.n_trials,
            "cv": self.cv,
            "scoring": self.scoring,
            "n_jobs": self.n_jobs,
            "refit": self.refit,
            "random_state": self.random_state,
            "epochs": self.epochs,
            "base_params": self.base_params,
        }

    def set_params(self, **params) -> "OptunaSearchCV":
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
