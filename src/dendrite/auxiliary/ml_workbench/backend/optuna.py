"""Optuna hyperparameter optimization utilities.

Provides both a general-purpose runner for custom objectives and an
sklearn-compatible wrapper for nested cross-validation.
"""

import json
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from optuna.trial import TrialState
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from dendrite.ml.search import OptunaConfig, create_sampler, suggest_params
from dendrite.utils.logger_central import get_logger

logger = get_logger(__name__)


class OptunaRunner:
    """Run hyperparameter optimization using Optuna with pruning support.

    Key features:
    - Pluggable objective function pattern
    - MedianPruner for early stopping of bad trials
    - Progress callbacks matching existing Dendrite patterns
    - JSON export of results

    Usage:
        from dendrite.auxiliary.ml_workbench.backend import OptunaRunner
        from dendrite.ml.search import OptunaConfig

        config = OptunaConfig(n_trials=20, pruner_type='median')
        runner = OptunaRunner(config)

        def objective(trial):
            lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            # ... train and evaluate ...
            return accuracy

        results = runner.optimize(objective)
        print(f"Best: {results['best_value']:.1%} with {results['best_params']}")
    """

    def __init__(self, config: OptunaConfig):
        """Initialize runner with configuration."""
        issues = config.validate()
        if issues:
            raise ValueError(f"Invalid config: {issues}")

        self.config = config
        self.study: optuna.Study | None = None
        self._progress_callback: Callable[[int, str], None] | None = None
        self._trial_count = 0

    def optimize(
        self,
        objective: Callable[[optuna.Trial], float],
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> dict[str, Any]:
        """Run optimization with pluggable objective function.

        The objective function receives an Optuna trial and should:
        1. Use trial.suggest_* to sample hyperparameters
        2. Train/evaluate the model
        3. Optionally call trial.report(value, step) for intermediate results
        4. Raise optuna.TrialPruned() if trial.should_prune()
        5. Return the final metric value

        Args:
            objective: Function(trial) -> float
            progress_callback: Optional callback(percent, message)

        Returns:
            Dict with best_params, best_value, n_trials, trials summary
        """
        self._progress_callback = progress_callback
        self._trial_count = 0

        self.study = optuna.create_study(
            study_name=self.config.study_name,
            storage=self.config.storage,
            load_if_exists=self.config.load_if_exists,
            sampler=create_sampler(
                self.config.sampler_type,
                self.config.seed,
                self.config.n_startup_trials,
            ),
            pruner=self._create_pruner(),
            direction=self.config.direction,
        )

        if not self.config.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        self._report_progress(0, "Starting optimization...")

        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            callbacks=[self._optuna_callback],
            show_progress_bar=False,
        )

        self._report_progress(100, "Optimization complete")

        results = self._build_results()

        if self.config.results_dir:
            self._save_results(results)

        return results

    def _create_pruner(self) -> optuna.pruners.BasePruner:
        """Create pruner based on config."""
        if self.config.pruner_type == "median":
            return optuna.pruners.MedianPruner(
                n_startup_trials=self.config.pruner_n_startup_trials,
                n_warmup_steps=self.config.pruner_n_warmup_steps,
            )
        elif self.config.pruner_type == "percentile":
            return optuna.pruners.PercentilePruner(
                percentile=self.config.pruner_percentile,
                n_startup_trials=self.config.pruner_n_startup_trials,
                n_warmup_steps=self.config.pruner_n_warmup_steps,
            )
        else:  # 'none' - validated in __init__
            return optuna.pruners.NopPruner()

    def _optuna_callback(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        """Callback after each trial completes."""
        self._trial_count += 1
        percent = int(100 * self._trial_count / self.config.n_trials)

        if trial.state == TrialState.COMPLETE:
            value_str = f"{trial.value:.3f}" if trial.value else "N/A"
            msg = f"Trial {self._trial_count}/{self.config.n_trials}: {value_str}"
        elif trial.state == TrialState.PRUNED:
            msg = f"Trial {self._trial_count}/{self.config.n_trials}: PRUNED"
        else:
            msg = f"Trial {self._trial_count}/{self.config.n_trials}: {trial.state.name}"

        if self.config.verbose:
            best_val = study.best_value if study.best_trial else None
            if best_val is not None:
                msg += f" (best: {best_val:.3f})"
            logger.info(msg)

        self._report_progress(percent, msg)

    def _report_progress(self, percent: int, message: str):
        """Report progress via callback if set."""
        if self._progress_callback:
            self._progress_callback(percent, message)

    def _build_results(self) -> dict[str, Any]:
        """Build results dictionary from completed study."""
        if not self.study:
            return {}

        state_counts = Counter(t.state for t in self.study.trials)
        completed = state_counts[TrialState.COMPLETE]
        pruned = state_counts[TrialState.PRUNED]
        failed = state_counts[TrialState.FAIL]

        results = {
            "best_params": self.study.best_params if self.study.best_trial else {},
            "best_value": self.study.best_value if self.study.best_trial else None,
            "n_trials": len(self.study.trials),
            "n_completed": completed,
            "n_pruned": pruned,
            "n_failed": failed,
            "direction": self.config.direction,
            "config": {
                "sampler": self.config.sampler_type,
                "pruner": self.config.pruner_type,
                "seed": self.config.seed,
            },
        }

        completed_trials = [t for t in self.study.trials if t.state == TrialState.COMPLETE]
        if completed_trials:
            sorted_trials = sorted(
                completed_trials,
                key=lambda t: t.value or float("-inf"),
                reverse=(self.config.direction == "maximize"),
            )
            results["top_trials"] = [
                {"rank": i + 1, "value": t.value, "params": t.params}
                for i, t in enumerate(sorted_trials[:5])
            ]

        return results

    def _save_results(self, results: dict[str, Any]):
        """Save results to JSON file."""
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optuna_search_{timestamp}.json"
        filepath = results_dir / filename

        save_data = {"timestamp": datetime.now().isoformat(), **results}

        with open(filepath, "w") as f:
            json.dump(save_data, f, indent=2, default=str)

        logger.info(f"Results saved to: {filepath}")

    def get_trials_dataframe(self):
        """Export trials to pandas DataFrame."""
        if self.study:
            return self.study.trials_dataframe()
        return None

    def get_param_importances(self) -> dict[str, float]:
        """Get parameter importances using fANOVA."""
        if not self.study or len(self.study.trials) < 2:
            return {}

        try:
            return optuna.importance.get_param_importances(self.study)
        except (RuntimeError, ValueError) as e:
            logger.warning(f"Could not compute param importances: {e}")
            return {}


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
