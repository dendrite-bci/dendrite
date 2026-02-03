"""
Unified Optuna Hyperparameter Optimization Runner.

Provides a single, reusable Optuna integration for hyperparameter optimization
with proper pruning support. Accepts pluggable objective functions for different
evaluation strategies (async ErrP, cross-validation, ITR, etc.).

Usage:
    from dendrite.ml.search import OptunaRunner, OptunaConfig

    config = OptunaConfig(n_trials=20, pruner_type='median')
    runner = OptunaRunner(config)

    def objective(trial):
        lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        # ... train and evaluate ...
        return accuracy

    results = runner.optimize(objective)
    print(f"Best: {results['best_value']:.1%} with {results['best_params']}")
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import optuna
from optuna.trial import TrialState

from dendrite.utils.logger_central import get_logger

logger = get_logger(__name__)


@dataclass
class OptunaConfig:
    """Configuration for Optuna optimization run."""

    # Core settings
    n_trials: int = 20
    direction: str = "maximize"  # 'maximize' or 'minimize'
    seed: int | None = None

    # Sampler settings
    sampler_type: str = "tpe"  # 'tpe', 'random', 'cmaes'
    n_startup_trials: int = 5  # Random trials before TPE kicks in

    # Pruner settings (key for speed!)
    pruner_type: str = "median"  # 'median', 'percentile', 'none'
    pruner_n_startup_trials: int = 3  # Trials before pruning starts
    pruner_n_warmup_steps: int = 1  # Steps before pruning per trial
    pruner_percentile: float = 25.0  # For percentile pruner

    # Persistence (optional)
    study_name: str | None = None
    storage: str | None = None  # sqlite:///path/to/db
    load_if_exists: bool = False
    results_dir: str | None = None
    verbose: bool = True

    def validate(self) -> list[str]:
        """Validate configuration, return list of issues."""
        issues = []
        if self.n_trials < 1:
            issues.append("n_trials must be >= 1")
        if self.direction not in ["maximize", "minimize"]:
            issues.append(f"direction must be 'maximize' or 'minimize', got '{self.direction}'")
        if self.sampler_type not in ["tpe", "random", "cmaes"]:
            issues.append(f"Unknown sampler_type: {self.sampler_type}")
        if self.pruner_type not in ["median", "percentile", "none"]:
            issues.append(f"Unknown pruner_type: {self.pruner_type}")
        return issues


class OptunaRunner:
    """
    Run hyperparameter optimization using Optuna with pruning support.

    Key features:
    - Pluggable objective function pattern
    - MedianPruner for early stopping of bad trials
    - Progress callbacks matching existing Dendrite patterns
    - JSON export of results
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
        """
        Run optimization with pluggable objective function.

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
            sampler=self._create_sampler(),
            pruner=self._create_pruner(),
            direction=self.config.direction,
        )

        # Suppress Optuna's verbose logging if not verbose
        if not self.config.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        self._report_progress(0, "Starting optimization...")

        # Run optimization
        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            callbacks=[self._optuna_callback],
            show_progress_bar=False,
        )

        self._report_progress(100, "Optimization complete")

        # Build results
        results = self._build_results()

        # Save if results_dir specified
        if self.config.results_dir:
            self._save_results(results)

        return results

    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create sampler based on config."""
        return _create_sampler(
            self.config.sampler_type, self.config.seed, self.config.n_startup_trials
        )

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

        # Count trial states
        completed = len([t for t in self.study.trials if t.state == TrialState.COMPLETE])
        pruned = len([t for t in self.study.trials if t.state == TrialState.PRUNED])
        failed = len([t for t in self.study.trials if t.state == TrialState.FAIL])

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

        # Add top trials summary
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

        # Make results JSON-serializable
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


def _create_sampler(
    sampler_type: str, seed: int | None = None, n_startup_trials: int = 10
) -> optuna.samplers.BaseSampler:
    """Create Optuna sampler based on type.

    Args:
        sampler_type: 'tpe', 'random', 'cmaes', or 'grid'
        seed: Random seed for reproducibility
        n_startup_trials: Random trials before TPE kicks in (TPE only)

    Returns:
        Optuna sampler instance
    """
    if sampler_type == "tpe":
        return optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials)
    elif sampler_type == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    elif sampler_type == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    elif sampler_type == "grid":
        return optuna.samplers.GridSampler({})
    else:
        logger.warning(f"Unknown sampler '{sampler_type}', using TPE")
        return optuna.samplers.TPESampler(seed=seed)
