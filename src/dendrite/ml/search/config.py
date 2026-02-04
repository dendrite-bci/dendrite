"""Configuration dataclasses for Optuna hyperparameter search."""

from dataclasses import dataclass


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
