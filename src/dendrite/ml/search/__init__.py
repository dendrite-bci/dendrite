"""
Optuna-based hyperparameter search utilities.

Provides search space definitions and configuration generation for
hyperparameter optimization across both real-time processing and
offline training workflows.

OptunaRunner and OptunaSearchCV have been moved to:
    dendrite.auxiliary.ml_workbench.backend
"""

from .config import OptunaConfig
from .optuna_utils import (
    DEFAULT_SEARCH_SPACE,
    create_optuna_search_configs,
    suggest_params,
)
from .search_space import (
    BENCHMARK_SEARCH_SPACE,
    PROFILES,
    TRAINING,
    get_profile,
    get_search_space_description,
)

__all__ = [
    "OptunaConfig",
    "create_optuna_search_configs",
    "suggest_params",
    "DEFAULT_SEARCH_SPACE",
    "BENCHMARK_SEARCH_SPACE",
    "TRAINING",
    "PROFILES",
    "get_profile",
    "get_search_space_description",
]
