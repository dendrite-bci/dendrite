"""
Optuna-based hyperparameter search utilities.

Provides search space definitions and configuration generation for
hyperparameter optimization across both real-time processing and
offline training workflows.
"""

from .optuna_runner import OptunaConfig, OptunaRunner
from .optuna_search_cv import OptunaSearchCV
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
    "OptunaRunner",
    "OptunaSearchCV",
    "create_optuna_search_configs",
    "suggest_params",
    "DEFAULT_SEARCH_SPACE",
    "BENCHMARK_SEARCH_SPACE",
    "TRAINING",
    "PROFILES",
    "get_profile",
    "get_search_space_description",
]
