"""ML Workbench - offline model training and evaluation.

For internal modules, import directly:
- `from dendrite.ml.search import MODEL_SPACES`
- `from dendrite.auxiliary.ml_workbench.backend.types import EvalResult`
"""

from .backend import (
    EvalResult,
    OfflineAsyncRunner,
    OfflineTrainer,
    OptunaConfig,
    OptunaRunner,
    TrainResult,
)

__all__ = [
    "EvalResult",
    "OfflineAsyncRunner",
    "OfflineTrainer",
    "OptunaConfig",
    "OptunaRunner",
    "TrainResult",
]
