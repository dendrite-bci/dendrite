"""ML Workbench - offline model training and evaluation.

For internal modules, import directly:
- `from dendrite.ml.search import MODEL_SPACES`
- `from dendrite.auxiliary.ml_workbench.backend.eval_types import EvalResult`
"""

from .app import TrainerApp, main
from .backend import (
    DecoderStorage,
    EpochEvaluator,
    OfflineAsyncRunner,
    OfflineTrainer,
    OptunaConfig,
    OptunaRunner,
    TrainResult,
)

__all__ = [
    "TrainerApp",
    "main",
    "TrainResult",
    "OfflineTrainer",
    "EpochEvaluator",
    "OfflineAsyncRunner",
    "OptunaRunner",
    "OptunaConfig",
    "DecoderStorage",
]
