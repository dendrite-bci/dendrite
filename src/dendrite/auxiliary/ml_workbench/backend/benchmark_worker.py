"""Benchmark worker for running model comparisons in a background thread."""

import random

import numpy as np
import torch
from PyQt6 import QtCore

from dendrite.ml.search import BENCHMARK_SEARCH_SPACE

from .optuna_search_cv import OptunaSearchCV
from dendrite.utils.logger_central import get_logger

logger = get_logger(__name__)

BENCHMARK_SEED = 42

# Evaluation type display names
EVAL_TYPE_NAMES = {
    "within_session": "Within-session",
    "cross_session": "Cross-session",
    "cross_subject": "Cross-subject",
}


def set_reproducibility_seed(seed: int = BENCHMARK_SEED) -> None:
    """Set random seeds for reproducibility across all frameworks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BenchmarkWorker(QtCore.QThread):
    """Worker thread for running benchmarks."""

    progress = QtCore.pyqtSignal(int, str)
    result_ready = QtCore.pyqtSignal(str, dict)
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        study_data: dict,
        models: list[str],
        epochs: int,
        batch_size: int,
        learning_rate: float,
        n_folds: int,
        optuna_config=None,
        eval_type: str = "within_session",
    ):
        super().__init__()
        self._study_data = study_data
        self._models = models
        self._epochs = epochs
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._n_folds = n_folds
        self._optuna_config = optuna_config  # OptunaConfig or None
        self._eval_type = eval_type  # 'within_session', 'cross_session', 'cross_subject'

    def run(self):
        try:
            set_reproducibility_seed(BENCHMARK_SEED)

            config = self._study_data["config"]
            # Unified MOABB evaluation for all datasets
            self._run_moabb_evaluation(config)
        except Exception as e:
            logger.exception("Benchmark error")
            self.error.emit(str(e))

    def _run_moabb_evaluation(self, config):
        """Run benchmark using MOABB's standardized evaluation."""

        try:
            from moabb.paradigms import P300, LeftRightImagery, MotorImagery

            from dendrite.data import InternalDatasetWrapper
            from dendrite.data.imports.moabb_loader import _get_moabb_dataset
        except ImportError as e:
            self.error.emit(f"MOABB not available: {e}")
            return

        # Get dataset - either MOABB or wrapped internal
        selected_subject = self._study_data.get("selected_subject")

        if config.source_type == "moabb":
            dataset = _get_moabb_dataset(config.moabb_dataset)
            if selected_subject is not None:
                dataset.subject_list = [selected_subject]
        else:
            # Internal dataset - wrap for MOABB compatibility
            loader = self._study_data["loader"]
            dataset = InternalDatasetWrapper(loader, config)
            if selected_subject is not None:
                dataset.subject_list = [selected_subject]

        # Determine paradigm from config
        paradigm_name = getattr(config, "moabb_paradigm", None) or "MotorImagery"

        # Build paradigm kwargs from config (consistent with moabb_loader.py)
        paradigm_kwargs = {}
        if hasattr(config, "moabb_n_classes") and config.moabb_n_classes:
            paradigm_kwargs["n_classes"] = config.moabb_n_classes
        if hasattr(config, "moabb_events") and config.moabb_events:
            paradigm_kwargs["events"] = config.moabb_events

        if paradigm_name == "LeftRightImagery":
            paradigm = LeftRightImagery(**paradigm_kwargs)
        elif paradigm_name == "P300":
            paradigm = P300(**paradigm_kwargs)
        else:
            paradigm = MotorImagery(**paradigm_kwargs)

        # Use MOABB's standardized evaluation for all types
        self._run_moabb_eval_loop(dataset, paradigm)

    def _run_moabb_eval_loop(self, dataset, paradigm):
        """Run MOABB evaluation for all evaluation types."""
        from moabb.evaluations import (
            CrossSessionEvaluation,
            CrossSubjectEvaluation,
            WithinSessionEvaluation,
        )
        from sklearn.pipeline import make_pipeline

        from dendrite.ml.decoders import create_decoder

        eval_class = {
            "within_session": WithinSessionEvaluation,
            "cross_session": CrossSessionEvaluation,
            "cross_subject": CrossSubjectEvaluation,
        }.get(self._eval_type, WithinSessionEvaluation)

        evaluation = eval_class(
            paradigm=paradigm,
            datasets=[dataset],
            overwrite=True,
            n_jobs=-1,
        )

        n_models = len(self._models)
        n_subjects = len(dataset.subject_list)
        eval_name = EVAL_TYPE_NAMES.get(self._eval_type, "Evaluation")

        for i, model_name in enumerate(self._models):
            self.progress.emit(
                int((i / n_models) * 100), f"{eval_name} | {model_name} | {n_subjects} subjects..."
            )

            try:
                # Get data shape from first subject
                X_sample, y_sample, _ = paradigm.get_data(
                    dataset, subjects=[dataset.subject_list[0]]
                )
                n_channels, n_times = X_sample.shape[1], X_sample.shape[2]
                n_classes = len(np.unique(y_sample))

                # Create decoder or OptunaSearchCV wrapper (nested CV)
                best_params = {}
                if self._optuna_config:
                    # Nested CV: HPO runs inside each MOABB fold (no data leakage)
                    self.progress.emit(
                        int((i / n_models) * 100),
                        f"Nested CV | {model_name} | {self._optuna_config.n_trials} trials/fold...",
                    )

                    def _factory(
                        n_classes: int, input_shape: list[int], _model=model_name, **params
                    ):
                        return create_decoder(
                            model_type=_model,
                            num_classes=n_classes,
                            input_shapes={"eeg": input_shape},
                            **params,
                        )

                    decoder = OptunaSearchCV(
                        estimator_factory=_factory,
                        param_distributions=BENCHMARK_SEARCH_SPACE,
                        n_trials=self._optuna_config.n_trials,
                        cv=self._n_folds,
                        n_jobs=1,  # Sequential for reproducibility
                        random_state=BENCHMARK_SEED,
                        epochs=self._epochs,
                        base_params={
                            "batch_size": self._batch_size,
                            "learning_rate": self._learning_rate,
                        },
                    )
                else:
                    decoder = create_decoder(
                        model_type=model_name,
                        num_classes=n_classes,
                        input_shapes={"eeg": [n_channels, n_times]},
                        epochs=self._epochs,
                        batch_size=self._batch_size,
                        learning_rate=self._learning_rate,
                    )

                pipelines = {model_name: make_pipeline(decoder)}
                results = evaluation.process(pipelines)

                # Extract best_params from OptunaSearchCV if used
                if self._optuna_config and hasattr(decoder, "best_params_"):
                    best_params = decoder.best_params_
                    logger.info(f"Optuna found best params for {model_name}: {best_params}")

                # Extract per-subject results from MOABB DataFrame
                per_subject = []
                if "subject" in results.columns:
                    for subj in results["subject"].unique():
                        subj_data = results[results["subject"] == subj]
                        per_subject.append(
                            {
                                "subject": subj,
                                "accuracy": float(subj_data["score"].mean()),
                                "accuracy_std": float(subj_data["score"].std())
                                if len(subj_data) > 1
                                else 0,
                            }
                        )

                scores = results["score"].values
                # Note: MOABB 'time' column is evaluation/prediction time, not training time
                eval_time = float(results["time"].sum()) if "time" in results.columns else 0.0
                metrics = {
                    "accuracy": float(np.mean(scores)),
                    "accuracy_std": float(np.std(scores)) if len(scores) > 1 else 0,
                    "kappa": float("nan"),  # MOABB cross-eval doesn't compute kappa
                    "f1": float("nan"),
                    "balanced_accuracy": float("nan"),
                    "eval_time": eval_time,
                    "n_subjects": len(dataset.subject_list),
                    "n_sessions": len(results),
                    "eval_type": self._eval_type,
                    "per_subject": per_subject,
                    "best_params": best_params,
                }

                # Show completion status
                self.progress.emit(
                    int(((i + 1) / n_models) * 100),
                    f"{model_name}: {metrics['accuracy'] * 100:.1f}% ({i + 1}/{n_models} models)",
                )
                self.result_ready.emit(model_name, metrics)

            except Exception as e:
                logger.error(f"MOABB eval failed for {model_name}: {e}")
                self.result_ready.emit(
                    model_name,
                    {
                        "accuracy": 0,
                        "accuracy_std": 0,
                        "kappa": float("nan"),
                        "f1": float("nan"),
                        "balanced_accuracy": float("nan"),
                        "eval_time": 0,
                        "error": str(e),
                    },
                )

        self.progress.emit(100, "Complete")
        self.finished.emit()
