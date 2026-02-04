"""Shared worker classes for async operations in offline ML GUI."""

from typing import Any

import numpy as np
from PyQt6 import QtCore

from dendrite.auxiliary.ml_workbench.utils import format_duration
from dendrite.utils.logger_central import get_logger

logger = get_logger(__name__)


__all__ = [
    "DataLoaderWorker",
    "DirectTrainingWorker",
    "OptunaTrainingWorker",
]


class DataLoaderWorker(QtCore.QObject):
    """Worker for loading study data in a background thread.

    Used by both training_tab and evaluation_tab to load epochs without
    blocking the UI.
    """

    progress = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object, object)  # (X, y) or (None, None)
    validation_ready = QtCore.pyqtSignal(object)  # (continuous, times, labels) or None
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        loader: Any,
        subjects: list[int],
        capability: str = "epochs",
        holdout_pct: int = 0,
    ):
        """Initialize the data loader worker.

        Args:
            loader: Dataset loader with load_epochs() method
            subjects: List of subject IDs to load
            capability: Dataset capability ('full' or 'epochs')
            holdout_pct: Percentage of events to hold out for evaluation (0-50)
        """
        super().__init__()
        self.loader = loader
        self.subjects = subjects
        self.capability = capability
        self.holdout_pct = holdout_pct
        self._stopped = False

    def stop(self):
        """Signal the worker to stop."""
        self._stopped = True

    @QtCore.pyqtSlot()
    def run(self):
        """Load data from all subjects."""
        try:
            all_X, all_y = [], []
            validation_chunks = []
            val_sample_offset = 0

            for i, subj in enumerate(self.subjects):
                if self._stopped:
                    self.finished.emit(None, None)
                    return

                self.progress.emit(f"Loading subject {subj} ({i + 1}/{len(self.subjects)})...")

                try:
                    # Warn if holdout requested but conditions not met
                    if self.holdout_pct > 0:
                        if self.capability != "full":
                            logger.warning(
                                f"Holdout requested ({self.holdout_pct}%) but capability is "
                                f"'{self.capability}', not 'full'. Holdout disabled."
                            )
                        elif not hasattr(self.loader, "load_data_split"):
                            logger.warning(
                                f"Holdout requested ({self.holdout_pct}%) but loader doesn't "
                                "support load_data_split. Holdout disabled."
                            )

                    # Use holdout split if enabled and capability is full
                    if (
                        self.holdout_pct > 0
                        and self.capability == "full"
                        and hasattr(self.loader, "load_data_split")
                    ):
                        self.progress.emit(f"Loading with {self.holdout_pct}% holdout...")
                        val_ratio = self.holdout_pct / 100.0
                        (X, y), val_data = self.loader.load_data_split(
                            subj, block=1, val_ratio=val_ratio
                        )

                        # Accumulate validation data from all subjects
                        if val_data is not None:
                            val_cont, val_times, val_labels = val_data
                            # Adjust times with offset for multi-subject concatenation
                            adjusted_times = val_times + val_sample_offset
                            validation_chunks.append((val_cont, adjusted_times, val_labels))
                            val_sample_offset += val_cont.shape[1]
                    else:
                        X, y = self.loader.load_epochs(subj)

                    # Check again after blocking load completes
                    if self._stopped:
                        self.progress.emit("Cancelled")
                        self.finished.emit(None, None)
                        return

                    all_X.append(X)
                    all_y.append(y)
                except Exception as e:
                    logger.warning(f"Failed to load subject {subj}: {e}")
                    continue

            if not all_X:
                self.error.emit("No data loaded from any subject")
                self.finished.emit(None, None)
                return

            X = np.concatenate(all_X, axis=0)
            y = np.concatenate(all_y, axis=0)

            # Merge validation data from all subjects
            validation_data = None
            if validation_chunks:
                validation_data = (
                    np.concatenate([c[0] for c in validation_chunks], axis=1),
                    np.concatenate([c[1] for c in validation_chunks]),
                    np.concatenate([c[2] for c in validation_chunks]),
                )
                val_cont, val_times, val_labels = validation_data
                logger.info(
                    f"Validation data merged: {len(val_labels)} events, "
                    f"continuous shape: {val_cont.shape}"
                )

            # Calculate duration for progress reporting
            sample_rate = self.loader.get_sample_rate()
            epoch_samples = X.shape[2]
            train_duration = len(X) * epoch_samples / sample_rate

            # Report split info with duration
            if validation_data:
                val_cont = validation_data[0]
                val_duration = val_cont.shape[1] / sample_rate
                self.progress.emit(
                    f"Loaded {len(X)} train epochs ({format_duration(train_duration)}), "
                    f"{len(val_labels)} eval events ({format_duration(val_duration)}) "
                    f"from {len(validation_chunks)} subjects"
                )
                self.validation_ready.emit(validation_data)
            else:
                self.progress.emit(
                    f"Loaded {len(X)} epochs ({format_duration(train_duration)}) "
                    f"from {len(all_X)} subjects"
                )
                self.validation_ready.emit(None)

            self.finished.emit(X, y)

        except Exception as e:
            logger.exception("Data loading failed")
            self.error.emit(str(e))
            self.finished.emit(None, None)


class DirectTrainingWorker(QtCore.QObject):
    """Worker for running direct training without hyperparameter search."""

    progress = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    epoch_completed = QtCore.pyqtSignal(
        int, int, float, float, object, object
    )  # epoch, total, train_loss, train_acc, val_loss, val_acc

    def __init__(self, X: np.ndarray, y: np.ndarray, config: Any):
        """Initialize training worker.

        Args:
            X: Training data, shape (n_samples, n_channels, n_times)
            y: Labels, shape (n_samples,)
            config: TrainerConfig with training parameters
        """
        super().__init__()
        self.X = X
        self.y = y
        self.config = config
        self._stopped = False

    def stop(self):
        """Request the worker to stop."""
        self._stopped = True

    @QtCore.pyqtSlot()
    def run(self):
        """Run training and return TrainResult."""
        try:
            if self._stopped:
                self.finished.emit(None)
                return

            from dendrite.auxiliary.ml_workbench import OfflineTrainer

            self.progress.emit("Training with default hyperparameters...")
            trainer = OfflineTrainer()

            def epoch_callback(epoch, total, train_loss, train_acc, val_loss, val_acc):
                self.epoch_completed.emit(epoch, total, train_loss, train_acc, val_loss, val_acc)

            result = trainer.train(
                self.X,
                self.y,
                self.config,
                progress_callback=lambda msg: self.progress.emit(msg),
                epoch_callback=epoch_callback,
            )

            if result:
                self.progress.emit(f"Training complete! Val acc: {result.val_accuracy * 100:.1f}%")
                self.finished.emit(result)
            else:
                self.error.emit("Training failed")
                self.finished.emit(None)
        except Exception as e:
            logger.exception("DirectTrainingWorker failed")
            self.error.emit(str(e))
            self.finished.emit(None)


class OptunaTrainingWorker(QtCore.QObject):
    """Worker for running Optuna hyperparameter search in a thread."""

    progress = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    optuna_finished = QtCore.pyqtSignal(dict)  # Full optuna results
    trial_completed = QtCore.pyqtSignal(int, float, bool)  # trial_num, accuracy, is_best

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_config: Any,
        optuna_config: Any,
        search_space: dict[str, Any],
    ):
        """Initialize Optuna search worker.

        Args:
            X: Training data
            y: Labels
            base_config: TrainerConfig with base training parameters
            optuna_config: OptunaConfig with search parameters
            search_space: Dict defining hyperparameter search space
        """
        super().__init__()
        self.X = X
        self.y = y
        self.base_config = base_config
        self.optuna_config = optuna_config
        self.search_space = search_space
        self._stopped = False
        self._best_result = None
        self._best_accuracy = 0.0

    def stop(self):
        """Request the worker to stop."""
        self._stopped = True

    @QtCore.pyqtSlot()
    def run(self):
        """Run Optuna search and return best TrainResult."""
        try:
            if self._stopped:
                self.finished.emit(None)
                return

            from dendrite.auxiliary.ml_workbench import OfflineTrainer
            from dendrite.ml.decoders.decoder_schemas import DecoderConfig
            from dendrite.ml.search import suggest_params

            from .optuna_runner import OptunaRunner

            self.progress.emit("Starting Optuna search...")
            runner = OptunaRunner(self.optuna_config)

            # Reset best tracking
            self._best_result = None
            self._best_accuracy = 0.0

            def objective(trial):
                if self._stopped:
                    raise Exception("Search stopped by user")

                # Sample hyperparameters from search space
                suggested = suggest_params(trial, self.search_space, skip_params=["model_type"])

                # Build config with suggested params
                config_dict = self.base_config.model_dump()
                config_dict.update(suggested)
                config = DecoderConfig(**config_dict)

                trainer = OfflineTrainer()
                result = trainer.train(
                    self.X,
                    self.y,
                    config,
                    progress_callback=lambda msg: self.progress.emit(
                        f"Trial {trial.number + 1}: {msg}"
                    ),
                )

                # Track best and emit progress
                accuracy = result.val_accuracy if result else 0.0
                is_best = result and accuracy > self._best_accuracy

                if is_best:
                    self._best_accuracy = accuracy
                    self._best_result = result

                # Emit structured trial data for live plot
                self.trial_completed.emit(trial.number + 1, accuracy, is_best)

                return accuracy

            def progress_callback(percent: int, message: str):
                self.progress.emit(message)

            optuna_results = runner.optimize(objective, progress_callback=progress_callback)

            # Emit optuna results for display
            self.optuna_finished.emit(optuna_results)

            # Return best result
            if self._best_result:
                self.progress.emit(
                    f"Search complete! Best val acc: {self._best_accuracy * 100:.1f}%"
                )
                self.finished.emit(self._best_result)
            else:
                self.error.emit("No successful trials")
                self.finished.emit(None)
        except Exception as e:
            logger.exception("OptunaTrainingWorker failed")
            self.error.emit(str(e))
            self.finished.emit(None)
