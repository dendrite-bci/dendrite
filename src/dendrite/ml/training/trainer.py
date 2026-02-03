"""Neural network training loop with config-driven behaviors.

The TrainingLoop handles the training loop with all behaviors (early stopping,
checkpointing, LR scheduling, SWA) controlled via config flags.

Usage:
    from dendrite.ml.training import TrainingLoop

    training_loop = TrainingLoop(model=model, config=config, prepare_input_fn=prepare_fn)
    results = training_loop.fit(X_train, y_train, X_val, y_val)
"""

import copy
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import SWALR, AveragedModel, update_bn
from torch.utils.data import DataLoader, TensorDataset

from dendrite.ml.training.augmentation import apply_cutmix, apply_mixup, get_augmentation_manager
from dendrite.ml.training.losses import FocalLoss


def create_optimizer(config, model: nn.Module) -> optim.Optimizer:
    """Create optimizer based on config."""
    if config.optimizer_type == "AdamW":
        return optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    return optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


def create_criterion(config, class_weights: torch.Tensor | None) -> nn.Module:
    """Create loss function based on config."""
    label_smoothing = config.label_smoothing_factor
    if config.loss_type == "focal":
        return FocalLoss(
            gamma=config.focal_gamma, weight=class_weights, label_smoothing=label_smoothing
        )
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)


class TrainingLoop:
    """Training loop with config-driven behaviors.

    Handles:
    - Epoch/batch iteration
    - Forward/backward passes
    - Validation
    - Early stopping (config.use_early_stopping)
    - Model checkpointing (saves best val_loss model)
    - LR scheduling (config.use_lr_scheduler)
    - LR warmup (config.use_lr_warmup)
    - SWA (config.use_swa)
    - Online augmentation (config.use_augmentation, config.mixup_alpha > 0)
    """

    GRAD_CLIP_MAX_NORM = 1.0

    def __init__(
        self,
        model: nn.Module,
        config: Any,
        prepare_input_fn: Callable[[np.ndarray], torch.Tensor],
    ):
        """Initialize trainer.

        Args:
            model: PyTorch model to train
            config: NeuralNetConfig with training parameters
            prepare_input_fn: Function to convert X array to tensor (required)
        """
        self.model = model
        self.config = config

        # Injected function for custom input handling
        self._prepare_input = prepare_input_fn

        # Augmentation setup
        self.use_augmentation = config.use_augmentation
        if self.use_augmentation:
            self.augmentation_manager = get_augmentation_manager()
            self.aug_strategy = config.aug_strategy
        else:
            self.augmentation_manager = None

        # Training state (initialized in fit)
        self.optimizer: optim.Optimizer | None = None
        self.criterion: nn.Module | None = None
        self.device: torch.device | None = None
        self.X_train: np.ndarray | None = None

        # Early stopping state
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.best_model_state: dict[str, Any] | None = None
        self.patience_counter = 0
        self.stop_training = False

        # LR scheduling state
        self.scheduler: optim.lr_scheduler.LRScheduler | None = None
        self.warmup_scheduler: optim.lr_scheduler.LinearLR | None = None
        self.warmup_done = False

        # SWA state
        self.swa_model: AveragedModel | None = None
        self.swa_scheduler: SWALR | None = None
        self.swa_start_epoch: int = 0
        self.swa_active = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epoch_callback: Callable[[int, int, float, float, float | None, float | None], None]
        | None = None,
    ) -> dict[str, Any]:
        """Run training loop.

        Args:
            X: Training data array (n_samples, n_channels, n_times)
            y: Training labels
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            epoch_callback: Called after each epoch with (epoch, total_epochs, train_loss, train_acc, val_loss, val_acc)

        Returns:
            Training results dict with history and final metrics
        """
        start_time = time.time()

        self._setup_training(X, y)
        self._setup_schedulers()
        if self.config.use_swa:
            self._setup_swa()

        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []

        for epoch in range(self.config.epochs):
            if self.stop_training:
                break

            # Training epoch
            train_loss, train_acc = self._train_epoch(X, y, epoch)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)

            val_loss, val_acc = None, None
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self._validate(X_val, y_val)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)

            # Epoch callback for progress reporting
            if epoch_callback:
                epoch_callback(epoch, self.config.epochs, train_loss, train_acc, val_loss, val_acc)

            # Post-epoch updates
            self._update_schedulers(epoch, val_loss)
            self._check_early_stopping(val_loss, val_acc)
            self._update_swa(epoch)

        self._finalize_training()

        training_time = time.time() - start_time
        return self._build_results(
            train_losses, train_accuracies, val_losses, val_accuracies, training_time
        )

    def _setup_training(self, X: np.ndarray, y: np.ndarray) -> None:
        """Initialize training components."""
        self.device = self.config.get_device()
        self.X_train = X

        self.optimizer = create_optimizer(self.config, self.model)
        class_weights = self._calculate_class_weights(y, self.device)
        self.criterion = create_criterion(self.config, class_weights)

        # Reset state
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.best_model_state = None
        self.patience_counter = 0
        self.stop_training = False
        self.warmup_done = False
        self.swa_active = False

    def _setup_schedulers(self) -> None:
        """Setup LR schedulers based on config."""
        # Warmup scheduler
        if self.config.use_lr_warmup:
            self.warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=self.config.warmup_start_factor,
                end_factor=1.0,
                total_iters=self.config.warmup_epochs,
            )

        # Main scheduler
        if not self.config.use_lr_scheduler:
            self.scheduler = None
            return

        scheduler_type = self.config.lr_scheduler_type

        if scheduler_type == "ReduceLROnPlateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.lr_factor,
                patience=self.config.lr_patience,
                min_lr=self.config.lr_min,
            )
        elif scheduler_type == "StepLR":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.config.lr_step_size, gamma=self.config.lr_factor
            )
        elif scheduler_type == "CosineAnnealingLR":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.epochs, eta_min=self.config.lr_min
            )
        elif scheduler_type == "OneCycleLR":
            # OneCycleLR needs steps_per_epoch (must match actual batch count)
            n_samples = len(self.X_train)
            batch_size = self.config.batch_size
            steps_per_epoch = max(1, (n_samples + batch_size - 1) // batch_size)
            max_lr = self.config.onecycle_max_lr or (self.config.learning_rate * 10)

            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lr,
                epochs=self.config.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=self.config.onecycle_pct_start,
            )

    def _setup_swa(self) -> None:
        """Setup Stochastic Weight Averaging."""
        self.swa_model = AveragedModel(self.model)
        self.swa_start_epoch = int(self.config.swa_start_epoch * self.config.epochs)

        swa_lr = self.config.swa_lr or self.optimizer.param_groups[0]["lr"]
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=swa_lr)

    def _calculate_class_weights(self, y: np.ndarray, device: torch.device) -> torch.Tensor | None:
        """Calculate class weights for imbalanced data."""
        if not self.config.use_class_weights:
            return None

        unique_classes, class_counts = np.unique(y, return_counts=True)
        n_samples = len(y)
        n_classes = len(unique_classes)

        if self.config.class_weight_strategy == "balanced":
            weights = n_samples / (n_classes * class_counts)
        elif self.config.class_weight_strategy == "inverse":
            weights = 1.0 / class_counts
            weights = weights / weights.sum() * n_classes
        else:
            weights = np.ones(n_classes)

        weight_tensor = torch.zeros(self.config.num_classes, device=device)
        for idx, class_label in enumerate(unique_classes):
            if class_label < self.config.num_classes:
                weight_tensor[class_label] = weights[idx]

        return weight_tensor

    def _apply_mixup_if_enabled(
        self, X_batch: np.ndarray, y_batch: np.ndarray, use_mixup: bool
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float | None]:
        """Apply mixup/cutmix augmentation if enabled.

        Returns:
            Tuple of (X_batch, y1, y2, lambda) - y1/y2/lambda are None if mixup not active
        """
        if not use_mixup or len(y_batch) <= 1:
            return X_batch, None, None, None

        mixup_fn = apply_cutmix if self.config.mixup_type == "cutmix" else apply_mixup
        if self.config.mixup_type == "both":
            mixup_fn = apply_cutmix if np.random.random() > 0.5 else apply_mixup

        X_mixed, y1, y2, lam = mixup_fn(X_batch, y_batch, self.config.mixup_alpha)
        return X_mixed, y1, y2, lam

    def _compute_loss_and_accuracy(
        self,
        outputs: torch.Tensor,
        y_batch: np.ndarray,
        y1: np.ndarray | None,
        y2: np.ndarray | None,
        lam: float | None,
    ) -> tuple[torch.Tensor, float]:
        """Compute loss and accuracy, handling mixup if active."""
        mixup_active = y1 is not None

        if mixup_active:
            y1_tensor = torch.LongTensor(y1).to(self.device)
            y2_tensor = torch.LongTensor(y2).to(self.device)
            loss = lam * self.criterion(outputs, y1_tensor) + (1 - lam) * self.criterion(
                outputs, y2_tensor
            )
            _, predicted = torch.max(outputs.data, 1)
            correct = (
                (
                    lam * (predicted == y1_tensor).float()
                    + (1 - lam) * (predicted == y2_tensor).float()
                )
                .sum()
                .item()
            )
        else:
            y_tensor = torch.LongTensor(y_batch).to(self.device)
            loss = self.criterion(outputs, y_tensor)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_tensor).sum().item()

        return loss, correct

    def _apply_max_norm_constraint(self) -> None:
        """Apply max norm constraint to last linear layer of any model.

        This regularization technique clamps the weight norms of the classifier layer
        after each optimizer step. Originally from EEGNet, it prevents any single
        neuron from dominating and improves generalization.

        Works with any model architecture by finding the last nn.Linear layer.
        """
        max_norm = self.config.max_norm_constraint
        if max_norm is None:
            return

        # Find the last Linear layer (classifier)
        last_linear = None
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                last_linear = module

        if last_linear is None:
            return

        with torch.no_grad():
            norm = last_linear.weight.norm(dim=1, keepdim=True)
            desired = torch.clamp(norm, max=max_norm)
            last_linear.weight.copy_(last_linear.weight * desired / (norm + 1e-8))

    def _train_epoch(self, X: np.ndarray, y: np.ndarray, epoch: int) -> tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        batch_size = self.config.batch_size
        n_samples = len(y)
        use_mixup = self.config.mixup_alpha > 0
        is_onecycle = self.config.lr_scheduler_type == "OneCycleLR"

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = X[i:end_idx]
            y_batch = y[i:end_idx]

            # Online augmentation
            if self.use_augmentation and self.augmentation_manager:
                X_batch = self.augmentation_manager.transform_array(
                    X_batch, strategy_name=self.aug_strategy, prob=0.5
                )

            # Mixup/CutMix
            X_batch, y1, y2, lam = self._apply_mixup_if_enabled(X_batch, y_batch, use_mixup)

            # Forward pass
            X_tensor = self._prepare_input(X_batch)
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)

            # Loss and accuracy
            loss, batch_correct = self._compute_loss_and_accuracy(outputs, y_batch, y1, y2, lam)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.GRAD_CLIP_MAX_NORM
            )
            self.optimizer.step()

            self._apply_max_norm_constraint()

            if is_onecycle and self.scheduler is not None and not self.swa_active:
                self.scheduler.step()

            epoch_loss += loss.item()
            correct += batch_correct
            total += outputs.size(0)

        avg_loss = epoch_loss / max(1, n_samples // batch_size)
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def _validate(self, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Evaluate on validation data."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        batch_size = self.config.batch_size
        n_samples = len(y)

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)

                X_batch = X[i:end_idx]
                y_batch = y[i:end_idx]

                X_tensor = self._prepare_input(X_batch)
                y_tensor = torch.LongTensor(y_batch).to(self.device)

                outputs = self.model(X_tensor)
                loss = self.criterion(outputs, y_tensor)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == y_tensor).sum().item()
                total += y_tensor.size(0)

        avg_loss = total_loss / max(1, n_samples // batch_size)
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def _update_schedulers(self, epoch: int, val_loss: float | None) -> None:
        """Update LR schedulers after epoch."""
        # Warmup phase
        if self.config.use_lr_warmup and not self.warmup_done:
            self.warmup_scheduler.step()
            if epoch + 1 >= self.config.warmup_epochs:
                self.warmup_done = True
            return

        # SWA phase uses its own scheduler
        if self.swa_active:
            self.swa_scheduler.step()
            return

        # Main scheduler (skip OneCycleLR - it steps per batch)
        if self.scheduler is None or self.config.lr_scheduler_type == "OneCycleLR":
            return

        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            metric = val_loss if val_loss is not None else 0.0
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def _check_early_stopping(self, val_loss: float | None, val_acc: float | None) -> None:
        """Check early stopping condition and save best model."""
        if val_loss is None:
            return

        min_delta = self.config.early_stopping_min_delta

        if val_loss < self.best_val_loss - min_delta:
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc or 0.0
            self.best_model_state = copy.deepcopy(self.model.state_dict())
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.config.use_early_stopping:
                if self.patience_counter >= self.config.early_stopping_patience:
                    self.stop_training = True

    def _update_swa(self, epoch: int) -> None:
        """Update SWA model if in SWA phase."""
        if not self.config.use_swa or self.swa_model is None:
            return

        if epoch >= self.swa_start_epoch:
            self.swa_model.update_parameters(self.model)
            self.swa_active = True

    def _finalize_training(self) -> None:
        """Finalize training: restore best model or apply SWA."""
        # SWA finalization
        if self.swa_active and self.swa_model is not None:
            self._update_swa_bn()
            # Copy SWA weights to main model
            self.model.load_state_dict(self.swa_model.module.state_dict())
        # Restore best model (if not using SWA)
        elif self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

    def _update_swa_bn(self) -> None:
        """Update batch norm stats for SWA model."""
        if self.X_train is None:
            return

        train_data = self.X_train

        # Add channel dimension if needed
        if len(train_data.shape) == 3:
            train_data = np.expand_dims(train_data, axis=1)

        dataset = TensorDataset(torch.FloatTensor(train_data))
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        update_bn(loader, self.swa_model, device=self.device)

    def _build_results(
        self,
        train_losses: list[float],
        train_accuracies: list[float],
        val_losses: list[float],
        val_accuracies: list[float],
        training_time: float,
    ) -> dict[str, Any]:
        """Build training results dictionary."""
        epochs_completed = len(train_losses)

        results = {
            "final_train_acc": train_accuracies[-1] if train_accuracies else 0.0,
            "final_train_loss": train_losses[-1] if train_losses else 0.0,
            "epochs_completed": epochs_completed,
            "training_time": training_time,
            "early_stopped": self.stop_training,
            "history": {
                "loss": train_losses,
                "accuracy": train_accuracies,
            },
        }

        if val_losses:
            results["final_val_acc"] = val_accuracies[-1]
            results["final_val_loss"] = val_losses[-1]
            results["history"]["val_loss"] = val_losses
            results["history"]["val_accuracy"] = val_accuracies

            if self.best_model_state is not None:
                results["best_val_loss"] = self.best_val_loss
                results["best_val_acc"] = self.best_val_acc

        if self.swa_active:
            results["swa_applied"] = True

        return results
