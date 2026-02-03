"""Inference utilities for uncertainty estimation.

MC Dropout (Monte Carlo Dropout) estimates prediction uncertainty by
running multiple forward passes with dropout enabled.

Usage:
    from dendrite.ml.training import mc_dropout_predict

    mean_proba, std_proba = mc_dropout_predict(model, X_tensor, n_samples=10)
"""

import numpy as np
import torch
import torch.nn as nn


def mc_dropout_predict(
    model: nn.Module, X_tensor: torch.Tensor, n_samples: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Run MC Dropout inference for uncertainty estimation.

    Performs multiple forward passes with dropout enabled to estimate
    prediction uncertainty. Higher std indicates lower confidence.

    Args:
        model: PyTorch model with dropout layers
        X_tensor: Input tensor, shape (batch, ...)
        n_samples: Number of forward passes (more = better estimate, slower)

    Returns:
        mean_proba: Mean class probabilities, shape (batch, n_classes)
        std_proba: Standard deviation of probabilities (uncertainty)
    """
    was_training = model.training
    model.train()  # Enable dropout

    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            outputs = model(X_tensor)
            proba = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions.append(proba)

    if not was_training:
        model.eval()

    predictions = np.stack(predictions)  # (n_samples, batch, n_classes)
    mean_proba = predictions.mean(axis=0)
    std_proba = predictions.std(axis=0)

    return mean_proba, std_proba
