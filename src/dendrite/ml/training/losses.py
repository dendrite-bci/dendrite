"""Custom loss functions for decoder training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance and hard examples.

    Down-weights easy examples to focus learning on hard ones.
    FL(p) = -alpha * (1-p)^gamma * log(p)
    """

    def __init__(
        self, gamma: float = 2.0, weight: torch.Tensor = None, label_smoothing: float = 0.0
    ):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
