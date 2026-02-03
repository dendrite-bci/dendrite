"""
Shared neural network components for BMI models.

Contains reusable building blocks used across multiple model architectures.
"""

import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):
    """
    Separable 2D Convolution (Depthwise + Pointwise).

    Implements a depthwise convolution followed by a pointwise convolution,
    drastically reducing parameters compared to regular convolution.

    This is a key component of EEGNet that enables efficient learning with
    fewer parameters, reducing overfitting on small EEG datasets.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride: int = 1,
        padding=0,
        bias: bool = True,
    ):
        super().__init__()

        # Depthwise convolution: Each input channel filtered independently
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # KEY: groups=in_channels makes it depthwise
            bias=False,  # No bias before batch norm
        )

        # Batch normalization after depthwise (improves training stability)
        self.depthwise_bn = nn.BatchNorm2d(in_channels)

        # Pointwise convolution: 1x1 conv for cross-channel mixing
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.depthwise_bn(x)  # Normalize between depthwise and pointwise
        x = self.pointwise(x)
        return x
