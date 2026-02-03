"""
EEGNet++ - Enhanced EEGNet with modern ML techniques.

Improvements over original EEGNet:
1. Squeeze-and-Excitation (SE) block for channel attention
2. Residual connection around separable conv block
3. Maintains parameter efficiency of original design

Reference:
- Original EEGNet: Lawhern et al., 2018 (J. Neural Eng.)
- SE Block: Hu et al., 2018 (CVPR)
"""

import torch
import torch.nn as nn

from dendrite.utils.logger_central import get_logger

from .base import ModelBase
from .components import SeparableConv2d

logger = get_logger()


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.

    Adaptively recalibrates channel-wise feature responses by explicitly
    modeling interdependencies between channels.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(channels // reduction, 4)  # Minimum 4 channels

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # Squeeze: global average pooling
        y = self.squeeze(x).view(b, c)
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale: channel-wise multiplication
        return x * y


class EEGNetPP(ModelBase):
    """
    EEGNet++ - Enhanced EEGNet with SE attention and residual connections.

    Architecture improvements:
    - SE block after depthwise conv for adaptive channel weighting
    - Residual connection around separable conv block
    - Maintains EEGNet's parameter efficiency

    Input Requirements:
    - Shape: (batch, n_channels, n_times) - standard 3D EEG format
    """

    # Class attributes for ModelBase
    _model_type = "EEGNetPP"
    _description = "EEGNet++ with SE attention and residual connections"

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        dropout_rate: float = 0.5,
        kern_length: int = 64,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        norm_rate: float = 0.25,
        se_reduction: int = 4,
        use_residual: bool = True,
    ):
        super().__init__(n_channels, n_times, n_classes)

        # Store parameters for get_model_summary()
        self._params = {
            "F1": F1,
            "D": D,
            "F2": F2,
            "dropout_rate": dropout_rate,
            "kern_length": kern_length,
            "se_reduction": se_reduction,
            "use_residual": use_residual,
        }
        self.dropout_rate = dropout_rate
        self.norm_rate = norm_rate
        self.use_residual = use_residual

        # Validate F2 = F1 * D relationship
        if F2 != F1 * D:
            logger.warning(f"F2 ({F2}) should equal F1 * D ({F1} * {D} = {F1 * D})")

        # Block 1: Temporal + Depthwise Spatial Conv
        self.temporal_conv = nn.Conv2d(1, F1, (1, kern_length), padding="same", bias=False)
        self.temporal_bn = nn.BatchNorm2d(F1)

        self.depthwise_conv = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(F1 * D)
        self.elu1 = nn.ELU()

        # SE Block (NEW) - channel attention after spatial conv
        self.se_block = SqueezeExcitation(F1 * D, reduction=se_reduction)

        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Block 2: Separable Conv with residual
        self.separable_conv = SeparableConv2d(F1 * D, F2, (1, 16), padding="same", bias=False)
        self.separable_bn = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()

        # Residual projection if channels don't match
        if use_residual and F1 * D != F2:
            self.residual_proj = nn.Conv2d(F1 * D, F2, kernel_size=1, bias=False)
        else:
            self.residual_proj = None

        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        self.flatten_size = self._get_flatten_size()
        self.classifier = nn.Linear(self.flatten_size, n_classes)

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"EEGNet++ created: input=(batch, 1, {n_channels}, {n_times}), "
            f"F1={F1}, D={D}, F2={F2}, SE_reduction={se_reduction}, "
            f"residual={use_residual}, params={total_params:,}"
        )

    def _get_flatten_size(self) -> int:
        """Calculate flattened size by running dummy forward pass."""
        with torch.no_grad():
            x = torch.randn(1, 1, self.n_channels, self.n_times, dtype=torch.float32)
            x = self._forward_features(x)
            return x.numel()

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Feature extraction through conv blocks."""
        # Block 1: Temporal conv
        x = self.temporal_conv(x)
        x = self.temporal_bn(x)

        # Depthwise spatial conv
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.elu1(x)

        # SE block (channel attention)
        x = self.se_block(x)

        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2: Separable conv with residual
        identity = x

        x = self.separable_conv(x)
        x = self.separable_bn(x)

        # Add residual connection
        if self.use_residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            x = x + identity

        x = self.elu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass accepting 3D input.

        Args:
            x: Input tensor (batch, n_channels, n_times)

        Returns:
            Output logits (batch, n_classes)
        """
        # Accept 3D input, add channel dim for Conv2D
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # (batch, n_ch, n_times) -> (batch, 1, n_ch, n_times)
        elif len(x.shape) != 4 or x.shape[1] != 1:
            raise ValueError(f"EEGNetPP expects (batch, n_channels, n_times), got {x.shape}")

        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
