import torch
import torch.nn as nn

from dendrite.utils.logger_central import get_logger

from .base import ModelBase
from .components import SeparableConv2d

logger = get_logger()


class EEGNet(ModelBase):
    """
    Official EEGNet implementation in PyTorch.

    Based on the official Keras implementation:
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta

    This implements the newest version of EEGNet with:
    1. Depthwise Convolutions to learn spatial filters within temporal convolution
    2. Separable Convolutions to combine spatial filters across temporal bands

    Input Requirements:
    - Shape: (batch, n_channels, n_times) - standard 3D EEG format
    - Data type: float32
    - Device: same as model
    """

    # Class attributes for ModelBase
    _model_type = "EEGNet"
    _description = "Official EEGNet with depthwise and separable convolutions"

    def __init__(
        self,
        n_channels,
        n_times,
        n_classes,
        dropout_rate=0.5,
        kern_length=64,
        F1=8,
        D=2,
        F2=16,
        norm_rate=0.25,
        dropout_type="Dropout",
    ):
        """Initialize EEGNet with architecture parameters.

        The F1, D, F2 parameters follow the naming convention from the original
        EEGNet paper. F2 should equal F1 * D for optimal performance.

        Args:
            n_channels: Number of EEG channels (electrodes).
            n_times: Number of time samples per trial.
            n_classes: Number of output classes.
            dropout_rate: Dropout probability. Default: 0.5
            kern_length: Length of temporal convolution kernel in samples.
                Typically set to half the sampling rate. Default: 64
            F1: Number of temporal filters in Block 1. Default: 8
            D: Depth multiplier for depthwise convolution. Controls how many
                spatial filters to learn per temporal filter. Default: 2
            F2: Number of pointwise filters in Block 2. Should equal F1 * D
                for the separable convolution to maintain information flow. Default: 16
            norm_rate: Max norm constraint for depthwise conv weights. Default: 0.25
            dropout_type: Type of dropout. One of: 'Dropout', 'SpatialDropout2D'.
                Default: 'Dropout'

        Example:
            >>> from dendrite.ml.models import EEGNet
            >>> model = EEGNet(
            ...     n_channels=32,
            ...     n_times=250,  # 1 second at 250 Hz
            ...     n_classes=4,
            ...     F1=8, D=2, F2=16,  # F2 = F1 * D
            ...     kern_length=125   # Half the sampling rate
            ... )
        """
        super().__init__(n_channels, n_times, n_classes)

        # Store parameters for get_model_summary()
        self._params = {
            "F1": F1,
            "D": D,
            "F2": F2,
            "dropout_rate": dropout_rate,
            "norm_rate": norm_rate,
            "kern_length": kern_length,
        }
        self.dropout_rate = dropout_rate
        self.norm_rate = norm_rate

        # Validate F2 = F1 * D relationship (as per original paper)
        if F2 != F1 * D:
            logger.warning(
                f"F2 ({F2}) should equal F1 * D ({F1} * {D} = {F1 * D}) for optimal performance"
            )

        # Select dropout type
        if dropout_type == "SpatialDropout2D":
            self.dropout_layer = nn.Dropout2d
        elif dropout_type == "Dropout":
            self.dropout_layer = nn.Dropout
        else:
            raise ValueError("dropout_type must be one of SpatialDropout2D or Dropout")

        # Block 1: Temporal Convolution + Depthwise Spatial Convolution
        self.temporal_conv = nn.Conv2d(1, F1, (1, kern_length), padding="same", bias=False)
        self.temporal_bn = nn.BatchNorm2d(F1)

        # Depthwise convolution - equivalent to DepthwiseConv2D in Keras
        self.depthwise_conv = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(F1 * D)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = self.dropout_layer(dropout_rate)

        # Block 2: Separable Convolution
        # True separable convolution (depthwise + pointwise) as in original EEGNet paper
        self.separable_conv = SeparableConv2d(F1 * D, F2, (1, 16), padding="same", bias=False)
        self.separable_bn = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = self.dropout_layer(dropout_rate)

        # Calculate flattened size
        self.flatten_size = self._get_flatten_size()

        self.classifier = nn.Linear(self.flatten_size, n_classes)

        logger.info(
            f"EEGNet created: input=(batch, {n_channels}, {n_times}, 1), "
            f"F1={F1}, D={D}, F2={F2}, classes={n_classes}"
        )

    def _get_flatten_size(self) -> int:
        """Calculate the flattened size by running a forward pass."""
        with torch.no_grad():
            x = torch.randn(1, 1, self.n_channels, self.n_times, dtype=torch.float32, device="cpu")
            x = self._forward_features(x)
            return x.numel()

    def _forward_features(self, x):
        """Forward pass through feature extraction layers."""
        # Block 1: Temporal Convolution
        x = self.temporal_conv(x)
        x = self.temporal_bn(x)

        # Depthwise Spatial Convolution
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2: Separable Convolution
        x = self.separable_conv(x)
        x = self.separable_bn(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass accepting 3D input.

        Args:
            x: Input tensor with shape (batch, n_channels, n_times).

        Returns:
            Output tensor with shape (batch, n_classes).
        """
        # Accept 3D input, add channel dim for Conv2D
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # (batch, n_ch, n_times) -> (batch, 1, n_ch, n_times)
        elif len(x.shape) != 4 or x.shape[1] != 1:
            raise ValueError(
                f"EEGNet expects input shape (batch, n_channels, n_times), got {x.shape}"
            )

        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
