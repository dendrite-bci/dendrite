import torch
import torch.nn as nn

from dendrite.utils.logger_central import get_logger

from .base import ModelBase

logger = get_logger()


class LinearEEG(ModelBase):
    """
    A simple linear model for any modality classification that can learn with very few samples.
    This model is much simpler than EEGNet and requires less data to train effectively.
    Supports any modality (EEG, EMG, EOG, ECG, etc.) due to its generic linear architecture.

    Uses temporal pooling to reduce parameter count while preserving channel information.

    Input Requirements:
    - Shape: (batch, n_channels, n_times)
    - Data type: float32
    - Device: same as model
    """

    # Class attributes for ModelBase
    _model_type = "LinearEEG"
    _modalities = ["any"]  # Works with any modality
    _description = "Simple linear classifier for any physiological modality"

    def __init__(
        self, n_channels, n_times, n_classes, hidden_size=32, dropout_rate=0.2, pool_size=8
    ):
        """Initialize LinearEEG model.

        Uses adaptive temporal pooling to reduce the time dimension before a
        linear projection, making the model parameter-efficient and fast to train.

        Args:
            n_channels: Number of input channels.
            n_times: Number of time samples per trial.
            n_classes: Number of output classes.
            hidden_size: Size of the hidden layer after pooling. Default: 32
            dropout_rate: Dropout probability after hidden layer. Default: 0.2
            pool_size: Target temporal dimension after pooling. The time axis
                is reduced from n_times to pool_size using adaptive average
                pooling. Lower values = fewer parameters. Default: 8

        Example:
            >>> from dendrite.ml.models import LinearEEG
            >>> model = LinearEEG(
            ...     n_channels=8,
            ...     n_times=500,
            ...     n_classes=2,
            ...     pool_size=16,
            ...     hidden_size=64
            ... )
        """
        super().__init__(n_channels, n_times, n_classes)

        # Store parameters for get_model_summary()
        self._params = {
            "hidden_size": hidden_size,
            "dropout_rate": dropout_rate,
            "pool_size": pool_size,
        }
        self.pool_size = pool_size

        # Temporal pooling to reduce time dimension before flattening
        self.temporal_pool = nn.AdaptiveAvgPool1d(pool_size)
        self.flattener = nn.Flatten()

        # Feature extractor: linear projection on pooled features
        self.feature_extractor = nn.Sequential(
            nn.Linear(n_channels * pool_size, hidden_size), nn.ReLU(), nn.Dropout(dropout_rate)
        )

        # Classification head
        self.classifier = nn.Linear(hidden_size, n_classes)

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"LinearEEG created: input=(batch, {n_channels}, {n_times}), pool_size={pool_size}, hidden_size={hidden_size}, params={total_params:,}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass expecting correctly shaped input.

        Args:
            x: Input tensor with shape (batch, n_channels, n_times).

        Returns:
            Output tensor with shape (batch, n_classes).
        """
        # Validate input shape
        if len(x.shape) != 3:
            raise ValueError(
                f"LinearEEG expects input shape (batch, n_channels, n_times), got {x.shape}"
            )

        # Temporal pooling: (batch, channels, times) → (batch, channels, pool_size)
        x = self.temporal_pool(x)

        # Flatten the pooled input
        x = self.flattener(x)

        features = self.feature_extractor(x)
        logits = self.classifier(features)

        return logits
