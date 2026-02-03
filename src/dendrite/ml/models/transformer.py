import torch
import torch.nn as nn

from dendrite.utils.logger_central import get_logger

from .base import ModelBase

logger = get_logger()


class TransformerEEG(ModelBase):
    """
    A transformer-based model for EEG classification.
    This model treats EEG as a sequence where each time point contains all channel values.

    Input Requirements:
    - Shape: (batch, n_channels, n_times)
    - Data type: float32
    - Device: same as model
    """

    # Class attributes for ModelBase
    _model_type = "TransformerEEG"
    _modalities = ["eeg"]
    _description = "Transformer-based architecture for EEG sequence modeling"

    def __init__(
        self,
        n_channels,
        n_times,
        n_classes,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        dropout_rate=0.2,
        positional_encoding=True,
    ):
        """Initialize TransformerEEG model.

        Treats EEG as a sequence where each time point is a token containing
        all channel values. Uses sinusoidal positional encoding (like the
        original Transformer paper) to preserve temporal information.

        Args:
            n_channels: Number of EEG channels (used as input feature dim).
            n_times: Number of time samples (sequence length).
            n_classes: Number of output classes.
            embed_dim: Embedding dimension for transformer layers. Must be
                divisible by num_heads. Default: 64
            num_heads: Number of attention heads. Default: 4
            num_layers: Number of transformer encoder layers. Default: 2
            dropout_rate: Dropout probability in transformer and classifier.
                Default: 0.2
            positional_encoding: Whether to add sinusoidal positional encoding.
                Recommended for EEG where temporal order matters. Default: True

        Example:
            >>> from dendrite.ml.models import TransformerEEG
            >>> model = TransformerEEG(
            ...     n_channels=32,
            ...     n_times=250,
            ...     n_classes=4,
            ...     embed_dim=128,
            ...     num_heads=8,
            ...     num_layers=3
            ... )
        """
        super().__init__(n_channels, n_times, n_classes)

        # Store parameters for get_model_summary()
        self._params = {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "dropout_rate": dropout_rate,
            "positional_encoding": positional_encoding,
        }

        # Initial projection to embedding dimension
        self.input_proj = nn.Linear(n_channels, embed_dim)

        # Positional encoding
        self.positional_encoding = positional_encoding
        if positional_encoding:
            self.pos_encoder = self._create_positional_encoding(self.n_times, embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=2 * embed_dim,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, n_classes),
        )

        logger.info(
            f"TransformerEEG created: input=(batch, {n_channels}, {n_times}), "
            f"embed_dim={embed_dim}, num_heads={num_heads}, num_layers={num_layers}"
        )

    def _create_positional_encoding(self, seq_len, d_model):
        """Create fixed positional encodings for the transformer model."""
        position = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model)
        )

        pos_enc = torch.zeros(seq_len, d_model)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        # Register buffer so it's saved with model
        self.register_buffer("pos_enc", pos_enc)
        return pos_enc

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
                f"TransformerEEG expects input shape (batch, n_channels, n_times), got {x.shape}"
            )

        # Transpose to (batch, n_times, n_channels) for transformer processing
        x = x.transpose(1, 2)

        # Project to embedding dimension: (batch, n_times, embed_dim)
        x = self.input_proj(x)

        # Add positional encoding if enabled
        if self.positional_encoding:
            seq_len = x.size(1)
            if seq_len <= self.pos_enc.size(0):
                x = x + self.pos_enc[:seq_len, :].unsqueeze(0)
            else:
                # If sequence is longer than encoding, use modulo to repeat
                x = x + self.pos_enc[torch.arange(seq_len) % self.pos_enc.size(0), :].unsqueeze(0)

        # Apply transformer
        x = self.transformer_encoder(x)

        # Global pooling (mean across time dimension)
        x = torch.mean(x, dim=1)

        # Apply final classification
        x = self.classifier(x)

        return x
