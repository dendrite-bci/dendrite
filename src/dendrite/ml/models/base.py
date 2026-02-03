"""Abstract base class for Dendrite neural network models.

This module defines the standard interface that all Dendrite models must implement,
ensuring consistency across different architectures and providing clear
documentation for model developers.

Example:
    Creating a custom model::

        import torch.nn as nn
        from dendrite.ml.models.base import ModelBase

        class MyCustomModel(ModelBase):
            _model_type = 'MyCustomModel'
            _description = 'Custom CNN for EEG classification'
            _modalities = ['eeg']

            def __init__(self, n_channels, n_times, n_classes, hidden_dim=64):
                super().__init__(n_channels, n_times, n_classes)
                self._params = {'hidden_dim': hidden_dim}

                self.conv = nn.Conv1d(n_channels, hidden_dim, kernel_size=5)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Linear(hidden_dim, n_classes)

            def forward(self, x):
                # x shape: (batch, n_channels, n_times)
                x = self.conv(x)
                x = self.pool(x).squeeze(-1)
                return self.fc(x)
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from dendrite.utils.logger_central import get_logger

logger = get_logger().getChild("models.base")


class ModelBase(nn.Module, ABC):
    """Abstract base class for all Dendrite neural network models.

    This class enforces a standard interface for EEG/EMG/EOG classification models
    used in the Dendrite system. All concrete model implementations must inherit from
    this class and implement the required abstract methods.

    Standard Interface Requirements:
        1. Constructor must accept (n_channels, n_times, n_classes) parameters
        2. Define class attributes for model metadata (see below)
        3. Must implement forward() method (inherited from nn.Module)

    Class Attributes (override in subclasses):
        _model_type: Model identifier (must match registry key).
        _input_domain: 'time-series' or 'time-frequency'.
        _modalities: Supported modalities ['eeg'], ['any'], etc.
        _description: Human-readable description.
        _default_parameters: Default hyperparameters.

    Input/Output Conventions:
        Input: torch.Tensor with physiological signals.
        Output: torch.Tensor with shape (batch_size, n_classes) logits.
        Device: Models should work on both CPU and GPU.
        Dtype: Expects float32 tensors.
    """

    # Class attributes - subclasses should override these
    _model_type: str | None = None
    _input_domain: str = "time-series"
    _modalities: list[str] = ["eeg"]
    _description: str = ""
    _config_class = None  # Pydantic config class (single source of truth)

    @abstractmethod
    def __init__(self, n_channels: int, n_times: int, n_classes: int, **kwargs):
        """Initialize the model with standard parameters.

        Args:
            n_channels: Number of input channels (e.g., EEG electrodes).
            n_times: Number of time samples per trial.
            n_classes: Number of output classification classes.
            **kwargs: Model-specific additional parameters.

        Note:
            Subclasses must call super().__init__() and store the standard parameters
            as instance attributes for use in get_model_summary().
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes

    @classmethod
    def get_config_class(cls):
        """Get the Pydantic configuration class for this model.

        Returns:
            The Pydantic config class for validating model parameters,
            or None if the model doesn't have a specific config class.
        """
        if cls._config_class is not None:
            return cls._config_class
        # Fallback: look up by model type in registry
        from .model_configs import get_model_config_class

        model_type = cls._model_type or cls.__name__
        return get_model_config_class(model_type)

    @classmethod
    def get_default_parameters(cls) -> dict[str, Any]:
        """Get default parameters from the Pydantic config class.

        Returns:
            Dictionary of default parameter values.
        """
        config_class = cls.get_config_class()
        if config_class is not None:
            return config_class().model_dump()
        return {}

    @classmethod
    def get_model_info(cls) -> dict[str, Any]:
        """Return model interface information and capabilities.

        Uses class attributes (_model_type, _input_domain, _modalities, etc.)
        to build the info dict. Subclasses can override class attributes
        or override this method entirely for custom behavior.

        Returns:
            Dictionary containing model_type, input_domain, input_format,
            input_shape, modalities, description, and default_parameters.
        """
        return {
            "model_type": cls._model_type or cls.__name__,
            "input_domain": cls._input_domain,
            "input_format": "3D",
            "input_shape": "(batch, n_channels, n_times)",
            "modalities": cls._modalities,
            "description": cls._description,
            "default_parameters": cls.get_default_parameters(),
        }

    def get_model_summary(self) -> dict[str, Any]:
        """Get detailed runtime summary of the model configuration.

        Uses class attributes and instance state to build summary.
        Subclasses can override for custom behavior.

        Returns:
            Dictionary containing model_type, n_classes, input_shape,
            parameters, total_params, and trainable_params.
        """
        param_counts = self.get_parameter_count()
        return {
            "model_type": self._model_type or self.__class__.__name__,
            "n_classes": self.n_classes,
            "input_shape": (self.n_channels, self.n_times),
            "parameters": getattr(self, "_params", self.get_default_parameters()),
            **param_counts,
        }

    def validate_input_shape(self, x: torch.Tensor) -> None:
        """Validate input tensor shape matches model expectations.

        Args:
            x: Input tensor to validate.

        Raises:
            ValueError: If input shape doesn't match model requirements.

        Note:
            This is a helper method that subclasses can use in their forward()
            method to provide clear error messages for shape mismatches.
        """
        expected_info = self.get_model_info()
        expected_shape = expected_info["input_shape"]

        # Basic validation - subclasses can override for more specific checks
        if len(x.shape) < 3:
            raise ValueError(
                f"Input tensor must have at least 3 dimensions (batch, ...), "
                f"got shape {x.shape}. Expected format: {expected_shape}"
            )

    def get_parameter_count(self) -> dict[str, int]:
        """Get parameter count statistics for the model.

        Returns:
            Dictionary with 'total_params' and 'trainable_params' counts.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {"total_params": total_params, "trainable_params": trainable_params}

    def __repr__(self) -> str:
        """String representation of the model."""
        try:
            summary = self.get_model_summary()
            return (
                f"{summary['model_type']}("
                f"n_channels={self.n_channels}, "
                f"n_times={self.n_times}, "
                f"n_classes={self.n_classes}, "
                f"params={summary['total_params']})"
            )
        except (RuntimeError, ValueError, AttributeError, KeyError):
            return (
                f"{self.__class__.__name__}("
                f"n_channels={self.n_channels}, "
                f"n_times={self.n_times}, "
                f"n_classes={self.n_classes})"
            )
