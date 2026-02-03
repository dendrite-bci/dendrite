"""
Braindecode model adapters for Dendrite system.

This module provides thin adapters to make braindecode's battle-tested model
implementations compatible with the ModelBase interface.

Available models:
- BDEEGNet: EEGNetv4 - Compact CNN for EEG (replaces native EEGNet)
- BDEEGConformer: EEGConformer - CNN-Transformer hybrid (replaces native Conformer)
- BDShallowNet: ShallowFBCSPNet - Fast FBCSP-inspired baseline
- BDDeep4Net: Deep4Net - Deep ConvNet baseline
- BDATCNet: ATCNet - SOTA on MI benchmarks
- BDTCN: TCN - Temporal Convolutional Network
- BDEEGInception: EEGInceptionMI - Inception-based architecture
"""

from typing import Any

from braindecode.models import (
    TCN,
    ATCNet,
    Deep4Net,
    EEGInceptionMI,
    ShallowFBCSPNet,
)
from braindecode.models import (
    EEGConformer as BDEEGConformerBase,
)
from braindecode.models import (
    EEGNet as BDEEGNetBase,
)

from .base import ModelBase
from .model_configs import (
    BDATCNetConfig,
    BDDeep4NetConfig,
    BDEEGConformerConfig,
    BDEEGInceptionConfig,
    BDEEGNetConfig,
    BDShallowNetConfig,
    BDTCNConfig,
)

# Map of braindecode model names to classes
_BD_MODELS = {
    "EEGNet": BDEEGNetBase,
    "EEGConformer": BDEEGConformerBase,
    "ShallowFBCSPNet": ShallowFBCSPNet,
    "Deep4Net": Deep4Net,
    "ATCNet": ATCNet,
    "TCN": TCN,
    "EEGInceptionMI": EEGInceptionMI,
}


class BraindecodeAdapter(ModelBase):
    """Base adapter to make braindecode models compatible with ModelBase.

    Wraps braindecode model implementations to provide a consistent interface
    for the Dendrite system. Each subclass wraps a specific braindecode model.

    Subclasses must set these class attributes:
        _braindecode_class_name: Name of the braindecode model class to wrap.
        _model_type: Identifier for this adapter (used in registry).
        _description: Human-readable description.
        _config_class: Pydantic config class for default parameters.
    """

    # Subclasses must override these
    _braindecode_class_name: str = None
    _model_type: str = None
    _description: str = None
    _config_class = None  # Pydantic config class (single source of truth)

    def __init__(self, n_channels: int, n_times: int, n_classes: int, **kwargs):
        """Initialize braindecode model with Dendrite interface parameters.

        Any additional keyword arguments are forwarded to the underlying
        braindecode model constructor after merging with default parameters
        from the config class.

        Args:
            n_channels: Number of input channels.
            n_times: Number of time samples per trial.
            n_classes: Number of output classes.
            **kwargs: Model-specific parameters forwarded to braindecode.
                See the specific adapter class (BDEEGNet, BDATCNet, etc.)
                for available parameters.

        Example:
            >>> from dendrite.ml.models import BDEEGNet
            >>> model = BDEEGNet(
            ...     n_channels=32,
            ...     n_times=250,
            ...     n_classes=4,
            ...     F1=8, D=2  # Forwarded to braindecode's EEGNet
            ... )
        """
        super().__init__(n_channels, n_times, n_classes)

        model_class = _BD_MODELS[self._braindecode_class_name]

        # Get defaults from config class, merge with provided kwargs
        defaults = self.get_default_parameters()
        params = {**defaults, **kwargs}
        self._params = params

        # Create the braindecode model
        # Note: braindecode uses n_chans, n_times, n_outputs naming
        self.model = model_class(n_chans=n_channels, n_times=n_times, n_outputs=n_classes, **params)

    def forward(self, x):
        """Forward pass through the braindecode model."""
        return self.model(x)

    @classmethod
    def get_model_info(cls) -> dict[str, Any]:
        """Return model interface information."""
        return {
            "model_type": cls._model_type,
            "input_domain": "time-series",
            "input_format": "3D",
            "input_shape": "(batch, n_channels, n_times)",
            "modalities": ["eeg"],
            "description": cls._description,
            "default_parameters": cls.get_default_parameters(),
        }

    def get_model_summary(self) -> dict[str, Any]:
        """Get runtime summary of the model."""
        param_counts = self.get_parameter_count()
        return {
            "model_type": self._model_type,
            "n_classes": self.n_classes,
            "input_shape": (self.n_channels, self.n_times),
            "parameters": self._params,
            **param_counts,
        }


# Format: (braindecode_class_name, description, config_class)
# Config classes are the single source of truth for default parameters

_ADAPTER_SPECS = {
    "BDEEGNet": ("EEGNet", "Compact CNN for EEG, temporal and spatial filtering", BDEEGNetConfig),
    "BDEEGConformer": (
        "EEGConformer",
        "CNN-Transformer hybrid combining local and global features",
        BDEEGConformerConfig,
    ),
    "BDShallowNet": (
        "ShallowFBCSPNet",
        "Shallow CNN inspired by FBCSP, fast baseline for motor imagery",
        BDShallowNetConfig,
    ),
    "BDDeep4Net": ("Deep4Net", "Deep ConvNet with 4 convolutional blocks", BDDeep4NetConfig),
    "BDATCNet": (
        "ATCNet",
        "Attention-based TCN, state-of-the-art on motor imagery benchmarks",
        BDATCNetConfig,
    ),
    "BDEEGInception": (
        "EEGInceptionMI",
        "Inception-inspired architecture for motor imagery",
        BDEEGInceptionConfig,
    ),
}


def _make_adapter(name: str, bd_class: str, desc: str, config_class):
    """Factory to create adapter classes from specifications."""
    return type(
        name,
        (BraindecodeAdapter,),
        {
            "_braindecode_class_name": bd_class,
            "_model_type": name,
            "_description": desc,
            "_config_class": config_class,
            "__doc__": f"{desc}\n\nWraps braindecode's {bd_class} implementation.",
        },
    )


# Generate standard adapter classes
BDEEGNet = _make_adapter("BDEEGNet", *_ADAPTER_SPECS["BDEEGNet"])
BDEEGConformer = _make_adapter("BDEEGConformer", *_ADAPTER_SPECS["BDEEGConformer"])
BDShallowNet = _make_adapter("BDShallowNet", *_ADAPTER_SPECS["BDShallowNet"])
BDDeep4Net = _make_adapter("BDDeep4Net", *_ADAPTER_SPECS["BDDeep4Net"])
BDATCNet = _make_adapter("BDATCNet", *_ADAPTER_SPECS["BDATCNet"])
BDEEGInception = _make_adapter("BDEEGInception", *_ADAPTER_SPECS["BDEEGInception"])


class BDTCN(BraindecodeAdapter):
    """Temporal Convolutional Network with dilated convolutions.

    Wraps braindecode's TCN implementation.

    Note: Braindecode's TCN outputs per-timestep predictions. This adapter
    adds global average pooling to return a single classification per sample.
    """

    _braindecode_class_name = "TCN"
    _model_type = "BDTCN"
    _description = "Temporal Convolutional Network with dilated convolutions"
    _config_class = BDTCNConfig

    def __init__(self, n_channels: int, n_times: int, n_classes: int, **kwargs):
        """Initialize TCN adapter.

        Note: The underlying TCN model does not use n_times as it handles
        variable-length sequences. The n_times parameter is stored for
        interface compatibility only.

        Args:
            n_channels: Number of input channels.
            n_times: Number of time samples (stored but not used by TCN).
            n_classes: Number of output classes.
            **kwargs: TCN-specific parameters forwarded to braindecode.
        """
        ModelBase.__init__(self, n_channels, n_times, n_classes)

        defaults = self.get_default_parameters()
        params = {**defaults, **kwargs}
        self._params = params

        # TCN doesn't accept n_times
        self.model = _BD_MODELS["TCN"](n_chans=n_channels, n_outputs=n_classes, **params)

    def forward(self, x):
        """Forward pass with global average pooling over time."""
        out = self.model(x)  # (batch, n_classes, time)
        return out.mean(dim=-1)  # (batch, n_classes)


# Export all adapter classes
__all__ = [
    "BraindecodeAdapter",
    "BDEEGNet",
    "BDEEGConformer",
    "BDShallowNet",
    "BDDeep4Net",
    "BDATCNet",
    "BDTCN",
    "BDEEGInception",
]
