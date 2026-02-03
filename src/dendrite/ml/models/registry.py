"""
Model Registry - Single source of truth for all model classes and configs.

MODEL_REGISTRY maps model names to their class and Pydantic config.
Includes both neural networks and classical ML models.
"""

from .braindecode_adapter import (
    BDTCN,
    BDATCNet,
    BDDeep4Net,
    BDEEGConformer,
    BDEEGInception,
    BDEEGNet,
    BDShallowNet,
)
from .classical import CSPModel, LDAModel, SVMModel
from .eegnet import EEGNet
from .eegnet_plus import EEGNetPP
from .linear import LinearEEG
from .model_configs import (
    BDATCNetConfig,
    BDDeep4NetConfig,
    BDEEGConformerConfig,
    BDEEGInceptionConfig,
    BDEEGNetConfig,
    BDShallowNetConfig,
    BDTCNConfig,
    EEGNetConfig,
    EEGNetPPConfig,
    LinearEEGConfig,
    TransformerEEGConfig,
)
from .transformer import TransformerEEG

MODEL_REGISTRY: dict[str, dict] = {
    # Braindecode models (primary)
    "BDEEGNet": {"class": BDEEGNet, "config": BDEEGNetConfig},
    "BDEEGConformer": {"class": BDEEGConformer, "config": BDEEGConformerConfig},
    "BDShallowNet": {"class": BDShallowNet, "config": BDShallowNetConfig},
    "BDDeep4Net": {"class": BDDeep4Net, "config": BDDeep4NetConfig},
    "BDATCNet": {"class": BDATCNet, "config": BDATCNetConfig},
    "BDTCN": {"class": BDTCN, "config": BDTCNConfig},
    "BDEEGInception": {"class": BDEEGInception, "config": BDEEGInceptionConfig},
    # Native models
    "EEGNet": {"class": EEGNet, "config": EEGNetConfig},
    "EEGNetPP": {"class": EEGNetPP, "config": EEGNetPPConfig},
    "LinearEEG": {"class": LinearEEG, "config": LinearEEGConfig},
    "TransformerEEG": {"class": TransformerEEG, "config": TransformerEEGConfig},
    # Classical ML models (no config class - use sklearn defaults or model_params)
    "CSP": {"class": CSPModel, "config": None},
    "LDA": {"class": LDAModel, "config": None},
    "SVM": {"class": SVMModel, "config": None},
}
