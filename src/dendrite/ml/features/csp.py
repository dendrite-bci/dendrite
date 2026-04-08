"""
CSP (Common Spatial Patterns) feature extractor.

Sklearn-compatible transformer that extracts spatial filters from EEG data.
Used as a pipeline step before classifiers (LDA, SVM) in composite decoders.
"""

from typing import Any

from mne.decoding import CSP
from pydantic import BaseModel, ConfigDict, Field


class CSPConfig(BaseModel):
    """CSP feature extractor configuration with HPO search metadata."""

    model_config = ConfigDict(extra="allow")

    n_components: int = Field(
        default=8, ge=2, le=32,
        description="Number of spatial filters",
        json_schema_extra={"hpo": {"type": "int", "low": 4, "high": 16, "step": 2}},
    )
    reg: str = Field(
        default="ledoit_wolf",
        description="Covariance regularization method",
        json_schema_extra={"hpo": {"type": "categorical", "choices": ["ledoit_wolf", "empirical"]}},
    )


class CSPModel(CSP):
    """Common Spatial Patterns feature extractor.

    Wraps mne.decoding.CSP with consistent model info interface.
    Used as a feature extraction step before classifiers like LDA or SVM.
    """

    _model_type = "CSP"
    _modalities = ["eeg"]
    _description = "Common Spatial Patterns feature extractor"

    def __init__(self, n_components: int = 8, reg: str = "ledoit_wolf", **kwargs):
        super().__init__(n_components=n_components, reg=reg, **kwargs)

    @classmethod
    def get_model_info(cls) -> dict[str, Any]:
        return {
            "model_type": cls._model_type,
            "modalities": cls._modalities,
            "description": cls._description,
            "component_type": "feature_extractor",
            "default_parameters": {
                "n_components": 8,
                "reg": "ledoit_wolf",
            },
        }
