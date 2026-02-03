"""
Classical ML model wrappers for pipeline components.

These wrappers provide a consistent interface (get_model_info()) for classical ML
algorithms, matching the pattern used by neural network models.

Note: These are sklearn-compatible estimators, not nn.Module subclasses.
"""

from typing import Any

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC


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
        """Return model interface information."""
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


class LDAModel(LinearDiscriminantAnalysis):
    """Linear Discriminant Analysis classifier.

    Wraps sklearn's LinearDiscriminantAnalysis with consistent model info interface.
    """

    _model_type = "LDA"
    _modalities = ["any"]
    _description = "Linear Discriminant Analysis classifier"

    def __init__(self, shrinkage: str = "auto", solver: str = "lsqr", **kwargs):
        super().__init__(shrinkage=shrinkage, solver=solver, **kwargs)

    @classmethod
    def get_model_info(cls) -> dict[str, Any]:
        """Return model interface information."""
        return {
            "model_type": cls._model_type,
            "modalities": cls._modalities,
            "description": cls._description,
            "component_type": "classifier",
            "default_parameters": {
                "shrinkage": "auto",
                "solver": "lsqr",
            },
        }


class SVMModel(SVC):
    """Support Vector Machine classifier.

    Wraps sklearn's SVC with consistent model info interface.
    """

    _model_type = "SVM"
    _modalities = ["any"]
    _description = "Support Vector Machine classifier"

    def __init__(self, kernel: str = "rbf", C: float = 1.0, probability: bool = True, **kwargs):
        super().__init__(kernel=kernel, C=C, probability=probability, **kwargs)

    @classmethod
    def get_model_info(cls) -> dict[str, Any]:
        """Return model interface information."""
        return {
            "model_type": cls._model_type,
            "modalities": cls._modalities,
            "description": cls._description,
            "component_type": "classifier",
            "default_parameters": {
                "kernel": "rbf",
                "C": 1.0,
                "probability": True,
            },
        }
