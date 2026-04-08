"""
Classical ML classifier wrappers.

Sklearn-compatible classifiers with a consistent get_model_info() interface.
Feature extractors (CSP) live in ml/features/.
"""

from typing import Any

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC


class LDAModel(LinearDiscriminantAnalysis):
    """Linear Discriminant Analysis classifier."""

    _model_type = "LDA"
    _modalities = ["any"]
    _description = "Linear Discriminant Analysis classifier"

    def __init__(self, shrinkage: str = "auto", solver: str = "lsqr", **kwargs):
        super().__init__(shrinkage=shrinkage, solver=solver, **kwargs)

    @classmethod
    def get_model_info(cls) -> dict[str, Any]:
        return {
            "model_type": cls._model_type,
            "modalities": cls._modalities,
            "description": cls._description,
            "component_type": "classifier",
            "default_parameters": {"shrinkage": "auto", "solver": "lsqr"},
        }


class SVMModel(SVC):
    """Support Vector Machine classifier."""

    _model_type = "SVM"
    _modalities = ["any"]
    _description = "Support Vector Machine classifier"

    def __init__(self, kernel: str = "rbf", C: float = 1.0, probability: bool = True, **kwargs):
        super().__init__(kernel=kernel, C=C, probability=probability, **kwargs)

    @classmethod
    def get_model_info(cls) -> dict[str, Any]:
        return {
            "model_type": cls._model_type,
            "modalities": cls._modalities,
            "description": cls._description,
            "component_type": "classifier",
            "default_parameters": {"kernel": "rbf", "C": 1.0, "probability": True},
        }
