"""Search space definitions for Optuna hyperparameter search.

Model parameters use clean names (F1, D, etc.) without prefixes.
MODEL_SPACES is generated from Pydantic config classes (single source of truth).
TRAINING contains training hyperparameter search ranges.
"""

import logging
from typing import Any

logger = logging.getLogger("OptunaConfig")

# Default models for search (when none specified)
DEFAULT_MODELS = ["EEGNet", "EEGNetPP", "LinearEEG", "TransformerEEG"]

# Fixed training defaults applied to all generated configs
FIXED_TRAINING_DEFAULTS = {
    "epochs": 150,
    "optimizer_type": "AdamW",
    "lr_scheduler_type": "OneCycleLR",
    "onecycle_pct_start": 0.3,
}


# Search Space Categories
# Organized by function for clear profile composition

# Optimizer - Learning dynamics (most impactful for training)
OPTIMIZER = {
    "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
    "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
    "optimizer_type": {"type": "categorical", "choices": ["Adam", "AdamW"]},
    "lr_scheduler_type": {
        "type": "categorical",
        "choices": ["OneCycleLR", "ReduceLROnPlateau", "CosineAnnealingLR"],
    },
}

# Regularization - Prevent overfitting
REGULARIZATION = {
    "weight_decay": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    "dropout_rate": {"type": "float", "low": 0.1, "high": 0.5},
    "early_stopping_patience": {"type": "int", "low": 5, "high": 15},
}

# Augmentation - Data/target modifications
AUGMENTATION = {
    "loss_type": {"type": "categorical", "choices": ["cross_entropy", "focal"]},
    "use_label_smoothing": {"type": "categorical", "choices": [True, False]},
    "use_mixup": {"type": "categorical", "choices": [True, False]},
}

# Combined training parameters (backwards compatible)
TRAINING = {**OPTIMIZER, **REGULARIZATION, **AUGMENTATION}


def _build_model_spaces() -> dict[str, dict[str, Any]]:
    """Generate MODEL_SPACES from Pydantic config classes.

    Reads 'hpo' metadata from Field definitions in model_configs.py.
    This makes config classes the single source of truth for both
    default values and HPO search ranges.

    Classical models (CSP, LDA, SVM) are skipped as they don't have
    Pydantic config classes - they use sklearn defaults.
    """
    from dendrite.ml.models import MODEL_REGISTRY

    spaces = {}
    for model_name, entry in MODEL_REGISTRY.items():
        config_class = entry["config"]
        if config_class is None:
            continue
        space = {}
        for field_name, field_info in config_class.model_fields.items():
            extra = field_info.json_schema_extra
            if extra and isinstance(extra, dict) and "hpo" in extra:
                space[field_name] = extra["hpo"]
        if space:
            spaces[model_name] = space
    return spaces


# Model Type -> Search Space Mapping (generated from Pydantic configs)
MODEL_SPACES: dict[str, dict[str, Any]] = _build_model_spaces()

# Derived parameters: computed from sampled params
DERIVED_PARAMS: dict[str, dict[str, Any]] = {
    "EEGNet": {"F2": lambda p: p["F1"] * p["D"]},
    "EEGNetPP": {"F2": lambda p: p["F1"] * p["D"]},
    "BDEEGNet": {"F2": lambda p: p["F1"] * p["D"]},
}

# Benchmark search space for MOABB nested CV evaluations
# Focused subset of training params suitable for HPO inside evaluation folds
BENCHMARK_SEARCH_SPACE = {
    "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
    "dropout_rate": {"type": "float", "low": 0.1, "high": 0.7},
    "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-2, "log": True},
}


# Profile metadata for UI display
PROFILES = {
    "quick": {
        "name": "Quick",
        "trials": 10,
        "space": OPTIMIZER,
        "description": "Learning rate, optimizer, scheduler",
        "use_case": "Fast baseline, small datasets",
    },
    "balanced": {
        "name": "Balanced",
        "trials": 30,
        "space": {**OPTIMIZER, **REGULARIZATION},
        "description": "Optimizer + regularization params",
        "use_case": "Good default for most cases",
    },
    "full": {
        "name": "Full",
        "trials": 50,
        "space": TRAINING,
        "description": "All training hyperparameters",
        "use_case": "Maximum optimization",
    },
}


def get_profile(name: str) -> dict:
    """Get profile by name (case-insensitive)."""
    return PROFILES.get(name.lower(), PROFILES["balanced"])


def get_search_space_description(space: dict) -> list[str]:
    """Generate human-readable descriptions of search space parameters.

    Returns list of strings like:
        - Learning rate: 1e-4 to 1e-2 (log)
        - Batch size: 16, 32, 64
    """
    descriptions = []

    # Human-readable names for parameters
    param_names = {
        "learning_rate": "Learning rate",
        "weight_decay": "Weight decay",
        "batch_size": "Batch size",
        "dropout_rate": "Dropout",
        "early_stopping_patience": "Early stopping patience",
        "lr_scheduler_type": "LR scheduler",
        "optimizer_type": "Optimizer",
        "loss_type": "Loss function",
        "use_label_smoothing": "Label smoothing",
        "use_mixup": "Mixup augmentation",
    }

    for param, config in space.items():
        name = param_names.get(param, param.replace("_", " ").title())
        param_type = config.get("type")

        if param_type == "float":
            low, high = config["low"], config["high"]
            log_str = " (log)" if config.get("log") else ""
            descriptions.append(f"{name}: {low:.0e} to {high:.0e}{log_str}")
        elif param_type == "int":
            low, high = config["low"], config["high"]
            descriptions.append(f"{name}: {low} to {high}")
        elif param_type == "categorical":
            choices = config["choices"]
            if all(isinstance(c, bool) for c in choices):
                descriptions.append(f"{name}: on/off")
            elif len(choices) <= 4:
                choices_str = ", ".join(str(c) for c in choices)
                descriptions.append(f"{name}: {choices_str}")
            else:
                descriptions.append(f"{name}: {len(choices)} options")

    return descriptions
