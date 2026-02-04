"""
Optuna-based hyperparameter search utilities.

Provides two levels of abstraction:
1. suggest_params: Low-level, samples from any flat search space
2. suggest_training_config: High-level, returns structured config with model_params

Search space definitions are in search_space.py.
"""

import logging
from typing import Any

import optuna

from dendrite.ml.models import get_available_models

from .config import create_sampler
from .search_space import (
    DEFAULT_MODELS,
    DERIVED_PARAMS,
    FIXED_TRAINING_DEFAULTS,
    MODEL_SPACES,
    TRAINING,
)

logger = logging.getLogger("OptunaSearch")


def _suggest_single(trial: optuna.Trial, param: str, spec: dict[str, Any]) -> Any:
    """Suggest a single parameter value from its spec.

    Args:
        trial: Optuna trial object
        param: Parameter name
        spec: Dict with 'type' and range/choices:
            - {'type': 'float', 'low': ..., 'high': ..., 'log': bool}
            - {'type': 'int', 'low': ..., 'high': ..., 'step': int}
            - {'type': 'categorical', 'choices': [...]}

    Returns:
        Suggested value, or None if spec format is invalid.
    """
    if not isinstance(spec, dict):
        return None
    param_type = spec.get("type", "categorical")

    if param_type == "float":
        return trial.suggest_float(param, spec["low"], spec["high"], log=spec.get("log", False))
    elif param_type == "int":
        return trial.suggest_int(param, spec["low"], spec["high"], step=spec.get("step", 1))
    elif param_type == "categorical":
        return trial.suggest_categorical(param, spec["choices"])

    return None


def suggest_params(
    trial: optuna.Trial,
    search_space: dict[str, Any],
    skip_params: list[str] | None = None,
) -> dict[str, Any]:
    """Suggest hyperparameters from a flat search space definition.

    Low-level function that samples from any search space dict.
    Handles conditional params automatically via two-pass approach.

    Args:
        trial: Optuna trial object
        search_space: Search space definition (param_name -> spec)
        skip_params: Parameter names to skip

    Returns:
        Flat dict of suggested parameter values
    """
    skip_params = skip_params or []
    config = {}

    # First pass: non-conditional params
    for param, spec in search_space.items():
        if param in skip_params:
            continue
        if isinstance(spec, dict) and "conditional" in spec:
            continue
        value = _suggest_single(trial, param, spec)
        if value is not None:
            config[param] = value

    # Second pass: conditional params (based on first pass values)
    for param, spec in search_space.items():
        if param in skip_params:
            continue
        if not (isinstance(spec, dict) and "conditional" in spec):
            continue
        cond_param, cond_value = spec["conditional"]
        if config.get(cond_param) == cond_value:
            value = _suggest_single(trial, param, spec)
            if value is not None:
                config[param] = value

    return config


def suggest_training_config(
    trial: optuna.Trial,
    model_types: list[str],
) -> dict[str, Any]:
    """Suggest complete training config with model-specific params.

    High-level function that returns a cleanly nested config by category:
    1. Samples training params from TRAINING space
    2. Samples model_type and model-specific params
    3. Computes derived params (e.g., F2 = F1 * D)

    Args:
        trial: Optuna trial object
        model_types: List of model types to search over

    Returns:
        Nested config dict:
        {
            'model': {'type': str, 'params': dict},
            'training': {...training params...},
        }
    """
    # 1. Sample training params
    training = suggest_params(trial, TRAINING)

    # 2. Sample model_type
    model_type = trial.suggest_categorical("model_type", model_types)

    # 3. Sample model-specific params
    model_space = MODEL_SPACES.get(model_type, {})
    if not model_space:
        logger.warning(f"No HPO search space for {model_type} - using defaults only")
    model_params = suggest_params(trial, model_space)

    # 4. Compute derived params
    derived = DERIVED_PARAMS.get(model_type, {})
    for param, compute_fn in derived.items():
        try:
            model_params[param] = compute_fn(model_params)
        except (KeyError, TypeError) as e:
            logger.debug(f"Skipping derived param '{param}' for {model_type}: {e}")

    # 5. Add dropout_rate to model_params if present
    if "dropout_rate" in training:
        model_params["dropout_rate"] = training["dropout_rate"]

    return {
        "model": {"type": model_type, "params": model_params},
        "training": training,
    }


def _flatten_config(nested: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested config to TrainerConfig-compatible format.

    Converts categorized nested config to flat dict expected by TrainerConfig.from_dict().

    Args:
        nested: Nested config from suggest_training_config with 'model' and 'training' keys

    Returns:
        Flat dict with 'model_type', 'model_params', and all training params
    """
    flat = {}

    # Extract model info
    model = nested.get("model", {})
    flat["model_type"] = model.get("type")
    if model.get("params"):
        flat["model_params"] = model["params"]

    # Flatten training params
    flat.update(nested.get("training", {}))

    return flat


def _format_config_for_training(nested: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested config and add fixed defaults for TrainerConfig.

    Takes nested config from suggest_training_config, flattens it,
    and merges with fixed defaults (epochs, optimizer, etc.).

    Args:
        nested: Nested config from suggest_training_config

    Returns:
        Flat config ready for TrainerConfig.from_dict()
    """
    flat = _flatten_config(nested)
    return {
        **FIXED_TRAINING_DEFAULTS,
        **flat,
    }


def get_search_models(model_types: list[str] | None = None) -> list[str]:
    """Get validated list of models for search."""
    available = set(get_available_models())

    if model_types is None:
        return [m for m in DEFAULT_MODELS if m in available]

    valid = [m for m in model_types if m in available]
    invalid = set(model_types) - set(valid)
    if invalid:
        logger.warning(
            f"Models not in registry, skipping: {sorted(invalid)}. Available: {sorted(available)}"
        )

    return valid or [m for m in DEFAULT_MODELS if m in available]


# Search space from search_space.py (training params, no model arch params)
DEFAULT_SEARCH_SPACE = TRAINING


def create_optuna_search_configs(
    n_trials: int = 10,
    model_types: list[str] | None = None,
    seed: int | None = None,
    sampler_type: str = "tpe",
) -> list[dict[str, Any]]:
    """Generate formatted search configurations using Optuna sampler.

    High-level convenience function for DecoderPool/synchronous_mode.
    Generates n_trials configs with model_params structure ready for decoders.

    Args:
        n_trials: Number of configurations to generate
        model_types: Models to search over (uses config defaults if None)
        seed: Random seed for reproducibility
        sampler_type: Optuna sampler type ('tpe', 'random')

    Returns:
        List of config dicts with model_type, training params, and model_params
    """
    # Validate model types against registry
    model_types = get_search_models(model_types)

    # Create study with sampler
    sampler = create_sampler(sampler_type, seed)
    study = optuna.create_study(sampler=sampler, direction="maximize")

    # Generate configs
    configs = []
    for i in range(n_trials):
        trial = study.ask()
        try:
            raw_config = suggest_training_config(trial, model_types)
            formatted = _format_config_for_training(raw_config)
            configs.append(formatted)
            study.tell(trial, 0.5)  # Dummy value for sampling
        except Exception as e:
            logger.error(f"Failed to generate config for trial {i}: {e}")

    logger.debug(f"Generated {len(configs)} search configurations using {sampler_type} sampler")
    return configs
