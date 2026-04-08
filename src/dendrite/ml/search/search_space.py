"""Search space definitions for Optuna hyperparameter search.

The search space for a decoder is derived from its pipeline steps and config:
- Architecture params: from model config classes (EEGNetConfig, CSPConfig, etc.)
- Training params: from DecoderConfig fields with HPO metadata
- All HPO ranges defined via json_schema_extra={"hpo": ...} on Pydantic Fields

Entry point: build_decoder_search_space(decoder_type, categories)
"""

from typing import Any

# Derived params: computed from suggested values, not directly searched
DERIVED_PARAMS: dict[str, dict[str, Any]] = {
    "EEGNet": {"F2": lambda p: p["F1"] * p["D"]},
    "EEGNetPP": {"F2": lambda p: p["F1"] * p["D"]},
    "BDEEGNet": {"F2": lambda p: p["F1"] * p["D"]},
}


def _extract_hpo_space(config_class) -> dict[str, Any]:
    """Extract HPO search ranges from a Pydantic config class."""
    space = {}
    for field_name, field_info in config_class.model_fields.items():
        extra = field_info.json_schema_extra
        if extra and isinstance(extra, dict) and "hpo" in extra:
            space[field_name] = extra["hpo"]
    return space


def _get_training_space() -> dict[str, Any]:
    """Get searchable training params from DecoderConfig HPO metadata."""
    from dendrite.ml.decoders.decoder_schemas import DecoderConfig
    return _extract_hpo_space(DecoderConfig)


def _get_step_space(step: str, decoder_type: str) -> dict[str, Any]:
    """Get searchable params for a single pipeline step."""
    if step == "csp":
        from dendrite.ml.features.csp import CSPConfig
        return _extract_hpo_space(CSPConfig)
    elif step == "classifier":
        from dendrite.ml.models import MODEL_REGISTRY
        config_class = MODEL_REGISTRY.get(decoder_type, {}).get("config")
        return _extract_hpo_space(config_class) if config_class else {}
    elif step in ("lda", "svm"):
        from dendrite.ml.models import MODEL_REGISTRY
        config_class = MODEL_REGISTRY.get(step.upper(), {}).get("config")
        return _extract_hpo_space(config_class) if config_class else {}
    return {}


def get_decoder_categories(decoder_type: str) -> dict[str, dict[str, Any]]:
    """Return searchable categories for a decoder, based on its pipeline."""
    from dendrite.ml.decoders.registry import DECODER_REGISTRY

    entry = DECODER_REGISTRY.get(decoder_type)
    if not entry:
        return {}

    is_neural = "model_class" in entry
    categories: dict[str, dict[str, Any]] = {}

    arch_params: dict[str, Any] = {}
    for step in entry.get("default_steps", []):
        arch_params.update(_get_step_space(step, decoder_type))
    if arch_params:
        categories["architecture"] = {
            "label": "Architecture",
            "params": list(arch_params.keys()),
        }

    if is_neural:
        training_params = _get_training_space()
        if training_params:
            categories["training"] = {
                "label": "Training",
                "params": list(training_params.keys()),
            }

    return categories


def build_decoder_search_space(
    decoder_type: str,
    categories: list[str] | None = None,
) -> dict[str, Any]:
    """Build search space from a decoder's pipeline steps.

    Args:
        decoder_type: Decoder name (e.g., "EEGNet", "CSP+LDA").
        categories: Filter to these categories. None = all available.
    """
    from dendrite.ml.decoders.registry import DECODER_REGISTRY

    entry = DECODER_REGISTRY.get(decoder_type)
    if not entry:
        return {}

    is_neural = "model_class" in entry
    available = get_decoder_categories(decoder_type)
    cats = categories or list(available.keys())

    space: dict[str, Any] = {}
    if "architecture" in cats:
        for step in entry.get("default_steps", []):
            space.update(_get_step_space(step, decoder_type))
    if is_neural and "training" in cats:
        space.update(_get_training_space())

    return space
