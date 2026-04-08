"""Decoder Registry — maps decoder types to pipeline builders."""

from sklearn.pipeline import Pipeline

from dendrite.ml.models.registry import MODEL_REGISTRY


def _create_component(name: str, config):
    """Create a pipeline component by name."""
    from dendrite.ml.decoders.neural_classifier import NeuralNetClassifier
    from dendrite.processing.preprocessing.scalers import ChannelScaler

    if name == "scaler":
        return ChannelScaler()
    elif name == "classifier":
        return NeuralNetClassifier(config)
    elif name == "csp":
        from dendrite.ml.features.csp import CSPConfig, CSPModel
        return _build_classical_component(CSPModel, CSPConfig, config)
    elif name in ("lda", "svm"):
        entry = MODEL_REGISTRY[name.upper()]
        return _build_classical_component(entry["class"], entry.get("config"), config)
    elif name == "covariances":
        from pyriemann.estimation import Covariances
        return Covariances(estimator="lwf")
    elif name == "fgmdm":
        from pyriemann.classification import FgMDM
        return FgMDM(metric="riemann")
    else:
        raise ValueError(f"Unknown component: {name}")


def _build_classical_component(model_class, config_class, config):
    """Build a classical component with defaults merged and params filtered."""
    defaults = model_class.get_model_info().get("default_parameters", {})
    valid_keys = set(config_class.model_fields) if config_class else set(defaults)
    overrides = {k: v for k, v in (config.model_params or {}).items() if k in valid_keys}
    return model_class(**{**defaults, **overrides})


def _build_pipeline(config, default_steps: list[str]) -> Pipeline:
    """Build sklearn Pipeline from config or default steps."""
    steps = config.pipeline_steps if config.pipeline_steps else default_steps
    return Pipeline([(name, _create_component(name, config)) for name in steps])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DECODER_REGISTRY = {}

# Neural decoders (from MODEL_REGISTRY, skip classical classifiers)
for model_name, model_entry in MODEL_REGISTRY.items():
    if model_entry.get("classical"):
        continue
    model_class = model_entry["class"]
    info = model_class.get_model_info() if hasattr(model_class, "get_model_info") else {}
    DECODER_REGISTRY[model_name] = {
        "pipeline_builder": lambda cfg, s=["scaler", "classifier"]: _build_pipeline(cfg, s),
        "model_class": model_class,
        "description": info.get("description", model_name),
        "modalities": info.get("modalities", ["any"]),
        "default_steps": ["scaler", "classifier"],
        "step_types": {"scaler": "preprocessing", "classifier": "classifier"},
    }

# Classical decoders
for name, steps, desc in [
    ("CSP+LDA", ["csp", "lda"], "CSP + LDA (Motor Imagery baseline)"),
    ("CSP+SVM", ["csp", "svm"], "CSP + SVM (Nonlinear classification)"),
    ("FgMDM", ["covariances", "fgmdm"], "Riemannian FgMDM (geometry-aware)"),
]:
    DECODER_REGISTRY[name] = {
        "pipeline_builder": lambda cfg, s=steps: _build_pipeline(cfg, s),
        "description": desc,
        "modalities": ["eeg"],
        "default_steps": steps,
        "step_types": {s: ("features" if i == 0 else "classifier") for i, s in enumerate(steps)},
    }


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def get_available_decoders() -> list[str]:
    return list(DECODER_REGISTRY.keys())


def get_decoder_entry(decoder_type: str) -> dict | None:
    return DECODER_REGISTRY.get(decoder_type)


def get_decoder_capabilities(decoder_name: str) -> list[str]:
    entry = DECODER_REGISTRY.get(decoder_name)
    return entry.get("modalities", ["any"]) if entry else ["any"]


def check_decoder_compatibility(
    decoder_name: str, selected_modalities: list[str],
) -> tuple[bool, list[str]]:
    capabilities = set(m.lower() for m in get_decoder_capabilities(decoder_name))
    if "any" in capabilities:
        return True, []
    unsupported = [m for m in selected_modalities if m.lower() not in capabilities]
    return len(unsupported) == 0, unsupported
