"""Unified Decoder Registry.

Single source of truth for all decoder types (neural and classical).
All decoders use pipeline_builder for consistent sklearn Pipeline architecture.

Architecture:
    Models are defined in `dendrite.ml.models.MODEL_REGISTRY`. This module
    builds the `DECODER_REGISTRY` by wrapping each model in a pipeline with
    optional preprocessing steps (scaler, CSP, etc.).

    DECODER_REGISTRY entries contain:
    - 'pipeline_builder': Callable that creates sklearn Pipeline from config
    - 'model_class': The neural network class (for neural decoders only)
    - 'description': Human-readable description
    - 'modalities': Supported input modalities (['eeg'], ['any'], etc.)
    - 'default_steps': Default pipeline step names

Extending with Custom Decoders:
    1. Add your model to MODEL_REGISTRY (see dendrite.ml.models.registry)
    2. The decoder will be automatically registered with neural pipeline
    3. For classical pipelines, add entry directly to DECODER_REGISTRY

Factory Functions:
    - get_available_decoders(): List all registered decoder names
    - get_decoder_entry(): Get full registry entry for a decoder
    - get_decoder_capabilities(): Get supported modalities
    - check_decoder_compatibility(): Validate modality compatibility

Example::

    from dendrite.ml.decoders.registry import get_available_decoders
    from dendrite.ml.decoders.registry import get_decoder_entry
    decoders = get_available_decoders()
    entry = get_decoder_entry('EEGNet')
    pipeline = entry['pipeline_builder'](config)
"""

from sklearn.pipeline import Pipeline

from dendrite.ml.models.registry import MODEL_REGISTRY


def _create_component(name: str, config):
    """Create a pipeline component by name.

    Uses MODEL_REGISTRY for CSP/LDA/SVM lookups, ensuring models are
    the single source of truth for all algorithms.
    """
    from dendrite.ml.decoders.neural_classifier import NeuralNetClassifier
    from dendrite.processing.preprocessing.scalers import ChannelScaler

    if name == "scaler":
        return ChannelScaler()
    elif name == "classifier":
        return NeuralNetClassifier(config)
    elif name == "csp":
        csp_class = MODEL_REGISTRY["CSP"]["class"]
        n_components = config.model_params.get("n_components", 8) if config.model_params else 8
        return csp_class(n_components=n_components)
    elif name == "lda":
        lda_class = MODEL_REGISTRY["LDA"]["class"]
        return lda_class()
    elif name == "svm":
        svm_class = MODEL_REGISTRY["SVM"]["class"]
        return svm_class()
    else:
        raise ValueError(f"Unknown component: {name}")


def build_pipeline(config, steps: list) -> Pipeline:
    """Build sklearn Pipeline from step list."""
    return Pipeline([(name, _create_component(name, config)) for name in steps])


def _build_neural_pipeline(config) -> Pipeline:
    """Build neural decoder pipeline from config."""
    if config.pipeline_steps:
        return build_pipeline(config, config.pipeline_steps)
    # Legacy fallback
    steps = ["scaler", "classifier"] if config.use_scaler else ["classifier"]
    return build_pipeline(config, steps)


def _build_classical_pipeline(default_steps: list):
    """Return a builder for classical decoders with given default steps."""

    def builder(config) -> Pipeline:
        steps = config.pipeline_steps if config.pipeline_steps else default_steps
        return build_pipeline(config, steps)

    return builder


DECODER_REGISTRY = {}

# Neural Decoders (from MODEL_REGISTRY)
for model_name, model_entry in MODEL_REGISTRY.items():
    model_class = model_entry["class"]
    description = model_name
    modalities = ["eeg"]
    if hasattr(model_class, "get_model_info"):
        info = model_class.get_model_info()
        description = info.get("description", model_name)
        modalities = info.get("modalities", ["eeg"])

    DECODER_REGISTRY[model_name] = {
        "pipeline_builder": _build_neural_pipeline,
        "model_class": model_class,
        "description": description,
        "modalities": modalities,
        "default_steps": ["scaler", "classifier"],
    }

# Classical Decoders
DECODER_REGISTRY["CSP+LDA"] = {
    "pipeline_builder": _build_classical_pipeline(["csp", "lda"]),
    "description": "CSP + LDA (Motor Imagery baseline)",
    "modalities": ["eeg"],
    "default_steps": ["csp", "lda"],
}

DECODER_REGISTRY["CSP+SVM"] = {
    "pipeline_builder": _build_classical_pipeline(["csp", "svm"]),
    "description": "CSP + SVM (Nonlinear classification)",
    "modalities": ["eeg"],
    "default_steps": ["csp", "svm"],
}


def get_available_decoders() -> list[str]:
    """Get list of all available decoder types.

    Returns:
        List of decoder type names including neural networks (EEGNet, etc.)
        and classical ML pipelines (CSP+LDA, CSP+SVM).
    """
    return list(DECODER_REGISTRY.keys())


def get_decoder_entry(decoder_type: str) -> dict | None:
    """Get registry entry for a decoder type.

    Args:
        decoder_type: Name of the decoder type to look up.

    Returns:
        Registry entry dict with keys: pipeline_builder, description,
        modalities, default_steps. Returns None if not found.
    """
    return DECODER_REGISTRY.get(decoder_type)


def get_decoder_capabilities(decoder_name: str) -> list[str]:
    """Get supported modalities for a decoder type.

    Args:
        decoder_name: Decoder type name (e.g., 'EEGNet', 'CSP+LDA').

    Returns:
        List of supported modality names. Returns ['eeg'] for most decoders,
        ['any'] for modality-agnostic decoders like LinearEEG.
    """
    entry = DECODER_REGISTRY.get(decoder_name)
    if entry:
        return entry.get("modalities", ["eeg"])
    return ["eeg"]


def check_decoder_compatibility(
    decoder_name: str, selected_modalities: list[str]
) -> tuple[bool, list[str]]:
    """Check if selected modalities are compatible with a decoder.

    Args:
        decoder_name: Decoder type name (e.g., 'EEGNet', 'CSP+LDA').
        selected_modalities: List of modalities to check (e.g., ['eeg'], ['eeg', 'emg']).

    Returns:
        Tuple of (is_compatible, unsupported_modalities). First element is True
        if all modalities are supported; second element lists any unsupported modalities.
    """
    capabilities = get_decoder_capabilities(decoder_name)
    if "any" in [m.lower() for m in capabilities]:
        return True, []
    supported_set = set(m.lower() for m in capabilities)
    selected_set = set(m.lower() for m in selected_modalities)
    unsupported = list(selected_set - supported_set)
    return len(unsupported) == 0, unsupported
