---
sidebar_label: registry
title: dendrite.ml.decoders.registry
---

Unified Decoder Registry.

Single source of truth for all decoder types (neural and classical).
All decoders use pipeline_builder for consistent sklearn Pipeline architecture.

Architecture:
Models are defined in `dendrite.ml.models.MODEL_REGISTRY`. This module
builds the `DECODER_REGISTRY` by wrapping each model in a pipeline with
optional preprocessing steps (scaler, CSP, etc.).

DECODER_REGISTRY entries contain:
- 'pipeline_builder': Callable that creates sklearn Pipeline from `config`
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

Example:

```python
from dendrite.ml.decoders.registry import get_available_decoders
from dendrite.ml.decoders.registry import get_decoder_entry
decoders = get_available_decoders()
entry = get_decoder_entry('EEGNet')
pipeline = entry['pipeline_builder'](config)
```

#### build\_pipeline

```python
def build_pipeline(config, steps: list) -> Pipeline
```

Build sklearn Pipeline from step list.

#### get\_available\_decoders

```python
def get_available_decoders() -> list[str]
```

Get list of all available decoder types.

**Returns**:

  List of decoder type names including neural networks (EEGNet, etc.)
  and classical ML pipelines (CSP+LDA, CSP+SVM).

#### get\_decoder\_entry

```python
def get_decoder_entry(decoder_type: str) -> dict | None
```

Get registry entry for a decoder type.

**Arguments**:

- `decoder_type` - Name of the decoder type to look up.
  

**Returns**:

  Registry entry dict with keys: pipeline_builder, description,
  modalities, default_steps. Returns None if not found.

#### get\_decoder\_capabilities

```python
def get_decoder_capabilities(decoder_name: str) -> list[str]
```

Get supported modalities for a decoder type.

**Arguments**:

- `decoder_name` - Decoder type name (e.g., 'EEGNet', 'CSP+LDA').
  

**Returns**:

  List of supported modality names. Returns ['eeg'] for most decoders,
  ['any'] for modality-agnostic decoders like LinearEEG.

#### check\_decoder\_compatibility

```python
def check_decoder_compatibility(
        decoder_name: str,
        selected_modalities: list[str]) -> tuple[bool, list[str]]
```

Check if selected modalities are compatible with a decoder.

**Arguments**:

- `decoder_name` - Decoder type name (e.g., 'EEGNet', 'CSP+LDA').
- `selected_modalities` - List of modalities to check (e.g., ['eeg'], ['eeg', 'emg']).
  

**Returns**:

  Tuple of (is_compatible, unsupported_modalities). First element is True
  if all modalities are supported; second element lists any unsupported modalities.

