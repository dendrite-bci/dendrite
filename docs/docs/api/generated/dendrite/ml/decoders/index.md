---
id: decoders
sidebar_label: decoders
title: dendrite.ml.decoders
---

Dendrite Decoder Module

This module contains the main decoder implementation for brain-machine interface applications.

The Decoder class supports both neural network and classical ML pipelines for
EEG signal classification with an sklearn-compatible interface.

#### create\_decoder

```python
def create_decoder(model_type: str = "EEGNet", **kwargs)
```

Create a Decoder instance using the complete decoder configuration.

**Arguments**:

- `model_type` - Model architecture name ('EEGNet', 'TransformerEEG', 'LinearEEG', etc.).
- `**kwargs` - Additional decoder config including num_classes, input_shapes,
  event_mapping, label_mapping, epochs, learning_rate, etc.
  

**Returns**:

  Configured Decoder instance ready for training.
  

**Example**:

    ```python
    decoder = create_decoder(
        model_type='EEGNet',
        num_classes=3,
        input_shapes=\{'eeg': (32, 250)\},
        event_mapping=\{1: 'left', 2: 'right'\},
        label_mapping=\{'left': 0, 'right': 1\}
    )
    ```

#### load\_decoder

```python
def load_decoder(decoder_path: str) -> Decoder
```

Load a pre-trained decoder from a saved decoder file.

**Arguments**:

- `decoder_path` - Path to the saved decoder JSON metadata file.
  

**Returns**:

  Decoder instance with loaded weights/parameters ready for inference.
  

**Raises**:

- `FileNotFoundError` - If decoder file doesn't exist.
- `RuntimeError` - If loading fails.

#### validate\_decoder\_file

```python
def validate_decoder_file(
    filepath: str,
    expected_shapes: dict[str, tuple[int, int]] | None = None,
    expected_sample_rate: float | None = None,
    expected_labels: dict[str, list[str]] | None = None
) -> tuple[dict[str, Any] | None, list[str]]
```

Validate decoder file with detailed error reporting.

**Arguments**:

- `filepath` - Path to decoder JSON file or base path
- `expected_shapes` - Expected input shapes per modality: \{'eeg': (channels, samples), ...\}
- `expected_sample_rate` - Expected sampling rate in Hz
- `expected_labels` - Expected channel labels per modality: \{'EEG': ['Fp1', 'Fp2', ...], ...\}
  

**Returns**:

  Tuple of (metadata_dict or None, list of validation issues)

#### get\_decoder\_metadata

```python
def get_decoder_metadata(filepath: str) -> dict[str, Any]
```

Get metadata from saved decoder file for inspection without loading.

Use this to inspect decoder properties (model type, input shapes, class
mappings, training info) without loading the full model weights.

**Arguments**:

- `filepath` - Path to saved decoder JSON file.
  

**Returns**:

  Decoder metadata including model_type, input_shapes, num_classes,
  event_mapping, label_mapping, and training configuration.
  

**Raises**:

- `FileNotFoundError` - If file doesn't exist.
- `ValueError` - If metadata validation fails.

