---
id: models
sidebar_label: models
title: dendrite.ml.models
---

Models for EEG/EMG classification.

This module provides all model classes (neural networks and classical ML) used in
brain-computer and muscle-computer interfaces. All models are registered in
MODEL_REGISTRY which is the single source of truth for model classes and configs.

Neural network models:
- BDEEGNet: Compact CNN for EEG (braindecode EEGNetv4)
- BDEEGConformer: CNN-Transformer hybrid (braindecode)
- BDShallowNet: Fast FBCSP-inspired baseline
- BDDeep4Net: Deep ConvNet with 4 blocks
- BDATCNet: Attention TCN (SOTA on MI)
- BDTCN: Temporal Convolutional Network
- BDEEGInception: Inception for Motor Imagery
- EEGNet: Native compact CNN implementation
- EEGNetPP: EEGNet with SE attention and residual connections
- LinearEEG: Simple linear model for few-shot learning (any modality)
- TransformerEEG: Transformer-based architecture

Classical ML models:
- CSP: Common Spatial Patterns feature extractor
- LDA: Linear Discriminant Analysis classifier
- SVM: Support Vector Machine classifier

Use get_available_models() to see all available models.
Use get_available_decoders() for pipeline configurations (e.g., CSP+LDA).

#### create\_model

```python
def create_model(model_type: str, num_classes: int,
                 input_shape: tuple[int, int], **kwargs)
```

Create a neural network model of the specified type.

**Arguments**:

- `model_type` - Type of model to create. Use get_available_models() to see all options.
- `num_classes` - Number of output classes.
- `input_shape` - Shape of input data (n_channels, n_times).
- `**kwargs` - Additional model-specific parameters.
  

**Returns**:

  Initialized PyTorch model (torch.nn.Module).
  

**Raises**:

- `ValueError` - If model_type is not recognized.

#### get\_available\_models

```python
def get_available_models() -> list[str]
```

Get a list of all available model types.

Returns all models from MODEL_REGISTRY including neural networks
(EEGNet, etc.) and classical ML models (CSP, LDA, SVM).

For pipeline configurations (CSP+LDA, CSP+SVM), use get_available_decoders().

**Returns**:

  List of supported model type names.

