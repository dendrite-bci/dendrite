---
sidebar_label: base
title: dendrite.ml.models.base
---

Abstract base class for Dendrite neural network models.

This module defines the standard interface that all Dendrite models must implement,
ensuring consistency across different architectures and providing clear
documentation for model developers.

**Example - Creating a custom model:**

```python
import torch.nn as nn
from dendrite.ml.models.base import ModelBase

class MyCustomModel(ModelBase):
    _model_type = 'MyCustomModel'
    _description = 'Custom CNN for EEG classification'
    _modalities = ['eeg']

    def __init__(self, n_channels, n_times, n_classes, hidden_dim=64):
        super().__init__(n_channels, n_times, n_classes)
        self._params = {'hidden_dim': hidden_dim}

        self.conv = nn.Conv1d(n_channels, hidden_dim, kernel_size=5)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        # x shape: (batch, n_channels, n_times)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)
```

## ModelBase Objects

```python
class ModelBase(nn.Module, ABC)
```

Abstract base class for all Dendrite neural network models.

This class enforces a standard interface for EEG/EMG/EOG classification models
used in the Dendrite system. All concrete model implementations must inherit from
this class and implement the required abstract methods.

Standard Interface Requirements:
    1. Constructor must accept (n_channels, n_times, n_classes) parameters
    2. Define class attributes for model metadata (see below)
    3. Must implement forward() method (inherited from nn.Module)

Class Attributes (override in subclasses):
    _model_type: Model identifier (must match registry key).
    _input_domain: 'time-series' or 'time-frequency'.
    _modalities: Supported modalities ['eeg'], ['any'], etc.
    _description: Human-readable description.
    _default_parameters: Default hyperparameters.

Input/Output Conventions:
    Input: torch.Tensor with physiological signals.
    Output: torch.Tensor with shape (batch_size, n_classes) logits.
    Device: Models should work on both CPU and GPU.
    Dtype: Expects float32 tensors.

#### \_\_init\_\_

```python
@abstractmethod
def __init__(n_channels: int, n_times: int, n_classes: int, **kwargs)
```

Initialize the model with standard parameters.

**Arguments**:

- `n_channels` - Number of input channels (e.g., EEG electrodes).
- `n_times` - Number of time samples per trial.
- `n_classes` - Number of output classification classes.
- `**kwargs` - Model-specific additional parameters.
  

**Notes**:

  Subclasses must call super().__init__() and store the standard parameters
  as instance attributes for use in get_model_summary().

#### get\_config\_class

```python
@classmethod
def get_config_class(cls)
```

Get the Pydantic configuration class for this model.

**Returns**:

  The Pydantic config class for validating model parameters,
  or None if the model doesn't have a specific config class.

#### get\_default\_parameters

```python
@classmethod
def get_default_parameters(cls) -> dict[str, Any]
```

Get default parameters from the Pydantic config class.

**Returns**:

  Dictionary of default parameter values.

#### get\_model\_info

```python
@classmethod
def get_model_info(cls) -> dict[str, Any]
```

Return model interface information and capabilities.

Uses class attributes (_model_type, _input_domain, _modalities, etc.)
to build the info dict. Subclasses can override class attributes
or override this method entirely for custom behavior.

**Returns**:

  Dictionary containing model_type, input_domain, input_format,
  input_shape, modalities, description, and default_parameters.

#### get\_model\_summary

```python
def get_model_summary() -> dict[str, Any]
```

Get detailed runtime summary of the model configuration.

Uses class attributes and instance state to build summary.
Subclasses can override for custom behavior.

**Returns**:

  Dictionary containing model_type, n_classes, input_shape,
  parameters, total_params, and trainable_params.

#### validate\_input\_shape

```python
def validate_input_shape(x: torch.Tensor) -> None
```

Validate input tensor shape matches model expectations.

**Arguments**:

- `x` - Input tensor to validate.
  

**Raises**:

- `ValueError` - If input shape doesn't match model requirements.
  

**Notes**:

  This is a helper method that subclasses can use in their forward()
  method to provide clear error messages for shape mismatches.

#### get\_parameter\_count

```python
def get_parameter_count() -> dict[str, int]
```

Get parameter count statistics for the model.

**Returns**:

  Dictionary with 'total_params' and 'trainable_params' counts.

