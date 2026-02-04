---
sidebar_label: decoder_schemas
title: dendrite.ml.decoders.decoder_schemas
---

Pydantic configuration schemas for Dendrite decoders.

This module defines the valid parameters for each classifier type using Pydantic models,
replacing the scattered manual parameter filtering with automatic validation.

## NeuralNetConfig Objects

```python
class NeuralNetConfig(BaseModel)
```

Configuration schema for neural network classifiers.

**Attributes:**

| Field | Type | Description |
|-------|------|-------------|
| `model_type` | `str` | Type of neural network model |
| `num_classes` | `int` | Number of output classes |
| `learning_rate` | `float` | Learning rate for optimizer |
| `device` | `str` | Device for model execution (auto/cpu/cuda/mps) |
| `optimizer_type` | `Literal["Adam", "AdamW"]` | Optimizer type (AdamW has better weight decay, recommended for few-shot) |
| `epochs` | `int` | Number of training epochs |
| `batch_size` | `int` | Batch size for training |
| `validation_split` | `float` | Validation split ratio |
| `use_early_stopping` | `bool` | Enable early stopping |
| `early_stopping_patience` | `int` | Early stopping patience (epochs) |
| `early_stopping_min_delta` | `float` | Minimum change to qualify as improvement |
| `weight_decay` | `float` | Weight decay for regularization (use with AdamW) |
| `max_norm_constraint` | `float \| None` | Max norm for classifier weights (None to disable). Applied after each optimizer step. |
| `label_smoothing_factor` | `float` | Label smoothing factor (0.1 recommended for few-shot, 0 = disabled) |
| `use_augmentation` | `bool` | Enable data augmentation during training |
| `aug_strategy` | `str` | Augmentation strategy (light/moderate/aggressive) |
| `use_class_weights` | `bool` | Use class weights for imbalanced data |
| `class_weight_strategy` | `Literal["balanced", "inverse", "equal"]` | Strategy for calculating class weights |
| `use_lr_scheduler` | `bool` | Enable learning rate scheduling |
| `lr_scheduler_type` | `Literal["ReduceLROnPlateau", "StepLR", "CosineAnnealingLR", "OneCycleLR"]` | Type of LR scheduler (OneCycleLR recommended for few-shot) |
| `lr_patience` | `int` | LR scheduler patience |
| `lr_factor` | `float` | LR reduction factor |
| `lr_min` | `float` | Minimum learning rate |
| `lr_step_size` | `int` | Step size for StepLR scheduler |
| `use_lr_warmup` | `bool` | Enable learning rate warmup |
| `warmup_epochs` | `int` | Number of warmup epochs |
| `warmup_start_factor` | `float` | Starting LR factor during warmup (fraction of base LR) |
| `onecycle_max_lr` | `float \| None` | Max LR for OneCycleLR (None = 10x learning_rate) |
| `onecycle_pct_start` | `float` | Fraction of cycle spent increasing LR |
| `mixup_alpha` | `float` | Beta distribution alpha for mixup (0 = disabled, 0.2 recommended when enabled) |
| `mixup_type` | `Literal["mixup", "cutmix", "both"]` | Type of mixing augmentation (cutmix better for preserving local structure) |
| `loss_type` | `Literal["cross_entropy", "focal"]` | Loss function type (focal helps with hard examples) |
| `focal_gamma` | `float` | Focal loss gamma (higher = more focus on hard examples) |
| `use_swa` | `bool` | Enable Stochastic Weight Averaging |
| `swa_start_epoch` | `float` | Start SWA at this fraction of total epochs (e.g., 0.75 = last 25%) |
| `swa_lr` | `float \| None` | SWA learning rate (None = use current LR) |
| `use_mc_dropout` | `bool` | Enable MC Dropout for prediction uncertainty estimation |
| `mc_dropout_samples` | `int` | Number of MC forward passes for uncertainty estimation |
| `model_params` | `dict[str, Any]` | Additional model-specific parameters |
| `input_shapes` | `dict[str, list[int]] \| None` | Input shapes for different modalities |
| `use_scaler` | `bool` | Apply per-channel z-score normalization before model |
| `pipeline_steps` | `list[str] \| None` | Explicit pipeline steps. When None, uses use_scaler to determine steps. |

#### validate\_model\_params

```python
@field_validator("model_params")
@classmethod
def validate_model_params(cls, v, info)
```

Validate model_params against model-specific config if available.

#### get\_device

```python
def get_device() -> torch.device
```

Get torch device from config.

#### get\_model\_specific\_params

```python
def get_model_specific_params() -> dict[str, Any]
```

Extract parameters specific to the configured model type.

Returns model_params dict. Each model's default_parameters (from
get_model_info) handles its own dropout parameter naming.

**Returns**:

  Dictionary of model-specific parameters

## DecoderConfig Objects

```python
class DecoderConfig(NeuralNetConfig)
```

Complete decoder configuration - used for training AND metadata storage.

**Attributes:**

| Field | Type | Description |
|-------|------|-------------|
| `event_mapping` | `dict[int, str] \| None` | Event ID to event name mapping |
| `label_mapping` | `dict[str, int] \| None` | Event name to class index mapping |
| `sample_rate` | `float \| None` | Original data sample rate in Hz |
| `target_sample_rate` | `float \| None` | Sample rate after resampling (None = no resampling) |
| `modality` | `str \| None` | Signal modality (EEG, EMG, etc.) |
| `input_shapes` | `dict[str, list[int]] \| None` | Input shapes for each modality |
| `channel_labels` | `dict[str, list[str]] \| None` | Channel labels per modality \{'eeg': ['Fp1', 'Fp2', ...]\} |
| `preprocessing_config` | `PreprocessingConfig \| None` | Preprocessing config used during training for reproducibility |

#### convert\_event\_mapping\_keys

```python
@field_validator("event_mapping", mode="before")
@classmethod
def convert_event_mapping_keys(cls, v)
```

Convert string keys to integers for JSON compatibility.

JSON serialization always converts object keys to strings, but we need
integer event IDs for proper decoder functionality.

#### validate\_label\_mapping\_consistency

```python
@field_validator("label_mapping")
@classmethod
def validate_label_mapping_consistency(cls, v, info)
```

Ensure label_mapping is consistent with num_classes if both are provided.

#### validate\_input\_shapes

```python
@field_validator("input_shapes")
@classmethod
def validate_input_shapes(cls, v)
```

Validate and normalize input shapes.

#### normalize\_channel\_labels

```python
@field_validator("channel_labels")
@classmethod
def normalize_channel_labels(cls, v)
```

Normalize channel_labels keys to lowercase.

#### effective\_sample\_rate

```python
@property
def effective_sample_rate() -> float
```

Get the rate at which model was trained (target if resampled, else original).

#### check\_compatibility

```python
def check_compatibility(system_shapes: dict[str, list[int]],
                        system_labels: dict[str, list[str]] | None = None,
                        system_sample_rate: float | None = None) -> list[str]
```

Check if this decoder is compatible with the given system configuration.

**Arguments**:

- `system_shapes` - Expected input shapes per modality \{modality: [channels, time_samples]\}
- `system_labels` - Channel labels per modality \{modality: ['ch1', 'ch2', ...]\}
- `system_sample_rate` - System sample rate in Hz
  

**Returns**:

  List of compatibility issues (empty if compatible)

