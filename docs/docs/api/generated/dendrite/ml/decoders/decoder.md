---
sidebar_label: decoder
title: dendrite.ml.decoders.decoder
---

Decoder for Dendrite applications.

Provides a unified interface for neural network and classical ML classifiers
for EEG signal classification. This is the primary decoder implementation.

Note: This decoder is a pure algorithm implementation. Data management and
workflow orchestration should be handled by the mode that uses this decoder.

## Decoder Objects

```python
class Decoder(BaseEstimator, ClassifierMixin)
```

Primary decoder for EEG classification.

Wraps neural network and classical ML classifiers in an sklearn-compatible
interface for training, prediction, and cross-validation.

Features:
- Neural network classifiers (EEGNet, TransformerEEG, LinearEEG, etc.)
- Classical ML pipelines (CSP+LDA, CSP+SVM, etc.)
- sklearn-compatible interface (works with cross_val_score, GridSearchCV, MOABB)

#### \_\_init\_\_

```python
def __init__(config: DecoderConfig)
```

Initialize decoder with configuration.

**Arguments**:

- `config` - DecoderConfig with model type, input shape, and training params
  

**Example**:

  config = DecoderConfig(
  model_type='EEGNet',
  num_classes=2,
  input_shapes={'eeg': [32, 250]},  # {modality: [channels, times]}
  epochs=200,
- `event_mapping=\{1` - 'left', 2: 'right'\},
- `label_mapping=\{'left'` - 0, 'right': 1\}
  )
  decoder = Decoder(config)

#### is\_fitted

```python
@property
def is_fitted() -> bool
```

Check if the decoder is fitted and ready.

#### fit

```python
def fit(X: np.ndarray, y: np.ndarray, epoch_callback=None) -> "Decoder"
```

Train the decoder on provided data.

For cross-validation, use sklearn's cross_val_score() externally:
scores = cross_val_score(decoder, X, y, cv=5)

**Arguments**:

- `X` - Training data with shape (n_samples, n_channels, n_times).
- `y` - Training labels with shape (n_samples,).
- `epoch_callback` - Optional callback invoked after each training epoch.
- `Signature` - callback(epoch, total_epochs, train_loss, train_acc,
  val_loss, val_acc) where val_loss and val_acc are None if no
  validation set is used.
  

**Returns**:

  Self for method chaining (sklearn compatible). After training,
  `training_metrics` attribute is populated for neural models with:
  - 'history': Dict with 'train_loss', 'train_acc', 'val_loss', 'val_acc' lists
  - 'final_train_acc', 'final_val_acc': Final epoch accuracies
  - 'epochs_completed': Number of epochs trained

#### get\_training\_metrics

```python
def get_training_metrics() -> dict[str, Any] | None
```

Get training metrics from the last training session.

**Returns**:

  Training metrics if available, None otherwise

#### get\_training\_history

```python
def get_training_history() -> dict[str, Any] | None
```

Get detailed training history (loss/accuracy curves) if available.

**Returns**:

  Training history if available, None otherwise

#### predict

```python
def predict(X: np.ndarray) -> int | np.ndarray
```

Predict class labels.

For Dendrite applications, consider using predict_sample() to also get confidence.

**Arguments**:

- `X` - Input data with shape (n_samples, n_channels, n_times) or (n_channels, n_times)
  

**Returns**:

  Predicted class(es). Single int for single sample, array of ints for batch.

#### predict\_proba

```python
def predict_proba(X: np.ndarray) -> np.ndarray
```

Predict class probabilities.

**Arguments**:

- `X` - Input data with shape (n_samples, n_channels, n_times) or (n_channels, n_times)
  

**Returns**:

  Class probabilities with shape (n_samples, n_classes)
  

**Raises**:

- `RuntimeError` - If model is not fitted or prediction fails.

#### save

```python
def save(file_identifier: str, study_name: str | None = None) -> str
```

Save complete decoder (pipeline + config + metadata) to disk.

Uses format: .json (metadata) + .pt (neural state) or .joblib (classical pipeline).
Neural models use torch.save for state_dict to handle parametrized modules.

**Arguments**:

- `file_identifier` - Unique identifier for the decoder file.
- `study_name` - Study name for study-scoped storage. If None, uses
  global decoders directory.
  

**Returns**:

  Path to the saved JSON metadata file.
  

**Raises**:

- `ValueError` - If file_identifier is empty or decoder is not fitted.

#### get\_expected\_sample\_rate

```python
def get_expected_sample_rate() -> float
```

Get the rate at which model was trained (for online validation).

#### predict\_sample

```python
def predict_sample(X: np.ndarray) -> tuple[int, float]
```

Primary prediction interface for BMI applications.

Returns both prediction and confidence for a single sample.

**Arguments**:

- `X` - Single sample with shape (n_channels, n_times) or (1, n_channels, n_times)
  

**Returns**:

  Tuple of (prediction, confidence)

#### get\_params

```python
def get_params(deep: bool = True) -> dict[str, Any]
```

Get parameters for sklearn compatibility.

Required for cross_val_score, GridSearchCV, clone(), etc.

**Arguments**:

- `deep` - If True, return nested parameters (ignored, config is atomic)
  

**Returns**:

  Parameter dict with 'config' key containing DecoderConfig

#### set\_params

```python
def set_params(**params) -> "Decoder"
```

Set parameters for sklearn compatibility.

**Arguments**:

- `**params` - Parameters to set. Supports 'config' for full config replacement.
  

**Returns**:

  Self for method chaining.

#### score

```python
def score(X: np.ndarray, y: np.ndarray) -> float
```

Return accuracy score for sklearn compatibility.

**Arguments**:

- `X` - Test samples with shape (n_samples, n_channels, n_times)
- `y` - True labels
  

**Returns**:

  Accuracy score (fraction of correct predictions)

