---
id: ml-layer
title: Machine Learning Layer
sidebar_label: ML Layer
---

# Machine Learning Layer

PyTorch models with sklearn-compatible `fit`/`predict` interface for cross-validation and real-time inference.

**Processing Flow:**
```text
Preprocessed Data → Decoder Pipeline → Neural Network → Predictions
   (array)           (NeuralNetClassifier)  (EEGNet/etc)  (labels + confidence)
```

---

## Decoders

Decoder (`src/dendrite/ml/decoders/decoder.py`) wraps PyTorch models with sklearn-compatible methods (`fit`/`predict`/`predict_proba`) for offline training and real-time BCI inference.

The decoder builds sklearn pipelines for both neural networks (NeuralNetClassifier) and classical ML (CSP spatial filtering with LDA/SVM). Both support the same interface, enabling offline cross-validation via StratifiedKFold and consistent training/inference for real-time sessions. All pipelines accept array input with shape `(n_samples, n_channels, n_times)`.

### Neural Network Classifier

NeuralNetClassifier (`src/dendrite/ml/decoders/neural_classifier.py`) bridges PyTorch models and sklearn interfaces with training infrastructure including weighted loss for class imbalance, early stopping, auto device detection (CUDA/MPS/CPU), learning rate scheduling, and L2 regularization.

### Configuration

Decoders use Pydantic schemas (`src/dendrite/ml/decoders/decoder_schemas.py`) with two-level hierarchy: `NeuralNetConfig` for training parameters (learning rate, batch size, regularization, early stopping) and `DecoderConfig` extending it with decoder metadata (model_type, event/label mappings, input shapes). See **[Decoders API](../../api/generated/dendrite/ml/decoders/decoder.md)** for complete parameter reference.

### Data Augmentation

Online augmentation applies random transformations during training to improve model generalization. Available strategies include noise injection, amplitude scaling, time/channel masking, and dropout. Presets (`light`, `moderate`, `aggressive`) configure strategy combinations and application probabilities. Mixup/CutMix augmentation also supported. Configure via `use_augmentation`, `aug_strategy`, and mixup parameters.

### Data Contracts

Array-based data structures throughout the pipeline.

**Decoder Input**

```python
X: np.ndarray  # (n_samples, n_channels, n_times)
y: np.ndarray  # (n_samples,) - class labels for fit
```

**Decoder Output**

```python
predictions: np.ndarray   # (n_samples,) - class indices
probabilities: np.ndarray # (n_samples, n_classes) - softmax confidence
```

Shape `(n_samples, n_channels, n_times)` is preserved until model-specific reshaping occurs in the classifier.

---

## Models

ModelBase (`src/dendrite/ml/models/base.py`) defines the abstract interface for PyTorch architectures designed for time-series neural signals.

**Available architectures:** Braindecode wrappers (BDEEGNet, BDEEGConformer, BDShallowNet, BDDeep4Net, BDATCNet, BDTCN, BDEEGInception) provide research-grade implementations with published benchmarks. Native models (EEGNet, EEGNetPP, LinearEEG, TransformerEEG) are custom implementations. Classical pipelines (CSP+LDA, CSP+SVM) use sklearn/MNE components with different training paths.

Neural models have Pydantic configuration schemas defining architecture-specific parameters, validated via `model_params` in `DecoderConfig`. Classical pipelines (CSP+LDA/SVM) use sklearn defaults. Tensor conversion handles numpy to PyTorch to device placement automatically, with model-specific reshaping for time-series vs time-frequency inputs.

Complete specification: **[Models API](../../api/generated/dendrite/ml/models/base.md)**

---

## Metrics

MetricsManager (`src/dendrite/ml/metrics/metrics_manager.py`) routes metrics calculations to mode-specific backends. Metrics are computed online during sessions and support post-hoc threshold analysis.

### Synchronous Metrics

SynchronousMetrics (`src/dendrite/ml/metrics/synchronous_metrics.py`) evaluates trial-based classification with discrete epochs.

| Metric | Description |
|--------|-------------|
| **Prequential Accuracy** | Exponentially weighted moving accuracy (forgetting factor: 0.95) |
| **Confusion Matrix** | Per-class prediction counts |
| **Cohen's Kappa** | Agreement beyond chance |
| **Chance Level** | Maximum class proportion (baseline for accuracy) |

### Asynchronous Metrics

AsynchronousMetrics (`src/dendrite/ml/metrics/asynchronous_metrics.py`) evaluates continuous classification using a confusion matrix foundation.

**Window-Based Ground Truth:**
Each prediction is evaluated against the majority label in its corresponding time window.

| Metric | Description |
|--------|-------------|
| **Balanced Accuracy** | Mean of per-class recalls (handles class imbalance) |
| **Per-class Accuracy** | Recall per class (correct / total per class) |
| **ITR** | Information Transfer Rate (bits/min, Wolpaw formula) |
| **FAR/min** | False alarm rate per minute (predictions outside trial windows) |
| **Mean TTD** | Mean time-to-detection in ms |
| **Per-class Trials** | Trial count per class |

**Background Handling:** The `background_class` parameter controls how windows with label `-1` are evaluated. Set to `None` (default) to skip background windows, or set to a class index (e.g., `0`) to evaluate background periods as that class. Useful for paradigms where idle time counts toward metrics (e.g., ErrP where background = "no error").

---

**Related Documentation:**
- **[Processing Layer](processing-layer.md)** - Real-time modes and decoder application
- **[Data Layer](data-layer.md)** - Data acquisition and storage

**API References:**
- **[Decoders API](../../api/generated/dendrite/ml/decoders/decoder.md)** - Decoder interface and usage
- **[Models API](../../api/generated/dendrite/ml/models/base.md)** - Model architectures and parameters
