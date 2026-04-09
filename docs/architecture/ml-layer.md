
# Machine Learning Layer

Decoder pipelines with sklearn-compatible `fit`/`predict` interface for cross-validation and real-time inference. All pipelines accept `(n_samples, n_channels, n_times)` and return class indices + softmax confidence arrays.

---

## Decoders

Decoder (`src/dendrite/ml/decoders/decoder.py`) wraps PyTorch models and classical ML (CSP+LDA/SVM) behind a unified sklearn interface. The registry (`src/dendrite/ml/decoders/registry.py`) maps decoder type strings to pipeline builders and modality metadata.

`DecoderConfig` (`src/dendrite/ml/decoders/decoder_schemas.py`) is the self-describing unit: model type, input shapes, event/label mappings, and per-modality `PreprocessingConfig` (bandpass, CAR, resampling). Saved with the decoder JSON so inference needs no external config.

---

## Training Infrastructure

`TrainingLoop` runs in a `ProcessPoolExecutor` subprocess, isolated from the async event loop. Progress broadcasts to `/ws/training` via epoch callbacks. Cancellation uses a `threading.Event` checked between epochs.

Key cross-module contracts:
- SWA (Stochastic Weight Averaging) skips early-stopping checkpoint restore — the averaged model replaces the best checkpoint
- `stop_event` must be a `multiprocessing.Manager().Event()` to cross the process boundary
- Loss function (`CrossEntropyLoss` or `FocalLoss`) and augmentation strategy are config-driven — see the source for available options

Source: `src/dendrite/ml/training/trainer.py`

---

## Hyperparameter Search

Optuna-based search (`src/dendrite/ml/search/`). Search spaces are auto-generated from Pydantic config `hpo` field metadata across optimizer, regularization, and augmentation categories. Profiles (`Quick`/`Balanced`/`Full`) scope the number of trials and which categories to search. Early stopping halts search after `n_trials / 3` consecutive non-improving trials.


---

## Evaluation

Offline decoder evaluation (`src/dendrite/ml/evaluation.py`) via epoch-based and sliding window methods.

### Epoch Evaluation

`evaluate_epochs()` -- each epoch passed through the decoder and compared to ground truth. Returns accuracy, confusion matrix, and classification report.

### Sliding Window Evaluation

`evaluate_sliding_window()` -- causal sliding window through continuous data, predicting at each step. Instantiates `DecisionGate.from_config(config)` and passes it to `compute_trial_metrics()` for metric aggregation — the same function used by online `AsynchronousMetrics`.

**Decoder window:** Always derived from `decoder.input_shapes` (e.g., `[60, 2000]` = 2000 samples = 4.0s at 500Hz). Both offline eval and online async mode use this window size.

**Trial window:** Predictions count toward a trial as soon as the sliding window overlaps the post-onset period — no artificial dead time. Early predictions (where the window mostly contains pre-event data) will be noisy, but the dwell gate handles this naturally by requiring N consecutive predictions of the same class before firing. This gives realistic TTD measurements from event onset rather than from the first full-window prediction.

**Key parameters:**
- **`epoch_tmin`/`epoch_tmax`** -- trial evaluation boundary. Auto-adjusted if smaller than decoder window. Offline only.
- **`step_ms`** -- step size between predictions (default 100ms)
- **`code_to_class`** -- event codes to class indices. Read from decoder's `label_mapping`.

**DecisionGate** (`src/dendrite/ml/decision_gate.py`) controls detection strategy, dwell length, and confidence threshold.

**Optimal gate:** The evaluator grid-searches over strategy, `dwell_n`, and `confidence_threshold` combinations to find the gate that maximizes balanced accuracy. The result includes `optimal_gate` with the best gate config. This uses the same raw per-trial data stored for reaggregation.

**Reaggregation:** The evaluator stores `per_trial` predictions and `background_preds`/`background_confs` in the job result, enabling post-hoc reaggregation with different gate parameters via `POST /api/ml/jobs/{job_id}/reaggregate` without re-running the decoder.

---

## Models

`ModelBase` (`src/dendrite/ml/models/base.py`) defines the abstract interface. Neural models use Pydantic schemas for architecture-specific parameters validated via `model_params` in `DecoderConfig`. Available models are listed at `/api/ml/models`.

Source: `src/dendrite/ml/models/`

---

**Related Documentation:**
- **[Processing Layer](processing-layer.md)** — Real-time modes and decoder application
- **[Data Layer](data-layer.md)** — Data acquisition and storage

