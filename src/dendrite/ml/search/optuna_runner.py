"""Optuna trial execution, config suggestion, and model search utilities."""

import logging
import time
from dataclasses import dataclass
from queue import Full
from typing import Any

import numpy as np
import optuna

from dendrite.ml.decoders.decoder import Decoder
from dendrite.ml.decoders.decoder_schemas import DecoderConfig
from dendrite.ml.training.runner import train_decoder

from .search_space import DERIVED_PARAMS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config suggestion
# ---------------------------------------------------------------------------


def _suggest_single(trial: optuna.Trial, param: str, spec: dict[str, Any]) -> Any:
    """Suggest a single parameter value from its spec."""
    if not isinstance(spec, dict):
        return None
    param_type = spec.get("type", "categorical")
    if param_type == "float":
        return trial.suggest_float(param, spec["low"], spec["high"], log=spec.get("log", False))
    elif param_type == "int":
        return trial.suggest_int(param, spec["low"], spec["high"], step=spec.get("step", 1))
    elif param_type == "categorical":
        return trial.suggest_categorical(param, spec["choices"])
    return None


def _suggest_params(trial: optuna.Trial, search_space: dict[str, Any]) -> dict[str, Any]:
    """Suggest hyperparameters from a flat search space (two-pass for conditionals)."""
    config = {}
    for param, spec in search_space.items():
        if isinstance(spec, dict) and "conditional" in spec:
            continue
        value = _suggest_single(trial, param, spec)
        if value is not None:
            config[param] = value
    for param, spec in search_space.items():
        if not (isinstance(spec, dict) and "conditional" in spec):
            continue
        cond_param, cond_value = spec["conditional"]
        if config.get(cond_param) == cond_value:
            value = _suggest_single(trial, param, spec)
            if value is not None:
                config[param] = value
    return config


def split_params(
    flat_params: dict[str, Any], base_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Split flat param dict into DecoderConfig kwargs + model_params.

    Params that match DecoderConfig fields go at the top level; everything
    else goes into model_params.  Derived params (e.g., EEGNet F2 = F1*D)
    are computed from the model_params after splitting.
    """
    from dendrite.ml.decoders.decoder_schemas import DecoderConfig

    kwargs = dict(base_kwargs)
    decoder_fields = set(DecoderConfig.model_fields)

    for key, val in flat_params.items():
        if key in decoder_fields:
            kwargs[key] = val
        elif key == "model_type":
            kwargs["model_type"] = val
        else:
            kwargs.setdefault("model_params", {})[key] = val

    model_type = kwargs.get("model_type")
    model_params = kwargs.get("model_params", {})
    if model_type is None:
        return kwargs
    for param, compute_fn in DERIVED_PARAMS.get(model_type, {}).items():
        try:
            model_params[param] = compute_fn(model_params)
        except (KeyError, TypeError):
            pass

    return kwargs


def suggest_decoder_kwargs(
    trial: optuna.Trial,
    model_types: list[str],
    base_kwargs: dict[str, Any],
    search_space: dict[str, Any],
) -> dict[str, Any]:
    """Suggest DecoderConfig kwargs from pipeline-aware search space.

    The search_space should be built by build_decoder_search_space() which
    includes only params relevant to this decoder's pipeline steps.

    Returns a flat dict ready for DecoderConfig(**kwargs).
    """
    suggested = _suggest_params(trial, search_space)
    suggested["model_type"] = trial.suggest_categorical("model_type", model_types)
    return split_params(suggested, base_kwargs)



# ---------------------------------------------------------------------------
# Trial execution
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    """Result of a single Optuna trial."""

    trial_num: int
    model_type: str
    accuracy: float
    decoder: Decoder | None
    decoder_kwargs: dict
    elapsed: float
    error: str | None = None


def prepare_holdout_split(
    X: np.ndarray,
    y: np.ndarray,
    holdout_ratio: float = 0.2,
    modality: str = "eeg",
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Split data and build base_kwargs for DecoderConfig.

    Returns (X_train, y_train, X_holdout, y_holdout, base_kwargs).
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    config = config or {}
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=holdout_ratio, random_state=42,
    )
    train_idx, holdout_idx = next(splitter.split(X, y))

    base_kwargs = {
        "num_classes": int(np.max(y) + 1),
        "input_shapes": {modality: list(X.shape[1:])},
        "epochs": config.get("epochs", 100),
        # Reduce inner val split during search — holdout already provides eval
        "validation_split": min(config.get("validation_split", 0.2), 0.1),
        "use_early_stopping": config.get("use_early_stopping", True),
        "early_stopping_patience": config.get("early_stopping_patience", 10),
        "event_mapping": config.get("event_mapping"),
        "label_mapping": config.get("label_mapping"),
    }

    return X[train_idx], y[train_idx], X[holdout_idx], y[holdout_idx], base_kwargs


def run_single_trial(
    study: optuna.Study,
    trial_num: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    model_types: list[str],
    base_kwargs: dict[str, Any],
    modality: str = "eeg",
    search_space: dict[str, Any] | None = None,
) -> TrialResult:
    """Run one Optuna trial: suggest config, train decoder, score on holdout.

    Pure synchronous function — caller handles threading/executor.
    """
    if search_space is None:
        from .search_space import build_decoder_search_space
        search_space = build_decoder_search_space(model_types[0])

    start = time.time()
    trial = study.ask()
    try:
        decoder_kwargs = suggest_decoder_kwargs(trial, model_types, base_kwargs, search_space)
        decoder = train_decoder(X_train, y_train, DecoderConfig(**decoder_kwargs), modality)
        accuracy = float(decoder.score(X_holdout, y_holdout))
        study.tell(trial, accuracy)
        return TrialResult(
            trial_num=trial_num,
            model_type=decoder_kwargs.get("model_type", "?"),
            accuracy=accuracy,
            decoder=decoder,
            decoder_kwargs=decoder_kwargs,
            elapsed=time.time() - start,
        )
    except Exception as e:
        study.tell(trial, state=optuna.trial.TrialState.FAIL)
        logger.warning(f"Trial {trial_num} failed: {e}")
        return TrialResult(
            trial_num=trial_num,
            model_type="?",
            accuracy=0.0,
            decoder=None,
            decoder_kwargs={},
            elapsed=time.time() - start,
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Subprocess-safe Optuna search (mirrors run_training pattern)
# ---------------------------------------------------------------------------


def run_optuna_search(
    X: np.ndarray, y: np.ndarray, config: dict, progress_queue=None,
    stop_event=None,
) -> dict:
    """Run Optuna hyperparameter search in a subprocess.

    Mirrors the run_training() pattern: module-level function, picklable,
    reports progress via multiprocessing Queue, sentinel-terminated.
    """
    import os

    max_threads = config.get("max_threads", 2)
    os.environ["OMP_NUM_THREADS"] = str(max_threads)
    os.environ["MKL_NUM_THREADS"] = str(max_threads)

    import torch
    torch.set_num_threads(max_threads)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from .search_space import build_decoder_search_space

    model_type = config["model_type"]
    categories = config.get("search_categories")
    search_space = build_decoder_search_space(model_type, categories)
    n_trials = config.get("optuna_n_trials") or 30
    holdout_ratio = config.get("holdout_ratio") or 0.2

    model_types = [model_type]

    X_train, y_train, X_hold, y_hold, base_kwargs = prepare_holdout_split(
        X, y, holdout_ratio, config=config,
    )

    study = optuna.create_study(direction="maximize")
    best_accuracy = 0.0
    trials_since_improvement = 0
    patience = max(5, n_trials // 3)
    trial_results = []
    start_time = time.time()
    early_stopped = False

    for i in range(n_trials):
        if stop_event is not None and stop_event.is_set():
            logger.info(f"Search cancelled at trial {i + 1}/{n_trials}")
            early_stopped = True
            break
        tr = run_single_trial(
            study, i + 1, X_train, y_train, X_hold, y_hold,
            model_types, base_kwargs, search_space=search_space,
        )
        if tr.error:
            trial_results.append({"trial": tr.trial_num, "error": tr.error})
        else:
            trial_results.append({
                "trial": tr.trial_num, "model_type": tr.model_type,
                "val_accuracy": round(tr.accuracy, 4),
                "elapsed": round(time.time() - start_time, 1),
            })
            if tr.accuracy > best_accuracy + 1e-4:
                best_accuracy = tr.accuracy
                trials_since_improvement = 0
            else:
                trials_since_improvement += 1

        if progress_queue is not None:
            try:
                progress_queue.put_nowait({
                    "trial": i + 1, "total_trials": n_trials,
                    "model_type": tr.model_type,
                    "val_accuracy": round(tr.accuracy, 4),
                    "best_accuracy": round(best_accuracy, 4),
                    "elapsed_seconds": round(time.time() - start_time, 1),
                })
            except (Full, OSError):
                pass

        # Early stop if no improvement for `patience` consecutive trials
        if trials_since_improvement >= patience and i >= patience:
            logger.info(
                f"Search early stopped at trial {i + 1}/{n_trials}: "
                f"no improvement in {patience} trials (best={best_accuracy:.4f})"
            )
            early_stopped = True
            break

    if progress_queue is not None:
        progress_queue.put_nowait(None)  # sentinel

    best_path = None
    if best_accuracy > 0:
        # Retrain best config on full data (train + holdout) with restored val split
        final_base = {**base_kwargs, "validation_split": config.get("validation_split", 0.2)}
        final_kwargs = split_params(dict(study.best_trial.params), final_base)
        X_full = np.concatenate([X_train, X_hold])
        y_full = np.concatenate([y_train, y_hold])
        try:
            decoder = train_decoder(X_full, y_full, DecoderConfig(**final_kwargs))
            best_path = decoder.save(
                f"optuna_{config.get('model_type', 'search')}_{int(time.time())}",
            )
        except Exception as e:
            logger.warning(f"Final retrain with best params failed: {e}")

    return {
        "search_type": "optuna",
        "categories": categories,
        "searched_params": list(search_space.keys()),
        "n_trials": len(trial_results), "max_trials": n_trials,
        "early_stopped": early_stopped,
        "best_accuracy": round(best_accuracy, 4),
        "best_model_type": (
            study.best_trial.params.get("model_type", config.get("model_type", "?"))
            if best_accuracy > 0 else None
        ),
        "best_params": dict(study.best_trial.params) if best_accuracy > 0 else None,
        "trial_results": trial_results,
        "path": best_path,
        "n_samples": len(y), "input_shape": list(X.shape),
    }
