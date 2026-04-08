"""Evaluation and benchmarking — pure ML functions.

No web/async/WS dependencies. Takes data in, returns results out.
Trial-level metric aggregation delegates to ``compute_trial_metrics()`` in
metrics_utils (shared with online AsynchronousMetrics).
"""

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from dendrite.ml.decision_gate import DecisionGate
from dendrite.ml.decoders.decoder import Decoder
from dendrite.ml.metrics_utils import calculate_itr, compute_trial_metrics, optimize_gate
from dendrite.ml.training.runner import decoder_config_from_dict, train_decoder

logger = logging.getLogger(__name__)

_BATCH_SIZE = 128


def evaluate_epochs(
    decoder: Decoder, X: np.ndarray, y: np.ndarray,
    progress_callback: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Run epoch-by-epoch evaluation. Returns accuracy + confusion matrix + report."""
    total = len(X)
    predictions = []

    for i in range(total):
        pred, conf = decoder.predict_sample(X[i])
        info = {
            "prediction": int(pred), "true_label": int(y[i]),
            "confidence": round(float(conf), 4),
            "correct": bool(int(pred) == int(y[i])),
        }
        predictions.append(info)
        if progress_callback and i % max(1, total // 50) == 0:
            progress_callback(i + 1, total, info)

    y_pred = np.array([p["prediction"] for p in predictions])
    return {
        "accuracy": round(float(np.mean(y_pred == y)), 4),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "classification_report": classification_report(
            y, y_pred, output_dict=True, zero_division=0,
        ),
        "n_samples": total,
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# Sliding-window helpers
# ---------------------------------------------------------------------------


def _build_trial_ranges(
    events: list[tuple[int, int]],
    event_mapping: dict[int, str],
    epoch_tmin: float,
    epoch_tmax: float,
    sample_rate: float,
    n_samples: int,
    code_to_class: dict[int, int],
    window_samples: int,
) -> list[tuple[int, int, int, str, int]]:
    """Filter events and build trial ranges with class labels.

    Returns list of (onset, trial_end, event_code, event_name, true_class).
    """
    trial_events = []
    for sample_idx, event_code in events:
        if event_mapping and event_code not in event_mapping:
            continue
        name = event_mapping.get(event_code, str(event_code))
        onset = sample_idx + int(epoch_tmin * sample_rate)
        offset = sample_idx + int(epoch_tmax * sample_rate)
        if onset >= 0 and offset <= n_samples:
            trial_events.append((onset, event_code, name))

    trial_ranges = []
    epoch_duration_samples = int((epoch_tmax - epoch_tmin) * sample_rate)
    for onset, event_code, event_name in trial_events:
        trial_end = min(onset + epoch_duration_samples + window_samples, n_samples)
        true_class = code_to_class.get(event_code, event_code)
        trial_ranges.append((onset, trial_end, event_code, event_name, true_class))

    return trial_ranges


def _batch_predict(
    decoder: Decoder,
    data: np.ndarray,
    positions: np.ndarray,
    window_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch sliding-window prediction across all positions.

    Returns (predictions, confidences) arrays.
    """
    n_positions = len(positions)
    all_preds = np.empty(n_positions, dtype=np.int32)
    all_confs = np.empty(n_positions, dtype=np.float32)

    for batch_start in range(0, n_positions, _BATCH_SIZE):
        batch_end = min(batch_start + _BATCH_SIZE, n_positions)
        batch_pos = positions[batch_start:batch_end]
        windows = np.stack([data[:, p - window_samples:p] for p in batch_pos])
        probas = decoder.predict_proba(windows)
        all_preds[batch_start:batch_end] = np.argmax(probas, axis=1)
        all_confs[batch_start:batch_end] = np.max(probas, axis=1)

    return all_preds, all_confs


def _tag_positions(
    positions: np.ndarray,
    trial_ranges: list[tuple[int, int, int, str, int]],
    window_samples: int,
) -> np.ndarray:
    """Tag each position as belonging to a trial or background (-1).

    Positions are window-end indices: position p covers [p - window, p].
    Only count predictions once the decoder window fully contains post-onset
    data (p >= onset + window).  Earlier predictions use mostly pre-event
    data and form wrong-class dwell streaks.
    """
    trial_idx_arr = np.full(len(positions), -1, dtype=np.int32)
    for t_idx, (t_onset, t_end, *_) in enumerate(trial_ranges):
        mask = (positions >= t_onset + window_samples) & (positions <= t_end)
        trial_idx_arr[mask] = t_idx
    return trial_idx_arr


def enrich_per_trial(per_trial: list[dict[str, Any]]) -> None:
    """Add vote_name and detection_time_s to per-trial dicts (in-place).

    Expects each trial dict to already have: label, event_name, vote,
    first_pred_s (optional), and ttd_ms (optional, from gate outcomes).
    """
    class_to_name = {t["label"]: t["event_name"] for t in per_trial}
    for t in per_trial:
        t["vote_name"] = class_to_name.get(t["vote"], str(t["vote"]))
        fps = t.get("first_pred_s")
        t["detection_time_s"] = (
            round(fps + t["ttd_ms"] / 1000.0, 3)
            if fps is not None and t.get("ttd_ms") is not None
            else None
        )


def _collect_trial_data(
    trial_ranges: list[tuple[int, int, int, str, int]],
    positions: np.ndarray,
    all_preds: np.ndarray,
    all_confs: np.ndarray,
    trial_idx_arr: np.ndarray,
) -> list[dict[str, Any]]:
    """Collect raw predictions + confidences per trial (strategy-independent).

    Stores raw (unfiltered) predictions and confidences so that
    ``compute_trial_metrics`` can re-apply any gate post-hoc.
    """
    per_trial = []
    for t_idx, (_t_onset, _t_end, event_code, event_name, true_class) in enumerate(
        trial_ranges,
    ):
        mask = trial_idx_arr == t_idx
        if not np.any(mask):
            continue
        preds_raw = all_preds[mask].tolist()
        confs_raw = all_confs[mask].tolist()
        n_correct = int(np.sum(all_preds[mask] == true_class))

        if t_idx < 3:
            logger.info(
                f"Trial {t_idx}: event={event_name} code={event_code} "
                f"true_class={true_class} preds={set(preds_raw)} "
                f"n_correct={n_correct}/{len(preds_raw)}"
            )

        per_trial.append({
            "trial": t_idx + 1, "event_code": int(event_code),
            "event_name": event_name,
            "predictions": preds_raw,
            "confidences": confs_raw,
            "label": true_class,
            "n_correct": n_correct,
            "n_steps": len(preds_raw),
            "step_accuracy": round(n_correct / len(preds_raw), 4),
        })

    return per_trial


def _build_timeline(
    positions: np.ndarray,
    all_preds: np.ndarray,
    all_confs: np.ndarray,
    trial_idx_arr: np.ndarray,
    trial_ranges: list[tuple[int, int, int, str, int]],
    sample_rate: float,
) -> list[dict]:
    """Build full continuous timeline for visualization."""
    pos_times = positions / sample_rate
    timeline: list[dict] = []
    for i in range(len(positions)):
        t_idx = int(trial_idx_arr[i])
        pred_int = int(all_preds[i])
        correct = None
        if t_idx >= 0:
            correct = pred_int == trial_ranges[t_idx][4]
        timeline.append({
            "time_s": round(float(pos_times[i]), 3),
            "prediction": pred_int,
            "confidence": round(float(all_confs[i]), 4),
            "correct": correct,
            "trial_idx": t_idx,
        })
    return timeline


# ---------------------------------------------------------------------------
# Main sliding-window evaluator
# ---------------------------------------------------------------------------


def evaluate_sliding_window(
    decoder: Decoder,
    raw_data: np.ndarray,
    events: list[tuple[int, int]],
    sample_rate: float,
    config: dict[str, Any],
    channel_indices: list[int] | None = None,
    progress_callback: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Sliding-window evaluation simulating real-time BCI operation.

    Collects per-trial predictions via causal sliding window, then delegates
    metric aggregation to ``compute_trial_metrics()`` (shared with online eval).
    """
    event_mapping = config.get("event_mapping", {})
    step_ms = config.get("step_ms", config.get("step_size_ms", 100))
    epoch_tmin = config.get("epoch_tmin", 0.0)
    epoch_tmax = config.get("epoch_tmax", 2.0)

    gate = DecisionGate.from_config(config)

    # Decoder window size (from training input shape)
    input_shapes = decoder.input_shapes or {}
    modality_shape = next(iter(input_shapes.values()), None)
    if modality_shape and len(modality_shape) >= 2:
        window_samples = int(modality_shape[1])
    else:
        window_samples = int((epoch_tmax - epoch_tmin) * sample_rate)
    step_samples = max(1, int(step_ms / 1000.0 * sample_rate))
    n_samples_total = int(raw_data.shape[1])
    window_sec = window_samples / sample_rate

    if epoch_tmax - epoch_tmin < window_sec:
        epoch_tmax = epoch_tmin + window_sec

    data = raw_data[channel_indices] if channel_indices else raw_data

    # Class mapping
    code_to_class = config.get("code_to_class")
    if code_to_class:
        code_to_class = {int(k): int(v) for k, v in code_to_class.items()}
    else:
        code_to_class = {code: i for i, code in enumerate(sorted(event_mapping.keys()))}
    n_classes = len(code_to_class)

    # Build trials
    trial_ranges = _build_trial_ranges(
        events, event_mapping, epoch_tmin, epoch_tmax,
        sample_rate, n_samples_total, code_to_class, window_samples,
    )
    if not trial_ranges:
        return {"error": "No valid trials found", "accuracy": 0, "n_trials": 0}

    logger.info(
        f"Sliding window eval: {len(trial_ranges)} trials, {n_classes} classes, "
        f"window={window_samples} ({window_sec:.3f}s), step={step_ms}ms"
    )

    # Batch predict across full recording
    positions = np.arange(window_samples, n_samples_total + 1, step_samples)
    all_preds, all_confs = _batch_predict(decoder, data, positions, window_samples)

    # Tag positions and build results
    trial_idx_arr = _tag_positions(positions, trial_ranges, window_samples)
    timeline = _build_timeline(
        positions, all_preds, all_confs, trial_idx_arr, trial_ranges, sample_rate,
    )
    # Raw preds + confidences per trial — gate applies filtering in compute_trial_metrics
    per_trial = _collect_trial_data(
        trial_ranges, positions, all_preds, all_confs, trial_idx_arr,
    )

    # Progress callbacks
    if progress_callback:
        for trial in per_trial:
            tr_idx = trial["trial"] - 1
            tl_slice = [e for e in timeline if e["trial_idx"] == tr_idx]
            progress_callback(trial["trial"], len(trial_ranges), trial, tl_slice)

    # Background predictions + confidences (raw — gate filters during aggregation)
    bg_mask = trial_idx_arr == -1
    bg_preds = all_preds[bg_mask].tolist()
    bg_confs = all_confs[bg_mask].tolist()

    # Aggregate metrics (gate applies confidence + strategy)
    agg = compute_trial_metrics(
        per_trial, bg_preds, gate,
        num_classes=n_classes, step_duration_ms=step_ms,
        background_confs=bg_confs,
    )

    # Merge per-trial outcomes from agg into per_trial dicts
    outcomes = agg.pop("trial_outcomes", [])
    for t, o in zip(per_trial, outcomes, strict=True):
        t.update(o)

    # Compute first_pred_s per trial (needs position arrays, only available here)
    for t in per_trial:
        t_idx = t["trial"] - 1
        trial_positions = positions[trial_idx_arr == t_idx]
        t["first_pred_s"] = round(float(trial_positions[0]) / sample_rate, 3) if len(trial_positions) > 0 else None

    enrich_per_trial(per_trial)

    # TTD stats from per-trial outcomes
    all_ttds = [o["ttd_ms"] for o in outcomes if o["ttd_ms"] is not None]
    ttd_stats = {}
    if all_ttds:
        ttd_arr = np.array(all_ttds)
        ttd_stats = {
            "mean_ms": round(float(np.mean(ttd_arr)), 1),
            "median_ms": round(float(np.median(ttd_arr)), 1),
            "min_ms": round(float(np.min(ttd_arr)), 1),
            "max_ms": round(float(np.max(ttd_arr)), 1),
            "n_detected": len(all_ttds),
            "n_total": len(per_trial),
        }

    # Event markers (correctness comes from per-trial outcomes)
    trial_window_sec = (epoch_tmax - epoch_tmin) + window_sec
    per_trial_by_idx = {t["trial"] - 1: t for t in per_trial}
    event_markers = [
        {
            "event_s": round(t_onset / sample_rate - epoch_tmin, 3),  # actual stimulus time
            "trial_start_s": round(t_onset / sample_rate, 3),  # epoch window start
            "trial_end_s": round(t_end / sample_rate, 3),  # epoch window end (+ lookahead)
            "name": name, "code": int(code),
            "correct": per_trial_by_idx[i]["correct"] if i in per_trial_by_idx else False,
        }
        for i, (t_onset, t_end, code, name, _cls) in enumerate(trial_ranges)
    ]

    accuracy = agg["accuracy"]
    return {
        "mode": "sliding_window",
        **agg,
        "ttd": ttd_stats,
        "per_trial": per_trial,
        "timeline": timeline,
        "event_markers": event_markers,
        "background_preds": bg_preds,
        "background_confs": bg_confs,
        "far": {
            "false_detections": agg["far_false_detections"],
            "background_steps": len(bg_preds),
            "background_duration_s": round(len(bg_preds) * step_ms / 1000.0, 1),
            "far_per_min": agg["far_per_min"],
        },
        "itr_bits_per_min": round(calculate_itr(
            n_classes, accuracy,
            ttd_stats["mean_ms"] / 1000.0 if ttd_stats.get("mean_ms") is not None else trial_window_sec,
        ), 2),
        "optimal_gate": optimize_gate(
            per_trial, bg_preds,
            num_classes=n_classes, step_duration_ms=step_ms,
            bg_confs=bg_confs,
        ),
        "config": {
            "window_samples": window_samples,
            "window_sec": window_sec,
            "step_ms": step_ms,
            "trial_window_sec": round(trial_window_sec, 3),
            "epoch_tmin": epoch_tmin, "epoch_tmax": epoch_tmax,
            "gate": gate.to_dict(),
        },
    }


def benchmark_cv(
    X: np.ndarray, y: np.ndarray,
    model_types: list[str],
    config: dict[str, Any],
    model_callback=None,
) -> list[dict[str, Any]]:
    """Run k-fold CV for each model type. Returns list of per-model results."""
    n_folds = config.get("n_folds", 5)
    num_classes = int(np.max(y) + 1)
    modality = config.get("modality", "eeg")
    input_shapes = {modality: list(X.shape[1:])}
    results = []

    for model_type in model_types:
        start = time.time()
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_accs = []

        for train_idx, test_idx in cv.split(X, y):
            dc = decoder_config_from_dict(
                {
                    "model_type": model_type,
                    "epochs": config.get("epochs", 100),
                    "batch_size": config.get("batch_size", 32),
                    "learning_rate": config.get("learning_rate", 0.001),
                    "validation_split": 0.0,
                    "use_early_stopping": False,
                },
                num_classes, input_shapes,
            )
            decoder = train_decoder(X[train_idx], y[train_idx], dc, modality)
            fold_accs.append(float(decoder.score(X[test_idx], y[test_idx])))

        elapsed = time.time() - start
        result = {
            "model_type": model_type,
            "accuracy": round(float(np.mean(fold_accs)), 4),
            "std": round(float(np.std(fold_accs)), 4),
            "elapsed_s": round(elapsed, 1),
            "n_folds": n_folds,
        }
        results.append(result)
        if model_callback:
            model_callback(result)

    return results
