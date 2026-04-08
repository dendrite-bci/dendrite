"""Shared metrics utilities for online (metrics/) and offline (evaluation.py).

All trial-level BCI metric computation lives here — single source of truth.
"""

import math
from collections import Counter
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix

from dendrite.ml.decision_gate import DecisionGate, dwell_detect_any


def class_distribution(labels: list[int]) -> dict[int, float]:
    """Calculate class distribution from labels."""
    if not labels:
        return {}
    counts = Counter(labels)
    total = sum(counts.values())
    return {cls: count / total for cls, count in counts.items()}


def calculate_itr(
    num_classes: int, accuracy: float, mean_selection_time_sec: float,
) -> float:
    """Information Transfer Rate (Wolpaw formula), bits/min."""
    if mean_selection_time_sec <= 0 or accuracy <= 0 or num_classes < 2:
        return 0.0
    P, N = accuracy, num_classes
    if P >= 1.0:
        bits = math.log2(N)
    else:
        bits = math.log2(N) + P * math.log2(P) + (1 - P) * math.log2((1 - P) / (N - 1))
    return max(0.0, bits * (60.0 / mean_selection_time_sec))


# ---------------------------------------------------------------------------
# Trial-level metric aggregation (shared by online + offline eval)
# ---------------------------------------------------------------------------


def compute_trial_metrics(
    trials: list[dict[str, Any]],
    background_preds: list[int],
    gate: DecisionGate,
    num_classes: int = 2,
    step_duration_ms: float = 100.0,
    label_mapping: dict[int, str] | None = None,
    background_confs: list[float] | None = None,
) -> dict[str, Any]:
    """Compute trial-level BCI metrics from collected predictions.

    Each trial dict must have ``predictions``, ``label``, ``n_correct``,
    and optionally ``confidences`` (for confidence-threshold re-filtering).
    """
    if not trials:
        return {
            "per_class_accuracy": {}, "per_class_accuracy_named": {},
            "balanced_accuracy": 0.0, "mean_step_accuracy": 0.0,
            "far_per_min": 0.0, "far_false_detections": 0,
            "mean_ttd_ms": None, "n_trials": 0,
            "per_class_trials": {}, "confusion_matrix": [],
        }

    label_mapping = label_mapping or {}

    by_class: dict[int, list[dict]] = {}
    for t in trials:
        by_class.setdefault(t["label"], []).append(t)

    # Step accuracy (raw classifier quality, independent of gate)
    step_accs = [t["n_correct"] / len(t["predictions"])
                 for t in trials if t["predictions"]]
    mean_step_accuracy = sum(step_accs) / len(step_accs) if step_accs else 0.0

    true_labels: list[int] = []
    pred_labels: list[int] = []
    trial_outcomes: list[dict[str, Any]] = []
    ttds: list[float] = []
    class_correct: dict[int, int] = {}
    class_total: dict[int, int] = {}
    all_classes = sorted(by_class.keys())

    for t in trials:
        label = t["label"]
        true_labels.append(label)
        class_total[label] = class_total.get(label, 0) + 1

        # Gate decides: confidence filter + strategy (dwell/majority)
        vote, detect_step = gate.decide(t["predictions"], t.get("confidences"))
        if detect_step is not None:
            ttd_ms: float | None = detect_step * step_duration_ms
            ttds.append(ttd_ms)
        else:
            ttd_ms = None

        # Fallback if gate couldn't decide: majority vote, or -1 (wrong) if
        # all predictions were abstained — never use the ground-truth label.
        if vote is None:
            filtered = [p for p in t["predictions"] if p >= 0]
            vote = Counter(filtered).most_common(1)[0][0] if filtered else -1

        correct = vote == label
        if correct:
            class_correct[label] = class_correct.get(label, 0) + 1

        pred_labels.append(vote)
        trial_outcomes.append({
            "vote": int(vote), "correct": bool(correct), "ttd_ms": ttd_ms,
        })

    per_class_accuracy = {
        cls: class_correct.get(cls, 0) / class_total[cls] for cls in all_classes
    }
    per_class_accuracy_named = {
        label_mapping.get(cls, str(cls)): acc for cls, acc in per_class_accuracy.items()
    }
    balanced_accuracy = (
        sum(per_class_accuracy.values()) / len(per_class_accuracy)
        if per_class_accuracy else 0.0
    )

    # FAR: dwell-based false detections in background
    false_detections = 0
    far_per_min = 0.0
    if gate.use_dwell:
        bg_filtered = gate.filter_predictions(background_preds, background_confs)
        false_detections = dwell_detect_any(bg_filtered, gate.dwell_n)
        bg_duration_s = len(background_preds) * step_duration_ms / 1000.0
        far_per_min = false_detections / (bg_duration_s / 60) if bg_duration_s > 0 else 0.0

    n_correct = sum(1 for o in trial_outcomes if o["correct"])
    accuracy = n_correct / len(trials)

    return {
        "accuracy": round(float(accuracy), 4),
        "per_class_accuracy": per_class_accuracy,
        "per_class_accuracy_named": per_class_accuracy_named,
        "balanced_accuracy": round(float(balanced_accuracy), 4),
        "mean_step_accuracy": round(float(mean_step_accuracy), 4),
        "far_per_min": round(float(far_per_min), 2),
        "far_false_detections": false_detections,
        "mean_ttd_ms": round(float(np.mean(ttds)), 1) if ttds else None,
        "n_trials": len(trials),
        "per_class_trials": {k: len(v) for k, v in by_class.items()},
        "confusion_matrix": confusion_matrix(
            true_labels, pred_labels, labels=all_classes,
        ).tolist() if true_labels else [],
        "trial_outcomes": trial_outcomes,
    }


def optimize_gate(
    per_trial: list[dict[str, Any]],
    bg_preds: list[int],
    num_classes: int = 2,
    step_duration_ms: float = 100.0,
    bg_confs: list[float] | None = None,
) -> dict[str, Any]:
    """Find the DecisionGate params that maximize balanced accuracy.

    Grid search over strategy, dwell_n, and confidence_threshold.
    Returns the best gate config and its metrics.
    """
    if not per_trial:
        return {"gate": DecisionGate().to_dict(), "accuracy": 0.0, "balanced_accuracy": 0.0}

    thresholds = [round(t * 0.05, 2) for t in range(20)]  # 0.0 .. 0.95
    candidates: list[DecisionGate] = []
    for threshold in thresholds:
        candidates.append(DecisionGate(strategy="majority", confidence_threshold=threshold))
        for dwell_n in range(1, 16):
            candidates.append(DecisionGate(
                strategy="dwell", dwell_n=dwell_n, confidence_threshold=threshold,
            ))

    best_score = -1.0
    best_gate = candidates[0]
    best_metrics: dict[str, Any] = {}

    for gate in candidates:
        agg = compute_trial_metrics(
            per_trial, bg_preds, gate,
            num_classes=num_classes, step_duration_ms=step_duration_ms,
            background_confs=bg_confs,
        )
        if agg["balanced_accuracy"] > best_score:
            best_score = agg["balanced_accuracy"]
            best_gate = gate
            best_metrics = agg

    return {
        "gate": best_gate.to_dict(),
        "accuracy": best_metrics.get("accuracy", 0.0),
        "balanced_accuracy": best_metrics.get("balanced_accuracy", 0.0),
    }
