"""Read the metrics HDF5 produced by a run and reduce it to a result row."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config


def latest_metrics_file(metrics_dir: Path) -> Path | None:
    """Return the most recently modified `*_metrics.h5` in the given dir."""
    if not metrics_dir.exists():
        return None
    files = sorted(metrics_dir.glob("*_metrics.h5"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _itr_wolpaw(p: float, n_classes: int, trials_per_min: float) -> float:
    """Wolpaw ITR in bits/min. Returns 0 if p <= 1/N (at or below chance)."""
    if n_classes < 2 or trials_per_min <= 0:
        return 0.0
    if p >= 1.0:
        return float(np.log2(n_classes) * trials_per_min)
    if p <= 1.0 / n_classes:
        return 0.0
    bits = (
        np.log2(n_classes)
        + p * np.log2(p)
        + (1 - p) * np.log2((1 - p) / (n_classes - 1))
    )
    return float(bits * trials_per_min)


def _reduce_async_group(grp, n_classes: int, prefix: str) -> tuple[dict, np.ndarray]:
    """Reduce an asynchronous-mode HDF5 group to prefix-keyed metrics.

    Shared by the online async mode (`async_`) and the pretrained-decoder mode
    (`pretrained_`) — both write the same group structure. Returns the metrics
    dict plus the inter-prediction intervals (ms) so the caller can derive
    step-interval percentiles if it wants them.
    """
    ba = grp["balanced_accuracy"][:] if "balanced_accuracy" in grp else np.array([])
    preds = grp["prediction"][:] if "prediction" in grp else np.array([])
    truth = grp["true_label"][:] if "true_label" in grp else np.array([])
    ts = grp["prediction_timestamps"][:] if "prediction_timestamps" in grp else np.array([])
    detected = grp["detected"][:] if "detected" in grp else np.array([])

    valid = (truth >= 0) if truth.size else np.array([], dtype=bool)
    n_valid = int(valid.sum())
    acc_overall = float(np.mean(preds[valid] == truth[valid])) if n_valid else 0.0
    duration_s = float(ts[-1] - ts[0]) if ts.size >= 2 else 0.0
    n_detections = int(detected.sum()) if detected.size else 0
    detections_per_min = (n_detections / duration_s * 60.0) if duration_s > 0 else 0.0
    itr = _itr_wolpaw(acc_overall, n_classes=n_classes, trials_per_min=detections_per_min)
    inter_pred_ms = np.diff(ts) * 1000.0 if ts.size >= 2 else np.array([])

    metrics = {
        f"{prefix}n_steps": int(preds.size),
        f"{prefix}n_with_truth": n_valid,
        f"{prefix}overall_accuracy": acc_overall,
        f"{prefix}final_balanced_accuracy": float(ba[-1]) if ba.size else None,
        f"{prefix}n_detections": n_detections,
        f"{prefix}detections_per_min": detections_per_min,
        f"{prefix}itr_bits_per_min": itr,
        f"{prefix}duration_s": duration_s,
    }
    return metrics, inter_pred_ms


def read_session_row(h5_path: Path, *, n_classes: int = 2) -> dict:
    """Reduce one metrics HDF5 to a per-session result row.

    `n_classes` only affects the Wolpaw ITR calc — the rest of the row is
    dataset-agnostic. The `pretrained_*` keys are present only when the run
    included a `BenchAsync_Pretrained` mode.
    """
    import h5py

    row: dict = {"metrics_file": str(h5_path)}

    with h5py.File(h5_path, "r") as h:
        sync = h.get(config.SYNC_MODE_NAME)
        if sync is not None:
            acc = sync["accuracy"][:] if "accuracy" in sync else np.array([])
            kappa = sync["cohens_kappa"][:] if "cohens_kappa" in sync else np.array([])
            train_pts = (
                sync["training_s_point"][:] if "training_s_point" in sync else np.array([])
            )
            event_types = sync["event_type"][:] if "event_type" in sync else np.array([])
            row.update({
                "sync_n_predictions": int(acc.size),
                "sync_n_train_events": int(train_pts.size),
                "sync_n_epochs": int(event_types.size),
                "sync_final_accuracy": float(acc[-1]) if acc.size else None,
                "sync_mean_accuracy": float(np.mean(acc)) if acc.size else None,
                "sync_final_kappa": float(kappa[-1]) if kappa.size else None,
                "prequential_accuracy": acc.tolist(),
            })

        async_grp = h.get(config.ASYNC_MODE_NAME)
        if async_grp is not None:
            metrics, inter_pred_ms = _reduce_async_group(async_grp, n_classes, "async_")
            row.update(metrics)
            row.update({
                "step_interval_ms_p50": (
                    float(np.percentile(inter_pred_ms, 50)) if inter_pred_ms.size else None
                ),
                "step_interval_ms_p95": (
                    float(np.percentile(inter_pred_ms, 95)) if inter_pred_ms.size else None
                ),
                "step_interval_ms_p99": (
                    float(np.percentile(inter_pred_ms, 99)) if inter_pred_ms.size else None
                ),
            })

        pretrained_grp = h.get(config.PRETRAINED_MODE_NAME)
        if pretrained_grp is not None:
            metrics, _ = _reduce_async_group(pretrained_grp, n_classes, "pretrained_")
            row.update(metrics)

    return row
