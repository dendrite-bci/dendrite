"""Committed regression baselines for the e2e harness.

A baseline is a small JSON file (numbers, not data — version-tracked) holding
the known-good metrics for a dataset. `compare()` diffs a fresh run against it
within a tolerance band, so real accuracy / latency regressions fail the test.

The local in-house dataset is the primary baseline: generate it on a machine
that has the data with `python -m tests.e2e.harness.runner --update-baseline`,
then commit `tests/e2e/baselines/local.json`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean

# Metrics the baseline tracks. Accuracy metrics regress downward; latency upward.
_ACC_METRICS = ("sync_mean_accuracy", "async_overall_accuracy")
_LATENCY_METRICS = ("step_interval_ms_p95",)


def aggregate_rows(rows: list[dict]) -> dict:
    """Reduce per-session result rows to the metrics the baseline tracks.

    Accuracy is averaged across sessions (damps single-run online variance);
    latency p95 is taken as the worst (max) across sessions.
    """
    def _mean(metric: str) -> float | None:
        vals = [r[metric] for r in rows if r.get(metric) is not None]
        return float(fmean(vals)) if vals else None

    def _max(metric: str) -> float | None:
        vals = [r[metric] for r in rows if r.get(metric) is not None]
        return float(max(vals)) if vals else None

    return {
        **{m: _mean(m) for m in _ACC_METRICS},
        **{m: _max(m) for m in _LATENCY_METRICS},
    }


def load_baseline(path: Path) -> dict | None:
    """Load a committed baseline, or None if it doesn't exist yet."""
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_baseline(
    path: Path, *, dataset_key: str, sessions: list[str], metrics: dict,
) -> None:
    """Write a baseline file (the caller is expected to commit it).

    `sessions` is the list of session labels the aggregate was built from — it
    is recorded so a later run can detect a config/baseline mismatch (e.g. a
    changed `DENDRITE_E2E_MOABB_PRESETS`).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "dataset": dataset_key,
        "generated_at": datetime.now(UTC).isoformat(),
        "n_sessions": len(sessions),
        "sessions": sessions,
        "metrics": metrics,
    }, indent=2))


def compare(
    baseline: dict, current: dict, *, acc_tol: float, latency_tol_ms: float,
) -> list[str]:
    """Return a list of regression violations (empty == within tolerance)."""
    violations: list[str] = []
    base = baseline.get("metrics", {})

    for metric in _ACC_METRICS:
        bv, cv = base.get(metric), current.get(metric)
        if bv is None or cv is None:
            continue
        if cv < bv - acc_tol:
            violations.append(
                f"{metric}: {cv:.4f} < baseline {bv:.4f} - tol {acc_tol:.4f} "
                f"(floor {bv - acc_tol:.4f})"
            )

    for metric in _LATENCY_METRICS:
        bv, cv = base.get(metric), current.get(metric)
        if bv is None or cv is None:
            continue
        if cv > bv + latency_tol_ms:
            violations.append(
                f"{metric}: {cv:.2f}ms > baseline {bv:.2f}ms + tol "
                f"{latency_tol_ms:.2f}ms (ceiling {bv + latency_tol_ms:.2f}ms)"
            )

    return violations
