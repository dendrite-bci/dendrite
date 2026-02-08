"""Export benchmark results to CSV and JSON files."""

import csv
import json
from dataclasses import dataclass
from typing import Any

from dendrite.auxiliary.ml_workbench.backend.benchmark_worker import BENCHMARK_SEED
from dendrite.utils.logger_central import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkRow:
    """A single benchmark result row."""

    model: str
    accuracy: str
    time: str
    n_subjects: int
    best_params: dict[str, Any]
    per_subject: list[dict[str, Any]]


def export_benchmark_results(filepath: str, rows: list[BenchmarkRow]) -> str | None:
    """Export benchmark results to CSV or JSON.

    Args:
        filepath: Destination file path (.csv or .json)
        rows: Benchmark result rows to export

    Returns:
        Path to per-subject CSV if generated, or None
    """
    summary = [
        {
            "model": r.model,
            "accuracy": r.accuracy,
            "time": r.time,
            "n_subjects": r.n_subjects,
            "best_params": r.best_params,
        }
        for r in rows
    ]

    all_per_subject = [
        {
            "model": r.model,
            "subject": subj.get("subject", ""),
            "accuracy": subj.get("accuracy", ""),
            "kappa": subj.get("kappa", ""),
            "f1": subj.get("f1", ""),
            "balanced_accuracy": subj.get("balanced_accuracy", ""),
        }
        for r in rows
        for subj in r.per_subject
    ]

    if filepath.endswith(".json"):
        _export_json(filepath, summary, all_per_subject)
        return None

    _export_csv(filepath, summary)

    if all_per_subject:
        per_subj_path = filepath.replace(".csv", "_per_subject.csv")
        _export_per_subject_csv(per_subj_path, all_per_subject)
        return per_subj_path

    return None


def _export_json(
    filepath: str,
    summary: list[dict],
    per_subject: list[dict],
) -> None:
    """Write JSON export with summary and per-subject breakdown."""
    export_data = {
        "summary": summary,
        "per_subject": per_subject,
        "seed": BENCHMARK_SEED,
    }
    with open(filepath, "w") as f:
        json.dump(export_data, f, indent=2)


def _export_csv(filepath: str, summary: list[dict]) -> None:
    """Write summary CSV with serialized best_params."""
    csv_rows = []
    for row in summary:
        csv_row = row.copy()
        csv_row["best_params"] = json.dumps(row.get("best_params", {}))
        csv_rows.append(csv_row)

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "accuracy", "time", "n_subjects", "best_params"]
        )
        writer.writeheader()
        writer.writerows(csv_rows)


def _export_per_subject_csv(filepath: str, per_subject: list[dict]) -> None:
    """Write per-subject breakdown CSV."""
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "subject", "accuracy", "kappa", "f1", "balanced_accuracy"],
        )
        writer.writeheader()
        writer.writerows(per_subject)
