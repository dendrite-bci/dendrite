"""
ML model regression tests.

Two-tier testing strategy:
- Smoke tests: Fast CI checks (2 epochs, 1 subject) - verify training works
- Full benchmarks: Periodic validation (100 epochs, all subjects) - verify accuracy vs baselines

Usage:
    # Smoke tests (CI)
    pytest tests/regression/ -m smoke -v

    # Full benchmarks (nightly)
    pytest tests/regression/ -m benchmark -v

    # Update baselines (manual)
    pytest tests/regression/test_model_regression.py::test_update_baselines -s -m manual
"""

import json
from pathlib import Path

import pytest

from dendrite.auxiliary.ml_workbench.backend import (
    BENCHMARK_SEED,
    run_benchmark,
    set_reproducibility_seed,
)

# Baselines file path
BASELINES_PATH = Path(__file__).parent / "baselines" / "regression_baselines.json"

# Models to test
REGRESSION_MODELS = ["CSP+LDA", "EEGNet", "BDEEGNet"]


def load_baselines() -> dict:
    """Load baseline accuracy thresholds from JSON."""
    with open(BASELINES_PATH) as f:
        return json.load(f)


def save_baselines(baselines: dict) -> None:
    """Save baseline accuracy thresholds to JSON."""
    with open(BASELINES_PATH, "w") as f:
        json.dump(baselines, f, indent=2)
    print(f"Baselines saved to: {BASELINES_PATH}")


@pytest.mark.parametrize("model_name", REGRESSION_MODELS)
def test_model_training_smoke(model_name: str) -> None:
    """
    Fast smoke test to verify model training works.

    Runs minimal training (2 epochs, 1 subject) to catch:
    - Import errors
    - Model initialization failures
    - Training pipeline breaks
    - Data shape mismatches

    Does NOT verify accuracy - just that training completes without errors.
    """
    set_reproducibility_seed(BENCHMARK_SEED)
    baselines = load_baselines()

    smoke_config = baselines["smoke_test_config"]
    dataset = baselines["dataset"]

    result = run_benchmark(
        dataset_name=dataset,
        models=[model_name],
        eval_type=baselines["eval_type"],
        subjects=smoke_config["subjects"],
        epochs=smoke_config["epochs"],
    )

    # Verify training completed without errors
    summary = result["summary"][0]
    assert "error" not in summary, f"Training failed: {summary.get('error')}"
    assert summary["accuracy"] > 0, "Model produced zero accuracy"

    print(f"\n{model_name} smoke test passed")
    print(f"  Accuracy: {summary['accuracy']*100:.1f}%")
    print(f"  Time: {summary['train_time']:.1f}s")


@pytest.mark.slow
@pytest.mark.parametrize("model_name", REGRESSION_MODELS)
def test_model_accuracy_regression(model_name: str) -> None:
    """
    Full benchmark test to verify model accuracy hasn't regressed.

    Runs complete evaluation (100 epochs, all subjects) and compares
    against baseline accuracy thresholds. Fails if accuracy drops
    more than the configured tolerance (default: 10%).
    """
    set_reproducibility_seed(BENCHMARK_SEED)
    baselines = load_baselines()

    model_baseline = baselines["models"].get(model_name)
    if model_baseline is None:
        pytest.skip(f"No baseline configured for {model_name}")

    baseline_accuracy = model_baseline.get("accuracy")
    if baseline_accuracy is None:
        # No baseline established yet - run and report
        full_config = baselines["full_benchmark_config"]
        result = run_benchmark(
            dataset_name=baselines["dataset"],
            models=[model_name],
            eval_type=baselines["eval_type"],
            subjects=full_config["subjects"],
            epochs=full_config["epochs"],
        )
        summary = result["summary"][0]
        if "error" not in summary:
            print(f"\n{model_name} baseline not set. Current accuracy: {summary['accuracy']*100:.1f}%")
            print("Run test_update_baselines to establish baseline.")
        pytest.skip(
            f"No baseline established for {model_name}. "
            f"Current accuracy: {summary['accuracy']*100:.1f}%"
        )

    tolerance_percent = model_baseline.get("tolerance_percent", 10)

    full_config = baselines["full_benchmark_config"]
    dataset = baselines["dataset"]

    result = run_benchmark(
        dataset_name=dataset,
        models=[model_name],
        eval_type=baselines["eval_type"],
        subjects=full_config["subjects"],
        epochs=full_config["epochs"],
    )

    summary = result["summary"][0]
    assert "error" not in summary, f"Training failed: {summary.get('error')}"

    current_accuracy = summary["accuracy"]
    min_acceptable = baseline_accuracy * (1 - tolerance_percent / 100)

    print(f"\n{model_name} regression test results:")
    print(f"  Baseline: {baseline_accuracy*100:.1f}%")
    print(f"  Current:  {current_accuracy*100:.1f}%")
    print(f"  Minimum:  {min_acceptable*100:.1f}% (tolerance: {tolerance_percent}%)")
    print(f"  Delta:    {(current_accuracy - baseline_accuracy)*100:+.1f}%")

    assert current_accuracy >= min_acceptable, (
        f"{model_name} accuracy regression detected! "
        f"Current: {current_accuracy*100:.1f}%, "
        f"Baseline: {baseline_accuracy*100:.1f}%, "
        f"Minimum acceptable: {min_acceptable*100:.1f}%"
    )


def test_update_baselines() -> None:
    """
    Utility to establish or update baseline accuracy thresholds.

    Runs full benchmark for all models and updates the baselines file.
    Should be run manually when intentionally improving models.

    Usage:
        pytest tests/regression/test_model_regression.py::test_update_baselines -s -m manual
    """
    set_reproducibility_seed(BENCHMARK_SEED)
    baselines = load_baselines()

    full_config = baselines["full_benchmark_config"]

    print("\nRunning full benchmark to update baselines...")
    print("=" * 60)

    result = run_benchmark(
        dataset_name=baselines["dataset"],
        models=REGRESSION_MODELS,
        eval_type=baselines["eval_type"],
        subjects=full_config["subjects"],
        epochs=full_config["epochs"],
    )

    print("\n" + "=" * 60)
    print("UPDATING BASELINES")
    print("=" * 60)

    for summary in result["summary"]:
        model_name = summary["model"]
        if "error" in summary:
            print(f"  {model_name}: SKIPPED (error: {summary['error']})")
            continue

        old_accuracy = baselines["models"].get(model_name, {}).get("accuracy")
        new_accuracy = summary["accuracy"]

        if model_name not in baselines["models"]:
            baselines["models"][model_name] = {"tolerance_percent": 10}

        baselines["models"][model_name]["accuracy"] = round(new_accuracy, 3)

        if "note" in baselines["models"][model_name]:
            del baselines["models"][model_name]["note"]

        if old_accuracy is None:
            print(f"  {model_name}: NEW baseline {new_accuracy*100:.1f}%")
        else:
            delta = (new_accuracy - old_accuracy) * 100
            print(f"  {model_name}: {old_accuracy*100:.1f}% -> {new_accuracy*100:.1f}% ({delta:+.1f}%)")

    save_baselines(baselines)
    print("\nBaselines updated successfully.")
