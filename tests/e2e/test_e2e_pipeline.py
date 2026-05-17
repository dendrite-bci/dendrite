"""End-to-end tests — drive the live backend over REST + LSL replay.

Two layers, two markers:

  test_pipeline_smoke       @e2e             one session, liveness + above-chance
  test_pipeline_regression  @e2e_regression  all sessions, aggregate vs baseline

Both exercise the seam the in-process tests can't: REST config load, LSL
acquisition, the mode subprocesses, the queue bridge, the metrics saver. Both
are excluded from the default `pytest` run.

  uv run pytest tests/e2e -m e2e -v -s                          # smoke (MOABB on CI)
  DENDRITE_E2E_LOCAL_DIR=<dir> uv run pytest tests/e2e -m e2e_regression -v -s

The regression test diffs against a committed baseline; generate one with
`uv run python -m tests.e2e.harness.runner --update-baseline`.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from tests.e2e.harness.baseline import aggregate_rows, compare, load_baseline
from tests.e2e.harness.datasets import Dataset, SessionSpec
from tests.e2e.harness.runner import run_session

_ERROR_MARKERS = ("ERROR", "Traceback (most recent call last)", "CRITICAL")


def _scan_log_errors(log_path: Path) -> list[str]:
    """Return backend log lines that look like errors or tracebacks."""
    if not log_path.exists():
        return []
    lines = log_path.read_text(errors="replace").splitlines()
    return [ln for ln in lines if any(m in ln for m in _ERROR_MARKERS)]


def _assert_liveness(row: dict, session: SessionSpec, live_backend: Path | None) -> None:
    """The pipeline ran start-to-finish and produced non-degenerate output."""
    assert row["replay"]["final_progress"] == pytest.approx(1.0), (
        f"[{session.label}] replay did not finish: {row['replay']}"
    )
    assert row.get("metrics_file"), f"[{session.label}] backend produced no metrics HDF5"
    assert row.get("sync_n_train_events", 0) >= 1, (
        f"[{session.label}] sync mode never trained a decoder"
    )

    # accuracy above chance — catches a degenerate / all-one-class decoder.
    # max() of the two modes: a degenerate decoder fails both, but a
    # hard-but-working run still passes via the robust mode — the online
    # prequential sync mean can sit near chance on a short dataset (few trials)
    # even when async is solidly above.
    chance = session.chance_level
    sync_acc = row.get("sync_mean_accuracy")
    async_acc = row.get("async_overall_accuracy")
    accs = [a for a in (sync_acc, async_acc) if a is not None]
    assert accs and max(accs) > chance, (
        f"[{session.label}] neither mode beat chance {chance:.3f} — "
        f"sync={sync_acc}, async={async_acc}"
    )

    # recorded data — the DataSaver wrote an intact, loadable recording
    rec = row.get("recording")
    assert rec, f"[{session.label}] backend produced no raw recording"
    assert rec["n_samples"] > 0, f"[{session.label}] recording has no samples"
    assert rec["n_channels"] > 0, f"[{session.label}] recording has no channels"
    assert rec["sample_rate"] > 0, f"[{session.label}] recording has no sample rate"
    assert rec["n_events"] > 0, f"[{session.label}] recording captured no events"
    assert not rec["has_nan"], f"[{session.label}] recording contains NaN/Inf samples"
    replay_dur = row["replay"]["duration_s"]
    if replay_dur is not None:
        tol = max(10.0, 0.10 * replay_dur)
        assert abs(rec["duration_s"] - replay_dur) <= tol, (
            f"[{session.label}] recording duration {rec['duration_s']:.1f}s diverges "
            f"from replay {replay_dur:.1f}s by more than {tol:.1f}s"
        )

    # broadcast output — the PredictionStream outlet carried structurally valid frames
    pred = row.get("predictions")
    assert pred, f"[{session.label}] no prediction-stream summary"
    assert pred["resolved"], (
        f"[{session.label}] PredictionStream LSL outlet never appeared"
        + (f" — {pred['error']}" if pred.get("error") else "")
    )
    assert pred["stream_type"] == "PredictionStream", (
        f"[{session.label}] unexpected output stream type {pred['stream_type']!r}"
    )
    assert pred["channel_format"] == "string" and pred["channel_count"] == 1, (
        f"[{session.label}] unexpected output stream shape: "
        f"{pred['channel_count']}ch {pred['channel_format']!r}"
    )
    assert pred["n_frames"] > 0, f"[{session.label}] no prediction frames broadcast"
    assert pred["n_invalid"] == 0, (
        f"[{session.label}] {pred['n_invalid']} malformed prediction frame(s):\n"
        + "\n".join(pred["invalid_samples"])
    )
    # the wire output should track the async steps the metrics recorded
    async_steps = row.get("async_n_steps", 0)
    if async_steps:
        assert pred["n_async"] >= async_steps * 0.8, (
            f"[{session.label}] only {pred['n_async']} async frames broadcast "
            f"vs {async_steps} async steps computed"
        )

    # clean backend log — only checkable when the harness owns the server
    if live_backend is not None:
        errors = _scan_log_errors(live_backend)
        assert not errors, (
            f"[{session.label}] backend log has {len(errors)} error/traceback line(s):\n"
            + "\n".join(errors[:20])
        )


def _assert_pretrained(row: dict, session: SessionSpec) -> None:
    """The `BenchAsync_Pretrained` mode loaded the pretrained decoder and ran.

    For an async `decoder_source: "database"` mode, producing predictions at
    all is only possible if load -> channel-validation -> activate all
    succeeded — so a non-empty pretrained group proves the cross-subject
    load path works. Cross-subject MI accuracy is near chance, so accuracy is
    recorded but not gated.
    """
    n_steps = row.get("pretrained_n_steps")
    assert n_steps is not None, (
        f"[{session.label}] no BenchAsync_Pretrained metrics group — the "
        f"pretrained mode never ran"
    )
    assert n_steps > 0, (
        f"[{session.label}] pretrained mode produced no prediction steps — the "
        f"decoder likely failed to load or activate"
    )
    assert row.get("pretrained_n_with_truth", 0) > 0, (
        f"[{session.label}] pretrained mode produced no ground-truth-matched "
        f"predictions"
    )
    assert row.get("pretrained_overall_accuracy") is not None, (
        f"[{session.label}] pretrained mode recorded no accuracy"
    )


@pytest.mark.e2e
def test_pipeline_smoke(
    dataset: Dataset, live_backend: Path | None, pretrained_decoder: str
) -> None:
    """One session end-to-end: configure -> replay -> sync+async -> metrics.

    Also adds a `BenchAsync_Pretrained` mode (`decoder_source: "database"`)
    loading a decoder trained offline on a *different* AlexMI subject — one run
    covers the online path *and* the pretrained-load / cross-subject path.
    """
    session = dataset.sessions[0]
    row = run_session(
        session, manage_server=False, pretrained_decoder_path=pretrained_decoder
    )
    print(f"\nsmoke row ({session.label}):")
    for k, v in row.items():
        if k != "prequential_accuracy":
            print(f"  {k}: {v}")

    _assert_liveness(row, session, live_backend)
    _assert_pretrained(row, session)


@pytest.mark.e2e_regression
def test_pipeline_regression(dataset: Dataset, live_backend: Path | None) -> None:
    """All sessions end-to-end, aggregate diffed against the committed baseline.

    The accuracy floor here is relative — `baseline - tolerance` — so a real
    decoder regression fails even though it would clear the above-chance smoke
    check. Skips if no baseline has been committed for the resolved dataset.
    """
    baseline = load_baseline(dataset.baseline_path)
    if baseline is None:
        pytest.skip(
            f"no regression baseline at {dataset.baseline_path} — generate one with "
            f"`uv run python -m tests.e2e.harness.runner --update-baseline`"
        )

    rows: list[dict] = []
    for session in dataset.sessions:
        row = run_session(session, manage_server=False)
        rows.append(row)
        _assert_liveness(row, session, live_backend)

    # warn (don't fail) if the run's session set diverges from the baseline's —
    # the comparison is then not strictly apples-to-apples
    baseline_sessions = set(baseline.get("sessions", []))
    current_sessions = {row["label"] for row in rows}
    if baseline_sessions and baseline_sessions != current_sessions:
        warnings.warn(
            f"session set differs from baseline {dataset.baseline_path.name}: "
            f"baseline={sorted(baseline_sessions)} current={sorted(current_sessions)} "
            f"— regenerate the baseline with `--update-baseline` if this is intentional",
            stacklevel=2,
        )

    current = aggregate_rows(rows)
    print(f"\nregression aggregate ({dataset.key}, {len(rows)} session(s)):")
    print(f"  baseline: {baseline.get('metrics')}")
    print(f"  current:  {current}")

    violations = compare(
        baseline, current,
        acc_tol=dataset.acc_regression_tol,
        latency_tol_ms=dataset.latency_regression_tol_ms,
    )
    assert not violations, (
        f"{len(violations)} regression(s) vs baseline {dataset.baseline_path.name}:\n"
        + "\n".join(violations)
    )
