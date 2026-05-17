"""Drive e2e sessions end-to-end through the live Dendrite backend.

Per session:
  configure -> spawn replay -> wait for LSL -> start pipeline -> wait for
  replay to finish -> stop pipeline -> read the metrics HDF5 -> return a row.

Run standalone (resolves the dataset from env vars, see datasets.py):
  uv run python -m tests.e2e.harness.runner                    # spawn a backend
  uv run python -m tests.e2e.harness.runner --no-server        # reuse :8321
  uv run python -m tests.e2e.harness.runner --limit 1          # first session only
  uv run python -m tests.e2e.harness.runner --update-baseline  # write the baseline
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import config
from .baseline import aggregate_rows, save_baseline
from .client import DendriteClient
from .datasets import Dataset, SessionSpec, resolve_dataset
from .metrics import latest_metrics_file, read_session_row
from .receiver import PredictionReceiver
from .recordings import latest_recording_file, read_recording_row
from .replay import start_replay, stop_replay, wait_for_replay
from .server import is_healthy, start_server, stop_server, wait_for_health
from .template import build_config, write_temp_config


def _wait_for_streams_ready(client: DendriteClient, timeout: float) -> None:
    """Poll preflight until streams_reachable passes (LSL needs time to advertise)."""
    deadline = time.monotonic() + timeout
    last: dict | None = None
    while time.monotonic() < deadline:
        last = client.preflight()
        checks = {c["id"]: c for c in last.get("checks", [])}
        sr = checks.get("streams_reachable")
        if sr and sr.get("passed"):
            return
        time.sleep(0.5)
    raise RuntimeError(f"Streams never became reachable within {timeout}s: {last}")


def run_session(
    session: SessionSpec,
    *,
    manage_server: bool = True,
    log_path: Path | None = None,
    pretrained_decoder_path: str | Path | None = None,
) -> dict:
    """Run one session and return its reduced metrics row.

    When `manage_server` is True the harness spawns and tears down its own
    backend (capturing stdout/stderr to `log_path`). When False it assumes a
    backend is already listening on the harness port.

    `pretrained_decoder_path` adds a `BenchAsync_Pretrained` mode to the run
    (`decoder_source: "database"`) — the row then carries `pretrained_*` keys.
    """
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    server = None
    if manage_server:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = log_path or config.LOG_DIR / f"server_{session.label}.log"
        server = start_server(log_path=log_path)
        wait_for_health(timeout=120.0)

    try:
        cfg = build_config(session, pretrained_decoder_path)
        cfg_path = write_temp_config(cfg, session)

        with DendriteClient() as client:
            client.load_config(cfg_path)

            metrics_before = (
                {p.name for p in session.metrics_dir.glob("*_metrics.h5")}
                if session.metrics_dir.exists()
                else set()
            )
            recordings_before = (
                {p.name for p in session.recordings_dir.glob("*_raw.h5")}
                if session.recordings_dir.exists()
                else set()
            )

            replay_handle = start_replay(session)
            receiver = PredictionReceiver()
            replay_info: dict | None = None
            pipeline_started = False
            prediction_summary: dict = {}
            try:
                _wait_for_streams_ready(client, timeout=session.streams_ready_timeout)
                client.start_pipeline()
                pipeline_started = True
                receiver.start()
                replay_info = wait_for_replay(replay_handle)
            finally:
                prediction_summary = receiver.stop()
                if replay_info is None:
                    stop_replay(replay_handle)
                if pipeline_started:
                    client.stop_pipeline()

            time.sleep(2.0)  # let the metrics saver flush

        metrics_file = latest_metrics_file(session.metrics_dir)
        if metrics_file is None or metrics_file.name in metrics_before:
            raise RuntimeError("No new metrics HDF5 produced for this run")

        recording_file = latest_recording_file(session.recordings_dir)
        if recording_file is None or recording_file.name in recordings_before:
            raise RuntimeError("No new raw recording produced for this run")

        row = read_session_row(metrics_file, n_classes=session.n_classes)
        row["dataset"] = session.key
        row["label"] = session.label
        row["replay"] = {
            "duration_s": replay_info["duration_s"],
            "wall_s": replay_info["wall_s"],
            "final_progress": replay_info["final_progress"],
        }
        row["recording"] = read_recording_row(recording_file)
        row["predictions"] = prediction_summary
        return row

    finally:
        if server is not None:
            stop_server(server)


def run_dataset(
    dataset: Dataset, *, manage_server: bool = True, limit: int | None = None,
) -> list[dict]:
    """Run every session in the dataset, returning one row per session.

    The backend is spawned once (when `manage_server`) and reused across all
    sessions — sessions are sequential (they share the fixed harness port).
    """
    sessions = dataset.sessions[:limit] if limit else dataset.sessions
    if not sessions:
        raise RuntimeError(f"dataset {dataset.key!r} has no sessions")

    server = None
    if manage_server:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = config.LOG_DIR / f"server_{dataset.key}.log"
        server = start_server(log_path=log_path)
        wait_for_health(timeout=120.0)

    try:
        rows: list[dict] = []
        for session in sessions:
            print(f"\n=== {session.label} ===")
            rows.append(run_session(session, manage_server=False))
        return rows
    finally:
        if server is not None:
            stop_server(server)


def main() -> None:
    p = argparse.ArgumentParser(description="Run the Dendrite e2e harness.")
    p.add_argument(
        "--no-server", action="store_true",
        help="Reuse a backend already running on the harness port",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Run only the first N sessions",
    )
    p.add_argument(
        "--update-baseline", action="store_true",
        help="Write the aggregate to tests/e2e/baselines/<dataset>.json",
    )
    args = p.parse_args()

    dataset = resolve_dataset()
    print(f"Resolved dataset: {dataset.key} ({len(dataset.sessions)} session(s))")

    manage = not (args.no_server or is_healthy())
    if not manage:
        print("Reusing the backend already running on :8321")

    rows = run_dataset(dataset, manage_server=manage, limit=args.limit)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for row in rows:
        out = config.OUTPUT_DIR / f"row_{row['label']}.json"
        out.write_text(json.dumps(row, indent=2, default=str))

    aggregate = aggregate_rows(rows)
    print(f"\nAggregate over {len(rows)} session(s):")
    print(json.dumps(aggregate, indent=2))

    if args.update_baseline:
        save_baseline(
            dataset.baseline_path,
            dataset_key=dataset.key,
            sessions=[row["label"] for row in rows],
            metrics=aggregate,
        )
        print(f"\nBaseline written to {dataset.baseline_path}")
    else:
        print(f"\n{len(rows)} row(s) written to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
