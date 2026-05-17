"""Manage the dendrite backend subprocess for e2e runs."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import httpx

from . import config


def is_healthy(timeout: float = 2.0) -> bool:
    """True if a backend is already responding on the harness port."""
    try:
        r = httpx.get(f"{config.SERVER_URL}/api/pipeline/status", timeout=timeout)
        return r.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


def start_server(log_path: Path | None = None) -> subprocess.Popen:
    """Spawn `uv run dendrite` in the dendrite repo. Returns the Popen handle."""
    log = open(log_path, "w") if log_path else subprocess.DEVNULL
    return subprocess.Popen(
        [
            "uv", "run", "dendrite",
            "--host", config.SERVER_HOST,
            "--port", str(config.SERVER_PORT),
        ],
        cwd=str(config.DENDRITE_REPO),
        stdout=log,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        start_new_session=True,
    )


def wait_for_health(timeout: float = 120.0) -> None:
    """Poll until the server responds to /api/pipeline/status, or raise."""
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{config.SERVER_URL}/api/pipeline/status", timeout=2.0)
            if r.status_code == 200:
                return
        except (httpx.HTTPError, OSError) as e:
            last_err = e
        time.sleep(0.5)
    raise RuntimeError(f"Server did not become healthy within {timeout}s: {last_err}")


def stop_server(proc: subprocess.Popen, timeout: float = 10.0) -> None:
    """Send SIGTERM to the process group, fall back to SIGKILL."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5.0)
