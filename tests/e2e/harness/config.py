"""Paths and constants for the e2e test harness."""

from __future__ import annotations

from pathlib import Path

# tests/e2e/harness/config.py -> repo root is parents[3]
DENDRITE_REPO = Path(__file__).resolve().parents[3]
STUDIES_DIR = DENDRITE_REPO / "data" / "studies"

HARNESS_DIR = Path(__file__).resolve().parent
TEMPLATE_CONFIG = HARNESS_DIR / "template_config.json"

# Harness run artefacts (gitignored) — temp configs, server logs, result rows.
OUTPUT_DIR = DENDRITE_REPO / "tests" / "e2e" / "_output"
CONFIG_OUTPUT_DIR = OUTPUT_DIR / "configs"
LOG_DIR = OUTPUT_DIR / "logs"

# Committed regression baselines (numbers, not data — these are version-tracked).
BASELINES_DIR = DENDRITE_REPO / "tests" / "e2e" / "baselines"

# Harness-owned study names. Keep the metrics each path writes in dedicated
# `data/studies/e2e_*/` dirs, isolated from real recordings (the replay *source*
# and the metrics *study* are decoupled). MOABB needs its own too — a single
# fixed study dir no longer fits a configurable preset list.
LOCAL_STUDY_NAME = "e2e_local"
MOABB_STUDY_NAME = "e2e_moabb"
# Where the offline-trained pretrained-decoder fixture is saved (gitignored).
PRETRAINED_STUDY_NAME = "e2e_pretrained"

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8321
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

# Mode-instance names in the vendored template; also the HDF5 group names the
# metrics reader looks for. PRETRAINED is an extra async mode the harness adds
# when given a pretrained decoder (decoder_source: "database").
SYNC_MODE_NAME = "BenchSync"
ASYNC_MODE_NAME = "BenchAsync"
PRETRAINED_MODE_NAME = "BenchAsync_Pretrained"
RECORDING_NAME = "e2e"
