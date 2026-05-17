"""Fixtures for the e2e harness tests.

These tests spawn (or reuse) a live Dendrite backend and replay a dataset
through the real pipeline — they are gated behind the `e2e` marker and
excluded from the default `pytest` run.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.e2e.harness import config as hconfig
from tests.e2e.harness import server as hserver
from tests.e2e.harness.datasets import Dataset, resolve_dataset


@pytest.fixture(scope="session")
def dataset() -> Dataset:
    """Resolve the dataset for this run (local in-house via env var, else MOABB)."""
    return resolve_dataset()


@pytest.fixture(scope="session")
def pretrained_decoder() -> str:
    """Offline-train a CSP+LDA decoder on a *different* AlexMI subject (sub-07)
    and return its saved `.json` path.

    Feeds the harness's `BenchAsync_Pretrained` mode so one run also exercises
    the `decoder_source: "database"` / cross-subject path. Trained in-process
    in ~seconds — no harness run, no chaining, no committed binary.
    """
    import numpy as np

    from dendrite.data.loaders._training_data import load_moabb_for_training
    from dendrite.ml.decoders.decoder import Decoder
    from dendrite.ml.decoders.decoder_schemas import DecoderConfig

    data = load_moabb_for_training(
        {"dataset_code": "AlexMI", "subject": 7, "paradigm": "MotorImagery"}
    )
    X, y = data.X, data.y
    input_shapes = {"eeg": list(X.shape[1:])}
    cfg = DecoderConfig(
        model_type="CSP+LDA",
        num_classes=int(np.max(y) + 1),
        input_shapes=input_shapes,
        sample_rate=512.0,  # AlexMI native rate — matches the harness replay
        epochs=1,
        batch_size=32,
        validation_split=0.0,
        use_early_stopping=False,
    )
    decoder = Decoder(cfg)
    decoder.input_shapes = input_shapes
    decoder.fit(X, y)
    return decoder.save(
        "pretrained_alexmi_sub07", study_name=hconfig.PRETRAINED_STUDY_NAME
    )


@pytest.fixture(scope="session")
def live_backend() -> Iterator[Path | None]:
    """Ensure a backend is running on the harness port.

    Reuses an already-healthy backend (common in dev — don't disturb the
    user's running instance). Otherwise spawns one for the test session and
    tears it down afterwards.

    Yields the server log path when the harness owns the process (so the test
    can assert the log is clean), or None when reusing an external backend.
    """
    if hserver.is_healthy():
        yield None
        return

    hconfig.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = hconfig.LOG_DIR / "server_e2e_session.log"
    proc = hserver.start_server(log_path=log_path)
    try:
        hserver.wait_for_health(timeout=120.0)
        yield log_path
    finally:
        hserver.stop_server(proc)
