"""Resolve which dataset the e2e harness runs against.

Resolution order:
  1. DENDRITE_E2E_DATASET=local|moabb        -- force a choice explicitly
  2. DENDRITE_E2E_LOCAL_DIR set + has H5s    -- replay local in-house sessions
  3. otherwise                               -- MOABB BNCI2014_001 (public, auto-downloads)

The env vars are picked up from the process environment; `tests/conftest.py`
loads the repo `.env` file at import time, so they can live there too.

MOABB is the portable default: it needs no in-house data, so a fresh checkout
or CI can run the harness. Point DENDRITE_E2E_LOCAL_DIR at a directory of
training `*_eeg.h5` files to exercise the real in-house recordings instead —
that local dataset is also what the regression baseline is built from.

A resolved `Dataset` has one or more `SessionSpec`s: the local dataset yields
one per H5 file, MOABB one per configured preset x subject. The smoke test runs
the first session; the regression test runs them all and averages.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from . import config

# --- MOABB target -----------------------------------------------------------
# Default: AlexMI = Alexandre motor imagery (16 EEG @ 512 Hz, 3-class MI). Chosen
# for its short per-subject recording (~8 min) — the harness replays in real
# time, so a short dataset keeps the smoke/regression loop iterable. Override
# the preset / subject lists via DENDRITE_E2E_MOABB_PRESETS /
# DENDRITE_E2E_MOABB_SUBJECTS (comma-separated) — e.g. set BNCI2014_001 for a
# fuller (but ~38 min) public benchmark. The dataset then resolves to one
# session per preset x subject. Per-dataset constants (n_classes, epoch window)
# are derived from the MOABB registry, so any registered MI preset works with
# no code change.
#
# Subject 4 is the default deliberately: AlexMI sub-01 sits at chance for
# CSP+LDA (~0.35 offline CV, 3-class chance 0.33), so the chance-aware smoke
# floor wouldn't be meaningful. Sub-04 is the strongest (~0.75 offline CV) —
# its online accuracy clears chance with real headroom.
_DEFAULT_MOABB_PRESETS = ["AlexMI"]
_DEFAULT_MOABB_SUBJECTS = [4]


def _moabb_presets() -> list[str]:
    raw = os.environ.get("DENDRITE_E2E_MOABB_PRESETS", "").strip()
    return [p.strip() for p in raw.split(",") if p.strip()] or _DEFAULT_MOABB_PRESETS


def _moabb_subjects() -> list[int]:
    raw = os.environ.get("DENDRITE_E2E_MOABB_SUBJECTS", "").strip()
    parsed = [int(s) for s in raw.split(",") if s.strip()]
    return parsed or _DEFAULT_MOABB_SUBJECTS


@dataclass
class SessionSpec:
    """One runnable session — enough to build a config, replay it, find metrics."""

    key: str                       # "local" | "moabb"
    study_name: str                # backend study -> data/studies/<study_name>/metrics/
    label: str                     # unique id for temp-config / log / row filenames
    eeg_stream_name: str           # must match what ReplayStreamer broadcasts over LSL
    events_stream_name: str
    n_classes: int
    epoch_tmin: float
    epoch_tmax: float
    streams_ready_timeout: float   # LSL preflight wait (MOABB needs longer: download)
    h5_path: Path | None = None
    moabb_preset: str | None = None
    moabb_subject: int | None = None
    moabb_session: str | None = None

    @property
    def metrics_dir(self) -> Path:
        return config.STUDIES_DIR / self.study_name / "metrics"

    @property
    def recordings_dir(self) -> Path:
        return config.STUDIES_DIR / self.study_name / "raw"

    @property
    def chance_level(self) -> float:
        """Accuracy a degenerate decoder would reach on balanced classes."""
        return 1.0 / self.n_classes


@dataclass
class Dataset:
    """A resolved dataset: its sessions plus regression thresholds + baseline."""

    key: str                       # "local" | "moabb"
    sessions: list[SessionSpec] = field(default_factory=list)
    acc_regression_tol: float = 0.07       # accuracy may drop this much vs baseline
    latency_regression_tol_ms: float = 30.0  # p95 may rise this much vs baseline

    @property
    def chance_level(self) -> float:
        return self.sessions[0].chance_level

    @property
    def baseline_path(self) -> Path:
        return config.BASELINES_DIR / f"{self.key}.json"


def _local_session_spec(h5_path: Path) -> SessionSpec:
    stem = h5_path.stem
    return SessionSpec(
        key="local",
        study_name=config.LOCAL_STUDY_NAME,
        label=f"local-{stem}",
        eeg_stream_name=stem,
        events_stream_name=f"{stem}_Events",
        n_classes=2,
        epoch_tmin=0.5,
        epoch_tmax=4.5,
        streams_ready_timeout=30.0,
        h5_path=h5_path,
    )


def _moabb_session_spec(preset: str, subject: int) -> SessionSpec:
    """Build a MOABB session spec, deriving n_classes and the epoch window from
    the MOABB registry (metadata only — no download)."""
    from dendrite.data.loaders import get_moabb_dataset_info

    info = get_moabb_dataset_info(preset)
    if not info:
        raise RuntimeError(f"Unknown MOABB preset: {preset!r}")
    tmin, tmax = info["interval"]
    return SessionSpec(
        key="moabb",
        study_name=config.MOABB_STUDY_NAME,
        label=f"moabb-{preset}-sub{subject:02d}",
        eeg_stream_name="MOABB_EEG",
        events_stream_name="MOABB_Events",
        n_classes=len(info["events"]),
        epoch_tmin=float(tmin),
        epoch_tmax=float(tmax),
        streams_ready_timeout=300.0,  # first run downloads ~150 MB
        moabb_preset=preset,
        moabb_subject=subject,
        moabb_session=None,
    )


def _find_local_sessions() -> list[Path]:
    """All `*_eeg.h5` training files under DENDRITE_E2E_LOCAL_DIR, if set."""
    local_dir = os.environ.get("DENDRITE_E2E_LOCAL_DIR", "").strip()
    if not local_dir:
        return []
    d = Path(local_dir).expanduser()
    if not d.is_dir():
        return []
    return sorted(d.glob("*_eeg.h5"))


def _local_dataset(h5_files: list[Path]) -> Dataset:
    return Dataset(
        key="local",
        sessions=[_local_session_spec(p) for p in h5_files],
    )


def _moabb_dataset() -> Dataset:
    sessions = [
        _moabb_session_spec(preset, subject)
        for preset in _moabb_presets()
        for subject in _moabb_subjects()
    ]
    return Dataset(key="moabb", sessions=sessions)


def resolve_dataset() -> Dataset:
    """Pick the dataset for this run — see module docstring for the order."""
    forced = os.environ.get("DENDRITE_E2E_DATASET", "").strip().lower()

    if forced == "moabb":
        return _moabb_dataset()
    if forced == "local":
        h5_files = _find_local_sessions()
        if not h5_files:
            raise RuntimeError(
                "DENDRITE_E2E_DATASET=local but DENDRITE_E2E_LOCAL_DIR is unset, "
                "not a directory, or contains no *_eeg.h5 files"
            )
        return _local_dataset(h5_files)
    if forced:
        raise RuntimeError(
            f"DENDRITE_E2E_DATASET must be 'local' or 'moabb', got {forced!r}"
        )

    h5_files = _find_local_sessions()
    return _local_dataset(h5_files) if h5_files else _moabb_dataset()
