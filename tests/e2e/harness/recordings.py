"""Read the raw recording the DataSaver wrote during a harness run.

Parallel to `metrics.py`: where that reads the MetricsSaver's `*_metrics.h5`,
this reads the DataSaver's `*_raw.h5` and round-trips it through the production
loader (`dendrite.data.loaders.load_file`) — so the test exercises the same
path real recordings go through, and proves the file was flushed + closed.
"""

from __future__ import annotations

from pathlib import Path


def latest_recording_file(raw_dir: Path) -> Path | None:
    """Return the most recently modified `*_raw.h5` in the given dir."""
    if not raw_dir.exists():
        return None
    files = sorted(raw_dir.glob("*_raw.h5"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def read_recording_row(h5_path: Path) -> dict:
    """Load a DataSaver recording via the production loader, reduce to a row.

    Loading through `load_file` is the point — it proves the recording is
    well-formed enough for the real pipeline to consume.
    """
    import numpy as np

    from dendrite.data.loaders import load_file

    loaded = load_file(str(h5_path))
    data = loaded.data
    has_nan = bool(np.isnan(data).any() or np.isinf(data).any())
    return {
        "recording_file": str(h5_path),
        "n_samples": int(loaded.n_samples),
        "n_channels": int(loaded.n_channels),
        "sample_rate": float(loaded.sample_rate),
        "n_events": len(loaded.events),
        "duration_s": float(loaded.duration),
        "has_nan": has_nan,
    }
