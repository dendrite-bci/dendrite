"""Capture and structurally verify the backend's broadcast prediction stream.

While the pipeline runs it publishes a `PredictionStream` LSL outlet — one JSON
frame per prediction (~10 Hz async, per-trial sync). This receiver opens an LSL
inlet against that outlet, collects every frame, and validates each one against
the *production* packet dataclasses (`ModeOutputPacket` + `AsyncPrediction` /
`SyncPrediction`) — so a regression in the output-protocol layer (no outlet,
wrong metadata, malformed payload, wire/dataclass drift) is caught.

Mirrors the inlet pattern in `src/dendrite/data/acquisition.py`
(`resolve_byprop` -> `StreamInlet` -> `pull_sample` loop).
"""

from __future__ import annotations

import json
import threading

# Expected outlet metadata — mirrors PREDICTION_STREAM_INFO in
# src/dendrite/data/streaming/output_protocol_manager.py
PREDICTION_STREAM_NAME = "PredictionStream"
PREDICTION_STREAM_TYPE = "PredictionStream"

# pylsl channel_format() returns an int enum; map the ones we might see.
_LSL_FORMATS = {
    0: "undefined", 1: "float32", 2: "double64", 3: "string",
    4: "int32", 5: "int16", 6: "int8", 7: "int64",
}


def _validate_frame(raw: str, packet_cls, data_validators: dict) -> tuple[str | None, dict | None]:
    """Return (error, parsed). `error` is None when `raw` is a structurally
    valid `ModeOutputPacket` carrying a known prediction dataclass."""
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        return f"invalid JSON ({e})", None
    if not isinstance(obj, dict):
        return f"frame is {type(obj).__name__}, not an object", None
    # reconstruct the production packet dataclass — catches missing/extra/renamed keys
    try:
        packet_cls(**obj)
    except TypeError as e:
        return f"not a ModeOutputPacket ({e})", obj
    data = obj["data"]
    if not isinstance(data, dict):
        return f"data is {type(data).__name__}, not an object", obj
    # Only `prediction`-type frames carry an Async/SyncPrediction payload. The
    # same outlet also carries other ModeOutputPacket types (e.g.
    # `training_point`) with their own `data` shape — those are valid frames,
    # just not predictions, so don't schema-check them against a prediction.
    if obj.get("type") == "prediction":
        mode_type = obj["mode_type"]
        validator = data_validators.get(mode_type)
        if validator is None:
            return f"unknown mode_type {mode_type!r}", obj
        try:
            validator(**data)
        except TypeError as e:
            return f"data does not match {mode_type} prediction schema ({e})", obj
    return None, obj


class PredictionReceiver:
    """LSL inlet that collects + structurally validates the prediction stream.

    Lifecycle: `.start()` after the pipeline is up, `.stop()` once the replay is
    done. `.stop()` is idempotent and safe even if `.start()` never ran.
    """

    def __init__(self, stream_type: str = PREDICTION_STREAM_TYPE, resolve_timeout: float = 2.0):
        self.stream_type = stream_type
        # per-attempt resolve timeout — the receiver resolve-loops for the whole
        # run (the prediction outlet appears late, see _run), so this is just
        # the poll granularity, not a give-up deadline.
        self.resolve_timeout = resolve_timeout
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._frames: list[tuple[float, str]] = []
        self._meta: dict = {}
        self._error: str | None = None
        self._started = False

    def start(self) -> None:
        self._started = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            from pylsl import StreamInlet, resolve_byprop

            # The backend creates the prediction outlet *lazily* — only on the
            # first pushed prediction, which is minutes into a run (after the
            # sync->async decoder handoff). So resolve-loop for the whole run
            # rather than giving up after one timeout.
            info = None
            while not self._stop.is_set():
                streams = resolve_byprop("type", self.stream_type, timeout=self.resolve_timeout)
                if streams:
                    info = streams[0]
                    break
            if info is None:
                self._error = (
                    f"stop requested before an LSL stream with "
                    f"type={self.stream_type!r} appeared"
                )
                return
            self._meta = {
                "stream_name": info.name(),
                "stream_type": info.type(),
                "channel_count": info.channel_count(),
                "channel_format": _LSL_FORMATS.get(
                    info.channel_format(), str(info.channel_format())
                ),
            }
            inlet = StreamInlet(info, max_buflen=360)
            try:
                while not self._stop.is_set():
                    sample, ts = inlet.pull_sample(timeout=0.5)
                    if sample:
                        self._frames.append((ts, sample[0]))
                # drain whatever is still buffered
                while True:
                    sample, ts = inlet.pull_sample(timeout=0.0)
                    if not sample:
                        break
                    self._frames.append((ts, sample[0]))
            finally:
                inlet.close_stream()
        except Exception as e:
            # surface any failure (import, resolve, inlet) in the summary
            self._error = f"{type(e).__name__}: {e}"

    def stop(self) -> dict:
        if not self._started:
            return _empty_summary()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        return self._summarize()

    def _summarize(self) -> dict:
        from dendrite.processing.modes.asynchronous_mode import AsyncPrediction
        from dendrite.processing.modes.base_mode import ModeOutputPacket
        from dendrite.processing.modes.synchronous_mode import SyncPrediction

        data_validators = {"asynchronous": AsyncPrediction, "synchronous": SyncPrediction}
        n_async = n_sync = n_other = n_invalid = 0
        invalid_samples: list[str] = []
        for _ts, raw in self._frames:
            err, parsed = _validate_frame(raw, ModeOutputPacket, data_validators)
            if err is not None:
                n_invalid += 1
                if len(invalid_samples) < 5:
                    invalid_samples.append(f"{err}: {raw[:200]}")
                continue
            assert parsed is not None  # narrows the union for type checking
            if parsed.get("type") != "prediction":
                n_other += 1  # valid frame, just not a prediction (e.g. training_point)
            elif parsed["mode_type"] == "asynchronous":
                n_async += 1
            elif parsed["mode_type"] == "synchronous":
                n_sync += 1
        return {
            "resolved": bool(self._meta),
            "error": self._error,
            "stream_name": self._meta.get("stream_name"),
            "stream_type": self._meta.get("stream_type"),
            "channel_count": self._meta.get("channel_count"),
            "channel_format": self._meta.get("channel_format"),
            "n_frames": len(self._frames),
            "n_async": n_async,
            "n_sync": n_sync,
            "n_other": n_other,
            "n_invalid": n_invalid,
            "invalid_samples": invalid_samples,
        }


def _empty_summary() -> dict:
    return {
        "resolved": False,
        "error": "receiver was never started",
        "stream_name": None,
        "stream_type": None,
        "channel_count": None,
        "channel_format": None,
        "n_frames": 0,
        "n_async": 0,
        "n_sync": 0,
        "n_other": 0,
        "n_invalid": 0,
        "invalid_samples": [],
    }
