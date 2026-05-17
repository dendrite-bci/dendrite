"""Spawn ReplayStreamer (local H5 or MOABB preset) and wait for completion.

Two-phase by design: the caller starts streaming first so the LSL outlets
become reachable, then starts the Dendrite pipeline against them.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from multiprocessing.synchronize import Event as EventT

from .datasets import SessionSpec


@dataclass
class ReplayHandle:
    streamer: mp.Process
    stop_event: EventT
    info_queue: mp.Queue
    started_at: float
    duration_s: float | None = None
    final_progress: float = 0.0


def start_replay(spec: SessionSpec) -> ReplayHandle:
    """Start the ReplayStreamer subprocess. Returns a handle for wait_for_replay."""
    from dendrite.data.streaming.replay import ReplayStreamer

    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    info_queue: mp.Queue = ctx.Queue()

    if spec.key == "moabb":
        streamer = ReplayStreamer(
            stop_event,
            moabb_preset=spec.moabb_preset,
            moabb_subject=spec.moabb_subject,
            moabb_session=spec.moabb_session,
            enable_event_stream=True,
            info_queue=info_queue,
        )
    else:
        streamer = ReplayStreamer(
            stop_event,
            data_file_path=str(spec.h5_path),
            enable_event_stream=True,
            info_queue=info_queue,
        )
    streamer.start()
    return ReplayHandle(
        streamer=streamer, stop_event=stop_event, info_queue=info_queue,
        started_at=time.monotonic(),
    )


def wait_for_replay(handle: ReplayHandle) -> dict:
    """Block until the streamer reports progress >= 1.0 (or exits)."""
    while True:
        try:
            msg = handle.info_queue.get(timeout=1.0)
        except queue.Empty:
            if not handle.streamer.is_alive():
                break
            continue
        kind = msg.get("type")
        if kind == "duration":
            handle.duration_s = float(msg["value"])
        elif kind == "progress":
            handle.final_progress = float(msg["value"])
            if handle.final_progress >= 1.0:
                break
    handle.streamer.join(timeout=10.0)
    if handle.streamer.is_alive():
        handle.stop_event.set()
        handle.streamer.join(timeout=5.0)
    if handle.streamer.is_alive():
        handle.streamer.terminate()

    return {
        "duration_s": handle.duration_s,
        "wall_s": time.monotonic() - handle.started_at,
        "final_progress": handle.final_progress,
    }


def stop_replay(handle: ReplayHandle) -> None:
    """Stop the streamer early (e.g. on error)."""
    handle.stop_event.set()
    handle.streamer.join(timeout=5.0)
    if handle.streamer.is_alive():
        handle.streamer.terminate()
