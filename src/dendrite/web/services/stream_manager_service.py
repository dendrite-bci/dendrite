"""Stream Manager Service — manages DataStreamer subprocesses."""

import multiprocessing
import queue
import uuid
from multiprocessing.queues import Queue as MpQueue
from multiprocessing.synchronize import Event as MpEvent
from typing import Any

from dendrite.data.loaders import is_supported_format
from dendrite.data.loaders.moabb_discovery import discover_moabb_datasets
from dendrite.data.storage.database import RecordingRepository
from dendrite.data.streaming.replay import ReplayStreamer
from dendrite.utils.logger_central import get_logger


class StreamManagerService:
    """Manages DataStreamer processes for file and MOABB streams."""

    def __init__(self, recording_repo: RecordingRepository | None = None):
        self.logger = get_logger("StreamManagerService")
        self._recording_repo = recording_repo
        self._streamers: dict[str, ReplayStreamer] = {}
        self._stop_events: dict[str, MpEvent] = {}
        self._info_queues: dict[str, MpQueue[Any]] = {}
        self._progress: dict[str, float] = {}
        self._configs: dict[str, dict] = {}  # store config for status display

    def start_stream(self, config: dict) -> str:
        """Start a ReplayStreamer subprocess.

        Config keys:
            source: 'file' | 'moabb'
            path: str (for file)
            dataset: str (for moabb)
            subject: int (for moabb)
            session: str | None (for moabb)
            enable_events: bool
            stream_name: str | None
        """
        stream_id = str(uuid.uuid4())[:8]
        stop_event = multiprocessing.Event()
        info_queue = multiprocessing.Queue()

        source = config.get("source", "file")

        streamer = ReplayStreamer(
            stop_event,
            data_file_path=config.get("path", ""),
            stream_name_prefix=config.get("stream_name"),
            moabb_preset=config.get("dataset") if source == "moabb" else None,
            moabb_subject=config.get("subject"),
            moabb_session=config.get("session"),
            enable_event_stream=config.get("enable_events", False),
            info_queue=info_queue,
        )
        streamer.daemon = True
        streamer.start()

        self._streamers[stream_id] = streamer
        self._stop_events[stream_id] = stop_event
        self._info_queues[stream_id] = info_queue
        self._progress[stream_id] = 0.0
        self._configs[stream_id] = {
            k: v for k, v in config.items() if v is not None and v != ""
        }

        self.logger.info(f"Started replay {stream_id}: {source}")
        return stream_id

    def stop_stream(self, stream_id: str) -> None:
        """Stop a specific streamer."""
        stop_event = self._stop_events.get(stream_id)
        if stop_event:
            stop_event.set()
        streamer = self._streamers.get(stream_id)
        if streamer and streamer.is_alive():
            streamer.join(timeout=3.0)
            if streamer.is_alive():
                streamer.terminate()
        self._cleanup_stream(stream_id)
        self.logger.info(f"Stopped stream {stream_id}")

    def get_status(self) -> list[dict]:
        """Return status of all streamers."""
        status = []
        dead_ids = []
        for stream_id, streamer in self._streamers.items():
            # Drain progress queue
            self._drain_progress(stream_id)
            running = streamer.is_alive()
            if not running:
                dead_ids.append(stream_id)
            info = {
                "id": stream_id,
                "running": running,
                "progress": self._progress.get(stream_id, 0.0),
                **self._configs.get(stream_id, {}),
            }
            status.append(info)
        # Clean up dead streamers
        for sid in dead_ids:
            self._cleanup_stream(sid)
        return status

    def list_moabb_datasets(self) -> list[dict]:
        """List available MOABB datasets."""
        try:
            configs = discover_moabb_datasets()
        except Exception as e:
            self.logger.warning(f"MOABB discovery failed: {e}")
            return []

        return [
            {
                "name": ds["code"],
                "paradigm": ds["paradigm"],
                "n_subjects": ds["n_subjects"],
                "subjects": ds["subjects"][:20],
                "events": ds["events"],
            }
            for ds in configs
        ]

    def list_internal_datasets(self) -> list[dict]:
        """List recordings from the database for stream picker."""
        results = []
        if not self._recording_repo:
            return results
        try:
            for rec in self._recording_repo.get_all_recordings():
                results.append({
                    "id": f"rec_{rec['recording_id']}",
                    "source_type": "recording",
                    "name": rec["recording_name"],
                    "file_path": rec["hdf5_file_path"],
                    "subject": rec.get("subject_id", ""),
                    "session": rec.get("session_id", ""),
                    "study": rec.get("study_name", ""),
                })
        except Exception as e:
            self.logger.warning(f"Failed to list internal datasets: {e}")

        return results

    def get_file_info(self, path: str) -> dict[str, Any]:
        """Get metadata for a data file."""
        if not is_supported_format(path):
            return {"error": f"Unsupported format: {path}"}
        try:
            from dendrite.data.loaders import load_file

            loaded = load_file(path)
            return {
                "path": path,
                "duration_s": loaded.duration,
                "sample_rate": loaded.sample_rate,
                "n_channels": loaded.n_channels,
                "channel_names": loaded.channel_names[:20],
                "n_events": len(loaded.events),
                "event_id": loaded.event_id,
            }
        except Exception as e:
            return {"error": str(e)}

    def stop_all(self) -> None:
        """Stop all active streamers."""
        for stream_id in list(self._streamers.keys()):
            self.stop_stream(stream_id)

    def _drain_progress(self, stream_id: str) -> None:
        """Drain the info queue for a streamer to get latest progress."""
        info_q = self._info_queues.get(stream_id)
        if not info_q:
            return
        while True:
            try:
                msg = info_q.get_nowait()
                if msg.get("type") == "progress":
                    self._progress[stream_id] = msg["value"]
            except queue.Empty:
                break

    def _cleanup_stream(self, stream_id: str) -> None:
        """Remove a stream from tracking."""
        self._streamers.pop(stream_id, None)
        self._stop_events.pop(stream_id, None)
        q = self._info_queues.pop(stream_id, None)
        if q:
            q.close()
            q.cancel_join_thread()
        self._progress.pop(stream_id, None)
        self._configs.pop(stream_id, None)
