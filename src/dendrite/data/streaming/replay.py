"""Replay streamer — replays file and MOABB data over LSL.

Runs as a subprocess, loads data from files or MOABB datasets,
and streams sample-by-sample via LSL outlets with real-time timing.

Streams pure data channels only — the orchestrator adds the markers column
to ring buffers (`raw_channels + 1`). Events reach consumers via
a separate Events LSL stream (`EventOutlet`).
"""

import logging
import time
from collections import Counter
from multiprocessing import Process
from pathlib import Path
from multiprocessing.queues import Queue

import numpy as np
from pylsl import local_clock

from dendrite.data.loaders import (
    MOAABLoader,
    get_moabb_dataset_info,
    is_supported_format,
    load_file,
)
from dendrite.data.lsl_helpers import LSLOutlet
from dendrite.data.stream_schemas import StreamConfig
from dendrite.data.streaming.event_outlet import EventOutlet

_CHANNEL_TYPE_TO_UNIT: dict[str, str] = {
    "eeg": "microvolts",
    "emg": "microvolts",
    "ecg": "microvolts",
    "eog": "microvolts",
    "seeg": "microvolts",
    "ecog": "microvolts",
    "meg": "tesla",
    "stim": "volts",
    "stimulus": "volts",
    "resp": "a.u.",
    "gsr": "microsiemens",
    "temperature": "celsius",
    "markers": "unknown",
    "position": "mm",
    "force": "N",
    "acceleration": "m/s^2",
}


class ReplayStreamer(Process):
    """Replays file or MOABB data over LSL with real-time timing."""

    def __init__(
        self,
        stop_event,
        *,
        data_file_path: str = "",
        stream_name_prefix: str | None = None,
        moabb_preset: str | None = None,
        moabb_subject: int | None = None,
        moabb_session: str | None = None,
        enable_event_stream: bool = False,
        info_queue: Queue | None = None,
    ):
        super().__init__()
        self.stop_event = stop_event
        self.data_file_path = data_file_path
        self.stream_name_prefix = stream_name_prefix
        self.moabb_preset = moabb_preset
        self.moabb_subject = moabb_subject
        self.moabb_session = moabb_session
        self.enable_event_stream = enable_event_stream
        self.info_queue = info_queue
        self.logger = logging.getLogger("ReplayStreamer")

    def _get_stream_name(self, base_name: str) -> str:
        if self.stream_name_prefix:
            return f"{base_name}_{self.stream_name_prefix}"
        return base_name

    def run(self):
        try:
            if self.moabb_preset:
                self._replay_moabb()
            elif self.data_file_path and is_supported_format(self.data_file_path):
                self._replay_file()
        except Exception as e:
            self.logger.exception(f"Replay error: {e}")

    # ------------------------------------------------------------------
    # File replay
    # ------------------------------------------------------------------

    def _replay_file(self):
        loaded = load_file(self.data_file_path)
        data = loaded.data
        n_samples = loaded.n_samples

        type_counts = Counter(t.upper() for t in loaded.channel_types)
        type_summary = ", ".join(f"{count} {t}" for t, count in type_counts.items())
        self.logger.info(f"Replaying {type_summary} @ {loaded.sample_rate} Hz")

        # Derive stream name from filename (e.g., "sub-01_ses-03_eeg")
        file_stem = Path(self.data_file_path).stem
        stream_name = self._get_stream_name(file_stem)

        config = _build_stream_config(
            stream_name,
            loaded.channel_names,
            loaded.channel_types,
            loaded.sample_rate,
        )
        outlet = LSLOutlet(config=config)

        # Event outlet (separate LSL stream for discrete events)
        event_outlet = None
        if self.enable_event_stream and loaded.events:
            event_mapping = loaded.event_id or {
                f"event_{code}": code
                for code in sorted({code for _, code in loaded.events})
            }
            event_outlet = EventOutlet(
                stream_name=self._get_stream_name(f"{file_stem}_Events"),
                events=event_mapping,
                stream_id=f"events_{file_stem}",
            )
            self.logger.info(f"Event stream: {len(loaded.events)} events, mapping: {event_mapping}")

        # Build sample_idx -> (marker_code, event_name) for event outlet
        event_id_rev = {v: k for k, v in (loaded.event_id or {}).items()}
        event_dict: dict[int, tuple[int, str]] = {
            idx: (code, event_id_rev.get(code, f"event_{code}"))
            for idx, code in loaded.events
        }

        if loaded.sample_rate <= 0:
            raise ValueError(f"Invalid sample rate: {loaded.sample_rate}")
        if self.info_queue:
            self.info_queue.put({"type": "duration", "value": n_samples / loaded.sample_rate})

        self._stream_loop(data, outlet, loaded.sample_rate, event_dict, event_outlet)

        outlet.close()
        if event_outlet:
            event_outlet.close()
        self.logger.info("File replay completed")

    # ------------------------------------------------------------------
    # MOABB replay
    # ------------------------------------------------------------------

    def _replay_moabb(self):
        info = get_moabb_dataset_info(self.moabb_preset)
        if not info:
            raise ValueError(f"Unknown MOABB dataset: {self.moabb_preset}")

        loader = MOAABLoader(info["config"])
        self.logger.info(
            f"Loading MOABB: {self.moabb_preset}, subject {self.moabb_subject}, session {self.moabb_session}"
        )

        loaded = loader.load_as_raw(self.moabb_subject, session=self.moabb_session)
        event_times = [e[0] for e in loaded.events]
        event_labels = [e[1] for e in loaded.events]
        event_mapping = loaded.event_id or {}
        channel_names = loaded.channel_names
        channel_types = loaded.channel_types
        sample_rate = loaded.sample_rate

        data = loaded.data  # (channels, samples) — matches _stream_loop's data[:, i]
        n_samples = loaded.n_samples
        self.logger.info(
            f"Loaded {n_samples} samples, {loaded.n_channels} channels @ {sample_rate} Hz, "
            f"{len(event_times)} events"
        )

        if self.info_queue:
            self.info_queue.put({"type": "duration", "value": n_samples / sample_rate})

        primary_type = Counter(
            t.upper() for t in channel_types
        ).most_common(1)[0][0] if channel_types else "DATA"
        config = _build_stream_config(
            self._get_stream_name(f"MOABB_{primary_type}"),
            channel_names, channel_types, sample_rate,
        )
        outlet = LSLOutlet(config=config)

        label_to_name = {label: name for name, label in event_mapping.items()}
        event_outlet = None
        if self.enable_event_stream and len(event_times) > 0:
            event_outlet = EventOutlet(
                stream_name=self._get_stream_name("MOABB_Events"),
                events=event_mapping,
                stream_id=f"moabb_events_{self.moabb_preset}",
            )

        # Build sample_idx -> (marker_code, event_name)
        event_dict: dict[int, tuple[int, str]] = {}
        for evt_time, evt_label in zip(event_times, event_labels, strict=True):
            event_dict[int(evt_time)] = (int(evt_label), label_to_name.get(evt_label, f"class_{evt_label}"))

        self._stream_loop(data, outlet, sample_rate, event_dict, event_outlet)

        outlet.close()
        if event_outlet:
            event_outlet.close()
        self.logger.info("MOABB replay completed")

    # ------------------------------------------------------------------
    # Shared streaming loop
    # ------------------------------------------------------------------

    def _stream_loop(
        self,
        data: np.ndarray,
        outlet: LSLOutlet,
        sample_rate: float,
        event_dict: dict[int, tuple[int, str]],
        event_outlet: EventOutlet | None,
    ):
        """Stream data sample-by-sample with real-time timing.

        Pushes pure data samples (no markers appended). Events are sent
        via the separate event_outlet if provided.
        """
        n_samples = data.shape[1]
        start_time = time.perf_counter()
        interval = 1.0 / sample_rate
        progress_interval = max(1, int(sample_rate // 2))

        for i in range(n_samples):
            if self.stop_event.is_set():
                break

            sleep_time = start_time + (i * interval) - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Send event via separate event outlet
            event_info = event_dict.get(i)
            if event_outlet and event_info:
                event_outlet.send_event(
                    event_type=event_info[1], additional_data={"sample_idx": i}
                )

            outlet.push_sample(data[:, i].tolist(), local_clock())

            if self.info_queue and i % progress_interval == 0:
                self.info_queue.put({"type": "progress", "value": i / n_samples})

        if self.info_queue:
            self.info_queue.put({"type": "progress", "value": 1.0})


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _build_stream_config(
    name: str, channel_names: list[str], channel_types: list[str], sample_rate: float
) -> StreamConfig:
    """Build StreamConfig for file/MOABB data with actual channel names.

    Streams pure data channels — no synthetic Markers column appended.
    The orchestrator adds the markers column to ring buffers separately.
    """
    units = [_CHANNEL_TYPE_TO_UNIT.get(t.lower(), "a.u.") for t in channel_types]

    primary_type = Counter(t.upper() for t in channel_types).most_common(1)[0][0] if channel_types else "DATA"

    return StreamConfig(
        name=name,
        type=primary_type,
        channel_count=len(channel_names),
        sample_rate=sample_rate,
        channel_format="float32",
        labels=list(channel_names),
        channel_types=list(channel_types),
        channel_units=units,
        source_id=f"replay_{name.lower().replace(' ', '_')}",
    )
