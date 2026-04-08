"""
Stream Service

LSL stream discovery and configuration management.
"""

import logging
import time
from typing import Any

from dendrite.data.stream_schemas import StreamMetadata
from dendrite.utils.logger_central import get_logger

METADATA_CHANNEL_TYPES = {"markers", "marker", "events", "triggers", "annotations"}


class StreamService:
    """Manages stream discovery results and configuration state."""

    def __init__(self):
        self.logger = get_logger("StreamService")
        self._streams: dict[str, StreamMetadata] = {}
        self._last_discovery: dict[str, StreamMetadata] = {}
        self._last_discovery_time: float = 0.0

    # ------------------------------------------------------------------
    # Discovery (runs in thread via asyncio.to_thread)
    # ------------------------------------------------------------------

    @staticmethod
    def discover_lsl_streams(timeout: float = 2.0) -> dict[str, StreamMetadata]:
        """Discover all available LSL streams. Blocking — call via to_thread.

        Returns:
            Dict of uid -> StreamMetadata.
        """
        from pylsl import LostError, StreamInlet, resolve_streams

        from dendrite.data.lsl_helpers import (
            infer_channel_types_from_labels,
            normalize_channel_type,
            normalize_channel_unit,
        )

        LSL_FORMAT_MAP = {
            0: "undefined",
            1: "float32",
            2: "double64",
            3: "string",
            4: "int32",
            5: "int16",
            6: "int8",
            7: "int64",
        }
        LSL_INLET_OPEN_TIMEOUT = 2.0

        logger = logging.getLogger("StreamDiscovery")
        discovered: dict[str, StreamMetadata] = {}

        try:
            all_stream_infos = resolve_streams(timeout)
        except Exception as e:
            logger.error(f"Stream discovery error: {e}", exc_info=True)
            return discovered

        for stream_info in all_stream_infos:
            name = stream_info.name()
            stream_type = stream_info.type()
            channel_count = stream_info.channel_count()
            sample_rate = stream_info.nominal_srate()
            channel_format = LSL_FORMAT_MAP.get(int(stream_info.channel_format()), "float32")
            source_id = stream_info.source_id()
            uid = stream_info.uid()

            labels = []
            channel_types = []
            channel_units = []
            has_metadata_issues = False
            metadata_issues = {}

            try:
                inlet = StreamInlet(stream_info, max_buflen=1)
                inlet.open_stream(timeout=LSL_INLET_OPEN_TIMEOUT)
                info = inlet.info()

                channels = info.desc().child("channels")
                if not channels.empty():
                    ch = channels.child("channel")
                    while not ch.empty():
                        labels.append(ch.child_value("label") or f"Ch_{len(labels) + 1}")
                        channel_types.append(ch.child_value("type") or "")
                        channel_units.append(ch.child_value("unit") or "unknown")
                        ch = ch.next_sibling("channel")

                inlet.close_stream()

                is_event_stream = stream_type.lower() in METADATA_CHANNEL_TYPES

                if len(labels) < channel_count:
                    if not is_event_stream:
                        has_metadata_issues = True
                        metadata_issues["channel_metadata_missing"] = channel_count - len(labels)
                    for i in range(len(labels), channel_count):
                        labels.append(f"Ch_{i + 1}")
                        channel_types.append("")
                        channel_units.append("unknown")

                inferred_count = 0
                for i, ch_type in enumerate(channel_types):
                    if not ch_type:
                        channel_types[i] = infer_channel_types_from_labels(
                            [labels[i]], default_type=stream_type
                        )[0]
                        inferred_count += 1

                if inferred_count > 0 and not is_event_stream:
                    has_metadata_issues = True
                    metadata_issues["types_inferred"] = inferred_count

                # Normalize to canonical names
                channel_types = [normalize_channel_type(ct) for ct in channel_types]
                channel_units = [normalize_channel_unit(cu) for cu in channel_units]

            except (LostError, RuntimeError, TimeoutError, OSError):
                logger.warning(f"Stream '{name}' disappeared during discovery")
                continue

            except Exception as e:
                logger.debug(f"Could not extract channel metadata from {name}: {e}")
                has_metadata_issues = True
                metadata_issues["extraction_failed"] = str(e)
                labels = [f"{stream_type}_{i + 1:02d}" for i in range(channel_count)]
                channel_types = [stream_type] * channel_count
                channel_units = ["unknown"] * channel_count

            meta = StreamMetadata(
                name=name,
                type=stream_type,
                channel_count=channel_count,
                sample_rate=sample_rate,
                channel_format=channel_format,
                source_id=source_id,
                uid=uid,
                labels=labels,
                channel_types=channel_types,
                channel_units=channel_units,
                has_metadata_issues=has_metadata_issues,
                metadata_issues=metadata_issues,
            )
            discovered[meta.stable_key] = meta

        return discovered

    def discover_and_cache(self, timeout: float = 2.0) -> dict[str, StreamMetadata]:
        """Discover LSL streams and cache the results.

        Clears configured streams to force re-selection from fresh discovery,
        preventing stale configs (e.g. old channel_types) from persisting
        after the physical stream setup has changed.
        """
        result = self.discover_lsl_streams(timeout)
        self._last_discovery = result
        self._last_discovery_time = time.monotonic()
        self._streams.clear()
        return result

    def get_cached_discovery(self, max_age: float = 30.0) -> dict[str, StreamMetadata] | None:
        """Return cached discovery results if recent enough, else None."""
        if self._last_discovery and time.monotonic() - self._last_discovery_time < max_age:
            return self._last_discovery
        return None

    def check_liveness(self, timeout: float = 1.0) -> dict[str, bool]:
        """Quick check which configured streams are still available on the network."""
        liveness, _ = self.check_streams(timeout)
        return liveness

    def check_streams(
        self, timeout: float = 1.0
    ) -> tuple[dict[str, bool], dict[str, tuple[int, int]]]:
        """Single LSL resolve returning liveness and channel mismatches.

        Returns:
            (liveness, channel_mismatches) where:
            - liveness: {uid: True/False}
            - channel_mismatches: {uid: (configured, live)} for mismatched streams only
        """
        if not self._streams:
            return {}, {}
        from pylsl import resolve_streams

        try:
            found = resolve_streams(timeout)
        except Exception:
            return {key: False for key in self._streams}, {}

        # Build lookups from resolved streams
        found_keys: dict[str, int] = {}
        for info in found:
            sid = info.source_id()
            key = sid if sid else f"{info.name()}:{info.type()}"
            found_keys[key] = info.channel_count()

        liveness = {key: key in found_keys for key in self._streams}

        mismatches = {}
        for uid, meta in self._streams.items():
            live_count = found_keys.get(uid)
            if live_count is not None and live_count != meta.channel_count:
                mismatches[uid] = (meta.channel_count, live_count)

        return liveness, mismatches

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure_streams(
        self,
        selected_uids: list[str],
        discovered_streams: dict[str, StreamMetadata],
        channel_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Select streams from discovery results and store them.

        Args:
            selected_uids: UIDs of streams to configure.
            discovered_streams: Full discovery results.
            channel_overrides: Per-stream channel edits (labels, channel_types, channel_units).

        Returns:
            Dict with 'streams' and 'issues' keys.
        """
        configured: dict[str, StreamMetadata] = {}
        issues = []
        key_counts: dict[str, int] = {}

        for uid in selected_uids:
            stream = discovered_streams.get(uid)
            if stream is None:
                continue

            # Apply user channel edits if provided
            if channel_overrides and uid in channel_overrides:
                allowed = {"labels", "channel_types", "channel_units"}
                updates = {k: v for k, v in channel_overrides[uid].items() if k in allowed}
                if updates:
                    stream = stream.model_copy(update=updates)

            # Assign unique stream_key: first of a type gets "EEG", second gets "EEG_2"
            base = stream.type
            key_counts[base] = key_counts.get(base, 0) + 1
            key = base if key_counts[base] == 1 else f"{base}_{key_counts[base]}"
            stream = stream.model_copy(update={"stream_key": key})

            configured[uid] = stream
            if stream.has_metadata_issues:
                issues.append({
                    "name": stream.name,
                    "type": stream.type,
                    "issues": stream.metadata_issues,
                })

        self._streams = configured
        return {"streams": configured, "issues": issues}

    def update_streams(self, streams: dict[str, StreamMetadata]) -> None:
        self._streams = streams.copy()

    def get_streams(self) -> dict[str, StreamMetadata]:
        return self._streams.copy()

    def get_stream(self, uid: str) -> StreamMetadata | None:
        return self._streams.get(uid)

    def has_streams(self) -> bool:
        return bool(self._streams)

    def get_modalities_by_stream(self) -> dict[str, dict[str, Any]]:
        """Get modality data grouped by stream.

        Returns a dict keyed by stream UID, each containing stream info and
        its modalities with channel lists. Channel indices are positions within
        each modality's list (matching ring buffer extraction order).
        """
        result = {}

        for stream_uid, stream in self._streams.items():
            if not stream.sample_rate or stream.sample_rate <= 0:
                continue

            stream_modalities: dict[str, list[dict[str, Any]]] = {}

            if stream.channel_types and len(stream.channel_types) == len(stream.labels):
                for i, (label, ch_type) in enumerate(
                    zip(stream.labels, stream.channel_types, strict=False)
                ):
                    normalized = ch_type.lower() if ch_type else ""
                    if normalized and normalized not in METADATA_CHANNEL_TYPES:
                        if normalized not in stream_modalities:
                            stream_modalities[normalized] = []
                        stream_modalities[normalized].append({
                            "label": label,
                            "local_index": i,
                        })

            if stream_modalities:
                result[stream_uid] = {
                    "stream_name": stream.name,
                    "stream_type": stream.type,
                    "stream_key": stream.stream_key or stream.type,
                    "sample_rate": stream.sample_rate,
                    "modalities": stream_modalities,
                }

        return result

    def restore_from_config(self, stream_configs: list[dict[str, Any] | str]) -> None:
        """Restore configured streams from a saved config (no discovery needed).

        Streams are restored with full channel maps/types/labels. Liveness
        polling will separately determine if they are currently online.
        Skips entries that are strings (old format from pre-fix saves).
        """
        self._streams.clear()
        for cfg in stream_configs:
            if not isinstance(cfg, dict):
                continue
            meta = StreamMetadata(**cfg)
            self._streams[meta.stable_key] = meta
