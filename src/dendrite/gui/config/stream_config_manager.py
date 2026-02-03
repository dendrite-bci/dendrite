"""
Central Stream Configuration Manager

This service acts as the single source of truth for all stream configuration and
modality-related information in the Dendrite GUI. It manages complete stream configurations
and provides reactive updates to all dependent components.

Key Features:
- Single source of truth for stream configurations and modality information
- Reactive updates via Qt signals
- Centralized stream config and modality aggregation logic
- Consistent API for all stream-related consumers
- Comprehensive stream configuration access
"""

from typing import Any

from PyQt6 import QtCore

from dendrite.data.stream_schemas import StreamMetadata
from dendrite.utils.logger_central import get_logger


class StreamConfigManager(QtCore.QObject):
    """
    Central manager for stream configurations using StreamMetadata.

    Stores validated StreamMetadata instances and provides access methods.
    All stream data is typed - use attribute access (stream.channel_count)
    not dict access (stream['channel_count']).
    """

    METADATA_CHANNEL_TYPES = {
        "markers",
        "marker",
        "events",
        "triggers",
        "annotations",
    }  # Non-signal channels

    streams_updated = QtCore.pyqtSignal(dict)  # Emitted when streams are updated
    modality_data_changed = QtCore.pyqtSignal(dict)  # Emitted when modality data changes

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = get_logger("StreamConfigManager")

        # Core data storage - StreamMetadata instances
        self._streams: dict[str, StreamMetadata] = {}
        self._updating: bool = False  # Guard against recursion

        # Stream configurations loaded from config files
        # Used by stream_selection dialog to match discovered streams with saved configs
        self._loaded_stream_configs: list[dict[str, Any]] = []

        self.logger.info("StreamConfigManager initialized")

    def _check_stream_issues(self, stream: StreamMetadata) -> dict[str, Any] | None:
        """Check if stream has metadata issues and return issue dict if so."""
        if stream.has_metadata_issues:
            return {"name": stream.name, "type": stream.type, "issues": stream.metadata_issues}
        return None

    def _streams_equal(self, a: dict[str, StreamMetadata], b: dict[str, StreamMetadata]) -> bool:
        """Compare streams by content, not just UIDs."""
        if set(a.keys()) != set(b.keys()):
            return False
        return all(a[uid].model_dump() == b[uid].model_dump() for uid in a)

    def process_discovered_streams(
        self, selected_streams: dict[str, str], discovered_streams: dict[str, StreamMetadata]
    ) -> dict[str, Any]:
        """
        Process discovered streams and return configuration result.

        Args:
            selected_streams: Dictionary mapping stream UIDs to LSL stream types
            discovered_streams: Dictionary of discovered StreamMetadata

        Returns:
            Dictionary with 'streams' (StreamMetadata) and 'issues' keys
        """
        configured_streams: dict[str, StreamMetadata] = {}
        streams_with_issues = []

        for uid in selected_streams:
            stream = discovered_streams[uid]

            # Check for metadata issues from LSL extraction
            issue = self._check_stream_issues(stream)
            if issue:
                streams_with_issues.append(issue)

            # Store StreamMetadata directly (no model_dump!)
            configured_streams[uid] = stream

        # Update internal state
        self._streams = configured_streams
        self._emit_updates()

        return {"streams": configured_streams, "issues": streams_with_issues}

    def update_streams(self, streams: dict[str, StreamMetadata]) -> None:
        """
        Update stream configurations and emit signals.

        Args:
            streams: Dictionary mapping stream UIDs to StreamMetadata
        """
        if self._updating:
            return  # Prevent recursion

        # Compare stream content, not just UIDs
        if self._streams_equal(streams, self._streams):
            self.logger.debug("No changes to stream configurations, skipping update")
            return

        self._updating = True
        try:
            self.logger.info(f"Updating {len(streams)} stream configurations")
            self._streams = streams.copy()
            self._emit_updates()
        finally:
            self._updating = False

    def _emit_updates(self) -> None:
        """Emit update signals to notify subscribers of configuration changes."""
        self.streams_updated.emit(self._streams.copy())
        self.modality_data_changed.emit(self.get_modality_data())

    def get_streams(self) -> dict[str, StreamMetadata]:
        """Get all configured streams."""
        return self._streams.copy()

    def get_stream(self, uid: str) -> StreamMetadata | None:
        """Get a specific stream by UID."""
        return self._streams.get(uid)

    @property
    def loaded_stream_configs(self) -> list[dict[str, Any]]:
        """Get loaded stream configurations from config files."""
        return self._loaded_stream_configs

    @loaded_stream_configs.setter
    def loaded_stream_configs(self, configs: list[dict[str, Any]]) -> None:
        """Set loaded stream configurations from config files."""
        self._loaded_stream_configs = configs

    def has_streams(self) -> bool:
        """Check if any streams are configured."""
        return bool(self._streams)

    def has_modalities(self) -> bool:
        """Check if any usable modalities are available (excluding markers/events)."""
        for stream in self._streams.values():
            if stream.channel_types:
                for ch_type in stream.channel_types:
                    normalized = ch_type.lower() if ch_type else ""
                    if normalized and normalized not in self.METADATA_CHANNEL_TYPES:
                        return True
        return False

    def get_system_sample_rate(self) -> int | None:
        """
        Get the system's master sample rate (EEG rate used for synchronization).

        The Dendrite system synchronizes all streams to the EEG sampling rate,
        which serves as the master clock. Individual streams may have
        different native rates but are resampled to this system rate.

        Returns:
            System sample rate in Hz, or None if no EEG streams configured
        """
        for stream in self._streams.values():
            if stream.type == "EEG" and stream.sample_rate:
                return int(stream.sample_rate)
        return None

    def get_modality_data(self) -> dict[str, dict[str, Any]]:
        """Get complete modality data for all available modalities."""
        stream_modalities = self.get_modalities_by_stream()

        modality_data: dict[str, dict[str, Any]] = {}
        for stream_info in stream_modalities.values():
            for modality, channels in stream_info["modalities"].items():
                if modality not in modality_data:
                    modality_data[modality] = {
                        "total_count": 0,
                        "channels": [],
                        "channel_labels": [],
                    }
                modality_data[modality]["channels"].extend(channels)
                modality_data[modality]["channel_labels"].extend(ch["label"] for ch in channels)
                modality_data[modality]["total_count"] += len(channels)

        return modality_data

    def get_modalities_by_stream(self) -> dict[str, dict[str, Any]]:
        """
        Get modality data organized hierarchically by stream.

        Returns:
            Dict mapping stream_uid to:
                - 'stream': StreamMetadata (name, type, sample_rate, etc.)
                - 'modalities': Dict[modality_type, List[channel_info]]
        """
        result = {}

        for stream_uid, stream in self._streams.items():
            # Skip streams with no/irregular sample rate (Events streams can't be preprocessed)
            if not stream.sample_rate or stream.sample_rate <= 0:
                continue

            stream_modalities: dict[str, list[dict[str, Any]]] = {}

            if stream.channel_types and len(stream.channel_types) == len(stream.labels):
                # Group channels by their type within this stream
                for i, (label, ch_type) in enumerate(
                    zip(stream.labels, stream.channel_types, strict=False)
                ):
                    normalized = ch_type.lower() if ch_type else ""
                    if normalized and normalized not in self.METADATA_CHANNEL_TYPES:
                        if normalized not in stream_modalities:
                            stream_modalities[normalized] = []
                        stream_modalities[normalized].append(
                            {
                                "label": label,
                                "stream_uid": stream_uid,
                                "stream_name": stream.name,
                                "local_index": i,
                            }
                        )

            if stream_modalities:
                result[stream_uid] = {"stream": stream, "modalities": stream_modalities}

        return result


# Global instance
_manager = None


def get_stream_config_manager() -> StreamConfigManager:
    """Get the global stream configuration manager instance."""
    global _manager
    if _manager is None:
        _manager = StreamConfigManager()
    return _manager


def initialize_stream_config_manager(parent=None) -> StreamConfigManager:
    """Initialize the global stream configuration manager."""
    global _manager
    if _manager is not None:
        _manager.deleteLater()
    _manager = StreamConfigManager(parent)
    return _manager
