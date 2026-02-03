"""
GUI Preflight Module

Stream discovery functionality for the Dendrite GUI.
Separated from the main Dendrite preflight module to keep GUI-specific functionality isolated.
"""

import logging
from collections.abc import Callable

from pylsl import LostError, StreamInlet, resolve_streams
from pylsl import StreamInfo as PylslStreamInfo

from dendrite.constants import LSL_FORMAT_MAP, LSL_INLET_OPEN_TIMEOUT
from dendrite.data.lsl_helpers import infer_channel_types_from_labels
from dendrite.data.stream_schemas import StreamMetadata

logger = logging.getLogger(__name__)


def _extract_stream_metadata(
    stream_info: PylslStreamInfo, log_callback: Callable[[str], None]
) -> StreamMetadata | None:
    """Extract metadata from LSL stream and return validated StreamMetadata.

    Returns:
        StreamMetadata if extraction succeeds, None if stream disappeared during discovery.
    """
    # Basic info (always available from discovery)
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

    # Try to extract channel metadata by connecting temporarily
    try:
        inlet = StreamInlet(stream_info, max_buflen=1)
        # Open with timeout to prevent hanging if stream disappears
        inlet.open_stream(timeout=LSL_INLET_OPEN_TIMEOUT)
        info = inlet.info()

        channels = info.desc().child("channels")
        if not channels.empty():
            ch = channels.child("channel")
            while not ch.empty():
                labels.append(ch.child_value("label") or f"Ch_{len(labels) + 1}")
                channel_types.append(ch.child_value("type") or "")  # Read LSL type
                channel_units.append(ch.child_value("unit") or "unknown")
                ch = ch.next_sibling("channel")

        inlet.close_stream()

        # If no/partial channel metadata, generate fallbacks
        if len(labels) < channel_count:
            has_metadata_issues = True
            metadata_issues["channel_metadata_missing"] = channel_count - len(labels)
            for i in range(len(labels), channel_count):
                labels.append(f"Ch_{i + 1}")
                channel_types.append("")
                channel_units.append("unknown")

        # Infer missing types from labels, flag for review
        inferred_count = 0
        for i, ch_type in enumerate(channel_types):
            if not ch_type:
                channel_types[i] = infer_channel_types_from_labels(
                    [labels[i]], default_type=stream_type
                )[0]
                inferred_count += 1

        if inferred_count > 0:
            has_metadata_issues = True
            metadata_issues["types_inferred"] = inferred_count

    except (LostError, RuntimeError, TimeoutError, OSError) as e:
        # Stream disappeared or connection failed - skip this stream entirely
        logger.warning(f"Stream '{name}' disappeared during discovery: {e}")
        log_callback(f"    Stream '{name}' unavailable (disappeared during discovery)")
        return None

    except Exception as e:
        # Other errors - use fallback metadata but continue
        logger.debug(f"Could not extract channel metadata from {name}: {e}")
        has_metadata_issues = True
        metadata_issues["extraction_failed"] = str(e)

        # Generate fallback labels
        labels = [f"{stream_type}_{i + 1:02d}" for i in range(channel_count)]
        channel_types = [stream_type] * channel_count
        channel_units = ["unknown"] * channel_count
        log_callback(f"    Using fallback labels for {name} (no channel metadata found)")

    return StreamMetadata(
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


def discover_all_lsl_streams(
    log_callback: Callable[[str], None], timeout: float = 2.0
) -> dict[str, StreamMetadata]:
    """
    Discover all available LSL streams on the network.

    Args:
        log_callback: Callback function to send log messages to the GUI.
        timeout: Timeout in seconds for stream discovery.

    Returns:
        Dictionary with stream UIDs as keys and StreamMetadata as values.
    """
    log_callback(f"--- Discovering All LSL Streams (timeout: {timeout}s) ---")

    discovered_streams: dict[str, StreamMetadata] = {}

    try:
        # Discover all streams without filtering
        all_stream_infos = resolve_streams(timeout)

        if not all_stream_infos:
            log_callback("No LSL streams found on the network.")
            return discovered_streams

        log_callback(f"Found {len(all_stream_infos)} LSL stream(s):")

        for i, stream_info in enumerate(all_stream_infos):
            stream_uid = stream_info.uid()
            stream_data = _extract_stream_metadata(stream_info, log_callback)

            # Skip streams that disappeared during metadata extraction
            if stream_data is None:
                continue

            log_callback(f"  Stream {i + 1}:")
            log_callback(f"    Name: {stream_data.name}")
            log_callback(f"    Type: {stream_data.type}")
            log_callback(f"    Channels: {stream_data.channel_count}")
            log_callback(f"    Sample Rate: {stream_data.sample_rate} Hz")
            log_callback(f"    Channel Format: {stream_data.channel_format}")
            log_callback(f"    Source ID: {stream_data.source_id}")
            log_callback(f"    UID: {stream_uid}")

            if stream_data.labels:
                log_callback(
                    f"    Channel Labels ({len(stream_data.labels)}): {stream_data.labels[:10]}{'...' if len(stream_data.labels) > 10 else ''}"
                )
            else:
                log_callback("    Channel Labels: None available")

            discovered_streams[stream_uid] = stream_data

    except Exception as e:
        log_callback(f"Error during stream discovery: {e}")
        logger.error(f"Stream discovery error: {e}", exc_info=True)

    log_callback("--- Stream Discovery Complete ---")
    return discovered_streams
