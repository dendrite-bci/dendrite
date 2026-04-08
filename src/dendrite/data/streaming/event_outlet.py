"""LSL Event Outlet - Sends structured events as JSON strings.

Standalone module with only pylsl dependency. Can be copied for use in task apps.

Example:
    Complete workflow for sending task events:

    ```python
    from dendrite.data import EventOutlet

    # Define events with unique integer IDs
    events = {'trial_start': 1, 'cue_left': 10, 'cue_right': 11, 'response': 20, 'trial_end': 2}

    outlet = EventOutlet(stream_name='TaskEvents', events=events)

    # Send events during task execution
    outlet.send_event('trial_start', {'trial_number': 1})
    outlet.send_event('cue_left')
    outlet.send_event('response', {'reaction_time_ms': 450})
    outlet.send_event('trial_end')

    # Clean up when done
    outlet.close()
    ```
"""

import json
from typing import Any

from pylsl import StreamInfo, StreamOutlet, local_clock


class EventOutlet:
    """LSL outlet for sending structured events as JSON strings."""

    def __init__(
        self,
        stream_name: str,
        events: dict[str, int] | None = None,
        stream_id: str | None = None,
    ) -> None:
        """
        Initialize event outlet.

        Args:
            stream_name: Name of the LSL stream
            events: Event name to ID mappings
            stream_id: Optional unique identifier (defaults to '{stream_name}_id')
        """
        if not stream_name:
            raise ValueError("stream_name cannot be empty")

        self.event_id_mapping = events or {}

        info = StreamInfo(
            name=stream_name,
            type="Events",
            channel_count=1,
            nominal_srate=0,
            channel_format="string",
            source_id=stream_id or f"{stream_name}_id",
        )
        self.outlet = StreamOutlet(info)

    def send_event(self, event_type: str, additional_data: dict[str, Any] | None = None) -> None:
        """Send event marker with optional metadata payload.

        Args:
            event_type: Event name from configured events mapping.
            additional_data: Optional metadata to include in event packet.

        Raises:
            ValueError: If event_type is not in configured events.
        """
        if event_type not in self.event_id_mapping:
            raise ValueError(
                f"Unknown event '{event_type}'. Valid: {list(self.event_id_mapping.keys())}"
            )

        event_data = {
            "event_id": self.event_id_mapping[event_type],
            "event_type": event_type,
        }
        if additional_data:
            event_data.update(additional_data)

        self.outlet.push_sample([json.dumps(event_data)], timestamp=local_clock())

    def close(self) -> None:
        """Close outlet and release LSL resources.

        Call this when done sending events to properly clean up.
        """
        self.outlet = None

    def have_consumers(self) -> bool:
        """Check if any consumers are connected to this LSL stream.

        Returns:
            True if at least one consumer is connected.
        """
        return self.outlet is not None and self.outlet.have_consumers()
