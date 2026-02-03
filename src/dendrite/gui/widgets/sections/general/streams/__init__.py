"""
Stream Configuration Subpackage

Provides LSL stream discovery, selection, and configuration widgets and dialogs.

Structure:
- stream_widget.py: Main StreamConfigurationWidget
- stream_setup_dialog.py: StreamSetupDialog for discovery and configuration
- components.py: Shared components, table models, and helpers
- preflight.py: LSL stream discovery functions
"""

from .components import (
    clear_layout,
    configure_channel_table,
    evaluate_metadata_issues,
    get_issue_severity,
    setup_channel_delegates,
    show_message_dialog,
)
from .stream_setup_dialog import StreamSetupDialog
from .stream_widget import StreamConfigurationWidget

__all__ = [
    "StreamConfigurationWidget",
    "StreamSetupDialog",
    # Helper functions
    "show_message_dialog",
    "clear_layout",
    "configure_channel_table",
    "setup_channel_delegates",
    "get_issue_severity",
    "evaluate_metadata_issues",
]
