"""UI widgets for the Dashboard app."""

from .components import (
    CHANNELS_PER_PAGE,
    ChannelConfigPopup,
    StatusBar,
    UIHelpers,
)
from .quality_indicators import QualityStripWidget

__all__ = [
    "CHANNELS_PER_PAGE",
    "ChannelConfigPopup",
    "QualityStripWidget",
    "StatusBar",
    "UIHelpers",
]
