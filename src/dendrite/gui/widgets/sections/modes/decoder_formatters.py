"""
Formatting utilities for decoder display in the decoder browser.

Contains helper functions for formatting accuracy, timestamps, and type badges.
"""

from datetime import datetime

from dendrite.gui.styles.design_tokens import (
    BG_ELEVATED,
    TEXT_MUTED,
)
from dendrite.gui.styles.widget_styles import FONTS, LAYOUT

_TIMESTAMP_FORMATS = ("%Y-%m-%d_%H-%M-%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")


def _parse_timestamp(ts: str) -> datetime | None:
    """Parse a timestamp string trying multiple known formats."""
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(str(ts).split(".")[0], fmt)
        except ValueError:
            continue
    return None


def relative_timestamp(ts: str) -> str:
    """Convert timestamp to relative format like '2 days ago'."""
    if not ts:
        return ""
    try:
        dt = _parse_timestamp(ts)
        if dt is None:
            return str(ts)

        now = datetime.now()
        delta = now - dt
        days = delta.days
        hours = delta.seconds // 3600

        if days == 0:
            if hours == 0:
                return "Just now"
            elif hours == 1:
                return "1 hour ago"
            return f"{hours} hours ago"
        elif days == 1:
            return "Yesterday"
        elif days < 7:
            return f"{days} days ago"
        elif days < 30:
            weeks = days // 7
            return f"{weeks} week{'s' if weeks > 1 else ''} ago"
        elif days < 365:
            months = days // 30
            return f"{months} month{'s' if months > 1 else ''} ago"
        else:
            years = days // 365
            return f"{years} year{'s' if years > 1 else ''} ago"
    except (ValueError, TypeError):
        return str(ts)


def format_timestamp(ts: str) -> str:
    """Format timestamp for display in details panel."""
    if not ts:
        return "-"
    try:
        dt = _parse_timestamp(ts)
        if dt is None:
            return str(ts)
        return dt.strftime("%b %d, %Y %I:%M %p")
    except (ValueError, TypeError):
        return str(ts)


def type_badge_style(size: str = "small") -> str:
    """Style for model type badges - subtle appearance."""
    if size == "tiny":
        padding = f"2px {LAYOUT['padding_xs']}px"
        font_size = FONTS["size_xs"]
    elif size == "small":
        padding = f"{LAYOUT['padding_xs']}px {LAYOUT['padding']}px"
        font_size = FONTS["size_sm"]
    else:
        padding = f"{LAYOUT['padding_sm']}px {LAYOUT['spacing_lg']}px"
        font_size = FONTS["size"]
    return f"""
        background-color: {BG_ELEVATED};
        color: {TEXT_MUTED};
        padding: {padding};
        border-radius: {LAYOUT["radius_sm"]}px;
        font-size: {font_size}px;
    """
