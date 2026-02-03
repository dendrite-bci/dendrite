#!/usr/bin/env python3
"""Design tokens for the Dendrite visual language. Modify here, not in widget_styles.py."""

# Neutral Colors
BG_MAIN = "#151515"  # Main window background (dark base)
BG_PANEL = "#1a1a1a"  # Panel/card backgrounds (standard)
BG_ELEVATED = "#2c2c2c"  # Elevated interactive elements (badges, primary buttons)
BG_INPUT = "#252525"  # Input field backgrounds (visible on dark bg)
BG_TAB_ACTIVE = "#3a3a3a"  # Dark grey active tab background

TEXT_MAIN = "#ffffff"  # Primary text (headings, body)
TEXT_LABEL = "#cccccc"  # Secondary text (labels, captions)
TEXT_DISABLED = "#666666"  # Disabled/placeholder text
TEXT_MUTED = "#a0a0a0"  # Muted secondary text (readable)
TEXT_MUTED_DARK = "#555555"  # Muted separators/details

BORDER = "#4a4a4a"  # Standard borders and dividers (clearly defined)
BORDER_INPUT = "rgba(255, 255, 255, 0.10)"  # Subtle input field borders

# Disabled state colors (muted, clearly non-interactive)
BG_DISABLED = "#1a1a1a"  # Darker than panel, clearly inactive
TEXT_DISABLED_MUTED = "#444444"  # Dimmer than TEXT_DISABLED for button text
BORDER_DISABLED = "#252525"  # Nearly invisible border for disabled elements
SEPARATOR_SUBTLE = "rgba(255, 255, 255, 0.06)"  # Ultra-subtle dividers

# Brand Colors
ACCENT = "#2a8be8"  # Primary brand color (buttons, links, selection)
ACCENT_HOVER = "#1e6fc4"  # Hover state for accent elements

# Semantic Colors
STATUS_OK = "#3da055"  # Success states, valid data, active status
STATUS_WARN = "#e8b422"  # Warnings, non-critical issues
STATUS_ERROR = "#d04555"  # Errors, invalid data, critical issues
STATUS_ERROR_PRESSED = "#c83d4d"  # Error button pressed state

# Interactive state colors (pressed for severity buttons)
STATUS_WARN_PRESSED = "#c09418"  # Warning button pressed state
STATUS_OK_PRESSED = "#2d803f"  # Success button pressed state

# Button hover states
BUTTON_HOVER_SUBTLE = "rgba(255, 255, 255, 0.10)"  # Subtle border highlight
BUTTON_HOVER_BG = "#2e2e2e"  # Primary button hover background

# Container backgrounds
BG_ERROR_SUBTLE = "rgba(208, 69, 85, 0.1)"  # Subtle error background for issue containers
BG_WARN_SUBTLE = "rgba(232, 180, 34, 0.1)"  # Subtle warning background
BG_OK_SUBTLE = "rgba(61, 160, 85, 0.1)"  # Subtle success background

# Resource Level Colors (telemetry-inspired)
STATUS_SUCCESS = "#20c8a0"  # Success/normal resource level
STATUS_WARNING_ALT = "#e8a020"  # Warning/medium resource level
STATUS_DANGER = "#e87575"  # Danger/high resource level

SPACING = {
    # Base spacing units (8pt grid)
    "xs": 4,  # 0.5x - Tight spacing (icon padding, dense lists)
    "sm": 6,  # 0.75x - Compact spacing (form labels, small gaps)
    "md": 8,  # 1x - DEFAULT - Standard element spacing
    "lg": 10,  # 1.25x - Comfortable spacing (between sections)
    "xl": 12,  # 1.5x - Generous spacing (section separators)
    "2xl": 16,  # 2x - Large spacing (dialog margins, major sections)
    "3xl": 20,  # 2.5x - Extra large (top-level containers)
    # Semantic aliases
    "form_field": 10,  # Between form fields
    "section": 12,  # Between sections
    "dialog_margin": 16,  # Dialog edge margins
}

PADDING = {
    "xs": 3,  # Minimal internal padding
    "sm": 4,  # Compact widget internals
    "md": 6,  # Standard widget padding
    "lg": 9,  # Generous widget padding
}

RADIUS = {
    "sm": 3,  # Small radius (inputs, badges)
    "md": 4,  # Standard radius (buttons, cards)
    "lg": 6,  # Large radius (major containers)
}

FONT_SIZE = {
    "xs": 9,  # Captions, minor labels
    "sm": 10,  # Small text, hints
    "md": 11,  # DEFAULT - Body text, standard labels
    "lg": 13,  # Section headers, emphasized text
    "xl": 15,  # Subsection titles
    "2xl": 17,  # Dialog titles
    "3xl": 22,  # Page titles, hero text
}

FONT_WEIGHT = {
    "normal": "normal",
    "medium": "500",
    "semibold": "600",  # Telemetry-inspired balanced emphasis
    "bold": "bold",
}

FONT_FAMILY = {
    "default": "inherit",
    "monospace": "'Consolas', 'Monaco', monospace",
}

#
# Typography:
#   3xl (24px): Main page/window titles
#   2xl (18px): Dialog titles, section headers
#   xl (16px): Subsection headers, group titles
#   lg (14px): Emphasized labels, secondary headers
#   md (12px): Body text, standard labels [DEFAULT]
#   sm (11px): Small text, metadata
#   xs (10px): Captions, hints, minor labels
#
# Spacing:
#   3xl (20px): Top-level container margins
#   2xl (16px): Dialog margins, major sections
#   xl (12px): Section separators
#   lg (10px): Form field vertical spacing
#   md (8px): Standard element spacing [DEFAULT]
#   sm (6px): Related controls, button groups
#   xs (4px): Tight groupings, icon padding
#
# Color Usage:
#   BG_MAIN: Main window background (dark base)
#   BG_PANEL: Standard panels and cards (clear separation from main)
#   BG_ELEVATED: Mode badges and primary buttons (elevated/prominent)
#   BG_INPUT: Text input fields
#   ACCENT: Interactive elements (buttons, links, selections)
#   STATUS_OK: Success, valid, active
#   STATUS_WARN: Warnings, validation issues
#   STATUS_ERROR: Errors, destructive actions
#   TEXT_MAIN: Primary content
#   TEXT_LABEL: Secondary info, labels
#   TEXT_DISABLED: Inactive, placeholders
#   TEXT_MUTED: Refined secondary text, muted hierarchy
#   TEXT_MUTED_DARK: Very subtle details, muted separators
#   STATUS_SUCCESS/WARNING_ALT/DANGER: Resource level indicators
