#!/usr/bin/env python3
"""Widget styling system with builder methods and design token integration."""

from PyQt6 import QtGui

from .design_tokens import (
    ACCENT,
    ACCENT_HOVER,
    BG_DISABLED,
    BG_ELEVATED,
    BG_ERROR_SUBTLE,
    BG_INPUT,
    BG_MAIN,
    BG_PANEL,
    BG_TAB_ACTIVE,
    BORDER,
    BORDER_DISABLED,
    BORDER_INPUT,
    BUTTON_HOVER_BG,
    BUTTON_HOVER_SUBTLE,
    FONT_FAMILY,
    FONT_SIZE,
    FONT_WEIGHT,
    PADDING,
    RADIUS,
    SEPARATOR_SUBTLE,
    SPACING,
    STATUS_DANGER,
    STATUS_ERROR,
    STATUS_ERROR_PRESSED,
    STATUS_OK,
    STATUS_OK_PRESSED,
    STATUS_SUCCESS,
    STATUS_WARN,
    STATUS_WARN_PRESSED,
    STATUS_WARNING_ALT,
    TEXT_DISABLED,
    TEXT_DISABLED_MUTED,
    TEXT_LABEL,
    TEXT_MAIN,
    TEXT_MUTED,
    TEXT_MUTED_DARK,
)

LAYOUT = {
    "spacing_xs": SPACING["xs"],
    "spacing_sm": SPACING["sm"],
    "spacing": SPACING["md"],
    "spacing_md": SPACING["md"],
    "spacing_lg": SPACING["lg"],
    "spacing_xl": SPACING["2xl"],
    "spacing_xxl": SPACING["3xl"],
    "padding_xs": PADDING["xs"],
    "padding_sm": PADDING["sm"],
    "padding": PADDING["md"],
    "padding_md": PADDING["md"],
    "padding_lg": PADDING["lg"],
    "margin": SPACING["md"],
    "margin_sm": SPACING["sm"],
    "margin_lg": SPACING["lg"],
    "radius": RADIUS["md"],
    "radius_sm": RADIUS["sm"],
    "radius_lg": RADIUS["lg"],
    "form_spacing": SPACING["form_field"],
    "form_vertical_spacing": SPACING["form_field"],
    "form_horizontal_spacing": 40,
    "dialog_margin": SPACING["dialog_margin"],
    "groupbox_padding_top": PADDING["lg"],
    "icon_size": 24,
}

FONTS = {
    "size_xs": FONT_SIZE["xs"],
    "size_sm": FONT_SIZE["sm"],
    "size": FONT_SIZE["md"],
    "size_md": FONT_SIZE["lg"],
    "size_lg": FONT_SIZE["xl"],
    "size_xl": FONT_SIZE["2xl"],
    "size_xxl": FONT_SIZE["3xl"],
    "weight_normal": FONT_WEIGHT["normal"],
    "weight_medium": FONT_WEIGHT["medium"],
    "weight_bold": FONT_WEIGHT["bold"],
    "family_default": FONT_FAMILY["default"],
    "family_monospace": FONT_FAMILY["monospace"],
}

_DIALOG_BASE = f"QDialog {{ background-color: {BG_MAIN}; color: {TEXT_LABEL}; border: none; }}"

_BUTTON_BASE = f"""
    QPushButton {{
        background-color: {BG_ELEVATED};
        color: {TEXT_LABEL};
        border: 1px solid {BORDER};
        padding: {LAYOUT["padding_sm"]}px {LAYOUT["padding_lg"]}px;
        border-radius: {LAYOUT["radius"]}px;
        font: 500 {FONTS["size"]}px;
        min-width: 80px;
        margin: 2px;
    }}
    QPushButton:hover {{ background-color: {ACCENT}; border-color: {ACCENT}; }}
    QPushButton:pressed {{ background-color: {ACCENT_HOVER}; }}
    QPushButton:disabled {{ background-color: {BG_DISABLED}; color: {TEXT_DISABLED_MUTED}; border: 1px solid {BORDER_DISABLED}; }}
"""

_INPUT_BASE = f"""
    QLineEdit, QComboBox {{
        background-color: {BG_INPUT};
        color: {TEXT_MAIN};
        border: none;
        padding: {LAYOUT["padding_sm"]}px {LAYOUT["padding"]}px;
        border-radius: {LAYOUT["radius_sm"]}px;
        min-height: 1.4em;
        min-width: 150px;  /* macOS compatibility - prevents field collapse */
    }}
    QLineEdit:focus, QComboBox:focus {{ background-color: {BG_PANEL}; }}
    QComboBox::drop-down {{
        border: none;
        background-color: transparent;
        border-left: 1px solid {BORDER};
        border-radius: 0px {LAYOUT["radius_sm"]}px {LAYOUT["radius_sm"]}px 0px;
        width: 16px;
    }}
    QComboBox::down-arrow {{
        border-style: solid;
        border-width: 2px 2px 0 2px;
        border-color: {TEXT_DISABLED} transparent transparent transparent;
        margin: 2px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {BG_PANEL};
        color: {TEXT_MAIN};
        border: 1px solid {BORDER};
        border-radius: {LAYOUT["radius_sm"]}px;
        selection-background-color: {ACCENT};
        selection-color: {TEXT_MAIN};
        outline: none;
    }}
    QComboBox QAbstractItemView::item {{
        min-height: 20px;
        padding: 3px {LAYOUT["padding_sm"]}px;
        background-color: transparent;
    }}
    QComboBox QAbstractItemView::item:hover {{
        background-color: {ACCENT};
        color: {TEXT_MAIN};
    }}
    QComboBox QAbstractItemView::item:selected {{
        background-color: {ACCENT};
        color: {TEXT_MAIN};
    }}
"""

_GROUPBOX_BASE = f"""
    QGroupBox {{
        background-color: transparent;
        border: none;
        margin-top: {LAYOUT["margin_sm"]}px;
        padding: {LAYOUT["spacing_xl"]}px;
        padding-top: 20px;
    }}
    QGroupBox::title {{
        color: {TEXT_MAIN};
        font: bold {FONTS["size_md"]}px;
        padding: 0 {LAYOUT["padding_xs"]}px;
        subcontrol-origin: margin;
        subcontrol-position: top left;
        margin-top: 0px;
        left: {LAYOUT["padding_lg"]}px;
    }}
"""


class WidgetStyles:
    """Unified widget styles using builder methods and design tokens."""

    colors = {
        "text_primary": TEXT_MAIN,
        "text_secondary": TEXT_LABEL,
        "text_disabled": TEXT_DISABLED,
        "text_muted": TEXT_MUTED,
        "text_muted_dark": TEXT_MUTED_DARK,
        "background_primary": BG_MAIN,
        "background_secondary": BG_PANEL,
        "background_input": BG_INPUT,
        "border": BORDER,
        "separator_subtle": SEPARATOR_SUBTLE,
        "accent_primary": ACCENT,
        "accent_hover": ACCENT_HOVER,
        "status_ok": STATUS_OK,
        "status_warn": STATUS_WARN,
        "status_error": STATUS_ERROR,
        "status_neutral": TEXT_DISABLED,
        "success": STATUS_OK,
        "warning": STATUS_WARN,
        "error": STATUS_ERROR,
        "status_success": STATUS_SUCCESS,
        "status_warning_alt": STATUS_WARNING_ALT,
        "status_danger": STATUS_DANGER,
    }

    @staticmethod
    def _build_border_css(border: bool | str, color: str) -> str:
        """Build border CSS from border spec.

        Args:
            border: Border style - False/None, True, 'top', 'bottom', 'left', 'right'
            color: Border color

        Returns:
            CSS border declaration string
        """
        if border is True:
            return f"border: 1px solid {color};"
        elif border == "top":
            return f"border: none; border-top: 1px solid {color};"
        elif border == "bottom":
            return f"border: none; border-bottom: 1px solid {color};"
        elif border == "left":
            return f"border: none; border-left: 1px solid {color};"
        elif border == "right":
            return f"border: none; border-right: 1px solid {color};"
        return "border: none;"

    @classmethod
    def label(
        cls,
        variant: str = "normal",
        color: str | None = None,
        size: int | None = None,
        weight: str = "normal",
        style: str = "normal",
        align: str = "left",
        padding: str | None = None,
        margin: str | None = None,
        bg: str | None = None,
        monospace: bool = False,
        uppercase: bool = False,
        letter_spacing: int | None = None,
        line_height: float | None = None,
        width: int | None = None,
        height: int | None = None,
        min_width: int | None = None,
        max_width: int | None = None,
        max_height: int | None = None,
    ) -> str:
        """Universal label builder with variant presets.

        Args:
            variant: 'normal', 'header', 'title', 'subtitle', 'form', 'small', 'tiny', 'muted', 'muted_header'
        """
        # Variant presets
        presets = {
            "normal": {
                "color": TEXT_LABEL,
                "size": FONTS["size"],
                "padding": f"{LAYOUT['padding_sm']}px {LAYOUT['padding']}px",
                "line_height": 1.4,
            },
            "header": {
                "color": TEXT_MAIN,
                "size": FONTS["size_md"],
                "weight": "bold",
                "padding": f"{LAYOUT['padding_xs']}px 0px",
                "line_height": 1.4,
            },
            "title": {
                "color": TEXT_MAIN,
                "size": FONTS["size_lg"],
                "weight": "bold",
                "padding": f"{LAYOUT['padding']}px",
                "line_height": 1.3,
            },
            "subtitle": {
                "color": TEXT_MAIN,
                "size": FONTS["size_md"],
                "weight": "bold",
                "padding": f"{LAYOUT['padding']}px",
                "line_height": 1.4,
            },
            "form": {
                "color": TEXT_LABEL,
                "size": FONTS["size"],
                "padding": "2px",
                "line_height": 1.4,
            },
            "small": {
                "color": TEXT_LABEL,
                "size": FONTS["size_sm"],
                "padding": f"{LAYOUT['padding_xs']}px",
                "line_height": 1.4,
            },
            "tiny": {
                "color": TEXT_DISABLED,
                "size": FONTS["size_xs"],
                "padding": f"{LAYOUT['padding_xs']}px",
                "line_height": 1.5,
            },
            "muted": {
                "color": TEXT_MUTED,
                "size": FONTS["size"],
                "padding": f"{LAYOUT['padding_xs']}px",
                "line_height": 1.4,
            },
            "muted_header": {
                "color": TEXT_MUTED,
                "size": FONTS["size_sm"],
                "weight": "500",
                "padding": f"{LAYOUT['padding_xs']}px 0px",
                "line_height": 1.4,
            },
        }

        preset = presets.get(variant, presets["normal"])

        # Apply overrides (explicit params override preset)
        final_color = color or preset.get("color", TEXT_LABEL)
        final_size = size or preset.get("size", FONTS["size"])
        final_weight = weight if weight != "normal" else preset.get("weight", "normal")
        final_padding = padding or preset.get(
            "padding", f"{LAYOUT['padding_sm']}px {LAYOUT['padding']}px"
        )
        final_bg = bg or "transparent"
        final_line_height = line_height or preset.get("line_height", 1.4)

        # Build CSS properties
        font_family = FONTS["family_monospace"] if monospace else FONTS["family_default"]
        font_weight = f"font-weight: {final_weight};" if final_weight != "normal" else ""
        font_style_css = "font-style: italic;" if style == "italic" else ""
        text_transform = "text-transform: uppercase;" if uppercase else ""
        letter_spacing_css = f"letter-spacing: {letter_spacing}px;" if letter_spacing else ""
        margin_css = f"margin: {margin};" if margin else ""
        text_align = f"text-align: {align};" if align != "left" else ""

        # Size constraints
        width_css = f"width: {width}px;" if width else ""
        height_css = f"height: {height}px;" if height else ""
        min_width_css = f"min-width: {min_width}px;" if min_width else ""
        max_width_css = f"max-width: {max_width}px;" if max_width else ""
        max_height_css = f"max-height: {max_height}px;" if max_height else ""

        return f"""
            QLabel {{
                color: {final_color};
                font-size: {final_size}px;
                font-family: {font_family};
                {font_weight}
                {font_style_css}
                background-color: {final_bg};
                border: none;
                padding: {final_padding};
                line-height: {final_line_height};
                {text_transform}
                {letter_spacing_css}
                {margin_css}
                {text_align}
                {width_css}
                {height_css}
                {min_width_css}
                {max_width_css}
                {max_height_css}
            }}
        """

    @classmethod
    def button(
        cls,
        variant: str = "primary",
        severity: str | None = None,
        align: str = "center",
        size: str = "normal",
        transparent: bool = False,
        blend: bool = False,
        padding: str | None = None,
        min_width: int | None = None,
        fixed_size: int | None = None,
        underline: bool = False,
        checked_bg: str | None = None,
        checked_text: str | None = None,
        outline: str = "none",
        text_color: str | None = None,
    ) -> str:
        """Universal button builder with variant and severity support.

        Args:
            variant: 'primary', 'text', 'icon'
            severity: 'error', 'warning', 'success' (optional)
            transparent: Use transparent background
            fixed_size: For square icon buttons (sets width=height)
        """
        if severity == "error":
            base_color = STATUS_ERROR
            pressed_color = STATUS_ERROR_PRESSED
        elif severity == "warning":
            base_color = STATUS_WARN
            pressed_color = STATUS_WARN_PRESSED
        elif severity == "success":
            base_color = STATUS_OK
            pressed_color = STATUS_OK_PRESSED
        else:
            base_color = ACCENT
            pressed_color = ACCENT_HOVER

        # Size-specific values
        if size == "small":
            final_padding = padding or f"{LAYOUT['padding_xs']}px {LAYOUT['padding']}px"
            font_size = FONTS["size_sm"]
            final_min_width = min_width or 50
        else:
            final_padding = padding or f"{LAYOUT['padding_sm']}px {LAYOUT['padding_lg']}px"
            font_size = FONTS["size"]
            final_min_width = min_width or 70

        # Variant-specific styling
        if variant == "text":
            # Text button (no background, optionally underlined)
            bg = "transparent"
            default_text_color = base_color
            border = "none"
            text_decoration = "text-decoration: underline;" if underline else ""
            hover_bg = base_color
            hover_text = TEXT_MAIN
            hover_border_color = base_color
            hover_decoration = "text-decoration: none;" if underline else ""
        elif variant == "icon":
            # Icon button (small, square)
            bg = BORDER if not transparent else "transparent"
            default_text_color = TEXT_LABEL
            border = "none"
            text_decoration = ""
            hover_bg = BG_ELEVATED
            hover_text = TEXT_MAIN
            hover_border_color = BG_ELEVATED
            hover_decoration = ""
        elif transparent:
            # Transparent variant
            bg = "transparent"
            default_text_color = TEXT_LABEL
            border = f"1px solid {BORDER}"
            text_decoration = ""

            if blend:
                # Subtle blend mode - no accent colors
                hover_bg = BG_PANEL
                hover_text = TEXT_MAIN
                hover_border_color = BORDER
            else:
                # Normal transparent mode - no accent, just subtle highlight
                hover_bg = BG_PANEL
                hover_text = TEXT_MAIN
                hover_border_color = BORDER
            hover_decoration = ""
        else:
            # Primary variant - minimal by default, accent on hover only for severity buttons
            bg = BG_ELEVATED
            default_text_color = TEXT_LABEL
            border = f"1px solid {BUTTON_HOVER_SUBTLE}"
            text_decoration = ""
            if severity:
                # Severity buttons get accent color on hover (Start/Stop/Add)
                hover_bg = base_color
                hover_text = TEXT_MAIN
                hover_border_color = base_color
            else:
                # Normal buttons get noticeable hover (lighter bg + subtle border highlight)
                hover_bg = BUTTON_HOVER_BG
                hover_text = TEXT_MAIN
                hover_border_color = BUTTON_HOVER_SUBTLE
            hover_decoration = ""

        # Use custom text_color if provided, otherwise use default
        final_text_color = text_color if text_color is not None else default_text_color

        # Fixed size for icon buttons
        size_constraint = f"width: {fixed_size}px; height: {fixed_size}px;" if fixed_size else ""
        text_align_css = f"text-align: {align};" if align != "center" else ""

        # Checked state (for toggle buttons)
        checked_state = ""
        if checked_bg or checked_text:
            checked_state = f"""
            QPushButton:checked {{
                background-color: {checked_bg or bg};
                color: {checked_text or final_text_color};
                font-weight: bold;
            }}"""

        # Focus state
        focus_state = (
            f"""
            QPushButton:focus {{
                outline: {outline};
            }}"""
            if outline
            else ""
        )

        # Pressed background: darker than hover for clear feedback
        if severity:
            pressed_bg = pressed_color
        elif transparent:
            pressed_bg = BG_INPUT
        else:
            pressed_bg = BG_MAIN  # Darker than hover for inset effect

        return f"""
            QPushButton {{
                background-color: {bg};
                color: {final_text_color};
                border: {border};
                border-radius: {LAYOUT["radius"]}px;
                padding: {final_padding};
                font-size: {font_size}px;
                font-weight: 500;
                min-width: {final_min_width}px;
                {size_constraint}
                {text_align_css}
                {text_decoration}
            }}
            QPushButton:hover {{
                background-color: {hover_bg};
                border-color: {hover_border_color};
                color: {hover_text};
                {hover_decoration}
            }}
            QPushButton:pressed {{
                background-color: {pressed_bg};
            }}
            QPushButton:disabled {{
                background-color: {BG_DISABLED};
                color: {TEXT_DISABLED_MUTED};
                border: 1px solid {BORDER_DISABLED};
            }}{checked_state}{focus_state}
        """

    @classmethod
    def groupbox(
        cls,
        variant: str = "normal",
        severity: str | None = None,
        title_align: str = "left",
        min_width: int | None = None,
        max_width: int | None = None,
        min_height: int | None = None,
    ) -> str:
        """Groupbox builder for cards and sections.

        Args:
            variant: 'normal' (borderless), 'card' (bordered), 'minimal' (subtle)
            severity: 'ok', 'warning', 'critical' for border color
        """
        if severity:
            border_color = cls.get_severity_color(severity)
        elif variant == "minimal":
            border_color = SEPARATOR_SUBTLE
        else:
            border_color = BORDER

        if variant == "card" or variant == "minimal":
            # Stream card style
            size_css = ""
            if min_width:
                size_css += f"min-width: {min_width}px; "
            if max_width:
                size_css += f"max-width: {max_width}px; "
            if min_height:
                size_css += f"min-height: {min_height}px; "

            title_position = f"top {title_align}"

            # Minimal variant has more subtle title (no border, muted color)
            if variant == "minimal":
                title_style = f"""
                    subcontrol-origin: margin;
                    subcontrol-position: {title_position};
                    padding: {LAYOUT["padding_xs"]}px 0;
                    color: {TEXT_MUTED};
                    background-color: transparent;
                    font: 500 {FONTS["size_sm"]}px;
                """
            else:
                title_style = f"""
                    subcontrol-origin: margin;
                    subcontrol-position: {title_position};
                    padding: {LAYOUT["padding_xs"]}px {LAYOUT["padding_xs"]}px;
                    color: {TEXT_MAIN};
                    background-color: {BG_MAIN};
                    font: bold {FONTS["size_sm"]}px;
                    border: 1px solid {border_color};
                    border-radius: {LAYOUT["radius_sm"]}px;
                """

            # Minimal variant has no border, card variant has visible border
            border_css = "none" if variant == "minimal" else f"1px solid {border_color}"

            return f"""
                QGroupBox {{
                    background-color: {BG_PANEL};
                    border: {border_css};
                    border-radius: {LAYOUT["radius"]}px;
                    margin: {LAYOUT["padding_xs"]}px;
                    padding: {LAYOUT["padding"]}px;
                    padding-top: {LAYOUT["spacing_xl"]}px;
                    {size_css}
                }}
                QGroupBox::title {{
                    {title_style}
                }}
            """
        else:
            # Normal groupbox
            return _GROUPBOX_BASE

    @classmethod
    def container(
        cls,
        bg: str = "transparent",
        border: bool | str = False,
        border_color: str | None = None,
        padding: int | str | None = None,
        margin: int | str | None = None,
        radius: int | None = None,
        severity: str | None = None,
        variant: str = "normal",
    ) -> str:
        """Universal container builder.

        Args:
            bg: 'transparent', 'panel', 'input', 'main', or hex color
            border: False, True, 'top', 'bottom'
            variant: 'normal', 'issue'
        """
        if variant == "issue":
            # Metadata issue container
            if severity == "critical":
                return f"""
                    QWidget {{
                        background-color: {BG_ERROR_SUBTLE};
                        border: none;
                        border-radius: {LAYOUT["radius"]}px;
                        padding: {LAYOUT["padding_xs"]}px;
                        margin: {LAYOUT["padding_xs"]}px 0px;
                    }}
                """
            else:
                return f"""
                    QWidget {{
                        background-color: transparent;
                        border: none;
                        padding: {LAYOUT["padding_xs"]}px;
                        margin: {LAYOUT["padding_xs"]}px 0px;
                    }}
                """
        # Background color mapping
        bg_map = {
            "transparent": "transparent",
            "panel": BG_PANEL,
            "input": BG_INPUT,
            "main": BG_MAIN,
        }
        final_bg = bg_map.get(bg, bg)  # Use mapping or direct color

        # Border styling
        border_css = cls._build_border_css(border, border_color or BORDER)

        # Padding/margin
        final_padding = f"{padding}px" if isinstance(padding, int) else (padding or "0px")
        final_margin = f"{margin}px" if isinstance(margin, int) else (margin or "0px")
        final_radius = radius if radius is not None else LAYOUT["radius"]

        return f"""
            QWidget {{
                background-color: {final_bg};
                {border_css}
                border-radius: {final_radius}px;
                padding: {final_padding};
                margin: {final_margin};
            }}
        """

    @classmethod
    def frame(
        cls,
        variant: str = "default",
        id_selector: str | None = None,
        bg: str | None = None,
        border: bool | str | None = None,
        border_color: str | None = None,
        radius: int | None = None,
        padding: int | str | None = None,
        margin: int | str | None = None,
        min_width: int | None = None,
        max_width: int | None = None,
        min_height: int | None = None,
        max_height: int | None = None,
        hover_bg: str | None = None,
        hover_border: str | None = None,
    ) -> str:
        """QFrame builder for badges and cards.

        Args:
            variant: 'default', 'pill'
            id_selector: Qt object name for QFrame#id_selector targeting
            hover_bg: Background color on hover
            hover_border: Border color on hover
        """
        # Variant presets
        presets = {
            "default": {
                "bg": BG_PANEL,
                "border": True,
                "border_color": BORDER,
                "radius": LAYOUT["radius"],
                "padding": LAYOUT["padding"],
            },
            "pill": {
                "bg": BG_PANEL,
                "border": False,
                "border_color": BORDER,
                "radius": 8,
                "padding": 14,
            },
        }

        preset = presets.get(variant, presets["default"])

        # Apply overrides (explicit params override preset)
        bg if bg is not None else preset["bg"]
        final_border = border if border is not None else preset["border"]
        final_border_color = border_color if border_color is not None else preset["border_color"]
        final_radius = radius if radius is not None else preset["radius"]
        padding_value = padding if padding is not None else preset["padding"]

        # Convert padding/margin to CSS strings
        final_padding = (
            f"{padding_value}px"
            if isinstance(padding_value, int)
            else (padding_value or f"{LAYOUT['padding']}px")
        )
        final_margin = f"{margin}px" if isinstance(margin, int) else (margin or "0px")

        # Border styling
        border_css = cls._build_border_css(final_border, final_border_color)

        # Size constraints
        size_css = ""
        if min_width:
            size_css += f"min-width: {min_width}px; "
        if max_width:
            size_css += f"max-width: {max_width}px; "
        if min_height:
            size_css += f"min-height: {min_height}px; "
        if max_height:
            size_css += f"max-height: {max_height}px; "

        selector = f"QFrame#{id_selector}" if id_selector else "QFrame"

        hover_state = ""
        if hover_bg or hover_border:
            hover_bg_css = f"background-color: {hover_bg};" if hover_bg else ""
            hover_border_css = f"border-color: {hover_border};" if hover_border else ""
            hover_state = f"""
            {selector}:hover {{
                {hover_bg_css}
                {hover_border_css}
            }}"""

        return f"""
            {selector} {{
                background-color: {bg};
                {border_css}
                border-radius: {final_radius}px;
                padding: {final_padding};
                margin: {final_margin};
                {size_css}
            }}{hover_state}
        """

    @classmethod
    def separator(cls, color: str | None = None, margin: int | None = None) -> str:
        """Horizontal separator line (QFrame with HLine shape)."""
        final_color = color or SEPARATOR_SUBTLE
        final_margin = margin if margin is not None else LAYOUT["spacing_sm"]
        return f"""
            QFrame {{
                background: transparent;
                border-top: 1px solid {final_color};
                max-height: 1px;
                margin: {final_margin}px 0px;
            }}
        """

    @classmethod
    def input(
        cls,
        state: str = "normal",
        clean: bool = False,
        size: int | None = None,
        weight: str = "normal",
    ) -> str:
        """Input field builder with state colors.

        Args:
            state: 'normal', 'error', 'warning', 'success'
            clean: Borderless style for large inline inputs
        """
        state_colors = {
            "normal": TEXT_MAIN,
            "error": STATUS_ERROR,
            "warning": STATUS_WARN,
            "success": STATUS_OK,
        }
        text_color = state_colors.get(state, TEXT_MAIN)

        if clean:
            # Clean style (borderless, transparent background)
            final_size = size or FONTS["size_xl"]
            font_weight = "font-weight: bold;" if weight == "bold" else ""

            return f"""
                QLineEdit {{
                    font-size: {final_size}px;
                    {font_weight}
                    padding: {LAYOUT["padding_lg"]}px;
                    border: none;
                    background-color: transparent;
                    color: {text_color};
                }}
                QLineEdit:focus {{
                    background-color: {BG_PANEL};
                    border-radius: {LAYOUT["radius"]}px;
                }}
            """
        else:
            # Standard input with state color - match _INPUT_BASE for consistency
            final_size = size or FONTS["size"]

            return f"""
                QLineEdit {{
                    background-color: {BG_INPUT};
                    color: {text_color};
                    border: 1px solid {BORDER_INPUT};
                    padding: {LAYOUT["padding_sm"]}px {LAYOUT["padding"]}px;
                    border-radius: {LAYOUT["radius_sm"]}px;
                    font-size: {final_size}px;
                    min-height: 1.4em;
                }}
                QLineEdit:focus {{
                    background-color: {BG_PANEL};
                    border-color: {ACCENT};
                }}
            """

    @classmethod
    def combobox(
        cls,
        state: str = "normal",
        size: int | None = None,
        weight: str = "normal",
        min_width: int | None = None,
    ) -> str:
        """ComboBox builder with state colors.

        Args:
            state: 'normal', 'error', 'warning', 'success'
        """
        state_colors = {
            "normal": TEXT_MAIN,
            "error": STATUS_ERROR,
            "warning": STATUS_WARN,
            "success": STATUS_OK,
        }
        text_color = state_colors.get(state, TEXT_MAIN)

        final_size = size or FONTS["size"]
        font_weight = "font-weight: bold;" if weight == "bold" else ""
        min_width_css = f"min-width: {min_width}px;" if min_width else ""

        return f"""
            QComboBox {{
                background-color: {BG_INPUT};
                color: {text_color};
                border: 1px solid {BORDER_INPUT};
                padding: {LAYOUT["padding_sm"]}px {LAYOUT["padding"]}px;
                border-radius: {LAYOUT["radius_sm"]}px;
                min-height: 1.4em;
                font-size: {final_size}px;
                {font_weight}
                {min_width_css}
            }}
            QComboBox:focus {{ background-color: {BG_PANEL}; border-color: {ACCENT}; }}
            QComboBox::drop-down {{
                border: none;
                background-color: transparent;
                border-left: 1px solid {BORDER_INPUT};
                border-radius: 0px {LAYOUT["radius_sm"]}px {LAYOUT["radius_sm"]}px 0px;
                width: 16px;
            }}
            QComboBox::down-arrow {{
                border-style: solid;
                border-width: 2px 2px 0 2px;
                border-color: {TEXT_DISABLED} transparent transparent transparent;
                margin: {LAYOUT["padding_xs"]}px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {BG_PANEL};
                color: {TEXT_MAIN};
                border: 1px solid {BORDER};
                border-radius: {LAYOUT["radius_sm"]}px;
                selection-background-color: {BG_ELEVATED};
                selection-color: {TEXT_MAIN};
                outline: none;
            }}
            QComboBox QAbstractItemView::item {{
                min-height: 20px;
                padding: 3px {LAYOUT["padding_sm"]}px;
                background-color: transparent;
            }}
            QComboBox QAbstractItemView::item:hover {{
                background-color: {BG_ELEVATED};
                color: {TEXT_MAIN};
            }}
            QComboBox QAbstractItemView::item:selected {{
                background-color: {BG_ELEVATED};
                color: {TEXT_MAIN};
            }}
        """

    @classmethod
    def status_icon(cls, color: str | None = None, size: int = 12) -> str:
        """Status icon builder with fixed 16x16 dimensions."""
        icon_color = color or TEXT_DISABLED
        return cls.label(
            color=icon_color,
            size=size,
            width=16,
            height=16,
            max_width=16,
            max_height=16,
            align="center",
        )

    @classmethod
    def navigation_button(cls, size: str = "normal") -> str:
        """Navigation switcher button style. Args: size='normal' or 'large'."""
        if size == "large":
            padding = f"{LAYOUT['padding_md']}px {LAYOUT['spacing_xxl']}px"
            font_size = FONTS["size_md"]
            min_width = 105
        else:
            padding = f"{LAYOUT['padding']}px {LAYOUT['spacing_xxl']}px"
            font_size = FONTS["size_sm"]
            min_width = 95

        return f"""
            QPushButton {{
                background-color: transparent;
                color: {TEXT_MUTED};
                border: none;
                padding: {padding};
                border-radius: {LAYOUT["radius"]}px;
                font-size: {font_size}px;
                font-weight: 500;
                min-width: {min_width}px;
                outline: none;
            }}
            QPushButton:hover {{
                color: {TEXT_LABEL};
                background-color: transparent;
            }}
            QPushButton:checked {{
                background-color: {BG_PANEL};
                color: {TEXT_MAIN};
                font-weight: 600;
            }}
        """

    @classmethod
    def checkbox_colored(cls, color: str) -> str:
        """Checkbox styling with custom text color."""
        return f"""
            QCheckBox {{ color: {color}; spacing: {LAYOUT["padding_sm"]}px; }}
            QCheckBox::indicator {{
                width: 13px; height: 13px;
                border: 1px solid {BORDER};
                background-color: {BG_INPUT};
                border-radius: 2px;
            }}
            QCheckBox::indicator:hover {{ border-color: {ACCENT}; background-color: {BG_PANEL}; }}
            QCheckBox::indicator:checked {{ background-color: {ACCENT}; border-color: {ACCENT}; }}
        """

    @classmethod
    def navigation_container(cls) -> str:
        """Navigation switcher container - minimal transparent design."""
        return f"""
            QWidget {{
                background-color: transparent;
                border-radius: {LAYOUT["radius_lg"]}px;
                padding: {LAYOUT["padding_xs"]}px;
            }}
        """

    @classmethod
    def table(cls) -> str:
        """QTableView styling for data tables."""
        return f"""
            QTableView {{
                background-color: {BG_PANEL};
                color: {TEXT_MAIN};
                gridline-color: transparent;
                border: 1px solid {BORDER};
                border-radius: {LAYOUT["radius_lg"]}px;
                selection-background-color: {BG_ELEVATED};
                selection-color: {TEXT_MAIN};
                font: {FONTS["size"]}px;
            }}
            QTableView::item {{
                padding: {LAYOUT["padding"]}px {LAYOUT["padding_lg"]}px;
                border: none;
            }}
            QTableView::item:selected {{
                background-color: {BG_ELEVATED};
                color: {TEXT_MAIN};
            }}
            QTableView::item:selected:hover {{
                background-color: {BG_ELEVATED};
                color: {TEXT_MAIN};
            }}
            QTableView::item:hover {{
                background-color: {BG_INPUT};
            }}
            QHeaderView::section {{
                background-color: {BG_MAIN};
                color: {TEXT_MAIN};
                padding: {LAYOUT["padding_sm"]}px {LAYOUT["padding"]}px;
                border: none;
                border-bottom: 2px solid {BORDER};
                border-right: 1px solid {BORDER};
                font: bold {FONTS["size"]}px;
            }}
            QHeaderView::section:hover {{
                background-color: {BG_INPUT};
            }}
            QTableView::corner {{
                background-color: {BG_MAIN};
                border: none;
            }}
        """

    @classmethod
    def slider(cls) -> str:
        """QSlider builder with custom groove and handle."""
        return f"""
            QSlider {{
                background: transparent;
            }}
            QSlider::groove:horizontal {{
                background: {BG_PANEL};
                border: 1px solid {BORDER};
                border-radius: 3px;
                height: 6px;
            }}
            QSlider::handle:horizontal {{
                background: {ACCENT};
                border: 1px solid {ACCENT_HOVER};
                border-radius: 4px;
                width: 14px;
                margin: -5px 0;
            }}
            QSlider::handle:horizontal:hover {{
                background: {ACCENT_HOVER};
            }}
            QSlider::sub-page:horizontal {{
                background: {ACCENT};
                border-radius: 3px;
            }}
        """

    @classmethod
    def progress_bar(cls) -> str:
        """QProgressBar builder with accent chunk."""
        return f"""
            QProgressBar {{
                border: 1px solid {BORDER};
                border-radius: 4px;
                text-align: center;
                background: {BG_PANEL};
            }}
            QProgressBar::chunk {{
                background: {ACCENT};
                border-radius: 3px;
            }}
        """

    @classmethod
    def tree_widget(cls) -> str:
        """QTreeWidget builder with hover/selection states."""
        return f"""
            QTreeWidget {{
                background: {BG_MAIN};
                border: none;
                border-radius: {RADIUS["md"]}px;
            }}
            QTreeWidget::item {{
                padding: {PADDING["xs"]}px;
            }}
            QTreeWidget::item:hover {{
                background: #1e1e1e;
            }}
            QTreeWidget::item:selected {{
                background: {BG_INPUT};
            }}
        """

    @classmethod
    def metric_card(cls) -> str:
        """Elevated metric card frame."""
        return f"""
            QFrame {{
                background: {BG_ELEVATED};
                border-radius: {RADIUS["md"]}px;
                padding: {PADDING["sm"]}px;
            }}
        """

    @classmethod
    def metric_card_title(cls) -> str:
        """Uppercase muted title label for metric cards."""
        return f"""
            color: {TEXT_MUTED};
            font-size: {FONT_SIZE["xs"]}px;
            font-weight: 600;
            letter-spacing: 1px;
            background-color: transparent;
        """

    @classmethod
    def metric_card_value(cls, size: int | None = None, weight: int = 700) -> str:
        """Large value label for metric cards."""
        final_size = size or FONT_SIZE["xl"]
        return f"""
            color: {TEXT_MAIN};
            font-size: {final_size}px;
            font-weight: {weight};
            background-color: transparent;
        """

    @classmethod
    def section_header(cls) -> str:
        """Small uppercase section header (e.g., 'METRICS')."""
        return f"""
            color: {TEXT_MUTED};
            font-size: {FONT_SIZE["xs"]}px;
            font-weight: 600;
            letter-spacing: 1px;
            text-transform: uppercase;
            background-color: transparent;
        """

    @classmethod
    def benchmark_table(cls) -> str:
        """QTableWidget style for benchmark results."""
        return f"""
            QTableWidget {{ background-color: {BG_MAIN}; gridline-color: {BORDER}; }}
            QTableWidget::item {{ padding: {SPACING["md"]}px; }}
            QHeaderView::section {{ background-color: {BG_PANEL}; padding: {SPACING["md"]}px; border: none; }}
        """

    @classmethod
    def inline_label(cls, color: str | None = None, size: int | None = None, weight: int | None = None) -> str:
        """Simple inline label style with transparent background.

        Use for status labels, metric displays, info labels, etc.
        """
        final_color = color or TEXT_MAIN
        css_parts = [f"color: {final_color}"]
        if size:
            css_parts.append(f"font-size: {size}px")
        if weight:
            css_parts.append(f"font-weight: {weight}")
        css_parts.append("background-color: transparent")
        return "; ".join(css_parts) + ";"

    @classmethod
    def section_card(cls) -> str:
        """Card-style section container with elevated background and border."""
        return f"""
            #section_card {{
                background-color: {BG_ELEVATED};
                border: 1px solid {BORDER_INPUT};
                border-radius: {RADIUS["md"]}px;
            }}
            QLabel {{
                background-color: transparent;
            }}
        """

    @classmethod
    def collapsible_header(cls) -> str:
        """Collapsible section header button style."""
        return f"""
            QPushButton {{
                background-color: {BG_ELEVATED};
                border: 1px solid {BORDER_INPUT};
                border-radius: {RADIUS["md"]}px;
                padding: {SPACING["lg"]}px {SPACING["xl"]}px;
                text-align: left;
                color: {TEXT_MUTED};
                font-size: {FONT_SIZE["sm"]}px;
                font-weight: 600;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background-color: {BG_TAB_ACTIVE};
            }}
            QPushButton:checked {{
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
                border-bottom: none;
            }}
        """

    @classmethod
    def collapsible_content(cls) -> str:
        """Collapsible section content container style."""
        return f"""
            #collapsible_content {{
                background-color: {BG_ELEVATED};
                border: 1px solid {BORDER_INPUT};
                border-top: none;
                border-top-left-radius: 0px;
                border-top-right-radius: 0px;
                border-bottom-left-radius: {RADIUS["md"]}px;
                border-bottom-right-radius: {RADIUS["md"]}px;
            }}
            QLabel {{
                background-color: transparent;
            }}
        """

    button_stop = f"""
        QPushButton {{
            background-color: transparent;
            color: {STATUS_ERROR};
            border: 1px solid {STATUS_ERROR};
            padding: {LAYOUT["padding_sm"]}px {LAYOUT["padding_lg"]}px;
            border-radius: {LAYOUT["radius"]}px;
            font: bold {FONTS["size"]}px;
            min-width: 70px;
        }}
        QPushButton:hover {{ background-color: {STATUS_ERROR}; color: {TEXT_MAIN}; border-color: {STATUS_ERROR}; }}
        QPushButton:pressed {{ background-color: {STATUS_ERROR_PRESSED}; color: {TEXT_MAIN}; }}
        QPushButton:disabled {{ background-color: {BG_DISABLED}; color: {TEXT_DISABLED_MUTED}; border: 1px solid {BORDER_DISABLED}; }}
    """

    spinbox = f"""
        QSpinBox {{
            background-color: {BG_INPUT};
            color: {TEXT_MAIN};
            border: none;
            padding: {LAYOUT["padding_sm"]}px {LAYOUT["padding"]}px;
            border-radius: {LAYOUT["radius_sm"]}px;
            min-height: 1.4em;
            min-width: 80px;  /* macOS compatibility */
        }}
        QSpinBox:focus {{ background-color: {BG_PANEL}; }}
        QSpinBox::up-button {{
            background-color: transparent;
            border: none;
            border-left: 1px solid {BORDER};
            border-radius: 0px {LAYOUT["radius_sm"]}px 0px 0px;
            width: 16px;
        }}
        QSpinBox::up-button:hover {{ background-color: {ACCENT}; }}
        QSpinBox::up-arrow {{
            border-style: solid;
            border-width: 0 2px 2px 2px;
            border-color: transparent transparent {TEXT_DISABLED} transparent;
            margin: {LAYOUT["padding_xs"]}px;
        }}
        QSpinBox::down-button {{
            background-color: transparent;
            border: none;
            border-left: 1px solid {BORDER};
            border-radius: 0px 0px {LAYOUT["radius_sm"]}px 0px;
            width: 16px;
        }}
        QSpinBox::down-button:hover {{ background-color: {ACCENT}; }}
        QSpinBox::down-arrow {{
            border-style: solid;
            border-width: 2px 2px 0 2px;
            border-color: {TEXT_DISABLED} transparent transparent transparent;
            margin: {LAYOUT["padding_xs"]}px;
        }}
    """

    checkbox = f"""
        QCheckBox {{ color: {TEXT_LABEL}; spacing: {LAYOUT["padding_sm"]}px; }}
        QCheckBox::indicator {{
            width: 13px; height: 13px;
            border: 1px solid {BORDER};
            background-color: {BG_INPUT};
            border-radius: 2px;
        }}
        QCheckBox::indicator:hover {{ border-color: {ACCENT}; background-color: {BG_PANEL}; }}
        QCheckBox::indicator:checked {{ background-color: {ACCENT}; border-color: {ACCENT}; }}
    """

    radiobutton = f"""
        QRadioButton {{
            color: {TEXT_LABEL};
            spacing: {LAYOUT["padding_sm"]}px;
            padding: {LAYOUT["padding_xs"]}px;
        }}
        QRadioButton::indicator {{
            width: 13px;
            height: 13px;
            border: 1px solid {BORDER};
            background-color: {BG_INPUT};
            border-radius: 7px;
        }}
        QRadioButton::indicator:hover {{
            border-color: {ACCENT};
            background-color: {BG_PANEL};
        }}
        QRadioButton::indicator:checked {{
            background-color: {ACCENT};
            border-color: {ACCENT};
            border-width: 3px;
        }}
        QRadioButton::indicator:checked:hover {{
            background-color: {ACCENT_HOVER};
            border-color: {ACCENT_HOVER};
        }}
    """

    scrollarea = """
        QScrollArea { background-color: transparent; border: none; }
        QScrollBar:vertical {
            background: transparent;
            width: 6px;
            margin: 0;
        }
        QScrollBar::handle:vertical {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            min-height: 20px;
        }
        QScrollBar::handle:vertical:hover {
            background: rgba(255, 255, 255, 0.2);
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0;
        }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
            background: transparent;
        }
        QScrollBar:horizontal {
            background: transparent;
            height: 6px;
            margin: 0;
        }
        QScrollBar::handle:horizontal {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            min-width: 20px;
        }
        QScrollBar::handle:horizontal:hover {
            background: rgba(255, 255, 255, 0.2);
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0;
        }
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
            background: transparent;
        }
    """

    tablewidget = f"""
        QTableWidget {{
            background-color: {BG_INPUT};
            color: {TEXT_LABEL};
            gridline-color: {BORDER};
            border: 1px solid {BORDER};
            border-radius: {LAYOUT["radius"]}px;
            selection-background-color: {ACCENT};
            selection-color: {TEXT_MAIN};
        }}
        QTableWidget::item {{
            padding: {LAYOUT["padding_sm"]}px {LAYOUT["padding"]}px;
        }}
        QTableWidget::item:selected {{
            background-color: {ACCENT};
            color: {TEXT_MAIN};
            border: 1px solid {ACCENT};
        }}
        QTableWidget::item:hover {{
            background-color: {BG_PANEL};
            border: 1px solid {ACCENT};
        }}
        QHeaderView::section {{
            background-color: {BG_PANEL};
            color: {TEXT_MAIN};
            padding: {LAYOUT["padding_sm"]}px {LAYOUT["padding"]}px;
            border: none;
            border-bottom: 2px solid {BORDER};
            border-right: 1px solid {BORDER};
            font: bold {FONTS["size"]}px;
        }}
        QHeaderView::section:hover {{
            background-color: {BG_INPUT};
            border-bottom: 2px solid {ACCENT};
        }}
    """

    tabwidget = f"""
        QTabWidget {{
            background-color: transparent;
            border: none;
        }}
        QTabWidget::pane {{
            background-color: transparent;
            border: none;
            border-radius: {LAYOUT["radius"]}px;
            padding: {LAYOUT["padding"]}px;
        }}
        QTabBar::tab {{
            background-color: {BG_MAIN};
            color: {TEXT_LABEL};
            padding: {LAYOUT["padding_md"]}px {LAYOUT["spacing_xxl"]}px;
            margin-right: {LAYOUT["spacing_xs"]}px;
            border-top-left-radius: {LAYOUT["radius"]}px;
            border-top-right-radius: {LAYOUT["radius"]}px;
            font-weight: 500;
        }}
        QTabBar::tab:selected {{
            background-color: {BG_PANEL};
            color: {TEXT_MAIN};
            border: 1px solid {BORDER};
            border-bottom: 2px solid {ACCENT};
            font-weight: bold;
        }}
        QTabBar::tab:hover {{
            background-color: {BG_INPUT};
            color: {TEXT_MAIN};
        }}
    """

    textedit_log = f"""
        QPlainTextEdit, QTextEdit {{
            background-color: {BG_MAIN};
            color: {TEXT_LABEL};
            border: none;
            border-radius: {LAYOUT["radius"]}px;
            padding: {LAYOUT["padding_sm"]}px;
            font-family: 'Consolas', monospace;
            font-size: 11px;
        }}
    """

    messagebox = f"""
        QMessageBox {{ background-color: {BG_MAIN}; color: {TEXT_MAIN}; }}
        QMessageBox QWidget {{ background-color: transparent; }}
        QMessageBox QFrame {{ background-color: transparent; border: none; }}
        QMessageBox QDialogButtonBox {{ background-color: transparent; }}
        QMessageBox QLabel {{ color: {TEXT_MAIN}; }}
        /* Default button style - emphasized (for OK, No, Cancel = safe options) */
        QMessageBox QPushButton {{
            background-color: {ACCENT};
            color: {TEXT_MAIN};
            border: none;
            padding: {LAYOUT["padding_sm"]}px {LAYOUT["padding_lg"]}px;
            border-radius: {LAYOUT["radius"]}px;
            font-weight: bold;
            min-width: 70px;
        }}
        QMessageBox QPushButton:hover {{ background-color: {ACCENT_HOVER}; }}
        /* Yes button - subtle for destructive confirmations */
        QMessageBox QPushButton[text="Yes"], QMessageBox QPushButton[text="&Yes"] {{
            background-color: transparent;
            color: {TEXT_LABEL};
            border: 1px solid {BORDER};
        }}
        QMessageBox QPushButton[text="Yes"]:hover, QMessageBox QPushButton[text="&Yes"]:hover {{
            background-color: {BG_PANEL};
            color: {TEXT_MAIN};
            border-color: {TEXT_LABEL};
        }}
    """

    dialog_buttonbox = f"""
        QDialogButtonBox {{
            background-color: transparent;
            spacing: {LAYOUT["padding"]}px;
        }}
        /* Cancel/Reject button - subtle, de-emphasized */
        QDialogButtonBox QPushButton[role="reject"] {{
            background-color: transparent;
            color: {TEXT_LABEL};
            border: 1px solid {BORDER};
            padding: {LAYOUT["padding"]}px {LAYOUT["spacing_xl"]}px;
            border-radius: {LAYOUT["radius"]}px;
            font-size: {FONTS["size"]}px;
            min-width: 80px;
            margin: {LAYOUT["spacing_xs"]}px 2px;
        }}
        QDialogButtonBox QPushButton[role="reject"]:hover {{
            background-color: {BG_PANEL};
            color: {TEXT_MAIN};
            border-color: {TEXT_LABEL};
        }}

        /* OK/Accept button - emphasized, primary action */
        QDialogButtonBox QPushButton[role="accept"] {{
            background-color: {BG_PANEL};
            color: {TEXT_MAIN};
            border: 2px solid {TEXT_LABEL};
            padding: {LAYOUT["padding"]}px {LAYOUT["spacing_xl"]}px;
            border-radius: {LAYOUT["radius"]}px;
            font: bold {FONTS["size"]}px;
            min-width: 80px;
            margin: {LAYOUT["spacing_xs"]}px 2px;
        }}
        QDialogButtonBox QPushButton[role="accept"]:hover {{
            background-color: {BORDER};
            border-color: {TEXT_MAIN};
        }}

        /* Fallback for buttons without explicit role */
        QDialogButtonBox QPushButton {{
            background-color: {BG_PANEL};
            color: {TEXT_LABEL};
            border: 1px solid {BORDER};
            padding: {LAYOUT["padding"]}px {LAYOUT["spacing_xl"]}px;
            border-radius: {LAYOUT["radius"]}px;
            font-size: {FONTS["size"]}px;
            min-width: 80px;
            margin: {LAYOUT["spacing_xs"]}px 2px;
        }}
        QDialogButtonBox QPushButton:hover {{
            background-color: {BORDER};
            color: {TEXT_MAIN};
        }}
    """

    @classmethod
    def get_severity_color(cls, severity: str = "ok") -> str:
        """Get color for severity level."""
        severity_map = {
            "ok": STATUS_OK,
            "minor": STATUS_WARN,
            "warning": STATUS_WARN,
            "critical": STATUS_ERROR,
            "error": STATUS_ERROR,
        }
        return severity_map.get(severity, STATUS_ERROR)

    @classmethod
    def dialog(cls) -> str:
        """Base dialog style."""
        return _DIALOG_BASE

    @classmethod
    def transparent_container(cls) -> str:
        """Transparent container style."""
        return "QWidget { background-color: transparent; }"

    layout = LAYOUT
    fonts = FONTS


def apply_app_styles(app) -> None:
    """Apply global styles to the entire application."""
    global_style = f"""
        QMainWindow {{ background-color: {BG_MAIN}; color: {TEXT_LABEL}; }}
        QDialog {{
            background-color: {BG_MAIN};
            color: {TEXT_LABEL};
            border: none;
        }}
        QToolTip {{
            background-color: {BG_PANEL};
            color: {TEXT_MAIN};
            border: none;
            border-radius: {LAYOUT["radius"]}px;
            padding: {LAYOUT["padding_xs"]}px;
        }}
        QGroupBox {{
            padding-top: {LAYOUT["groupbox_padding_top"]}px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            margin-top: 0px;
        }}
        QScrollBar:vertical {{
            background: transparent;
            width: 6px;
            margin: 0;
        }}
        QScrollBar::handle:vertical {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            min-height: 20px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: rgba(255, 255, 255, 0.2);
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0;
        }}
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
            background: transparent;
        }}
        QScrollBar:horizontal {{
            background: transparent;
            height: 6px;
            margin: 0;
        }}
        QScrollBar::handle:horizontal {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            min-width: 20px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background: rgba(255, 255, 255, 0.2);
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0;
        }}
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
            background: transparent;
        }}
        {WidgetStyles.messagebox}
    """
    app.setStyleSheet(global_style)

    palette = app.palette()
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(ACCENT))
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#ffffff"))
    app.setPalette(palette)
