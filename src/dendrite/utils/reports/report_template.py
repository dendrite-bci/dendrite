"""
Shared HTML report template and styling utilities.

Provides reusable components for generating consistent, clean HTML reports
with a modern dark theme across all Dendrite report types.

Uses the centralized design tokens from dendrite.gui.styles.design_tokens to ensure
visual consistency between GUI and HTML reports.
"""

import re
from datetime import datetime
from typing import Any

# Import design tokens for consistent styling across GUI and reports
from dendrite.gui.styles.design_tokens import (
    ACCENT,
    BG_MAIN,
    BG_PANEL,
    BORDER,
    FONT_FAMILY,
    FONT_SIZE,
    PADDING,
    SPACING,
    TEXT_DISABLED,
    TEXT_LABEL,
    TEXT_MAIN,
    TEXT_MUTED,
)

# Syntax highlighting colors for code blocks (VS Code inspired)
CODE_COLORS = {
    "keyword": "#569cd6",  # Blue for keywords/keys
    "string": "#ce9178",  # Orange for strings/values
    "type": "#4ec9b0",  # Teal for types/datasets
}


def get_base_style() -> str:
    """Get the base CSS styling for all reports using design tokens."""
    return f"""
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
            background-color: {BG_MAIN};
            color: {TEXT_LABEL};
            line-height: 1.6;
            font-size: {FONT_SIZE["md"]}px;
            padding: 0;
            margin: 0;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: {SPACING["3xl"]}px;
        }}

        .header {{
            padding: {SPACING["2xl"]}px 0;
            margin-bottom: {SPACING["2xl"]}px;
            text-align: center;
            border-bottom: 2px solid {BORDER};
        }}

        .header h1 {{
            font-size: {FONT_SIZE["2xl"]}px;
            font-weight: 600;
            color: {TEXT_MAIN};
            margin-bottom: {SPACING["md"]}px;
        }}

        .header .subtitle {{
            color: {ACCENT};
            font-size: {FONT_SIZE["md"]}px;
            margin-bottom: {SPACING["xs"]}px;
        }}

        .header .timestamp {{
            color: {TEXT_DISABLED};
            font-size: {FONT_SIZE["sm"]}px;
        }}

        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: {SPACING["xl"]}px;
            margin-bottom: {SPACING["2xl"]}px;
        }}

        .info-item {{
            background-color: {BG_PANEL};
            padding: {PADDING["md"]}px {PADDING["lg"]}px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: {SPACING["xl"]}px;
        }}

        .info-label {{
            font-weight: 500;
            color: {TEXT_LABEL};
            font-size: {FONT_SIZE["sm"]}px;
        }}

        .info-value {{
            color: {ACCENT};
            font-family: {FONT_FAMILY["monospace"]};
            font-size: {FONT_SIZE["sm"]}px;
            word-break: break-word;
        }}

        .section {{
            margin-bottom: {SPACING["2xl"]}px;
        }}

        .section h2 {{
            color: {TEXT_MAIN};
            padding: {SPACING["xl"]}px 0;
            margin: 0 0 {SPACING["2xl"]}px 0;
            font-size: {FONT_SIZE["xl"]}px;
            font-weight: 600;
            border-bottom: 1px solid {BORDER};
        }}

        .section-content {{
            padding: 0;
        }}

        .plot-container {{
            background-color: {BG_PANEL};
            padding: {SPACING["2xl"]}px;
            margin: {SPACING["xl"]}px 0;
        }}

        /* Plotly plot styling */
        .plotly-graph-div {{
            background-color: transparent !important;
        }}

        .js-plotly-plot .plotly {{
            background-color: transparent !important;
        }}

        .modebar {{
            background-color: rgba(30, 30, 30, 0.8) !important;
        }}

        .modebar-btn {{
            color: {TEXT_LABEL} !important;
        }}

        .modebar-btn:hover {{
            background-color: rgba(50, 50, 50, 0.8) !important;
        }}

        .no-data {{
            color: {TEXT_DISABLED};
            font-style: italic;
            text-align: center;
            padding: {SPACING["2xl"]}px;
        }}

        .code-block {{
            background-color: {BG_PANEL};
            padding: {SPACING["2xl"]}px;
            margin: {SPACING["xl"]}px 0;
            font-family: {FONT_FAMILY["monospace"]};
            font-size: {FONT_SIZE["sm"]}px;
            line-height: 1.5;
            overflow-x: auto;
        }}

        .code-block .file-path {{
            color: {ACCENT};
            font-weight: 600;
            margin-bottom: {SPACING["xl"]}px;
        }}

        .code-block .section-title {{
            color: {TEXT_MAIN};
            font-weight: 600;
            margin-top: {SPACING["2xl"]}px;
            margin-bottom: {SPACING["md"]}px;
        }}

        .code-block .section-title:first-child {{
            margin-top: 0;
        }}

        .code-block .key {{
            color: {CODE_COLORS["keyword"]};
        }}

        .code-block .value {{
            color: {CODE_COLORS["string"]};
        }}

        .code-block .dataset {{
            color: {CODE_COLORS["type"]};
            font-weight: 500;
        }}

        .code-block .indent {{
            margin-left: {SPACING["3xl"]}px;
        }}

        .footer {{
            text-align: center;
            padding: {SPACING["2xl"]}px 0;
            color: {TEXT_DISABLED};
            font-size: {FONT_SIZE["sm"]}px;
            border-top: 1px solid {BORDER};
            margin-top: {SPACING["3xl"]}px;
        }}

        /* Info Table Styles - Clean table for detailed data */
        .info-table {{
            width: 100%;
            border-collapse: collapse;
            margin: {SPACING["xl"]}px 0;
        }}

        .info-table tr {{
            border-bottom: 1px solid rgba(51, 51, 51, 0.5);
        }}

        .info-table tr:last-child {{
            border-bottom: none;
        }}

        .info-table td {{
            padding: {SPACING["md"]}px {SPACING["xl"]}px;
            font-size: {FONT_SIZE["sm"]}px;
        }}

        .info-table td:first-child {{
            color: {TEXT_LABEL};
            font-weight: 500;
            width: 40%;
        }}

        .info-table td:last-child {{
            color: {TEXT_MAIN};
            text-align: right;
        }}

        /* Value type formatting */
        .info-table .value-number {{
            font-family: {FONT_FAMILY["monospace"]};
            color: {ACCENT};
        }}

        .info-table .value-path {{
            font-family: {FONT_FAMILY["monospace"]};
            color: {TEXT_MUTED};
            font-size: {FONT_SIZE["xs"]}px;
        }}
    """


def render_header(title: str, session_id: str) -> str:
    """
    Render report header section.

    Args:
        title: Report title
        session_id: Session identifier

    Returns:
        HTML string for header
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""
        <div class="header">
            <h1>{title}</h1>
            <div class="subtitle">Session: {session_id}</div>
            <div class="timestamp">Generated: {timestamp}</div>
        </div>
    """


def render_info_grid(items: dict[str, Any]) -> str:
    """
    Render info grid with key-value pairs for summary cards (3-5 items).

    Args:
        items: Dictionary of label -> value pairs

    Returns:
        HTML string for info grid
    """
    html = ['<div class="info-grid">']

    for label, value in items.items():
        # Format value based on type
        if isinstance(value, float):
            formatted_value = f"{value:,.2f}" if value > 100 else f"{value:.3f}"
        elif isinstance(value, int):
            formatted_value = f"{value:,}"
        else:
            formatted_value = str(value)

        html.append(f"""
            <div class="info-item">
                <span class="info-label">{label}:</span>
                <span class="info-value">{formatted_value}</span>
            </div>
        """)

    html.append("</div>")
    return "\n".join(html)


def render_info_table(items: dict[str, Any]) -> str:
    """
    Render clean table for detailed key-value data (many rows).

    Auto-detects value types for appropriate formatting:
    - Numbers: monospace font in accent color
    - Paths (contains /): monospace font in muted color
    - Text: normal font in white

    Args:
        items: Dictionary of label -> value pairs

    Returns:
        HTML string for info table
    """
    html = ['<table class="info-table">']

    for label, value in items.items():
        # Determine value type and format accordingly
        value_class = ""
        formatted_value = str(value)

        if isinstance(value, (int, float)):
            # Numbers - monospace in accent color
            if isinstance(value, float):
                formatted_value = f"{value:,.2f}" if abs(value) > 100 else f"{value:.3f}"
            else:
                formatted_value = f"{value:,}"
            value_class = "value-number"
        elif isinstance(value, str):
            # Check if it's a file path
            if "/" in value or "\\" in value:
                value_class = "value-path"
            # Otherwise normal text formatting (no special class)

        html.append(f'''
            <tr>
                <td>{label}</td>
                <td class="{value_class}">{formatted_value}</td>
            </tr>
        ''')

    html.append("</table>")
    return "\n".join(html)


def render_section(title: str, content: str) -> str:
    """
    Render a section with title and content.

    Args:
        title: Section title
        content: Section content (HTML)

    Returns:
        HTML string for section
    """
    return f"""
        <div class="section">
            <h2>{title}</h2>
            <div class="section-content">
                {content}
            </div>
        </div>
    """


def render_plot_container(plot_html: str) -> str:
    """
    Render a plot container.

    Args:
        plot_html: Plotly HTML div string

    Returns:
        HTML string for plot container
    """
    return f"""
        <div class="plot-container">
            {plot_html}
        </div>
    """


def render_footer(lab_name: str = "Neural Brain Recovery Laboratory") -> str:
    """
    Render report footer.

    Args:
        lab_name: Laboratory name

    Returns:
        HTML string for footer
    """
    return f"""
        <div class="footer">
            <p>Report generated by {lab_name}</p>
        </div>
    """


def format_hdf5_structure(inspection_text: str) -> str:
    """
    Format HDF5 file inspection output with syntax highlighting.

    Args:
        inspection_text: Raw inspection output text

    Returns:
        HTML formatted inspection output
    """
    lines = inspection_text.split("\n")
    formatted = []

    for line in lines:
        # File path
        if "Opened HDF5 file:" in line or "File not found:" in line:
            path = line.split(":", 1)[1].strip() if ":" in line else line
            formatted.append(f'<div class="file-path">📁 {path}</div>')
            continue

        # Section headers
        if line.strip().endswith(":") and not line.strip().startswith((" ", "-")):
            section_name = line.strip().rstrip(":")
            if section_name in [
                "File Attributes",
                "Datasets and Groups in the file",
                "Datasets found",
                "Event Dataset Contents",
                "Fields",
            ]:
                formatted.append(f'<div class="section-title">{section_name}</div>')
                continue

        # Dataset names
        if re.match(r"^Dataset '(.+)':", line.strip()):
            dataset_name = re.search(r"Dataset '(.+)':", line).group(1)
            formatted.append(f'<div class="dataset">📊 {dataset_name}</div>')
            continue

        # Key-value pairs with indentation
        if ":" in line and line.strip().startswith((" ", "-")):
            indent_level = len(line) - len(line.lstrip())
            indent_class = "indent" if indent_level > 2 else ""

            # Split on first colon only
            parts = line.split(":", 1)
            if len(parts) == 2:
                key = parts[0].strip().lstrip("- ")
                value = parts[1].strip()

                # Format special values
                if key in ["Shape", "Data Type", "dtype"]:
                    formatted.append(
                        f'<div class="{indent_class}"><span class="key">{key}:</span> '
                        f'<span class="value">{value}</span></div>'
                    )
                else:
                    formatted.append(
                        f'<div class="{indent_class}"><span class="key">{key}:</span> {value}</div>'
                    )
                continue

        # Lists (arrays, etc.)
        if line.strip().startswith("[") or "Datasets found:" in line:
            formatted.append(f'<div class="indent">{line.strip()}</div>')
            continue

        # Empty lines
        if not line.strip():
            formatted.append('<div style="height: 8px;"></div>')
            continue

        # Default: regular text
        if line.strip():
            formatted.append(f"<div>{line.strip()}</div>")

    return "\n".join(formatted)


def generate_html_document(
    title: str, session_id: str, body_content: str, custom_css: str = ""
) -> str:
    """
    Generate complete HTML document.

    Args:
        title: Page title
        session_id: Session identifier
        body_content: HTML content for body (inside container)
        custom_css: Additional CSS to append to base styles

    Returns:
        Complete HTML document string
    """
    base_css = get_base_style()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - {session_id}</title>
    <style>
{base_css}
{custom_css}
    </style>
</head>
<body>
    <div class="container">
        {body_content}
    </div>
</body>
</html>
"""
