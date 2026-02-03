"""
Shared plotting utilities for report generation.

Provides common functions for downsampling, statistics computation,
and Plotly plot configuration to ensure consistency across all reports.
"""

import numpy as np
import plotly.graph_objects as go

# Plotly dark theme configuration
PLOTLY_THEME = "plotly_dark"
PLOT_COLORS = {
    "primary": "#007bff",
    "secondary": "#28a745",
    "warning": "#ffc107",
    "grid": "rgba(128,128,128,0.2)",
    "baseline": "#888888",
}


def downsample_1d(array_1d: np.ndarray, max_points: int) -> np.ndarray:
    """
    Downsample a 1D array to max_points using stride.

    Args:
        array_1d: Input array
        max_points: Maximum number of points to keep

    Returns:
        Downsampled array
    """
    if array_1d.size <= max_points:
        return array_1d
    stride = int(np.ceil(array_1d.size / max_points))
    return array_1d[::stride]


def compute_statistics(values: np.ndarray) -> dict[str, float]:
    """
    Compute comprehensive statistics for a metric.

    Args:
        values: Array of values

    Returns:
        Dictionary with statistical measures
    """
    if len(values) == 0:
        return {}

    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def create_time_series_plot(
    x_data: np.ndarray,
    y_data: np.ndarray,
    title: str,
    x_label: str = "Time (seconds)",
    y_label: str = "Value",
    subtitle: str = "",
    height: int = 320,
    show_stats: bool = False,
    stats: dict[str, float] | None = None,
) -> go.Figure:
    """
    Create a standardized time series plot.

    Args:
        x_data: X-axis data (time)
        y_data: Y-axis data (values)
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        subtitle: Optional subtitle with statistics
        height: Plot height in pixels
        show_stats: Whether to show mean/p95 lines
        stats: Pre-computed statistics (if show_stats=True)

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Main trace
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=y_data,
            mode="lines",
            name=title,
            line=dict(color=PLOT_COLORS["primary"], width=2),
            hovertemplate=f"<b>{title}</b><br>{x_label}: %{{x:.3f}}<br>{y_label}: %{{y:.3f}}<extra></extra>",
        )
    )

    # Add statistics lines if requested
    if show_stats and stats:
        # Mean line
        fig.add_hline(
            y=stats["mean"],
            line_dash="dash",
            line_color=PLOT_COLORS["secondary"],
            annotation_text=f"Mean: {stats['mean']:.3f}",
            annotation_position="right",
        )
        # P95 line
        fig.add_hline(
            y=stats["p95"],
            line_dash="dot",
            line_color=PLOT_COLORS["warning"],
            annotation_text=f"P95: {stats['p95']:.3f}",
            annotation_position="right",
        )

    # Build full title with subtitle
    full_title = f"{title}"
    if subtitle:
        full_title += f"<br><sub>{subtitle}</sub>"

    fig.update_layout(
        title=full_title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template=PLOTLY_THEME,
        height=height,
        hovermode="x unified",
        showlegend=True,
    )

    # Set axis ranges with margins
    if len(x_data) > 1:
        x_margin = (x_data[-1] - x_data[0]) * 0.02
    else:
        x_margin = 1.0

    y_margin = (float(np.max(y_data)) - float(np.min(y_data))) * 0.1 if y_data.size > 0 else 1.0

    fig.update_xaxes(
        range=[x_data[0] - x_margin, x_data[-1] + x_margin] if len(x_data) > 0 else None,
        showgrid=True,
        gridcolor=PLOT_COLORS["grid"],
    )

    if y_data.size > 0:
        fig.update_yaxes(
            range=[float(np.min(y_data)) - y_margin, float(np.max(y_data)) + y_margin],
            showgrid=True,
            gridcolor=PLOT_COLORS["grid"],
        )

    return fig


def create_relative_time_axis(timestamps: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Create relative time axis from timestamps.

    Args:
        timestamps: Array of timestamps

    Returns:
        Tuple of (time_axis, is_time_based)
        - time_axis: Relative time in seconds or sample indices
        - is_time_based: True if timestamps are numeric (converted to relative seconds)
    """
    try:
        if np.issubdtype(timestamps.dtype, np.number):
            # Numeric timestamps - convert to relative seconds
            if len(timestamps) > 0:
                return timestamps - timestamps[0], True
    except (ValueError, TypeError):
        pass

    # Fallback to sample indices
    return np.arange(len(timestamps), dtype=float), False


def fig_to_html_div(fig: go.Figure, div_id: str | None = None) -> str:
    """
    Convert Plotly figure to HTML div string.

    Args:
        fig: Plotly figure
        div_id: Optional div ID for the plot

    Returns:
        HTML div string
    """
    if div_id:
        return fig.to_html(include_plotlyjs="cdn", div_id=div_id)
    return fig.to_html(include_plotlyjs="cdn", div_id=f"plot_{hash(str(fig))}")


def add_baseline_marker(
    fig: go.Figure, baseline_time: float = 0.0, rows: int = 1, cols: int = 1
) -> None:
    """
    Add vertical baseline marker to plot(s).

    Args:
        fig: Plotly figure (can have subplots)
        baseline_time: Time value for baseline (default 0)
        rows: Number of subplot rows
        cols: Number of subplot columns
    """
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            fig.add_vline(
                x=baseline_time,
                line_width=1,
                line_dash="dash",
                line_color=PLOT_COLORS["baseline"],
                row=r,
                col=c,
            )


def format_stats_subtitle(stats: dict[str, float], include_keys: list | None = None) -> str:
    """
    Format statistics into a subtitle string.

    Args:
        stats: Statistics dictionary
        include_keys: Keys to include (default: ['count', 'mean', 'p95'])

    Returns:
        Formatted subtitle string
    """
    if not stats:
        return ""

    if include_keys is None:
        include_keys = ["count", "mean", "p95"]

    parts = []
    for key in include_keys:
        if key in stats:
            value = stats[key]
            if key == "count":
                parts.append(f"{key.title()}: {value:,}")
            else:
                parts.append(f"{key.upper()}: {value:.3f}")

    return " | ".join(parts)
