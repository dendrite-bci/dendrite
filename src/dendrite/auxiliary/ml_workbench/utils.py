"""Shared utilities for offline ML GUI tabs."""

import pyqtgraph as pg
from PyQt6 import QtWidgets

from dendrite.gui.styles.design_tokens import BG_MAIN, TEXT_LABEL, TEXT_MUTED, TEXT_MUTED_DARK
from dendrite.ml.decoders.registry import get_decoder_entry
from dendrite.utils.logger_central import get_logger

logger = get_logger(__name__)


def format_duration(seconds: float) -> str:
    """Format seconds to human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}min"
    hours = minutes / 60
    return f"{hours:.1f}h"


def style_plot(
    plot: pg.PlotWidget,
    title: str,
    show_grid: bool = True,
    y_range: tuple[float, float] | None = None,
) -> None:
    """Apply minimal styling to a pyqtgraph plot widget.

    Args:
        plot: The pyqtgraph PlotWidget to style
        title: Title text to display
        show_grid: Whether to show grid lines (default True)
        y_range: Optional (min, max) tuple for Y axis range
    """
    plot.setBackground(BG_MAIN)

    # Subtle title (smaller, not bold)
    plot.setTitle(title, color=TEXT_LABEL, size="10pt")

    # Minimal axis styling
    for axis_name in ["left", "bottom"]:
        axis = plot.getAxis(axis_name)
        axis.setPen(pg.mkPen(TEXT_MUTED_DARK, width=1))  # Subtle (#555555)
        axis.setTextPen(TEXT_MUTED)  # Muted tick labels
        axis.setStyle(tickLength=3, tickTextOffset=4)  # Tighter ticks

    if show_grid:
        plot.showGrid(x=True, y=True, alpha=0.08)  # Subtler grid

    if y_range:
        plot.setYRange(*y_range, padding=0.02)

    plot.getViewBox().setDefaultPadding(0.01)


def create_styled_plot(
    title: str,
    height: int = 200,
    y_range: tuple[float, float] | None = None,
    show_grid: bool = True,
) -> pg.PlotWidget:
    """Create a styled pyqtgraph PlotWidget with standard settings.

    Args:
        title: Title text to display
        height: Minimum height in pixels (default 200)
        y_range: Optional (min, max) tuple for Y axis range
        show_grid: Whether to show grid lines (default True)

    Returns:
        Styled PlotWidget ready to use
    """
    plot = pg.PlotWidget()
    plot.setMinimumHeight(height)
    style_plot(plot, title, show_grid=show_grid, y_range=y_range)
    return plot


def create_plot_container(name: str = "plotContainer") -> QtWidgets.QFrame:
    """Create minimal styled container for plots.

    Args:
        name: Object name for stylesheet targeting

    Returns:
        Styled QFrame ready to hold plot widgets
    """
    container = QtWidgets.QFrame()
    container.setObjectName(name)
    container.setStyleSheet(f"background: {BG_MAIN}; border: none;")
    return container


def is_model_compatible(model_name: str, modality: str) -> bool:
    """Check if a decoder is compatible with a given modality."""
    entry = get_decoder_entry(model_name)
    if not entry:
        return True

    decoder_modalities = entry.get("modalities", ["any"])

    if "any" in decoder_modalities or "multimodal" in decoder_modalities:
        return True

    return modality.lower() in [m.lower() for m in decoder_modalities]


def setup_worker_thread(
    worker, thread, on_finished=None, on_error=None, on_progress=None, quit_on_finished: bool = True
) -> None:
    """Connect worker signals to thread lifecycle.

    Sets up the standard pattern for moving a worker to a thread and
    connecting common signals.

    Args:
        worker: QObject worker with run() slot
        thread: QThread to move worker to
        on_finished: Callback for worker.finished signal
        on_error: Callback for worker.error signal
        on_progress: Callback for worker.progress signal
        quit_on_finished: If True, connect finished to thread.quit
    """
    worker.moveToThread(thread)
    thread.started.connect(worker.run)

    if on_finished:
        worker.finished.connect(on_finished)
    if on_error:
        worker.error.connect(on_error)
    if on_progress and hasattr(worker, "progress"):
        worker.progress.connect(on_progress)
    if quit_on_finished:
        worker.finished.connect(thread.quit)
