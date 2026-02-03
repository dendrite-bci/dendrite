"""
Unified Session Report Generator

Single report combining raw data analysis and metrics visualization.
Shows: Session Summary, Telemetry, Signal Preview, Mode Performance, ERP Analysis.
"""

import argparse
import gc
import glob
import json
import logging
import os
import re
from typing import Any

import h5py
import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dendrite.constants import TEMP_REPORTS_DIR
from dendrite.data.io import load_events
from dendrite.utils.reports.plot_utils import (
    PLOTLY_THEME,
    add_baseline_marker,
    compute_statistics,
    create_relative_time_axis,
    create_time_series_plot,
    fig_to_html_div,
)
from dendrite.utils.reports.report_template import (
    format_hdf5_structure,
    generate_html_document,
    render_footer,
    render_header,
    render_info_grid,
    render_info_table,
    render_plot_container,
    render_section,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MAX_SAMPLES_PER_PLOT = 15000
MAX_POINTS_LATENCY = 2000
ERP_CHUNK_TRIALS = 32
EEG_LABELS = [
    "Fp1",
    "Fz",
    "F3",
    "F7",
    "FT9",
    "FC5",
    "FC1",
    "C3",
    "T7",
    "TP9",
    "CP5",
    "CP1",
    "Pz",
    "P3",
    "P7",
    "O1",
    "Oz",
    "O2",
    "P4",
    "P8",
    "TP10",
    "CP6",
    "CP2",
    "Cz",
    "C4",
    "T8",
    "FT10",
    "FC6",
    "FC2",
    "F4",
    "F8",
    "Fp2",
    "AF7",
    "AF3",
    "AFz",
    "F1",
    "F5",
    "FT7",
    "FC3",
    "C1",
    "C5",
    "TP7",
    "CP3",
    "P1",
    "P5",
    "PO7",
    "PO3",
    "POz",
    "PO4",
    "PO8",
    "P6",
    "P2",
    "CPz",
    "CP4",
    "TP8",
    "C6",
    "C2",
    "FC4",
    "FT8",
    "F6",
    "AF8",
    "AF4",
    "F2",
    "FCz",
]


def plot_numeric_events(df, time_column: str = "timestamp") -> list[tuple[str, str]]:
    """Plot all numeric columns from events DataFrame using Plotly subplots."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference([time_column])
    if len(numeric_cols) == 0:
        return []

    ncols = 2
    nrows = int(np.ceil(len(numeric_cols) / ncols))

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[col for col in numeric_cols],
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
    )

    colors = pc.qualitative.Set1[: len(numeric_cols)]

    if time_column in df.columns:
        x_data = df[time_column]
        x_title = time_column
    else:
        x_data = df.index
        x_title = "Index"

    for i, col in enumerate(numeric_cols):
        row = i // ncols + 1
        col_idx = i % ncols + 1

        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=df[col],
                mode="lines+markers",
                name=col,
                line=dict(color=colors[i % len(colors)]),
                marker=dict(size=4),
            ),
            row=row,
            col=col_idx,
        )

        fig.update_xaxes(title_text=x_title, row=row, col=col_idx)
        fig.update_yaxes(title_text=col, row=row, col=col_idx)

    fig.update_layout(
        height=400 * nrows,
        title_text="Numeric Events Analysis",
        showlegend=False,
        template=PLOTLY_THEME,
    )

    html_div = fig_to_html_div(fig, "numeric_events_plot")
    return [("numeric_events", html_div)]


def plot_categorical_distributions(df) -> list[tuple[str, str]]:
    """Plot bar charts for all categorical columns using Plotly subplots."""
    cat_cols = df.select_dtypes(include=["object"]).columns
    if len(cat_cols) == 0:
        return []

    ncols = 2
    nrows = int(np.ceil(len(cat_cols) / ncols))

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[f"Distribution of {col}" for col in cat_cols],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    colors = pc.qualitative.Set2[: len(cat_cols)]

    for i, col in enumerate(cat_cols):
        row = i // ncols + 1
        col_idx = i % ncols + 1

        value_counts = df[col].value_counts()
        if len(value_counts) > 0:
            fig.add_trace(
                go.Bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    name=col,
                    marker_color=colors[i % len(colors)],
                ),
                row=row,
                col=col_idx,
            )

        fig.update_xaxes(title_text="Value", row=row, col=col_idx)
        fig.update_yaxes(title_text="Count", row=row, col=col_idx)

    fig.update_layout(
        height=400 * nrows,
        title_text="Categorical Events Analysis",
        showlegend=False,
        template=PLOTLY_THEME,
    )

    html_div = fig_to_html_div(fig, "categorical_events_plot")
    return [("categorical_events", html_div)]


def inspect_hdf5(file_path: str) -> str:
    """Inspect HDF5 file structure and return detailed string output."""
    output_lines = []
    try:
        with h5py.File(file_path, "r") as h5f:
            output_lines.append(f"Opened HDF5 file: {file_path}\n")
            output_lines.append("File Attributes:")
            for attr in h5f.attrs:
                output_lines.append(f"  {attr}: {h5f.attrs[attr]}")
            output_lines.append("\nDatasets and Groups in the file:")

            def gather_name(name, obj):
                output_lines.append(f"  {name} ({type(obj).__name__})")

            h5f.visititems(gather_name)
            output_lines.append("")

            # List top-level datasets
            datasets = [key for key in h5f.keys()]
            output_lines.append(f"Datasets found: {datasets}\n")

            # Inspect each dataset
            for ds_name in datasets:
                dataset = h5f[ds_name]
                if isinstance(dataset, h5py.Dataset):
                    output_lines.append(f"Dataset '{ds_name}':")
                    output_lines.append(f"  Shape: {dataset.shape}")
                    output_lines.append(f"  Data Type: {dataset.dtype}")
                    output_lines.append("  Attributes:")
                    for attr in dataset.attrs:
                        output_lines.append(f"    {attr}: {dataset.attrs[attr]}")
                    output_lines.append("")
                elif isinstance(dataset, h5py.Group):
                    output_lines.append(f"Group '{ds_name}':")
                    output_lines.append(f"  Contents: {list(dataset.keys())[:10]}")
                    output_lines.append("")

            # Check 'Event' dataset
            if "Event" in h5f:
                event_dataset = h5f["Event"]
                output_lines.append("Event Dataset Contents:")
                if hasattr(event_dataset, "dtype") and event_dataset.dtype.names:
                    field_names = event_dataset.dtype.names
                    output_lines.append(f"  Fields: {field_names}")
                    output_lines.append("  Sample of Entries:")
                    for i, event in enumerate(event_dataset[:5]):
                        event_dict = {name: event[name] for name in field_names}
                        for key, val in event_dict.items():
                            if isinstance(val, bytes):
                                event_dict[key] = val.decode("utf-8")
                        output_lines.append(f"    [{i}] {event_dict}")
                    output_lines.append("... (truncated)\n")

    except (OSError, KeyError, ValueError) as e:
        output_lines.append(f"Error inspecting file: {e}")

    return "\n".join(output_lines)


def read_script_metadata(h5f: h5py.File) -> dict[str, Any]:
    """Read script metadata from metrics HDF5 file."""
    metadata = {}
    if "script_metadata" not in h5f:
        return metadata

    meta_group = h5f["script_metadata"]
    for key, value in meta_group.attrs.items():
        if isinstance(value, str):
            if (value.startswith("{") and value.endswith("}")) or (
                value.startswith("[") and value.endswith("]")
            ):
                try:
                    metadata[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    metadata[key] = value
            else:
                metadata[key] = value
        else:
            # Convert numpy scalars to Python types
            metadata[key] = value.item() if hasattr(value, "item") else value

    return metadata


def detect_file_type(file_path: str) -> str:
    """Detect if HDF5 file is raw data or metrics."""
    with h5py.File(file_path, "r") as h5f:
        if "script_metadata" in h5f or "telemetry" in h5f:
            return "metrics"
        if "EEG" in h5f or "EMG" in h5f or "Event" in h5f:
            return "raw"
    return "unknown"


def extract_session_id(file_path: str) -> str:
    """Extract session ID from filename."""
    basename = os.path.basename(file_path)
    for prefix in ["metrics_", "eeg_data_"]:
        if basename.startswith(prefix) and basename.endswith(".h5"):
            return basename[len(prefix) : -len(".h5")]
    return os.path.splitext(basename)[0]


def extract_session_summary(raw_h5_path: str | None, metrics_h5_path: str | None) -> dict[str, Any]:
    """Extract high-level session summary from available files."""
    summary = {"duration_seconds": 0, "sample_rate": 0, "channels": 0, "datasets": []}

    if raw_h5_path and os.path.exists(raw_h5_path):
        with h5py.File(raw_h5_path, "r") as h5f:
            summary["datasets"] = list(h5f.keys())
            if "EEG" in h5f:
                eeg = h5f["EEG"]
                summary["sample_rate"] = eeg.attrs.get("sampling_frequency", eeg.attrs.get("fs", 0))
                summary["channels"] = eeg.shape[1] if len(eeg.shape) > 1 else 1
                summary["duration_seconds"] = eeg.shape[0] / max(summary["sample_rate"], 1)

    if metrics_h5_path and os.path.exists(metrics_h5_path):
        with h5py.File(metrics_h5_path, "r") as h5f:
            if "script_metadata" in h5f:
                meta = h5f["script_metadata"]
                summary["sample_rate"] = summary["sample_rate"] or meta.attrs.get("sample_rate", 0)
            # Get mode names
            summary["modes"] = [k for k in h5f.keys() if k not in ["script_metadata", "telemetry"]]

    return summary


def extract_telemetry(metrics_h5_path: str) -> dict[str, Any]:
    """Extract telemetry data from metrics HDF5."""
    if not metrics_h5_path or not os.path.exists(metrics_h5_path):
        return {}

    telemetry = {"latencies": {}, "mode_metrics": {}, "bandwidth": {}}

    with h5py.File(metrics_h5_path, "r") as h5f:
        if "telemetry" not in h5f:
            return {}

        telem_group = h5f["telemetry"]

        for key in telem_group.keys():
            if key.endswith("_timestamps"):
                continue

            dataset = telem_group[key]
            data = dataset[:]
            if len(data) == 0:
                continue

            timestamps_key = f"{key}_timestamps"
            timestamps = telem_group[timestamps_key][:] if timestamps_key in telem_group else None
            stats = compute_statistics(data)

            metric_info = {"data": data, "timestamps": timestamps, "stats": stats}

            # Categorize by metric type
            if "latency" in key.lower():
                telemetry["latencies"][key] = metric_info
            elif (
                key.endswith("_internal_ms")
                or key.endswith("_inference_ms")
                or key.endswith("_gpu_mb")
            ):
                telemetry["mode_metrics"][key] = metric_info
            elif key.endswith("_bandwidth_kbps"):
                telemetry["bandwidth"][key] = metric_info

    return telemetry


def create_telemetry_plot(telemetry: dict[str, Any]) -> str:
    """Create combined latency plot for all streams."""
    latencies = telemetry.get("latencies", {})
    if not latencies:
        return ""

    fig = go.Figure()
    colors = pc.qualitative.Set1

    for i, (key, data) in enumerate(latencies.items()):
        values = data["data"]
        timestamps = data["timestamps"]

        # Downsample for plotting
        if len(values) > MAX_POINTS_LATENCY:
            stride = len(values) // MAX_POINTS_LATENCY
            values = values[::stride]
            if timestamps is not None:
                timestamps = timestamps[::stride]

        # Create time axis
        if timestamps is not None and len(timestamps) > 0:
            x_axis = timestamps - timestamps[0]
            x_label = "Time (seconds)"
        else:
            x_axis = np.arange(len(values))
            x_label = "Sample"

        # Clean name for display
        display_name = key.replace("_latency_ms", "").replace("_", " ").upper()

        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=values,
                mode="lines",
                name=display_name,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f"<b>{display_name}</b><br>Time: %{{x:.1f}}s<br>Latency: %{{y:.2f}}ms<extra></extra>",
            )
        )

    fig.update_layout(
        title="Stream Latencies Over Session",
        xaxis_title=x_label,
        yaxis_title="Latency (ms)",
        template=PLOTLY_THEME,
        height=350,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig_to_html_div(fig, "telemetry_plot")


def extract_signal_preview(
    raw_h5_path: str, max_samples: int = MAX_SAMPLES_PER_PLOT
) -> dict[str, Any]:
    """Extract signal preview data from raw HDF5."""
    if not raw_h5_path or not os.path.exists(raw_h5_path):
        return {}

    preview = {}

    with h5py.File(raw_h5_path, "r") as h5f:
        for dataset_name in h5f.keys():
            try:
                dataset = h5f[dataset_name]
                total_samples = dataset.shape[0] if len(dataset.shape) > 0 else 0

                # Downsample if needed
                step = max(1, total_samples // max_samples)
                data = dataset[::step]

                fs = dataset.attrs.get("sampling_frequency", dataset.attrs.get("fs", 1000))
                effective_fs = fs / step if step > 1 else fs

                preview[dataset_name] = {
                    "data": data,
                    "fs": effective_fs,
                    "stored_fs": fs,
                    "total_samples": total_samples,
                    "display_samples": len(data),
                }
            except Exception as e:
                logger.warning(f"Failed to load {dataset_name}: {e}")

    return preview


def create_signal_preview_plot(preview: dict[str, Any]) -> str:
    """Create multimodal signal preview plot."""
    if not preview:
        return ""

    # Filter to plottable datasets
    plottable = [(k, v) for k, v in preview.items() if v["data"] is not None and len(v["data"]) > 0]
    if not plottable:
        return ""

    n_plots = len(plottable)
    subplot_titles = [
        f"{name} ({v['display_samples']:,} samples @ {v['stored_fs']:.0f}Hz)"
        for name, v in plottable
    ]

    fig = make_subplots(rows=n_plots, cols=1, subplot_titles=subplot_titles, vertical_spacing=0.12)
    colors = pc.qualitative.Set1

    for plot_idx, (name, dataset) in enumerate(plottable, 1):
        data = dataset["data"]
        fs = dataset["fs"]
        n_samples = len(data)
        time_axis = np.arange(n_samples) / fs

        # Handle different data shapes
        if hasattr(data, "dtype") and data.dtype.names:
            # Structured array (events) - plot as markers
            _plot_structured_data(fig, data, time_axis, name, plot_idx, colors)
        elif len(data.shape) == 2:
            # 2D: plot first 3 channels
            for ch in range(min(3, data.shape[1])):
                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=data[:, ch] + ch * np.std(data[:, ch]) * 3,
                        mode="lines",
                        name=f"Ch{ch + 1}",
                        line=dict(color=colors[ch % len(colors)], width=1),
                        showlegend=(plot_idx == 1),
                    ),
                    row=plot_idx,
                    col=1,
                )
        else:
            # 1D
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=data,
                    mode="lines",
                    name=name,
                    line=dict(color=colors[0], width=1),
                ),
                row=plot_idx,
                col=1,
            )

        fig.update_xaxes(title_text="Time (seconds)", row=plot_idx, col=1)
        fig.update_yaxes(title_text="Amplitude", row=plot_idx, col=1)

    fig.update_layout(
        height=280 * n_plots, title_text="Signal Overview", template=PLOTLY_THEME, showlegend=True
    )

    return fig_to_html_div(fig, "signal_preview")


def _plot_structured_data(fig, data, time_axis, name, plot_idx, colors):
    """Helper to plot structured array data (events)."""
    # Build lowercase field name mapping once (supports legacy PascalCase)
    field_map = {f.lower(): f for f in data.dtype.names}

    # Find timestamp and event type fields using normalized lookup
    ts_field = field_map.get("timestamp")
    ts_data = data[ts_field] if ts_field else None

    type_field = field_map.get("event_type") or field_map.get("type")
    event_types = data[type_field] if type_field else None

    if ts_data is not None and event_types is not None:
        ts_data = ts_data - ts_data[0] if len(ts_data) > 0 else ts_data
        event_types = [
            et.decode("utf-8") if isinstance(et, bytes) else str(et) for et in event_types
        ]
        unique_events = list(set(event_types))

        for i, event_type in enumerate(unique_events):
            event_times = [
                t for t, et in zip(ts_data, event_types, strict=False) if et == event_type
            ]
            if event_times:
                fig.add_trace(
                    go.Scatter(
                        x=event_times,
                        y=[i] * len(event_times),
                        mode="markers",
                        name=event_type,
                        marker=dict(size=8, color=colors[i % len(colors)]),
                        showlegend=True,
                    ),
                    row=plot_idx,
                    col=1,
                )


# Skip these metadata keys when displaying mode datasets
MODE_SKIP_KEYS = {"event_type", "mode_name", "packet_output_type", "source_mode", "type"}


def extract_mode_performance(metrics_h5_path: str) -> dict[str, Any]:
    """Extract mode performance metrics with detailed dataset info."""
    if not metrics_h5_path or not os.path.exists(metrics_h5_path):
        return {}

    modes = {}

    with h5py.File(metrics_h5_path, "r") as h5f:
        for group_name in h5f.keys():
            if group_name in ["script_metadata", "telemetry"]:
                continue

            group = h5f[group_name]
            if not isinstance(group, h5py.Group):
                continue

            mode_data = {"datasets": {}, "eeg_data_shape": None}

            for ds_key in group.keys():
                if ds_key.endswith("_timestamps"):
                    continue

                # Skip metadata fields
                if ds_key.lower() in MODE_SKIP_KEYS:
                    continue

                dataset = group[ds_key]

                # Store EEG data shape for ERP
                if ds_key == "eeg_data" and dataset.ndim == 3:
                    mode_data["eeg_data_shape"] = dataset.shape
                    continue

                # Store dataset info with shape, dtype, and last value/timestamp
                ds_info = {
                    "dtype": str(dataset.dtype),
                    "shape": dataset.shape,
                    "count": dataset.shape[0] if dataset.shape else 0,
                    "last_value": None,
                    "last_timestamp": None,
                }

                # Get last value
                if dataset.shape and dataset.shape[0] > 0:
                    try:
                        lv = dataset[-1]
                        if hasattr(lv, "dtype"):
                            if lv.dtype.kind == "S":
                                lv = lv.astype("U")
                            else:
                                lv = lv.tolist()
                        elif isinstance(lv, bytes):
                            lv = lv.decode("utf-8", errors="replace")
                        ds_info["last_value"] = lv

                        # Get last timestamp if available
                        ts_key = f"{ds_key}_timestamps"
                        if ts_key in group and group[ts_key].shape[0] > 0:
                            ts_val = group[ts_key][-1]
                            if isinstance(ts_val, bytes):
                                ts_val = ts_val.decode("utf-8", errors="replace")
                            ds_info["last_timestamp"] = ts_val
                    except (KeyError, IndexError, ValueError, UnicodeDecodeError):
                        pass

                mode_data["datasets"][ds_key] = ds_info

            if mode_data["datasets"] or mode_data["eeg_data_shape"]:
                modes[group_name] = mode_data

    return modes


def create_mode_performance_plots(
    metrics_h5_path: str, modes: dict[str, Any]
) -> list[tuple[str, str]]:
    """Create performance plots for each mode."""
    plots = []

    if not metrics_h5_path or not os.path.exists(metrics_h5_path):
        return plots

    with h5py.File(metrics_h5_path, "r") as h5f:
        for mode_name in modes:
            if mode_name not in h5f:
                continue
            group = h5f[mode_name]

            # Plot key metrics
            for ds_key in ["accuracy", "confidence", "prediction"]:
                if ds_key not in group:
                    continue
                dataset = group[ds_key]
                if dataset.ndim != 1 or dataset.shape[0] == 0:
                    continue

                data = dataset[:]
                if len(data) > MAX_POINTS_LATENCY:
                    data = data[:: len(data) // MAX_POINTS_LATENCY]

                # Get timestamps if available
                ts_key = f"{ds_key}_timestamps"
                if ts_key in group:
                    ts = group[ts_key][:]
                    if len(ts) > MAX_POINTS_LATENCY:
                        ts = ts[:: len(ts) // MAX_POINTS_LATENCY]
                    x_axis, is_time = create_relative_time_axis(ts[: len(data)])
                    x_label = "Time (seconds)" if is_time else "Sample"
                else:
                    x_axis = np.arange(len(data))
                    x_label = "Sample"

                fig = create_time_series_plot(
                    x_data=x_axis,
                    y_data=data,
                    title=f"{mode_name} - {ds_key.title()}",
                    x_label=x_label,
                    y_label="Value",
                    height=300,
                )
                plots.append((f"{mode_name}_{ds_key}", fig_to_html_div(fig)))

    return plots


def create_erp_plot(metrics_h5_path: str, mode_name: str) -> str | None:
    """Create ERP plot for z-channels only."""
    if not metrics_h5_path or not os.path.exists(metrics_h5_path):
        return None

    with h5py.File(metrics_h5_path, "r") as h5f:
        if mode_name not in h5f or "eeg_data" not in h5f[mode_name]:
            return None

        dataset = h5f[mode_name]["eeg_data"]
        if dataset.ndim != 3:
            return None

        n_trials, n_channels, n_time = dataset.shape

        # Chunked mean computation
        running_sum = np.zeros((n_channels, n_time), dtype=np.float64)
        processed = 0
        chunk = max(1, ERP_CHUNK_TRIALS)
        while processed < n_trials:
            end = min(n_trials, processed + chunk)
            batch = dataset[processed:end, :, :]
            running_sum += batch.sum(axis=0)
            processed = end
        erp = running_sum / float(n_trials)

        # Find z-channels
        z_channels = [i for i, ch in enumerate(EEG_LABELS) if "z" in ch and i < n_channels]
        if not z_channels:
            return None

        # Get sampling rate
        fs = next(
            (
                float(dataset.attrs[a])
                for a in ["sampling_frequency", "fs", "sample_rate"]
                if a in dataset.attrs
            ),
            1000.0,
        )

        # Downsample time if needed
        if n_time > MAX_POINTS_LATENCY:
            stride = n_time // MAX_POINTS_LATENCY
            erp = erp[:, ::stride]
            n_time = erp.shape[1]

        # Create time axis with baseline assumption
        total_duration = n_time / fs
        baseline = 0.2 if total_duration >= 0.5 else 0
        time_axis = (np.arange(n_time) / fs) - baseline

        # Create subplot
        n_plot = len(z_channels)
        cols = 2
        rows = int(np.ceil(n_plot / cols))
        subplot_titles = [f"ERP - {EEG_LABELS[z_channels[i]]}" for i in range(n_plot)]

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
        )
        colors = pc.qualitative.Set1

        for i, ch_idx in enumerate(z_channels):
            row = i // cols + 1
            col = i % cols + 1
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=erp[ch_idx, :],
                    mode="lines",
                    name=EEG_LABELS[ch_idx],
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f"<b>{EEG_LABELS[ch_idx]}</b><br>Time: %{{x:.3f}}s<br>Amplitude: %{{y:.3f}}<extra></extra>",
                    showlegend=True,
                ),
                row=row,
                col=col,
            )
            fig.update_xaxes(title_text="Time (s)", row=row, col=col)
            fig.update_yaxes(title_text="uV", row=row, col=col)

        # Set x-axis to show full time range
        fig.update_xaxes(range=[time_axis[0], time_axis[-1]])

        # Add baseline marker at t=0 if within range
        if time_axis[0] < 0 < time_axis[-1]:
            add_baseline_marker(fig, baseline_time=0.0, rows=rows, cols=cols)

        # Auto-scale y-axis per channel with 10% margin
        for i, ch_idx in enumerate(z_channels):
            row = i // cols + 1
            col = i % cols + 1
            y_data = erp[ch_idx, :]
            y_margin = (np.max(y_data) - np.min(y_data)) * 0.1
            fig.update_yaxes(
                range=[np.min(y_data) - y_margin, np.max(y_data) + y_margin], row=row, col=col
            )

        fig.update_layout(
            height=280 * rows,
            title_text=f"ERP Analysis: {mode_name} ({n_trials} trials)",
            template=PLOTLY_THEME,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        return fig_to_html_div(fig, f"erp_{mode_name}")

    return None


def _generate_summary_section(raw_h5_path: str | None, metrics_h5_path: str | None) -> str:
    """Generate session summary section HTML."""
    summary = extract_session_summary(raw_h5_path, metrics_h5_path)
    summary_items = {
        "Duration": f"{summary['duration_seconds']:.1f} seconds",
        "Sample Rate": f"{summary['sample_rate']:.0f} Hz",
        "Channels": summary["channels"],
        "Datasets": ", ".join(summary["datasets"][:5]) if summary["datasets"] else "None",
    }
    if summary.get("modes"):
        summary_items["Modes"] = ", ".join(summary["modes"])
    return render_info_grid(summary_items)


def _generate_metadata_section(metrics_h5_path: str) -> str | None:
    """Generate experiment configuration section from metrics file."""
    if not metrics_h5_path or not os.path.exists(metrics_h5_path):
        return None

    with h5py.File(metrics_h5_path, "r") as h5f:
        script_metadata = read_script_metadata(h5f)
        if not script_metadata:
            return None

        config_items = {}
        for key, value in script_metadata.items():
            if key in [
                "experiment_description",
                "mode_name",
                "sampling_rate",
                "duration_seconds",
                "script_name",
            ]:
                if isinstance(value, (dict, list)):
                    config_items[key.replace("_", " ").title()] = (
                        json.dumps(value, indent=2)
                        if len(str(value)) < 100
                        else str(type(value).__name__)
                    )
                else:
                    config_items[key.replace("_", " ").title()] = str(value)

        if config_items:
            return render_section("Experiment Configuration", render_info_grid(config_items))

    return None


def _generate_telemetry_section(metrics_h5_path: str) -> str | None:
    """Generate telemetry section with latency, mode metrics, and bandwidth stats."""
    if not metrics_h5_path:
        return None

    telemetry = extract_telemetry(metrics_h5_path)
    if not (
        telemetry.get("latencies") or telemetry.get("mode_metrics") or telemetry.get("bandwidth")
    ):
        return None

    content_parts = []

    # Latency stats
    if telemetry.get("latencies"):
        content_parts.append("<h3>Stream Latencies</h3>")
        stats_items = {}
        for key, data in telemetry["latencies"].items():
            stats = data["stats"]
            display_name = key.replace("_latency_ms", "").replace("_", " ").upper()
            stats_items[display_name] = (
                f"mean {stats['mean']:.1f}ms | p50 {stats['p50']:.1f}ms | "
                f"p95 {stats['p95']:.1f}ms | p99 {stats['p99']:.1f}ms (n={stats['count']:,})"
            )
        content_parts.append(render_info_table(stats_items))

    # Mode metrics (inference time, GPU)
    if telemetry.get("mode_metrics"):
        content_parts.append("<h3>Mode Performance</h3>")
        stats_items = {}
        for key, data in telemetry["mode_metrics"].items():
            stats = data["stats"]
            if "_internal_ms" in key:
                metric_label = f"{key.replace('_internal_ms', '')} (internal)"
                unit = "ms"
            elif "_inference_ms" in key:
                metric_label = f"{key.replace('_inference_ms', '')} (inference)"
                unit = "ms"
            elif "_gpu_mb" in key:
                metric_label = f"{key.replace('_gpu_mb', '')} (GPU)"
                unit = "MB"
            else:
                metric_label = key
                unit = ""
            stats_items[metric_label] = (
                f"mean {stats['mean']:.1f}{unit} | p50 {stats['p50']:.1f}{unit} | "
                f"p95 {stats['p95']:.1f}{unit} (n={stats['count']:,})"
            )
        content_parts.append(render_info_table(stats_items))

    # Bandwidth stats
    if telemetry.get("bandwidth"):
        content_parts.append("<h3>Streamer Bandwidth</h3>")
        stats_items = {}
        for key, data in telemetry["bandwidth"].items():
            stats = data["stats"]
            display_name = key.replace("_bandwidth_kbps", "").replace("_", " ")
            stats_items[display_name] = (
                f"mean {stats['mean']:.1f} kbps | p50 {stats['p50']:.1f} kbps | "
                f"max {stats['max']:.1f} kbps (n={stats['count']:,})"
            )
        content_parts.append(render_info_table(stats_items))

    # Latency plot
    telem_plot = create_telemetry_plot(telemetry)
    if telem_plot:
        content_parts.append(render_plot_container(telem_plot))

    return render_section("Telemetry", "\n".join(content_parts))


def _generate_signal_preview_section(raw_h5_path: str) -> str | None:
    """Generate signal preview section with plots."""
    if not raw_h5_path:
        return None

    preview = extract_signal_preview(raw_h5_path)
    if not preview:
        return None

    preview_plot = create_signal_preview_plot(preview)
    if not preview_plot:
        return None

    gc.collect()
    return render_section("Signal Preview", render_plot_container(preview_plot))


def _generate_event_analysis_section(raw_h5_path: str) -> str | None:
    """Generate event analysis section with numeric and categorical plots."""
    if not raw_h5_path or not os.path.exists(raw_h5_path):
        return None

    with h5py.File(raw_h5_path, "r") as h5f:
        if "Event" not in h5f:
            return None

    try:
        cleaned_events_df = load_events(raw_h5_path)
        content_parts = []

        for _, plot_html in plot_numeric_events(cleaned_events_df):
            content_parts.append(render_plot_container(plot_html))

        for _, plot_html in plot_categorical_distributions(cleaned_events_df):
            content_parts.append(render_plot_container(plot_html))

        del cleaned_events_df
        gc.collect()

        if content_parts:
            return render_section("Event Analysis", "\n".join(content_parts))
    except (OSError, ValueError, KeyError) as e:
        logger.warning(f"Failed to load events: {e}")

    return None


def _generate_structure_section(raw_h5_path: str) -> str | None:
    """Generate HDF5 file structure inspection section."""
    if not raw_h5_path or not os.path.exists(raw_h5_path):
        return None

    inspection_output = inspect_hdf5(raw_h5_path)
    if not inspection_output:
        return None

    formatted_inspection = format_hdf5_structure(inspection_output)
    content = f'<div class="code-block">{formatted_inspection}</div>'
    return render_section("File Structure Analysis", content)


def _generate_mode_performance_section(metrics_h5_path: str) -> str | None:
    """Generate mode performance section with metrics and ERP plots."""
    if not metrics_h5_path:
        return None

    modes = extract_mode_performance(metrics_h5_path)
    if not modes:
        return None

    content_parts = []

    # Summary table with detailed dataset info
    for mode_name, mode_data in modes.items():
        mode_items = {}
        meaningful_count = len(mode_data["datasets"])
        total_entries = 0

        for ds_info in mode_data["datasets"].values():
            samples = ds_info.get("count", 0)
            if samples > 0:
                total_entries = max(total_entries, samples)

        if total_entries > 0:
            mode_items["Total Entries"] = total_entries

        for ds_key, ds_info in list(mode_data["datasets"].items())[:5]:
            samples = ds_info.get("count", 0)
            if samples > 0:
                display_name = ds_key.replace("_", " ").title()
                last_val = ds_info.get("last_value")
                if last_val is not None and isinstance(last_val, (int, float)):
                    mode_items[display_name] = f"{samples} entries (last: {last_val:.4f})"
                else:
                    mode_items[display_name] = f"{samples} entries"

        if mode_data.get("eeg_data_shape"):
            shape = mode_data["eeg_data_shape"]
            mode_items["EEG Data"] = f"{shape[0]} trials x {shape[1]} ch x {shape[2]} samples"

        if mode_items:
            content_parts.append(f"<h3>{mode_name} ({meaningful_count} metrics)</h3>")
            content_parts.append(render_info_table(mode_items))

    # Performance plots
    for _, plot_html in create_mode_performance_plots(metrics_h5_path, modes):
        content_parts.append(render_plot_container(plot_html))

    # ERP plots
    for mode_name, mode_data in modes.items():
        if mode_data.get("eeg_data_shape"):
            erp_html = create_erp_plot(metrics_h5_path, mode_name)
            if erp_html:
                content_parts.append(render_plot_container(erp_html))

    if content_parts:
        return render_section("Mode Performance", "\n".join(content_parts))

    return None


def generate_session_report(
    raw_h5_path: str | None, metrics_h5_path: str | None, output_path: str, session_id: str
) -> str:
    """Generate unified session report."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    body_parts = [render_header("Session Report", session_id)]

    # Session Summary
    body_parts.append(_generate_summary_section(raw_h5_path, metrics_h5_path))

    # Optional sections
    for section in [
        _generate_metadata_section(metrics_h5_path),
        _generate_telemetry_section(metrics_h5_path),
        _generate_signal_preview_section(raw_h5_path),
        _generate_event_analysis_section(raw_h5_path),
        _generate_structure_section(raw_h5_path),
        _generate_mode_performance_section(metrics_h5_path),
    ]:
        if section:
            body_parts.append(section)

    body_parts.append(render_footer("Dendrite Analysis Pipeline"))

    html_content = generate_html_document(
        title="Session Report", session_id=session_id, body_content="\n".join(body_parts)
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"Report generated: {output_path}")
    return output_path


def find_paired_files(file_path: str) -> tuple[str | None, str | None]:
    """Find raw and metrics files from a given path."""
    basename = os.path.basename(file_path)
    dirname = os.path.dirname(file_path)

    file_type = detect_file_type(file_path)

    if file_type == "metrics":
        metrics_path = file_path
        # Try to find matching raw file
        session_id = extract_session_id(file_path)
        raw_candidates = glob.glob(
            os.path.join(dirname.replace("metrics", "raw"), f"*{session_id}*.h5")
        )
        raw_path = raw_candidates[0] if raw_candidates else None
    elif file_type == "raw":
        raw_path = file_path
        # Try to find matching metrics file
        session_id = extract_session_id(file_path)
        metrics_candidates = glob.glob(
            os.path.join(dirname.replace("raw", "metrics"), f"*{session_id}*.h5")
        )
        metrics_path = metrics_candidates[0] if metrics_candidates else None
    else:
        # Unknown - use as-is
        raw_path = file_path if "eeg" in basename.lower() else None
        metrics_path = file_path if "metrics" in basename.lower() else None

    return raw_path, metrics_path


def main():
    parser = argparse.ArgumentParser(description="Generate unified session report from HDF5 files.")
    parser.add_argument(
        "--file", type=str, required=True, help="Path to HDF5 file (raw or metrics)"
    )
    parser.add_argument("--raw", type=str, help="Path to raw HDF5 file (optional)")
    parser.add_argument("--metrics", type=str, help="Path to metrics HDF5 file (optional)")
    parser.add_argument(
        "--study", type=str, default=None, help="Study name (auto-detected if not provided)"
    )
    parser.add_argument("--output", type=str, default=None, help="Output HTML path")
    args = parser.parse_args()

    # Determine file paths - start with auto-detected, then override with explicit args
    raw_path, metrics_path = find_paired_files(args.file)

    if args.raw:
        raw_path = args.raw
    if args.metrics:
        metrics_path = args.metrics

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Extract timestamp from filename (format: *_YYYYMMDD_HHMMSS.h5)
        basename = os.path.basename(args.file)
        match = re.search(r"(\d{8}_\d{6})\.h5$", basename)
        timestamp = match.group(1) if match else basename.replace(".h5", "")
        TEMP_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = TEMP_REPORTS_DIR / f"session_report_{timestamp}.html"

    # Generate session ID
    session_id = extract_session_id(args.file)

    # Generate report
    generate_session_report(raw_path, metrics_path, output_path, session_id)


if __name__ == "__main__":
    main()
