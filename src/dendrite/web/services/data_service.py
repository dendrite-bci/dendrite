"""
Data service — recording file inspection, study folder import, metrics views.

Thin CRUD is handled by repositories directly via deps.py.
This service owns logic that reads H5 files or aggregates across repos.
"""

import glob
import json
import logging
import math
import os
import re
import shutil
import threading
from typing import Any

import h5py
import numpy as np

from dendrite.constants import STUDIES_DIR, get_study_paths
from dendrite.data.io.h5_explorer import (
    get_channel_info,
    get_channel_labels,
    get_h5_info,
    get_h5_metadata,
    load_events,
)
from dendrite.data.quality import detect_bad_channels
from dendrite.data.storage.database import (
    Database,
    DecoderRepository,
    RecordingRepository,
    StudyRepository,
)

logger = logging.getLogger(__name__)

_MODE_SKIP_KEYS = frozenset({
    "event_type", "mode_name", "packet_output_type", "source_mode", "type",
})
_METRICS_SKIP_KEYS = frozenset({"script_metadata", "telemetry"})


def _extract_session_id(file_path: str) -> str:
    """Extract session ID (timestamp portion) from a raw/metrics filename."""
    basename = os.path.basename(file_path)
    for prefix in ("metrics_", "eeg_data_"):
        if basename.startswith(prefix) and basename.endswith(".h5"):
            return basename[len(prefix):-len(".h5")]
    m = re.search(r'(\d{8}_\d{6})', basename)
    if m:
        return m.group(1)
    return os.path.splitext(basename)[0]


def _sanitize_floats(values: list) -> list:
    """Replace NaN/Inf with None so JSON serialization doesn't crash."""
    return [
        None if (isinstance(v, float) and not math.isfinite(v)) else v
        for v in values
    ]


class DataService:
    """H5 file inspection, study folder import, and metrics views."""

    def __init__(
        self,
        db: Database | None = None,
        db_path: str | None = None,
    ) -> None:
        if db is not None:
            self.db = db
        else:
            self.db = Database(db_path)
            self.db.init_db()
        self.studies = StudyRepository(self.db)
        self.recordings = RecordingRepository(self.db)
        # QC preview cache: (recording_id, param_hash) → computed data
        self._qc_cache: dict[tuple, dict[str, Any]] = {}
        self.decoders = DecoderRepository(self.db)
        # Per-file lock: h5py SWMR reads aren't thread-safe within a single process
        self._h5_locks: dict[str, threading.Lock] = {}
        self._h5_locks_guard = threading.Lock()

    def _h5_lock(self, path: str) -> threading.Lock:
        with self._h5_locks_guard:
            if path not in self._h5_locks:
                self._h5_locks[path] = threading.Lock()
            return self._h5_locks[path]

    # --- Study folder import ---

    def import_study_folder(
        self,
        folder_path: str,
        study_name: str,
        description: str = "",
    ) -> dict[str, Any]:
        """Scan a folder for .h5 files, copy into managed directory, and register."""
        folder = os.path.normpath(folder_path)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")

        h5_files = sorted(glob.glob(os.path.join(folder, "**", "*.h5"), recursive=True))
        h5_files = [f for f in h5_files if not f.endswith(("_metrics.h5", "_session.h5"))]
        imported = 0
        skipped = 0
        errors: list[str] = []

        paths = get_study_paths(study_name)
        studies_dir_str = str(STUDIES_DIR)

        with self.db.transaction() as conn:
            study = self.studies.get_or_create(study_name, description or None, _conn=conn)
            study_id = study["study_id"]

            for h5_path in h5_files:
                h5_path = os.path.normpath(h5_path)
                try:
                    meta = get_h5_metadata(h5_path)

                    # Prefer embedded file attributes; fall back to filename parsing
                    basename = os.path.splitext(os.path.basename(h5_path))[0]
                    file_id = (
                        meta.get("file_identifier")
                        or basename.removesuffix("_raw").removesuffix("_eeg")
                    )
                    session_id = (
                        meta.get("session_id") or _extract_session_id(h5_path)
                    )
                    task = meta.get("recording_name") or session_id
                    if session_id and len(session_id) <= 4:
                        recording_name = f"{task} — ses-{session_id}"
                    else:
                        recording_name = task
                    timestamp = meta.get("session_timestamp", "")
                    if not timestamp:
                        m = re.search(r'(\d{8}_\d{6})', os.path.basename(h5_path))
                        timestamp = m.group(1) if m else h5_path

                    # Copy into managed studies directory (skip if already there)
                    if os.path.normpath(h5_path).startswith(studies_dir_str):
                        managed_path = h5_path
                    else:
                        paths["raw"].mkdir(parents=True, exist_ok=True)
                        dst = paths["raw"] / os.path.basename(h5_path)
                        if not dst.exists():
                            shutil.copy2(h5_path, dst)
                        managed_path = str(dst)

                        # Copy paired metrics file if present
                        for old_sfx, new_sfx in [("_eeg.h5", "_metrics.h5"),
                                                  ("_raw.h5", "_metrics.h5")]:
                            if h5_path.endswith(old_sfx):
                                src_m = h5_path[:-len(old_sfx)] + new_sfx
                                if os.path.exists(src_m):
                                    paths["metrics"].mkdir(parents=True, exist_ok=True)
                                    dst_m = paths["metrics"] / os.path.basename(src_m)
                                    if not dst_m.exists():
                                        shutil.copy2(src_m, dst_m)
                                break

                    result = self.recordings.add_recording(
                        study_id=study_id,
                        recording_name=recording_name,
                        session_timestamp=str(timestamp),
                        hdf5_file_path=managed_path,
                        subject_id=str(meta.get("subject_id", "")),
                        session_id=session_id,
                        file_identifier=file_id,
                        _conn=conn,
                    )
                    if result is None:
                        skipped += 1
                    else:
                        imported += 1
                except Exception as e:
                    errors.append(f"{os.path.basename(h5_path)}: {e}")
                    logger.warning(f"Failed to import {h5_path}: {e}")

        logger.info(
            f"Study folder import: {imported} imported, {skipped} skipped, "
            f"{len(errors)} errors from {folder}"
        )
        return {
            "study_id": study_id,
            "study_name": study_name,
            "imported_count": imported,
            "skipped": skipped,
            "errors": errors,
            "total_found": len(h5_files),
        }

    # --- Recording file inspection ---

    def _get_recording_path(self, recording_id: int) -> str | None:
        """Resolve a recording ID to its validated H5 file path."""
        rec = self.recordings.get_by_id(recording_id)
        if not rec:
            return None
        path = rec["hdf5_file_path"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Recording file not found: {path}")
        return path

    def get_recording_file_info(self, recording_id: int) -> dict[str, Any] | None:
        path = self._get_recording_path(recording_id)
        return get_h5_info(path) if path else None

    def get_recording_channels(self, recording_id: int) -> dict[str, Any] | None:
        path = self._get_recording_path(recording_id)
        if not path:
            return None
        try:
            return get_channel_info(path)
        except KeyError:
            return None

    @staticmethod
    def _read_dataset(
        ds: "h5py.Dataset",
    ) -> tuple[np.ndarray, list[str], float, int] | None:
        """Read data, labels, sfreq from an H5 dataset.

        Handles both compound dtype and plain 2D formats.

        Returns:
            (data, labels, sfreq, total_samples) where data is (samples, channels),
            or None if the dataset can't be read.
        """
        if ds.shape[0] == 0:
            return None

        if ds.dtype.names:
            fields = [f for f in ds.dtype.names if np.issubdtype(ds.dtype[f], np.number)]
            if not fields:
                return None
            total_samples = ds.shape[0]
            n_ch = len(fields)
            raw = ds[:]
            data = np.column_stack([raw[f] for f in fields])
            labels = list(fields)
        elif len(ds.shape) >= 2:
            total_samples, n_ch = ds.shape
            data = ds[:]
            labels = get_channel_labels(ds, n_ch)
        else:
            return None

        sfreq = float(ds.attrs.get("sampling_frequency", ds.attrs.get("sample_rate", 0)))
        if not labels:
            labels = get_channel_labels(ds, n_ch)

        return data, labels, sfreq, total_samples

    def get_signal_preview(
        self,
        recording_id: int,
        max_points: int = 15000,
        max_channels: int = 8,
    ) -> dict[str, Any] | None:
        """Get downsampled signal data for preview plotting."""
        path = self._get_recording_path(recording_id)
        if path is None:
            return None

        _EVENT_NAMES = {"Event", "Event_Clean", "events"}
        result: dict[str, Any] = {}
        with self._h5_lock(path), h5py.File(path, "r", swmr=True) as h5f:
            for name in h5f.keys():
                if name in _EVENT_NAMES:
                    continue
                ds = h5f[name]
                if not isinstance(ds, h5py.Dataset):
                    continue

                parsed = self._read_dataset(ds)
                if parsed is None:
                    continue
                data, labels, sfreq, total_samples = parsed

                n_ch = min(len(labels), max_channels)
                step = max(1, total_samples // max_points)
                data = data[::step, :n_ch]
                labels = labels[:n_ch]

                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                time_axis = (np.arange(len(data)) * step / sfreq).tolist() if sfreq else []

                result[name] = {
                    "time": time_axis,
                    "channels": [
                        {"label": labels[i], "data": data[:, i].tolist()}
                        for i in range(n_ch)
                    ],
                    "sample_rate": sfreq,
                    "total_samples": total_samples,
                    "display_samples": len(data),
                }

        return result

    def get_event_summary(self, recording_id: int) -> dict[str, Any] | None:
        """Get event type distribution and event list for a recording."""
        path = self._get_recording_path(recording_id)
        if path is None:
            return None
        try:
            df = load_events(path, save=False)
        except KeyError:
            return {"total_count": 0, "event_types": {}, "events": []}

        # Zero-base event timestamps to match signal preview (which uses sample_index/sfreq)
        if "timestamp" in df.columns:
            _EVENT_NAMES = {"Event", "Event_Clean", "events"}
            t0 = df["timestamp"].min()
            try:
                with self._h5_lock(path), h5py.File(path, "r", swmr=True) as h5f:
                    for name in h5f.keys():
                        if name in _EVENT_NAMES:
                            continue
                        ds = h5f[name]
                        if isinstance(ds, h5py.Dataset) and ds.dtype.names and "timestamp" in ds.dtype.names and ds.shape[0] > 0:
                            t0 = float(ds["timestamp"][0])
                            break
            except Exception:
                pass
            df["timestamp"] = df["timestamp"] - t0

        event_types = {}
        if "event_type" in df.columns:
            event_types = df["event_type"].value_counts().to_dict()

        # Build event_id → event_type mapping from the data
        event_ids: dict[str, int] = {}
        if "event_id" in df.columns and "event_type" in df.columns:
            pairs = df[["event_type", "event_id"]].drop_duplicates()
            event_ids = {str(row["event_type"]): int(row["event_id"]) for _, row in pairs.iterrows()}

        # Use pandas JSON serialization (handles NaN natively) then parse back
        events_json = df.to_json(orient="records", default_handler=str) or "[]"
        events = json.loads(events_json)
        return {
            "total_count": len(df),
            "event_types": {str(k): v for k, v in event_types.items()},
            "event_ids": event_ids,
            "events": events,
        }

    # --- Metrics views ---

    def _get_session_file(self, rec: dict[str, Any], subdir: str, suffix: str) -> str | None:
        """Resolve a session file from file_identifier + study paths."""
        file_id = rec.get("file_identifier")
        study_name = rec.get("study_name")
        if not file_id or not study_name:
            return None
        paths = get_study_paths(study_name)
        path = str(paths[subdir] / f"{file_id}{suffix}")
        return path if os.path.exists(path) else None

    def _get_metrics_path(self, rec: dict[str, Any]) -> str | None:
        return self._get_session_file(rec, "metrics", "_metrics.h5")

    def get_session_summary(self, recording_id: int) -> dict[str, Any] | None:
        """Get high-level session summary from raw + metrics H5."""
        rec = self.recordings.get_by_id(recording_id)
        if not rec:
            return None
        raw_path = rec["hdf5_file_path"]
        summary: dict[str, Any] = {
            "duration_seconds": 0,
            "sample_rate": 0,
            "channels": 0,
            "datasets": [],
            "modes": [],
            "has_metrics": False,
        }

        if os.path.exists(raw_path):
            with self._h5_lock(raw_path), h5py.File(raw_path, "r", swmr=True) as h5f:
                summary["datasets"] = list(h5f.keys())
                # Find first data dataset (skip events/timestamps)
                for ds_name in h5f.keys():
                    if ds_name.lower().startswith("event") or ds_name.endswith("_timestamps"):
                        continue
                    ds = h5f[ds_name]
                    if not isinstance(ds, h5py.Dataset) or len(ds.shape) < 2:
                        continue
                    sfreq = float(
                        ds.attrs.get("sampling_frequency", ds.attrs.get("fs", 0))
                    )
                    summary["sample_rate"] = sfreq
                    summary["channels"] = ds.shape[1]
                    summary["duration_seconds"] = ds.shape[0] / max(sfreq, 1)
                    break

        metrics_path = self._get_metrics_path(rec)
        if metrics_path:
            summary["has_metrics"] = True
            with h5py.File(metrics_path, "r") as h5f:
                summary["modes"] = [
                    k for k in h5f.keys() if k not in _METRICS_SKIP_KEYS
                ]

        return summary

    @staticmethod
    def _read_h5_timeseries(
        group: h5py.Group, key: str, data: np.ndarray,
    ) -> dict[str, list]:
        ts_key = f"{key}_timestamps"
        timestamps: np.ndarray | None = None
        if ts_key in group:
            ts_ds = group[ts_key]
            if isinstance(ts_ds, h5py.Dataset):
                timestamps = ts_ds[:]
        if timestamps is not None and len(timestamps) > 0:
            time = _sanitize_floats((timestamps - timestamps[0]).tolist())
        else:
            time = list(range(len(data)))
        return {"time": time, "values": _sanitize_floats(data.tolist())}

    def get_telemetry(self, recording_id: int) -> dict[str, Any] | None:
        """Get telemetry time-series from a recording's paired metrics H5."""
        rec = self.recordings.get_by_id(recording_id)
        if not rec:
            return None

        result: dict[str, Any] = {"latencies": {}, "mode_metrics": {}, "bandwidth": {}}
        metrics_path = self._get_metrics_path(rec)
        if not metrics_path:
            return result

        with h5py.File(metrics_path, "r") as h5f:
            if "telemetry" not in h5f:
                return result
            tg = h5f["telemetry"]
            if not isinstance(tg, h5py.Group):
                return result

            for key in tg.keys():
                if key.endswith("_timestamps"):
                    continue
                ds = tg[key]
                if not isinstance(ds, h5py.Dataset):
                    continue
                data = ds[:]
                if len(data) == 0:
                    continue

                metric_info = self._read_h5_timeseries(tg, key, data)

                if "latency" in key.lower():
                    result["latencies"][key] = metric_info
                elif (
                    key.endswith("_internal_ms")
                    or key.endswith("_inference_ms")
                    or key.endswith("_gpu_mb")
                ):
                    result["mode_metrics"][key] = metric_info
                elif key.endswith("_bandwidth_kbps"):
                    result["bandwidth"][key] = metric_info

        return result

    def get_mode_performance(self, recording_id: int) -> dict[str, Any] | None:
        """Get mode performance time-series from a recording's paired metrics H5."""
        rec = self.recordings.get_by_id(recording_id)
        if not rec:
            return None

        metrics_path = self._get_metrics_path(rec)
        if not metrics_path:
            return {}

        modes: dict[str, Any] = {}

        with h5py.File(metrics_path, "r") as h5f:
            for group_name in h5f.keys():
                if group_name in _METRICS_SKIP_KEYS:
                    continue
                group = h5f[group_name]
                if not isinstance(group, h5py.Group):
                    continue

                mode_data: dict[str, Any] = {}
                for ds_key in group.keys():
                    if ds_key.endswith("_timestamps"):
                        continue
                    if ds_key.lower() in _MODE_SKIP_KEYS:
                        continue
                    dataset = group[ds_key]
                    if not isinstance(dataset, h5py.Dataset):
                        continue
                    if dataset.dtype.kind not in ("f", "i", "u") or dataset.ndim != 1:
                        continue
                    data = dataset[:]
                    if len(data) == 0:
                        continue
                    mode_data[ds_key] = self._read_h5_timeseries(group, ds_key, data)

                if mode_data:
                    modes[group_name] = mode_data

        return modes

    # --- ERP Preview ---

    def get_erp_preview(
        self,
        recording_id: int,
        epoch_tmin: float = -0.2,
        epoch_tmax: float = 0.8,
        lowcut: float = 0.5,
        highcut: float = 30.0,
        apply_rereferencing: bool = False,
        modality: str = "eeg",
    ) -> dict[str, Any] | None:
        """Compute averaged ERP waveforms per event type from a recording."""
        from dendrite.data.loaders.raw_h5_loader import RawH5Loader

        path = self._get_recording_path(recording_id)
        if path is None:
            return None

        with self._h5_lock(path):
            loaded = RawH5Loader(path, swmr=True).load()

        def _empty() -> dict[str, Any]:
            return {"erp_by_event": {}, "time_axis": [], "sample_rate": loaded.sample_rate,
                    "n_epochs": 0, "event_counts": {}, "epoch_tmin": epoch_tmin,
                    "epoch_tmax": epoch_tmax}

        if not loaded.events:
            return _empty()

        loaded.filter_modality(modality)
        loaded.preprocess({
            "lowcut": lowcut, "highcut": highcut,
            "apply_rereferencing": apply_rereferencing,
        })

        event_mapping: dict[int, str] = (
            {code: name for name, code in loaded.event_id.items()} if loaded.event_id else {}
        )

        epoched = loaded.epoch({
            "event_mapping": event_mapping,
            "epoch_tmin": epoch_tmin,
            "epoch_tmax": epoch_tmax,
        })
        X, y = epoched.X, epoched.y

        if len(X) == 0:
            return _empty()

        # Average per event class
        n_times = X.shape[2]
        time_axis = np.round(
            np.linspace(epoch_tmin, epoch_tmax, n_times), decimals=4
        ).tolist()

        erp_by_event: dict[str, Any] = {}
        event_counts: dict[str, int] = {}

        for label in sorted(set(y.tolist())):
            mask = y == label
            mean_waveform = np.nanmean(X[mask], axis=0)  # (n_channels, n_times)
            name = event_mapping.get(int(label), str(int(label)))
            count = int(mask.sum())
            erp_by_event[name] = {
                "channels": np.nan_to_num(mean_waveform).tolist(),
                "labels": loaded.channel_names,
                "count": count,
            }
            event_counts[name] = count

        return {
            "erp_by_event": erp_by_event,
            "time_axis": time_axis,
            "sample_rate": loaded.sample_rate,
            "n_epochs": len(X),
            "event_counts": event_counts,
            "epoch_tmin": epoch_tmin,
            "epoch_tmax": epoch_tmax,
        }

    # --- QC Preview ---

    def _compute_qc(
        self,
        recording_id: int,
        lowcut: float,
        highcut: float,
        apply_rereferencing: bool,
        bad_channel_mode: str,
    ) -> dict[str, Any] | None:
        """Load H5, preprocess, cache result. Returns cached data dict or None."""
        from dendrite.processing.preprocessing.preprocessor import ModalityProcessor

        cache_key = (recording_id, lowcut, highcut, apply_rereferencing, bad_channel_mode)
        if cache_key in self._qc_cache:
            return self._qc_cache[cache_key]

        path = self._get_recording_path(recording_id)
        if path is None:
            return None

        # Find primary data dataset
        with self._h5_lock(path), h5py.File(path, "r", swmr=True) as h5f:
            ds = None
            for name in h5f.keys():
                if name.lower().startswith("event") or name.endswith("_timestamps"):
                    continue
                obj = h5f[name]
                if not isinstance(obj, h5py.Dataset):
                    continue
                if obj.dtype.names or len(obj.shape) >= 2:
                    ds = obj
                    break
            if ds is None:
                return None

            parsed = self._read_dataset(ds)
            if parsed is None:
                return None
            raw_data, labels, sfreq, total_samples = parsed

            # Filter to EEG channels for QC metrics (quality detection is EEG-specific)
            n_ch = len(labels)
            channel_types_raw = ds.attrs.get("channel_types", None)
            if channel_types_raw is not None:
                from dendrite.data.loaders.raw_h5_loader import _decode_labels
                ch_types = _decode_labels(channel_types_raw)
                eeg_indices = [i for i, t in enumerate(ch_types) if t.lower() == "eeg"]
                if eeg_indices and len(eeg_indices) < n_ch:
                    raw_data = raw_data[:, eeg_indices]
                    labels = [labels[i] for i in eeg_indices if i < len(labels)]
                    n_ch = len(eeg_indices)

        if sfreq <= 0:
            return None

        raw_t = raw_data.T.astype(np.float64)

        # Bad channel detection (shared with ChannelQualityMonitor)
        quality_result = detect_bad_channels(raw_t)
        bad_indices = quality_result["bad_channels"]
        channels_quality = quality_result["channels"]

        for i, cq in enumerate(channels_quality):
            cq["label"] = labels[i] if i < len(labels) else f"ch_{i}"
            ch_data = raw_t[i]
            cq["std"] = float(np.std(ch_data))
            diff = np.abs(np.diff(ch_data))
            cq["max_deriv"] = float(np.max(diff)) if len(diff) > 0 else 0.0

        processor = ModalityProcessor({
            "num_channels": n_ch, "sample_rate": sfreq,
            "lowcut": lowcut, "highcut": highcut, "filter_order": 4,
            "apply_rereferencing": apply_rereferencing, "downsample_factor": 1,
            "channel_labels": labels,
        })

        bad_for_processing = None
        if bad_channel_mode == "interpolate" and bad_indices:
            processor.freeze_interpolation(bad_indices)
        elif bad_channel_mode == "exclude" and bad_indices:
            bad_for_processing = bad_indices

        preprocessed_t = processor.process_chunk(raw_t.copy(), bad_channels=bad_for_processing)

        result = {
            "raw_t": raw_t, "preprocessed_t": preprocessed_t,
            "labels": labels, "bad_indices": bad_indices,
            "channels_quality": channels_quality, "sfreq": sfreq,
            "total_samples": total_samples, "n_ch": n_ch,
        }

        # Keep cache small — max 2 entries
        if len(self._qc_cache) >= 2:
            oldest = next(iter(self._qc_cache))
            del self._qc_cache[oldest]
        self._qc_cache[cache_key] = result
        logger.info(f"QC computed for recording {recording_id}: {n_ch}ch, {total_samples} samples")
        return result

    def get_qc_preview(
        self,
        recording_id: int,
        lowcut: float = 0.5,
        highcut: float = 50.0,
        apply_rereferencing: bool = True,
        bad_channel_mode: str = "exclude",
        max_points: int = 50000,
        channel_indices: list[int] | None = None,
    ) -> dict[str, Any] | None:
        """Get QC preview — uses cached preprocessing, only slices + downsamples."""
        cached = self._compute_qc(recording_id, lowcut, highcut, apply_rereferencing, bad_channel_mode)
        if cached is None:
            return None

        raw_t = cached["raw_t"]
        preprocessed_t = cached["preprocessed_t"]
        labels = cached["labels"]
        bad_indices = cached["bad_indices"]
        channels_quality = cached["channels_quality"]
        sfreq = cached["sfreq"]
        total_samples = cached["total_samples"]
        n_ch = cached["n_ch"]

        # Select channels
        if channel_indices is not None:
            vis_indices = [i for i in channel_indices if 0 <= i < n_ch]
        else:
            vis_indices = list(range(min(8, n_ch)))
        if not vis_indices:
            vis_indices = list(range(min(8, n_ch)))

        # Downsample
        step = max(1, total_samples // max_points)
        raw_display = raw_t[vis_indices, ::step]
        preproc_display = preprocessed_t[vis_indices, ::step]
        display_samples = raw_display.shape[1]
        time_axis = np.round(np.arange(display_samples) * step / sfreq, decimals=4).tolist()

        raw_display = np.nan_to_num(raw_display, nan=0.0, posinf=0.0, neginf=0.0)
        preproc_display = np.nan_to_num(preproc_display, nan=0.0, posinf=0.0, neginf=0.0)

        bad_set = set(bad_indices)

        def _build_channels(data_2d: np.ndarray) -> list[dict[str, Any]]:
            return [
                {
                    "label": labels[vis_indices[i]] if vis_indices[i] < len(labels) else f"ch_{vis_indices[i]}",
                    "data": data_2d[i].tolist(),
                    "is_bad": vis_indices[i] in bad_set,
                }
                for i in range(len(vis_indices))
            ]

        return {
            "raw": {"time": time_axis, "channels": _build_channels(raw_display)},
            "preprocessed": {"time": time_axis, "channels": _build_channels(preproc_display)},
            "quality": {
                "channels": channels_quality,
                "bad_channels": bad_indices,
            },
            "sample_rate": sfreq,
            "total_samples": total_samples,
            "total_channels": n_ch,
            "display_samples": display_samples,
            "channel_indices": vis_indices,
            "preprocessing": {
                "lowcut": lowcut,
                "highcut": highcut,
                "apply_rereferencing": apply_rereferencing,
                "bad_channel_mode": bad_channel_mode,
            },
        }


