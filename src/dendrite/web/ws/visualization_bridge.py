"""
Visualization Bridge

Reads from shared ring buffers, preprocesses per-modality, monitors channel
quality, computes PSD, and broadcasts to WebSocket clients via QueueBridge.
"""

import asyncio
import queue
import time
from typing import Any

import numpy as np

from dendrite.data.quality import ChannelQualityMonitor
from dendrite.data.shared_buffers import OverrunError, SharedRingBuffer
from dendrite.processing.preprocessing.preprocessor import OnlinePreprocessor
from dendrite.utils.logger_central import get_logger
from dendrite.utils.state_keys import (
    calibration_corr_key,
    channel_quality_key,
    manual_bad_channels_key,
)
from dendrite.web.ws.bridge import QueueBridge

TARGET_VIZ_RATE = 100  # Hz — all streams decimate to this regardless of native rate
VIZ_BATCH_SIZE = 10  # samples per WS message (higher = fewer messages, more latency)
QUALITY_PUBLISH_INTERVAL_S = 2.0
PSD_WINDOW_S = 1.0  # accumulate 1s of full-rate data for Welch
PSD_UPDATE_S = 1.0
MODALITY_PSD_MAX_FREQ: dict[str, float] = {"eeg": 50.0, "emg": 200.0, "eog": 15.0}
DEFAULT_PSD_MAX_FREQ = 50.0


# ---------------------------------------------------------------------------
# Quality tracking (channel quality + interpolation management)
# ---------------------------------------------------------------------------

class QualityTracker:
    """Monitors EEG channel quality and manages interpolation freeze/unfreeze.

    After a warmup period, auto-detected bad channels are frozen.  New detections
    still show in QC dots but don't enter the effective set unless the operator
    manually confirms them via channel flags.
    """

    def __init__(self, eeg_indices: list[int], sample_rate: float,
                 shared_state, logger, warmup_s: float = 10.0):
        self._eeg_indices = eeg_indices
        self._n_eeg = len(eeg_indices)
        self._shared_state = shared_state
        self._logger = logger
        self._monitor = ChannelQualityMonitor(self._n_eeg, sample_rate)
        self._warmup_s = warmup_s
        self._start_time = time.time()
        self._warmup_done = False
        self._frozen_auto_bad: dict[str, list[int]] = {}
        self._interp_version = 0
        self._last_effective_bad: dict[str, list[int]] = {}
        self._last_publish_time = 0.0

        # Warmup data accumulation for correlation-based interpolation
        self._warmup_chunks: list[np.ndarray] = []
        self._calibration_corr: np.ndarray | None = None

    @property
    def last_effective_bad(self) -> dict[str, list[int]]:
        return self._last_effective_bad

    def update(self, raw_eeg: np.ndarray) -> None:
        """Feed raw EEG samples (n_samples, n_channels) to the quality monitor."""
        if not self._warmup_done:
            self._warmup_chunks.append(raw_eeg.copy())
        for i in range(len(raw_eeg)):
            self._monitor.update(raw_eeg[i].reshape(-1, 1))

    def publish_if_due(self, preprocessor: OnlinePreprocessor | None) -> None:
        """Every ~2s: compute effective bad channels, publish to SharedState,
        and update interpolation on the preprocessor if the set changed."""
        now = time.time()
        if now - self._last_publish_time < QUALITY_PUBLISH_INTERVAL_S:
            return
        self._last_publish_time = now

        q = self._monitor.get_quality()
        eeg_bad = q.get("bad_channels", [])
        q["bad_channels"] = {"eeg": eeg_bad} if eeg_bad else {}

        auto_bad = dict(q["bad_channels"])
        if not self._warmup_done and (now - self._start_time) >= self._warmup_s:
            self._warmup_done = True
            self._frozen_auto_bad = dict(auto_bad)
            self._logger.info(f"Auto bad channels frozen at warmup: {self._frozen_auto_bad}")
            self._compute_calibration_corr()

        # Merge frozen auto-detected + manual flags → effective bad
        manual = self._shared_state.get(manual_bad_channels_key()) or {}
        manual_flagged = manual.get("flagged", {})
        manual_unflagged = manual.get("unflagged", {})
        base_bad = self._frozen_auto_bad if self._warmup_done else auto_bad

        effective_bad: dict[str, list[int]] = {}
        for mod in set(list(base_bad.keys()) + list(manual_flagged.keys())):
            base_set = set(base_bad.get(mod, []))
            flagged_set = set(manual_flagged.get(mod, []))
            unflagged_set = set(manual_unflagged.get(mod, []))
            effective = sorted((base_set | flagged_set) - unflagged_set)
            if effective:
                effective_bad[mod] = effective

        if self._warmup_done and effective_bad != self._last_effective_bad:
            self._interp_version += 1
            self._last_effective_bad = dict(effective_bad)
            self._logger.info(
                f"Interpolation v{self._interp_version}: effective_bad={effective_bad}"
            )
            self._apply_interpolation(preprocessor)

        if self._warmup_done:
            q["effective_bad"] = self._last_effective_bad
            q["interp_version"] = self._interp_version
            q["manual_flags"] = manual_flagged
            q["manual_unflagged"] = manual_unflagged

        self._shared_state.set(channel_quality_key(), q)

    def reapply_interpolation(self, preprocessor: OnlinePreprocessor | None) -> None:
        """Re-apply current interpolation to a (new) preprocessor."""
        if self._warmup_done and self._last_effective_bad.get("eeg"):
            self._apply_interpolation(preprocessor)

    def _apply_interpolation(self, preprocessor: OnlinePreprocessor | None) -> None:
        """Freeze interpolation on the preprocessor's EEG processor."""
        eeg_bad = self._last_effective_bad.get("eeg")
        if not preprocessor or not eeg_bad:
            return
        eeg_proc = preprocessor.processors.get("eeg")
        if eeg_proc:
            bad_at_warmup = self._frozen_auto_bad.get("eeg", [])
            eeg_proc.freeze_interpolation(
                eeg_bad,
                corr_matrix=self._calibration_corr,
                bad_during_warmup=bad_at_warmup,
            )

    def _compute_calibration_corr(self) -> None:
        """Compute Pearson correlation matrix from accumulated warmup data."""
        if not self._warmup_chunks:
            return
        # (n_samples, n_channels) → (n_channels, n_samples)
        data = np.vstack(self._warmup_chunks).T.astype(np.float64)
        self._warmup_chunks.clear()  # free memory

        if data.shape[1] < 2:
            self._logger.warning("Not enough warmup samples for correlation")
            return

        # Common-average reference before correlation: raw data shares a common
        # reference, so common-mode inflates every pairwise correlation toward 1
        # and washes out the spatial neighbourhood structure the weights need.
        # (Channels bad throughout warmup still have no usable correlation and
        # fall back to equal weights — geometry/spline is the only fix there.)
        data = data - data.mean(axis=0, keepdims=True)

        corr: np.ndarray = np.asarray(np.corrcoef(data))
        corr = np.nan_to_num(corr, nan=0.0)

        self._calibration_corr = corr
        self._shared_state.set(calibration_corr_key(), self._calibration_corr)
        n_valid = int((np.diag(corr) != 0).sum())
        n_ch = data.shape[0]
        self._logger.info(
            f"Calibration correlation matrix computed: {n_ch}x{n_ch} "
            f"({n_valid}/{n_ch} channels with valid variance)"
        )


# ---------------------------------------------------------------------------
# PSD accumulation + Welch computation
# ---------------------------------------------------------------------------

class PSDAccumulator:
    """Accumulates preprocessed samples in circular buffers and periodically
    computes Welch PSD per modality."""

    def __init__(self, modalities: dict[str, list[int]], sample_rate: float):
        self._sample_rate = sample_rate
        self._buf_size = int(sample_rate * PSD_WINDOW_S)
        self._buffers: dict[str, np.ndarray] = {}
        self._write_pos: dict[str, int] = {}
        self._filled: dict[str, int] = {}
        self._last_compute_time = 0.0

        for mod, indices in modalities.items():
            n_ch = len(indices)
            self._buffers[mod] = np.zeros((n_ch, self._buf_size), dtype=np.float64)
            self._write_pos[mod] = 0
            self._filled[mod] = 0

    def accumulate(self, data: np.ndarray, modalities: dict[str, list[int]]) -> None:
        """Write preprocessed full-rate samples into per-modality circular buffers.

        data: (n_samples, total_columns) — the full ring buffer read.
        """
        for mod, indices in modalities.items():
            buf = self._buffers.get(mod)
            if buf is None:
                continue
            mod_data = data[:, indices].T.astype(np.float64)  # (n_ch, n_samples)
            n_samples = mod_data.shape[1]
            pos = self._write_pos[mod]
            buf_size = buf.shape[1]

            # If chunk >= buffer, just keep the last buf_size samples
            if n_samples >= buf_size:
                buf[:] = mod_data[:, -buf_size:]
                self._write_pos[mod] = 0
                self._filled[mod] = buf_size
                continue

            if n_samples <= buf_size - pos:
                buf[:, pos:pos + n_samples] = mod_data
                self._write_pos[mod] = (pos + n_samples) % buf_size
            else:
                first = buf_size - pos
                buf[:, pos:] = mod_data[:, :first]
                rem = n_samples - first
                buf[:, :rem] = mod_data[:, first:]
                self._write_pos[mod] = rem
            self._filled[mod] = min(self._filled[mod] + n_samples, buf_size)

    def maybe_compute(self) -> dict | None:
        """If >=1s since last compute and buffers are sufficiently filled,
        run Welch PSD and return a broadcast-ready payload (or None)."""
        now = time.time()
        if now - self._last_compute_time < PSD_UPDATE_S:
            return None
        half = self._buf_size // 2
        if not any(f >= half for f in self._filled.values()):
            return None
        self._last_compute_time = now

        # Snapshot data for computation (avoids racing with accumulate)
        snapshots = {
            mod: buf[:, :self._filled[mod]].copy()
            for mod, buf in self._buffers.items()
            if self._filled[mod] >= half
        }
        return _compute_psd(snapshots, self._sample_rate)


def _compute_psd(
    snapshots: dict[str, np.ndarray],
    sample_rate: float,
) -> dict | None:
    """Welch PSD per modality → channel-averaged spectrum in dB."""
    from scipy.signal import welch

    result: dict[str, dict] = {}
    for mod, data in snapshots.items():
        nperseg = min(256, data.shape[1])
        freqs, psd = welch(data, fs=sample_rate, nperseg=nperseg, axis=-1)
        # psd: (n_channels, n_freqs)

        max_freq = MODALITY_PSD_MAX_FREQ.get(mod, DEFAULT_PSD_MAX_FREQ)
        mask = freqs <= max_freq
        freqs_t = freqs[mask]
        psd_t = psd[:, mask]

        mean_psd = np.mean(psd_t, axis=0)
        mean_db = 10.0 * np.log10(np.maximum(mean_psd, 1e-20))

        result[mod] = {
            "freqs": freqs_t.astype(np.float32).tobytes(),
            "power": mean_db.astype(np.float32).tobytes(),
            "n": len(freqs_t),
        }

    return {"ch": "psd", "d": result} if result else None


# ---------------------------------------------------------------------------
# Top-level bridge orchestrator
# ---------------------------------------------------------------------------

async def run_visualization_bridge(
    bridge: QueueBridge,
    get_pipeline_service,
) -> None:
    """Drain ring buffer + mode queue, broadcast to WebSocket clients."""
    logger = get_logger("VisualizationBridge")
    logger.info("Visualization bridge started")

    bridge._viz_drain_task_count = 0

    while True:
        try:
            service = get_pipeline_service()
        except RuntimeError:
            await asyncio.sleep(1.0)
            continue

        await service._recording_event.wait()
        bridge.clear_history("mode_data")

        drain_tasks: list[asyncio.Task] = []

        # Raw data from ring buffers (one drain task per stream).
        # Only the first (primary) stream forwards markers to avoid
        # Nx event duplication when multiple streams are configured.
        rb_channel_maps = service.visualization_data_queue
        if rb_channel_maps:
            primary_chosen = False
            for stream_type, channel_map in rb_channel_maps.items():
                is_primary = not primary_chosen
                primary_chosen = True
                drain_tasks.append(
                    asyncio.create_task(
                        _drain_ring_buffer(
                            bridge, stream_type, channel_map,
                            get_pipeline_service, logger,
                            forward_markers=is_primary,
                        )
                    )
                )

        # Mode outputs from queue
        viz_q = service.visualization_queue
        if viz_q is not None:
            drain_tasks.append(
                asyncio.create_task(
                    _drain_mode_data(bridge, viz_q, get_pipeline_service, logger)
                )
            )

        if drain_tasks:
            bridge._viz_drain_task_count = len(drain_tasks)
            logger.info(f"Started {len(drain_tasks)} drain tasks")

        await service._stopped_event.wait()

        for task in drain_tasks:
            task.cancel()
        await asyncio.gather(*drain_tasks, return_exceptions=True)
        bridge.clear_history("mode_data")
        bridge._viz_drain_task_count = 0
        logger.info("Pipeline stopped, drain tasks cancelled")


# ---------------------------------------------------------------------------
# Ring buffer drain (raw data → preprocess → quality → PSD → broadcast)
# ---------------------------------------------------------------------------

async def _drain_ring_buffer(
    bridge, stream_type, channel_map, get_pipeline_service, logger,
    *, forward_markers: bool = True,
):
    """Read from a single ring buffer, preprocess, monitor quality, broadcast."""
    loop = asyncio.get_event_loop()

    service = get_pipeline_service()
    shared_state = service.shared_state

    buf_name = channel_map["buffer_name"]
    modalities = channel_map["modalities"]
    marker_col = channel_map["marker_col"]
    sample_rate = channel_map.get("sample_rate", 500.0)
    eeg_indices = modalities.get("eeg", [])
    channel_labels = channel_map.get("modality_labels", {})

    decimate = max(1, round(sample_rate / TARGET_VIZ_RATE))
    stream_viz_rate = sample_rate / decimate
    logger.info(f"Viz drain: ring buffer={buf_name}, native={sample_rate}Hz, "
                f"decimate={decimate}, viz_rate={stream_viz_rate}Hz")

    try:
        rb = SharedRingBuffer.connect(buf_name)
    except Exception as e:
        logger.error(f"Failed to connect to ring buffer: {e}")
        return

    # Preprocessing
    user_config = service.viz_preproc_config
    preprocessor = _create_viz_preprocessor(modalities, sample_rate, user_config, channel_labels)
    active_config = user_config

    # Quality tracking (EEG only)
    quality: QualityTracker | None = None
    if eeg_indices and shared_state:
        quality = QualityTracker(eeg_indices, sample_rate, shared_state, logger)

    # PSD accumulator (all modalities)
    psd = PSDAccumulator(modalities, sample_rate)

    read_pos = rb.write_pos
    count = 0
    sample_counter = 0
    pending_marker = 0.0

    # Rolling raw buffer (~warmstart window) replayed into a freshly-built
    # preprocessor so adaptive EOG correction engages immediately on toggle.
    # Only kept when the stream has EOG channels — warm-start is a no-op otherwise.
    warmstart_target = int(_VIZ_WARMSTART_S * sample_rate)
    track_raw = "eog" in modalities
    raw_buffer: list[np.ndarray] = []
    raw_buffer_samples = 0

    try:
        while True:
            # Check for dynamic preprocessor config changes
            new_config = service.viz_preproc_config
            if new_config != active_config:
                preprocessor = _create_viz_preprocessor(
                    modalities, sample_rate, new_config, channel_labels
                )
                active_config = new_config
                if quality:
                    quality.reapply_interpolation(preprocessor)
                _warmstart_eog(preprocessor, raw_buffer, modalities)
                logger.info(f"Viz preprocessor reconfigured: {new_config or 'defaults'}")

            # Poll ring buffer in executor
            data, timestamps, new_pos = await loop.run_in_executor(
                None, _read_with_sleep, rb, read_pos
            )
            if len(data) == 0:
                continue
            read_pos = new_pos

            # Maintain the rolling raw buffer for warm-starting on the next toggle.
            if track_raw:
                raw_buffer.append(data.copy())
                raw_buffer_samples += len(data)
                while (len(raw_buffer) > 1
                       and raw_buffer_samples - len(raw_buffer[0]) >= warmstart_target):
                    raw_buffer_samples -= len(raw_buffer.pop(0))

            # Quality monitoring on raw EEG (before preprocessing)
            if quality and eeg_indices:
                quality.update(data[:, eeg_indices])

            # Preprocess batch
            if preprocessor:
                data_dict = {
                    mod: data[:, idx].T.astype(np.float64)
                    for mod, idx in modalities.items()
                }
                processed = preprocessor.process(data_dict)
                for mod, idx in modalities.items():
                    if mod in processed and processed[mod].shape[1] == len(data):
                        data[:, idx] = processed[mod].T.astype(np.float32)

            # Quality publish + interpolation update
            if quality:
                quality.publish_if_due(preprocessor)

            # PSD accumulation (full-rate preprocessed data, before decimation)
            psd.accumulate(data, modalities)
            psd_payload = psd.maybe_compute()
            if psd_payload:
                await bridge.broadcast("visualization", psd_payload)

            # Decimate, batch, and broadcast
            batch_data: dict[str, list] = {}
            batch_markers: list = []
            batch_ts: list = []

            async def _flush():
                nonlocal count, batch_data, batch_markers, batch_ts
                if not batch_ts:
                    return
                payload: dict = {
                    "ch": "raw_data",
                    "ts": batch_ts[-1],
                    "d": {
                        mod: {"bytes": np.stack(samples).astype(np.float32).tobytes(),
                              "shape": [len(samples), len(samples[0])]}
                        for mod, samples in batch_data.items()
                    },
                    "meta": {"sample_rate": stream_viz_rate, "batch": len(batch_ts)},
                }
                if any(m != 0.0 for m in batch_markers):
                    payload["d"]["markers"] = batch_markers
                if channel_labels:
                    payload["channel_labels"] = channel_labels
                await bridge.broadcast("visualization", payload)
                count += len(batch_ts)
                batch_data, batch_markers, batch_ts = {}, [], []

            for i in range(len(data)):
                sample_counter += 1

                if forward_markers:
                    marker_val = data[i, marker_col]
                    if marker_val != 0.0:
                        pending_marker = float(marker_val)

                if sample_counter % decimate != 0:
                    continue

                effective_marker = pending_marker
                pending_marker = 0.0

                for mod, indices in modalities.items():
                    batch_data.setdefault(mod, []).append(data[i, indices])
                batch_markers.append(effective_marker)
                batch_ts.append(timestamps[i])

                if len(batch_ts) >= VIZ_BATCH_SIZE:
                    await _flush()

            await _flush()

            if count % 500 == 0 and count > 0:
                logger.info(f"Viz bridge: {count} samples broadcast")

    except asyncio.CancelledError:
        logger.info(f"Ring buffer drain stopped after {count} samples")
    except Exception as e:
        logger.warning(f"Ring buffer drain error: {e}", exc_info=True)
    finally:
        rb.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Rolling raw buffer replayed into a new viz preprocessor so adaptive EOG correction
# is already converged when a config toggle goes live. Sized just above the estimator's
# 30 s min-fit window (AdaptiveEOGRegression.min_fit_s) so the first refit can fire.
_VIZ_WARMSTART_S = 35.0


def _warmstart_eog(
    preprocessor: OnlinePreprocessor | None,
    raw_chunks: list[np.ndarray],
    modalities: dict[str, list[int]],
) -> None:
    """Replay buffered raw chunks through a freshly-built preprocessor so its
    adaptive EOG estimator (and causal filter states) are warm before going live.

    Without this, enabling EOG correction shows nothing for ~30 s (cold fit). With
    it, the regression is already fit on recent data, so the effect is visible
    within a frame. No-op when EOG correction isn't configured (EOG off or high-pass
    above the ocular band) — the replay itself is what builds the lazy estimator, so
    gate on `eog_correction_enabled`, not on the not-yet-built estimator.
    """
    if preprocessor is None or not preprocessor.eog_correction_enabled:
        return
    for chunk in raw_chunks:
        data_dict = {
            mod: chunk[:, idx].T.astype(np.float64) for mod, idx in modalities.items()
        }
        preprocessor.process(data_dict)


def _create_viz_preprocessor(
    modalities: dict[str, list[int]],
    sample_rate: float,
    user_config: dict | None = None,
    modality_labels: dict[str, list[str]] | None = None,
) -> OnlinePreprocessor | None:
    """Build preprocessor for viz — all modalities, no downsampling."""
    from dendrite.processing.preprocessing.preprocessing_schemas import MODALITY_DEFAULTS

    if not modalities:
        return None
    config: dict[str, dict] = {}
    for mod, indices in modalities.items():
        mod_config: dict[str, Any] = {
            "num_channels": len(indices),
            "sample_rate": sample_rate,
            "downsample_factor": 1,
            **MODALITY_DEFAULTS.get(mod, {}),
        }
        if modality_labels and mod in modality_labels:
            mod_config["channel_labels"] = modality_labels[mod]
        if user_config and mod in user_config:
            mod_config.update(user_config[mod])
        mod_config["downsample_factor"] = 1  # never downsample for viz
        config[mod] = mod_config
    return OnlinePreprocessor(config)


def _read_with_sleep(rb: SharedRingBuffer, read_pos: int):
    """Blocking read from ring buffer with sleep on empty."""
    try:
        data, ts, _local_ts, _receive_ns, new_pos = rb.read_new(read_pos)
    except OverrunError:
        return np.empty((0, 0), np.float32), np.empty(0, np.float64), rb.write_pos
    if len(data) == 0:
        time.sleep(0.005)
    return data, ts, new_pos


async def _drain_mode_data(bridge, mp_queue, get_pipeline_service, logger):
    """Drain mode output queue and broadcast to 'mode_data'."""
    loop = asyncio.get_event_loop()
    count = 0
    last_heartbeat = time.monotonic()

    while True:
        try:
            item = await loop.run_in_executor(
                None, _get_with_timeout, mp_queue, 0.1
            )

            now = time.monotonic()
            if now - last_heartbeat >= 60.0:
                logger.info(f"Mode drain heartbeat: {count} total outputs broadcast")
                last_heartbeat = now

            if item is None:
                continue

            if isinstance(item, dict):
                item.setdefault("ch", "mode_history")
            item = _serialize_mode_data(item)
            await bridge.broadcast("mode_data", item)

            count += 1
            if count % 100 == 0:
                logger.info(f"Viz bridge: {count} mode outputs broadcast")

        except asyncio.CancelledError:
            logger.info(f"Mode drain stopped after {count} outputs")
            return
        except ValueError:
            logger.warning(f"Mode drain: queue broken after {count} outputs (ValueError)")
            return
        except Exception as e:
            logger.warning(f"Mode data drain error: {e}", exc_info=True)
            await asyncio.sleep(0.1)


def _serialize_mode_data(obj):
    if isinstance(obj, dict):
        return {k: _serialize_mode_data(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_mode_data(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return {"bytes": obj.astype(np.float32).tobytes(), "shape": list(obj.shape)}
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def _get_with_timeout(mp_queue, timeout: float):
    try:
        return mp_queue.get(timeout=timeout)
    except queue.Empty:
        return None
