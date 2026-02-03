"""Simulation worker for running async evaluation in a background thread."""

import threading
from typing import Any

import numpy as np
from PyQt6 import QtCore

from dendrite.auxiliary.ml_workbench.backend.offline_async_runner import OfflineAsyncRunner
from dendrite.utils.logger_central import get_logger

logger = get_logger(__name__)


class SimulationWorker(QtCore.QObject):
    """Worker for running simulation in a thread.

    Uses thread-safe list for results - GUI polls via timer instead of signals.
    This eliminates signal overhead and lets the evaluation loop run uninterrupted.

    Uses OfflineAsyncRunner which feeds data through the actual AsyncMode code path.
    """

    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        decoder: Any,
        X: np.ndarray,
        event_times: np.ndarray,
        event_labels: np.ndarray,
        epoch_length: int,
        step_size: int,
        modality: str,
        sample_rate: float = 250.0,
        gating_config: dict[str, Any] | None = None,
        real_time: bool = False,
        class_names: dict[int, str] | None = None,
    ):
        super().__init__()
        self.decoder = decoder
        self.X = X
        self.event_times = event_times
        self.event_labels = event_labels
        self.epoch_length = epoch_length
        self.step_size = step_size
        self.modality = modality
        self.sample_rate = sample_rate
        self.gating_config = gating_config or {}
        self.real_time = real_time
        self.class_names = class_names
        self._stop_event = threading.Event()
        # Thread-safe results accumulator (GUI polls this)
        self.results: list[dict] = []
        self._lock = threading.Lock()
        # Store metrics manager from runner
        self.metrics_manager = None

    def stop(self):
        self._stop_event.set()

    def get_results_since(self, start_idx: int) -> list[dict]:
        """Get new results since last poll (thread-safe)."""
        with self._lock:
            return self.results[start_idx:]

    def _on_prediction(self, pred: dict):
        """Just append to list, no signals (fast)."""
        with self._lock:
            self.results.append(pred)

    @QtCore.pyqtSlot()
    def run(self):
        try:
            runner = OfflineAsyncRunner()
            # Convert epoch_length to window_length_sec
            window_length_sec = self.epoch_length / self.sample_rate
            # Convert step_size to step_size_ms
            step_size_ms = (self.step_size / self.sample_rate) * 1000

            self.metrics_manager = runner.run(
                decoder=self.decoder,
                X=self.X,
                event_times=self.event_times,
                event_labels=self.event_labels,
                modality=self.modality,
                sample_rate=self.sample_rate,
                window_length_sec=window_length_sec,
                step_size_ms=step_size_ms,
                gating_config=self.gating_config,
                callback=self._on_prediction,
                stop_event=self._stop_event,
                real_time=self.real_time,
                class_names=self.class_names,
            )
            self.finished.emit()
        except Exception as e:
            logger.exception("Simulation failed")
            self.error.emit(str(e))
            self.finished.emit()
