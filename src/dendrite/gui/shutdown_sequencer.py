"""
Shutdown Sequencer

Manages ordered teardown of processes and threads with timeouts.
Extracted from MainWindow to isolate the stop state machine logic.
"""

import time
from typing import Any

from PyQt6 import QtCore, QtWidgets

from dendrite.utils.logger_central import get_logger


class ShutdownSequencer(QtCore.QObject):
    """Manages ordered shutdown of processes/threads with per-target timeouts.

    Supports both non-blocking (timer-polled) and blocking shutdown modes.
    Emits `finished` when all targets have been stopped.
    """

    finished = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = get_logger("ShutdownSequencer")
        self._targets: list[tuple[str, Any, float]] = []
        self._index = 0
        self._start_time: float | None = None
        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.timeout.connect(self._poll_progress)

    def stop(self, targets: list[tuple[str, Any, float]], blocking: bool = False):
        """Begin ordered shutdown of targets.

        Args:
            targets: List of (name, process_or_thread, timeout_seconds).
            blocking: If True, block (while pumping Qt events) until all stopped.
        """
        self._targets = targets
        self._index = 0
        self._start_time = None

        if blocking:
            self._stop_blocking()
        else:
            self._poll_progress()

    def _stop_blocking(self):
        """Stop all targets, blocking but processing Qt events."""
        for name, target, timeout in self._targets:
            self.logger.info(f"Stopping {name}...")
            start = time.time()
            while target.is_alive() and (time.time() - start) < timeout:
                QtWidgets.QApplication.processEvents()
                time.sleep(0.05)
            if target.is_alive():
                self.logger.warning(f"Force terminating {name}")
                if hasattr(target, "terminate"):
                    target.terminate()
        self._finalize()

    def _poll_progress(self):
        """Poll current target, advance when done or timed out."""
        if self._index >= len(self._targets):
            self._poll_timer.stop()
            self._finalize()
            return

        name, target, timeout = self._targets[self._index]

        if self._start_time is None:
            self._start_time = time.time()
            self.logger.info(f"Stopping {name}...")
            self._poll_timer.start(50)
            return

        if not target.is_alive():
            self._start_time = None
            self._index += 1
            self._poll_progress()
            return

        if time.time() - self._start_time > timeout:
            self.logger.warning(f"Force terminating {name}")
            if hasattr(target, "terminate"):
                target.terminate()
            self._start_time = None
            self._index += 1
            self._poll_progress()

    def _finalize(self):
        """Emit finished signal when all targets are stopped."""
        self._targets = []
        self._index = 0
        self._start_time = None
        self.finished.emit()
