#!/usr/bin/env python
"""
LSL Data Receiver

Handles connection to and data reception from Lab Streaming Layer (LSL) streams.
Optimized for clean code, efficiency, and reduced logging.
"""

import json
import logging
import threading
import time

import pylsl
from PyQt6 import QtCore

DEFAULT_STREAM_NAME = "Dendrite_Visualization"
RAW_DATA_LOG_INTERVAL = 100


class OptimizedLSLReceiver(QtCore.QObject):
    """Receives and parses LSL stream data"""

    new_payload_signal = QtCore.pyqtSignal(dict)
    connection_changed_signal = QtCore.pyqtSignal(bool)  # True = connected, False = disconnected
    stream_info_signal = QtCore.pyqtSignal(str)  # Stream info string for status display

    def __init__(self, stream_name: str = DEFAULT_STREAM_NAME, parent=None):
        super().__init__(parent)
        self.stream_name = stream_name
        self.inlet = None
        self.stop_event = threading.Event()
        self.history_consumed_modes = set()
        self.raw_data_log_counter = 0
        self._connected = False

        self._thread = threading.Thread(
            target=self._receiver_loop, daemon=True, name="LSLReceiverThread"
        )
        self._thread.start()

    def _connect(self) -> bool:
        """Connect to LSL stream. Returns True if successful."""
        try:
            streams = pylsl.resolve_byprop("name", self.stream_name, minimum=1, timeout=2.0)
            if streams:
                self.inlet = pylsl.StreamInlet(streams[0])
                logging.info(f"Connected to LSL stream '{self.stream_name}'")

                # Emit connection status
                self._connected = True
                self.connection_changed_signal.emit(True)

                # Extract and emit stream info
                info = self.inlet.info()
                stream_info = (
                    f"{info.name()}: {info.channel_count()}ch @ {info.nominal_srate():.0f}Hz"
                )
                self.stream_info_signal.emit(stream_info)

                return True
            else:
                logging.debug(f"Stream '{self.stream_name}' not found, retrying...")
                return False
        except Exception as e:
            logging.error(f"Error connecting to stream: {e}")
            return False

    def _set_disconnected(self):
        """Set disconnected state and emit signal."""
        if self._connected:
            self._connected = False
            self.connection_changed_signal.emit(False)

    def _receiver_loop(self):
        """Main receiver loop"""
        logging.info(f"Resolving LSL stream: '{self.stream_name}'...")
        while not self.stop_event.is_set():
            if self._connect():
                break
            time.sleep(2)

        if self.stop_event.is_set():
            return

        while not self.stop_event.is_set():
            try:
                sample, timestamp = self.inlet.pull_sample(timeout=0.5)
                if sample is None:
                    continue

                self._process_sample(sample[0], timestamp)

            except pylsl.LostError:
                logging.error("LSL connection lost. Reconnecting...")
                self._set_disconnected()
                if self.inlet:
                    self.inlet.close_stream()
                self.inlet = None
                if not self._connect():
                    break
            except Exception as e:
                logging.error(f"Error in receiver loop: {e}", exc_info=True)
                time.sleep(1)

        logging.info("LSL receiver loop finished")
        if self.inlet:
            self.inlet.close_stream()

    def _process_sample(self, sample_data: str, lsl_timestamp: float):
        """Parse and emit LSL sample"""
        try:
            payload = json.loads(sample_data)
        except json.JSONDecodeError as e:
            logging.warning(f"Invalid JSON: {e} - Sample: '{sample_data[:100]}'")
            return

        payload_type = payload.get("type", "unknown")
        timestamp = payload.get("timestamp", lsl_timestamp)

        # Handle different payload types
        if payload_type == "raw_data":
            self._handle_raw_data(payload, timestamp)
        elif payload_type == "mode_history":
            self._handle_mode_history(payload)
        else:
            self._handle_mode_output(payload, payload_type, timestamp)

    def _handle_raw_data(self, payload: dict, timestamp: float):
        """Handle raw data payloads with reduced logging"""
        self.raw_data_log_counter = (self.raw_data_log_counter + 1) % RAW_DATA_LOG_INTERVAL
        if self.raw_data_log_counter == 0:
            logging.debug(f"Received raw_data (ts={timestamp:.3f})")
        self.new_payload_signal.emit(payload)

    def _handle_mode_history(self, payload: dict):
        """Handle mode history - emit once, let dashboard unpack"""
        mode_name = payload.get("mode_name", "unknown")

        if mode_name in self.history_consumed_modes:
            logging.debug(f"Skipping duplicate history for '{mode_name}'")
            return

        self.history_consumed_modes.add(mode_name)
        packet_count = payload.get("packet_count", len(payload.get("packets", [])))
        mode_type = payload.get("mode_type", "unknown")

        logging.info(
            f"Processing {packet_count} history packets for '{mode_name}' (type: {mode_type})"
        )
        self.new_payload_signal.emit(payload)

    def _handle_mode_output(self, payload: dict, payload_type: str, timestamp: float):
        """Handle regular mode output payloads"""
        mode_name = payload.get("mode_name", "unknown")
        mode_type = payload.get("mode_type")

        if mode_type is None:
            logging.warning(f"Received '{payload_type}' from '{mode_name}' missing mode_type field")
            mode_type = "missing"

        self.new_payload_signal.emit(payload)

    def stop(self):
        """Stop receiver cleanly"""
        logging.info("Stopping LSL receiver...")
        self._set_disconnected()
        self.stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logging.warning("Receiver thread did not terminate gracefully")
        logging.info("LSL receiver stopped")

    def trigger_reconnect(self):
        """Force reconnect by restarting thread"""
        logging.info("Triggering reconnect...")
        self._set_disconnected()

        # Stop current thread
        if self._thread.is_alive():
            self.stop_event.set()
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                logging.warning("Thread did not stop in time")

        # Reset state
        self.stop_event.clear()
        self.inlet = None
        self.history_consumed_modes.clear()
        self.raw_data_log_counter = 0

        # Start new thread
        self._thread = threading.Thread(
            target=self._receiver_loop, daemon=True, name="LSLReceiverThread-Reconnect"
        )
        self._thread.start()
        logging.info("Reconnect triggered")
