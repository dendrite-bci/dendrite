"""Offline evaluation using actual AsyncMode code path.

Feeds data through the existing AsynchronousMode pipeline without modifications,
ensuring offline evaluation tests the real mode code.
"""

import logging
import threading
import time
from collections.abc import Callable
from queue import Empty, Queue
from typing import Any

import numpy as np

from dendrite.processing.gating import PredictionGating
from dendrite.processing.modes.asynchronous_mode import AsynchronousMode


class OfflineAsyncRunner:
    """Run AsyncMode on offline data by feeding through its existing pipeline.

    This runner uses the actual AsynchronousMode code path for evaluation,
    ensuring offline metrics match real-time behavior exactly.
    """

    def run(
        self,
        decoder: Any,
        X: np.ndarray,
        event_times: np.ndarray,
        event_labels: np.ndarray,
        modality: str,
        sample_rate: float,
        window_length_sec: float,
        step_size_ms: float,
        gating_config: dict[str, Any],
        callback: Callable[[dict[str, Any]], None],
        stop_event: threading.Event | None = None,
        real_time: bool = False,
        class_names: dict[int, str] | None = None,
    ):
        """Run evaluation using AsyncMode's existing methods.

        Args:
            decoder: Trained decoder with predict_sample() method
            X: Continuous data, shape (n_samples, n_channels)
            event_times: Sample indices of events
            event_labels: Labels for each event
            modality: Data modality key (e.g., 'eeg')
            sample_rate: Sampling rate in Hz
            window_length_sec: Epoch window length in seconds
            step_size_ms: Step size between predictions in ms
            gating_config: Dict with confidence_threshold, dwell_time_sec, etc.
            callback: Called with each prediction result
            stop_event: Optional event to stop early
            real_time: If True, simulate real-time by delaying between prediction steps
            class_names: Optional mapping from class index to name (e.g., {0: 'left', 1: 'right'})

        Returns:
            MetricsManager with final metrics
        """
        n_samples, n_channels = X.shape

        # Build event lookup for fast marker injection
        event_lookup = {int(t): int(l) for t, l in zip(event_times, event_labels, strict=False)}

        # Create event_mapping dict {event_id: event_label}
        unique_labels = sorted(set(int(l) for l in event_labels if l >= 0))
        if class_names:
            event_mapping = {idx: class_names.get(idx, str(idx)) for idx in unique_labels}
        else:
            event_mapping = {l: str(l) for l in unique_labels}

        # Create queues for mode communication
        data_queue = Queue()
        output_queue = Queue()
        mode_stop = threading.Event()

        # Create mode instance config (only background_class for metrics evaluation)
        instance_config = {
            "name": "OfflineEval",
            "channel_selection": {modality: list(range(n_channels))},
            "decoder_config": {"model_config": {}},
            "decoder_source": "external",  # Decoder injected after init
            "window_length_sec": window_length_sec,
            "step_size_ms": step_size_ms,
            "evaluation_config": {"background_class": gating_config.get("background_class", 0)},
            "event_mapping": event_mapping,
        }

        # Create mode instance (not as a Process, just an object)
        mode = AsynchronousMode(
            data_queue=data_queue,
            output_queue=output_queue,
            stop_event=mode_stop,
            instance_config=instance_config,
            sample_rate=sample_rate,
        )

        # Initialize mode (sets up buffer, metrics)
        # Must call setup methods in same order as base mode's run()
        mode._setup_logger()
        mode._initialize_mode()

        # Suppress verbose prediction logs during offline evaluation
        mode.logger.setLevel(logging.WARNING)

        # Inject pre-trained decoder (overwrites any decoder created during init)
        mode.decoder = decoder

        # Create gating instance for post-prediction filtering
        gating = self._create_gating(gating_config)

        # Per-step delay for real-time pacing (sleep per prediction, not per sample)
        # time.sleep() has OS overhead (1-15ms), so sleeping per-sample is too slow
        step_delay_sec = (step_size_ms / 1000.0) if real_time else 0
        last_prediction_time = time.time()

        # Process samples through mode's existing methods
        for idx in range(n_samples):
            if stop_event and stop_event.is_set():
                break

            # Build sample dict matching expected format
            # Each sample must be (n_channels, 1) for buffer concatenation
            sample = {
                modality: X[idx][:, np.newaxis],  # (n_channels, 1)
                "markers": event_lookup.get(idx, -1),
            }

            # Feed through mode's processing pipeline
            mode._process_data(sample)

            # Check if prediction should be triggered
            if mode.buffer.is_ready_for_step(mode.samples_per_prediction_step):
                # Pace predictions to real-time (1s of data ≈ 1s wall time)
                if real_time:
                    elapsed = time.time() - last_prediction_time
                    sleep_time = step_delay_sec - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    last_prediction_time = time.time()

                mode._trigger_prediction()

            # Collect any outputs
            while not output_queue.empty():
                try:
                    packet = output_queue.get_nowait()
                    if isinstance(packet, dict):
                        data = packet.get("data", {})
                        raw_prediction = data.get("prediction")
                        confidence = data.get("confidence")

                        # Apply gating if configured
                        if gating and raw_prediction is not None and confidence is not None:
                            gated_prediction = gating.apply(raw_prediction, confidence)
                        else:
                            gated_prediction = raw_prediction

                        # Pass through all metrics from mode's output
                        result_dict = {
                            "prediction": gated_prediction,
                            "raw_prediction": raw_prediction,
                            "confidence": confidence,
                            "true_label": data.get("true_label"),
                            "sample_idx": idx,
                            "accuracy": data.get("accuracy", 0.0),
                        }
                        # Add per-class accuracy from metrics_manager for live display
                        if mode.metrics_manager:
                            current_metrics = mode.metrics_manager.get_current_metrics()
                            result_dict["per_class_accuracy_named"] = current_metrics.get(
                                "per_class_accuracy_named", {}
                            )
                        callback(result_dict)
                except Empty:
                    break

        return mode.metrics_manager

    def _create_gating(self, gating_config: dict[str, Any]) -> PredictionGating | None:
        """Create a PredictionGating instance from config.

        Args:
            gating_config: Dict with use_confidence_gating, confidence_threshold,
                          use_dwell_time_gating, dwell_time_sec, background_class

        Returns:
            PredictionGating instance or None if gating disabled
        """
        use_confidence = gating_config.get("use_confidence_gating", False)
        use_dwell = gating_config.get("use_dwell_time_gating", False)

        if not use_confidence and not use_dwell:
            return None

        # Convert dwell_time_sec to window_size (assuming ~10 predictions/sec for 100ms step)
        dwell_time_sec = gating_config.get("dwell_time_sec", 0.5)
        dwell_window_size = max(1, int(dwell_time_sec * 10))  # Approximate based on step_size

        return PredictionGating(
            confidence_threshold=gating_config.get("confidence_threshold", 0.6),
            dwell_window_size=dwell_window_size,
            background_class=gating_config.get("background_class", 0),
            use_confidence=use_confidence,
            use_dwell=use_dwell,
        )
