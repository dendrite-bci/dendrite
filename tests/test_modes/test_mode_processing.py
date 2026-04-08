"""Direct mode processing tests — feed synthetic data, verify output.

Tests the core processing pipeline of each mode without ring buffers or
multiprocessing. Creates mode instances via __new__ (bypassing Process.__init__),
configures them minimally, and feeds samples through _process_data().
"""

import logging
import multiprocessing
import queue

import numpy as np
import pytest
from unittest.mock import MagicMock

from dendrite.processing.modes.mode_utils import Buffer, FanOutQueue, extract_event_code


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample(n_channels=4, value=0.0, marker=0, modality="eeg", timestamp=0.0):
    """Build a single sample dict as modes expect from the ring buffer."""
    return {
        modality: np.full((n_channels, 1), value, dtype=np.float32),
        "markers": np.array([[marker]], dtype=np.float32),
        "lsl_timestamp": timestamp,
        "_receive_ns": int(timestamp * 1e9),
    }


def _drain(q):
    """Drain a queue into a list."""
    items = []
    while True:
        try:
            items.append(q.get_nowait())
        except (queue.Empty, EOFError):
            break
    return items


# ---------------------------------------------------------------------------
# Synchronous Mode
# ---------------------------------------------------------------------------


def _make_sync_mode(n_channels=4, sample_rate=250.0, modality="eeg"):
    """Create a SynchronousMode instance ready to process data."""
    from dendrite.processing.modes.synchronous_mode import SynchronousMode

    main_q = queue.Queue()
    pred_q = queue.Queue()

    mode = SynchronousMode.__new__(SynchronousMode)
    mode.logger = logging.getLogger("test_sync")
    mode.stop_event = multiprocessing.Event()
    mode.output_queue = FanOutQueue([main_q])
    mode.prediction_queue = pred_q
    mode.training_queue = MagicMock()
    mode.shared_state = MagicMock()
    mode.shared_state.get.return_value = None
    mode.mode_name = "test_sync"
    mode.mode_type = "synchronous"
    mode.file_identifier = "test"
    mode.study_name = "test"

    # Channel / modality config
    mode.channel_selection = {modality: list(range(n_channels))}
    mode.modality_labels = {modality: [f"Ch{i}" for i in range(n_channels)]}
    mode.modalities = [modality]
    mode.sample_rate = sample_rate
    mode.effective_sample_rate = sample_rate

    # Synchronous-specific
    mode.event_mapping = {1: "left", 2: "right"}
    mode.label_mapping = {"left": 0, "right": 1}
    mode.reverse_label_mapping = {0: "left", 1: "right"}
    mode.epoch_tmin = 0.0
    mode.epoch_tmax = 0.5
    mode.tmin_samples = 0
    mode.tmax_samples = int(0.5 * sample_rate)
    mode.epoch_length_samples = int(0.5 * sample_rate)
    mode.training_interval = 999  # Don't trigger training in tests
    mode.decoder_source = "online"

    # State
    mode.epoch_count = 0
    mode.current_sample_index = 0
    mode.last_lsl_timestamp = 0.0
    mode.pending_epochs = []
    mode.decoder = None
    mode.decoder_config = {}
    mode._training_pending = False
    mode._pending_decoder_load = None
    mode._sample_preprocessor = None
    mode._reader = None
    mode.metrics_manager = None
    mode._rb_config = None
    mode._mode_type = "synchronous"
    mode._gpu_last_emit_time = 0.0

    # Buffer: needs enough room for pre-event + epoch
    buf_size = mode.tmin_samples + mode.epoch_length_samples + int(sample_rate)
    mode.buffer = Buffer(
        modalities=[modality], buffer_size=buf_size, logger=mode.logger,
    )

    return mode, main_q, pred_q


class TestSynchronousProcessing:
    """Test synchronous mode event detection and epoch extraction."""

    def test_event_triggers_epoch_extraction(self):
        """Feed samples with an event marker, verify ERP output."""
        mode, main_q, _ = _make_sync_mode(n_channels=4, sample_rate=250.0)

        # Fill buffer to capacity first (needs to be full before extraction works)
        for i in range(mode.buffer.buffer_size):
            mode._process_data(_make_sample(timestamp=i / 250.0))

        # Send event marker
        mode._process_data(_make_sample(marker=1, timestamp=1.0))

        # Feed enough post-event samples for epoch extraction
        for i in range(mode.tmax_samples + 5):
            mode._process_data(_make_sample(timestamp=1.0 + (i + 1) / 250.0))

        assert mode.epoch_count == 1
        outputs = _drain(main_q)
        erp_outputs = [o for o in outputs if o.get("type") == "erp"]
        assert len(erp_outputs) == 1

    def test_multiple_events_extract_multiple_epochs(self):
        """Two different events should produce two epochs."""
        mode, main_q, _ = _make_sync_mode(n_channels=4, sample_rate=250.0)

        # Fill buffer
        for i in range(mode.buffer.buffer_size):
            mode._process_data(_make_sample(timestamp=i / 250.0))

        # Event 1 (left)
        mode._process_data(_make_sample(marker=1, timestamp=2.0))
        for i in range(mode.tmax_samples + 5):
            mode._process_data(_make_sample(timestamp=2.0 + (i + 1) / 250.0))

        # Event 2 (right)
        mode._process_data(_make_sample(marker=2, timestamp=3.0))
        for i in range(mode.tmax_samples + 5):
            mode._process_data(_make_sample(timestamp=3.0 + (i + 1) / 250.0))

        assert mode.epoch_count == 2

    def test_unknown_event_ignored(self):
        """Events not in event_mapping should not trigger epochs."""
        mode, _, _ = _make_sync_mode()

        for i in range(mode.buffer.buffer_size):
            mode._process_data(_make_sample(timestamp=i / 250.0))

        mode._process_data(_make_sample(marker=99, timestamp=2.0))
        for i in range(mode.tmax_samples + 5):
            mode._process_data(_make_sample(timestamp=2.0 + (i + 1) / 250.0))

        assert mode.epoch_count == 0

    def test_epoch_data_has_correct_shape(self):
        """Extracted ERP data should have (n_channels, epoch_samples) shape."""
        mode, main_q, _ = _make_sync_mode(n_channels=8, sample_rate=250.0)

        for i in range(mode.buffer.buffer_size):
            mode._process_data(_make_sample(n_channels=8, timestamp=i / 250.0))

        mode._process_data(_make_sample(n_channels=8, marker=1, timestamp=2.0))
        for i in range(mode.tmax_samples + 5):
            mode._process_data(_make_sample(n_channels=8, timestamp=2.0 + (i + 1) / 250.0))

        outputs = _drain(main_q)
        erp = next(o for o in outputs if o.get("type") == "erp")
        epoch_data = erp["data"]["data"]
        assert epoch_data.shape[0] == 8  # channels


# ---------------------------------------------------------------------------
# Neurofeedback Mode
# ---------------------------------------------------------------------------


def _make_nfb_mode(n_channels=4, sample_rate=250.0, modality="eeg"):
    """Create a NeurofeedbackMode instance ready to process data."""
    from dendrite.processing.modes.neurofeedback_mode import NeurofeedbackMode

    main_q = queue.Queue()
    pred_q = queue.Queue()

    mode = NeurofeedbackMode.__new__(NeurofeedbackMode)
    mode.logger = logging.getLogger("test_nfb")
    mode.stop_event = multiprocessing.Event()
    mode.output_queue = FanOutQueue([main_q])
    mode.prediction_queue = pred_q
    mode.training_queue = MagicMock()
    mode.shared_state = MagicMock()
    mode.shared_state.get.return_value = None
    mode.mode_name = "test_nfb"
    mode.mode_type = "neurofeedback"
    mode.file_identifier = "test"
    mode.study_name = "test"

    # Channel config
    mode.channel_selection = {modality: list(range(n_channels))}
    mode.modality_labels = {modality: [f"Ch{i}" for i in range(n_channels)]}
    mode.modalities = [modality]
    mode.sample_rate = sample_rate
    mode.effective_sample_rate = sample_rate
    mode.modality_name = modality

    # NFB-specific
    window_sec = 1.0
    step_ms = 250
    mode.window_length_samples = int(window_sec * sample_rate)
    mode.window_step_samples = int(step_ms * sample_rate / 1000)
    mode.feature_config = {
        "target_bands": {"alpha": [8.0, 12.0], "beta": [13.0, 30.0]},
        "use_relative_power": True,
        "use_cluster_mode": False,
    }
    mode.use_cluster_mode = False
    mode.channel_labels = [f"Ch{i}" for i in range(n_channels)]
    mode.selected_channel_indices = list(range(n_channels))
    mode.target_bands = {"alpha": [8.0, 12.0], "beta": [13.0, 30.0]}

    # State
    mode.current_sample_index = 0
    mode.last_lsl_timestamp = 0.0
    mode.decoder = None
    mode.decoder_config = {}
    mode._sample_preprocessor = None
    mode._reader = None
    mode.metrics_manager = None
    mode._rb_config = None
    mode._mode_type = "neurofeedback"
    mode._gpu_last_emit_time = 0.0

    # IAF defaults (disabled)
    mode.iaf_event_id = None
    mode.iaf_baseline_sec = 5.0
    mode.iaf_range = (7.0, 14.0)
    mode.iaf_state = "idle"
    mode.iaf_baseline_buf = None
    mode.iaf_baseline_pos = 0
    mode.iaf_baseline_samples = 0
    mode.iaf_value = None
    mode._original_bands = None
    mode._event_handlers = {}
    mode.use_relative_power = True

    # Initialize band power transform
    from dendrite.ml.features.transforms import BandPowerTransform
    nperseg = max(int(sample_rate / 0.5), int(sample_rate * 0.5))
    mode.band_power_transform = BandPowerTransform(
        bands=mode.target_bands, fs=sample_rate, nperseg=nperseg,
    )

    # Buffer
    mode.buffer = Buffer(
        modalities=[modality], buffer_size=mode.window_length_samples, logger=mode.logger,
    )

    return mode, main_q, pred_q


def _feed_nfb(mode, n_channels, sample_rate, modality="eeg", rng=None):
    """Feed samples through NFB mode (inline _run_main_loop logic)."""
    rng = rng or np.random.default_rng(42)
    total = mode.window_length_samples + mode.window_step_samples
    for i in range(total):
        sample = _make_sample(n_channels=n_channels, modality=modality, timestamp=i / sample_rate)
        sample[modality] = (rng.standard_normal((n_channels, 1)) +
                            np.sin(2 * np.pi * 10 * i / sample_rate)).astype(np.float32)
        # Inline main loop: preprocess → buffer → step check
        processed = mode._preprocess_sample(sample)
        if processed is None:
            continue
        mode.last_lsl_timestamp = processed.get("lsl_timestamp", 0.0)
        mode.buffer.add_sample(processed)
        if mode.buffer.is_ready_for_step(mode.window_step_samples):
            mode._extract_and_send_features()


class TestNeurofeedbackProcessing:
    """Test neurofeedback mode band power extraction."""

    def test_produces_band_powers_after_window_filled(self):
        """After filling one window + step, NFB should output band powers."""
        mode, main_q, pred_q = _make_nfb_mode(n_channels=4, sample_rate=250.0)
        _feed_nfb(mode, 4, 250.0)

        outputs = _drain(pred_q)
        nfb = [o for o in outputs if isinstance(o, dict) and o.get("type") == "neurofeedback"]
        assert len(nfb) >= 1

        payload = nfb[0]["data"]
        assert "channel_powers" in payload
        powers = payload["channel_powers"]
        assert len(powers) >= 1
        first_ch = next(iter(powers.values()))
        assert "alpha" in first_ch
        assert "beta" in first_ch

    def test_band_powers_are_positive(self):
        """Band power values should be non-negative."""
        mode, _, pred_q = _make_nfb_mode(n_channels=2, sample_rate=250.0)
        _feed_nfb(mode, 2, 250.0)

        outputs = _drain(pred_q)
        nfb = [o for o in outputs if isinstance(o, dict) and o.get("type") == "neurofeedback"]
        if nfb:
            for ch_powers in nfb[0]["data"]["channel_powers"].values():
                for band, power in ch_powers.items():
                    assert power >= 0, f"Negative power in {band}: {power}"

    def test_emg_modality_works(self):
        """NFB should work with non-EEG modalities."""
        mode, _, pred_q = _make_nfb_mode(n_channels=3, sample_rate=1000.0, modality="emg")
        _feed_nfb(mode, 3, 1000.0, modality="emg")

        outputs = _drain(pred_q)
        nfb = [o for o in outputs if isinstance(o, dict) and o.get("type") == "neurofeedback"]
        assert len(nfb) >= 1


# ---------------------------------------------------------------------------
# IAF Detection (pure functions + integration)
# ---------------------------------------------------------------------------


class TestIAFDetection:
    """Test IAF computation and band shifting."""

    def test_compute_iaf_known_peak(self):
        """CoG on a pure 10 Hz sinusoid should return ~10 Hz."""
        from dendrite.ml.features.iaf import compute_iaf

        fs = 250.0
        n_channels, n_samples = 4, int(5 * fs)
        t = np.arange(n_samples) / fs
        data = np.sin(2 * np.pi * 10 * t)[np.newaxis, :].repeat(n_channels, axis=0)
        data += np.random.default_rng(0).standard_normal(data.shape) * 0.1

        iaf = compute_iaf(data, fs, (7.0, 14.0))
        assert 9.5 < iaf < 10.5, f"Expected ~10 Hz, got {iaf:.2f}"

    def test_compute_iaf_shifted_peak(self):
        """CoG on a 9.5 Hz sinusoid should return ~9.5 Hz."""
        from dendrite.ml.features.iaf import compute_iaf

        fs = 250.0
        n_channels, n_samples = 2, int(5 * fs)
        t = np.arange(n_samples) / fs
        data = np.sin(2 * np.pi * 9.5 * t)[np.newaxis, :].repeat(n_channels, axis=0)

        iaf = compute_iaf(data, fs, (7.0, 14.0))
        assert 9.0 < iaf < 10.0, f"Expected ~9.5 Hz, got {iaf:.2f}"

    def test_shift_bands_overlapping(self):
        """Bands overlapping iaf_range should be shifted; others untouched."""
        from dendrite.ml.features.iaf import shift_bands

        bands = {"alpha": [8.0, 12.0], "beta": [15.0, 30.0]}
        shifted = shift_bands(bands, iaf=9.5, iaf_range=(7.0, 14.0))
        assert shifted["alpha"] == [7.5, 11.5], f"Got {shifted['alpha']}"
        assert shifted["beta"] == [15.0, 30.0], "Beta should not shift"

    def test_shift_bands_partial_overlap(self):
        """Band partially overlapping iaf_range should be shifted."""
        from dendrite.ml.features.iaf import shift_bands

        bands = {"smr": [12.0, 15.0], "theta": [4.0, 7.0]}
        shifted = shift_bands(bands, iaf=11.0, iaf_range=(7.0, 14.0))
        assert shifted["smr"] == [13.0, 16.0], f"Got {shifted['smr']}"
        assert shifted["theta"] == [4.0, 7.0], "Theta should not shift"

    def test_shift_bands_clamp_zero(self):
        """Shifted band should not go below 0 Hz."""
        from dendrite.ml.features.iaf import shift_bands

        bands = {"low_alpha": [7.0, 10.0]}
        shifted = shift_bands(bands, iaf=6.0, iaf_range=(5.0, 14.0))
        assert shifted["low_alpha"][0] >= 0, "Clamped to 0"
        assert shifted["low_alpha"] == [3.0, 6.0]

    def test_iaf_integration(self):
        """Full integration: trigger event → collect baseline → verify IAF."""
        from dendrite.processing.modes.neurofeedback_mode import NeurofeedbackMode

        iaf_event_id = 99
        baseline_sec = 0.5  # short for test
        fs = 250.0
        n_channels = 2
        target_freq = 9.0

        mode, main_q, pred_q = _make_nfb_mode(n_channels=n_channels, sample_rate=fs)
        mode.feature_config["iaf_event_id"] = iaf_event_id
        mode.feature_config["iaf_baseline_sec"] = baseline_sec
        mode.feature_config["iaf_range"] = [7.0, 14.0]
        mode.iaf_event_id = iaf_event_id
        mode.iaf_baseline_sec = baseline_sec
        mode.iaf_range = (7.0, 14.0)
        mode.iaf_state = "idle"
        mode.iaf_baseline_buf = None
        mode.iaf_baseline_pos = 0
        mode.iaf_baseline_samples = int(baseline_sec * fs)
        mode._original_bands = {k: list(v) for k, v in mode.target_bands.items()}
        mode.iaf_value = None
        mode._event_handlers = {iaf_event_id: mode._on_iaf_trigger}

        rng = np.random.default_rng(42)

        # Fill buffer first (window_length_samples + step)
        total_fill = mode.window_length_samples + mode.window_step_samples
        for i in range(total_fill):
            sample = _make_sample(n_channels=n_channels, modality="eeg", timestamp=i / fs)
            sample["eeg"] = rng.standard_normal((n_channels, 1)).astype(np.float32)
            processed = mode._preprocess_sample(sample)
            if processed is None:
                continue
            mode.last_lsl_timestamp = processed.get("lsl_timestamp", 0.0)
            mode.buffer.add_sample(processed)

        # Now send IAF trigger event
        trigger = _make_sample(n_channels=n_channels, modality="eeg", timestamp=total_fill / fs)
        trigger["markers"] = np.array([[iaf_event_id]])
        trigger["eeg"] = rng.standard_normal((n_channels, 1)).astype(np.float32)
        processed = mode._preprocess_sample(trigger)
        mode.last_lsl_timestamp = processed.get("lsl_timestamp", 0.0)
        if extract_event_code(processed) == iaf_event_id:
            mode._on_iaf_trigger(processed)
        mode.buffer.add_sample(processed)

        assert mode.iaf_state == "collecting"

        # Feed baseline samples with 9 Hz tone
        baseline_samples = mode.iaf_baseline_samples
        for i in range(baseline_samples):
            t = (total_fill + 1 + i) / fs
            sample = _make_sample(n_channels=n_channels, modality="eeg", timestamp=t)
            tone = np.sin(2 * np.pi * target_freq * t).astype(np.float32)
            sample["eeg"] = np.full((n_channels, 1), tone, dtype=np.float32)
            processed = mode._preprocess_sample(sample)
            if processed is None:
                continue
            mode.last_lsl_timestamp = processed.get("lsl_timestamp", 0.0)
            mode._accumulate_iaf_sample(processed)
            mode.buffer.add_sample(processed)

        assert mode.iaf_state == "done"
        assert mode.iaf_value is not None
        assert 8.0 < mode.iaf_value < 10.5, f"Expected ~{target_freq}, got {mode.iaf_value:.2f}"

        # Bands should have been shifted
        assert mode.target_bands != mode._original_bands

        # IAF result should be in the output queue
        outputs = _drain(main_q)
        iaf_outputs = [o for o in outputs if isinstance(o, dict) and o.get("type") == "iaf_result"]
        assert len(iaf_outputs) == 1
        assert "iaf_hz" in iaf_outputs[0]["data"]

    def test_iaf_disabled_ignores_markers(self):
        """Without iaf_event_id, markers should be ignored."""
        mode, main_q, _ = _make_nfb_mode()
        mode.iaf_event_id = None
        mode.iaf_state = "idle"
        mode.iaf_baseline_buf = None
        mode.iaf_baseline_pos = 0
        mode.iaf_baseline_samples = 0
        mode._original_bands = None
        mode.iaf_value = None
        mode._event_handlers = {}

        _feed_nfb(mode, 4, 250.0)
        assert mode.iaf_state == "idle"
        assert mode.iaf_value is None


# ---------------------------------------------------------------------------
# Asynchronous Mode (without decoder — just buffer filling)
# ---------------------------------------------------------------------------


def _make_async_mode(n_channels=4, sample_rate=250.0, modality="eeg"):
    """Create an AsynchronousMode instance (no decoder — tests buffer/window logic)."""
    from dendrite.processing.modes.asynchronous_mode import AsynchronousMode

    main_q = queue.Queue()
    pred_q = queue.Queue()

    mode = AsynchronousMode.__new__(AsynchronousMode)
    mode.logger = logging.getLogger("test_async")
    mode.stop_event = multiprocessing.Event()
    mode.output_queue = FanOutQueue([main_q])
    mode.prediction_queue = pred_q
    mode.training_queue = MagicMock()
    mode.shared_state = MagicMock()
    mode.shared_state.get.return_value = None
    mode.mode_name = "test_async"
    mode.mode_type = "asynchronous"
    mode.file_identifier = "test"
    mode.study_name = "test"

    # Channel config
    mode.channel_selection = {modality: list(range(n_channels))}
    mode.modality_labels = {modality: [f"Ch{i}" for i in range(n_channels)]}
    mode.modalities = [modality]
    mode.sample_rate = sample_rate
    mode.effective_sample_rate = sample_rate

    # Async-specific
    mode.window_length_sec = 1.0
    mode.step_size_ms = 100
    mode.epoch_length_samples = int(1.0 * sample_rate)
    mode.samples_per_prediction_step = int(0.1 * sample_rate)
    mode.decoder_source = "online"
    mode.decoder_config = {}
    mode._source_mode = None

    # State
    mode.epoch_count = 0
    mode.current_sample_index = 0
    mode.prediction_count = 0
    mode.last_lsl_timestamp = 0.0
    mode.decoder = None
    mode._training_pending = False
    mode._pending_decoder_load = None
    mode._last_decoder_check_ts = 0.0
    mode._sample_preprocessor = None
    mode._reader = None
    mode.metrics_manager = None
    mode._rb_config = None
    mode._active_label = None
    mode._active_label_remaining = 0
    mode.event_mapping = {1: "left", 2: "right"}
    mode.label_mapping = {"left": 0, "right": 1}
    mode.reverse_label_mapping = {0: "left", 1: "right"}
    mode._mode_type = "asynchronous"
    mode._gpu_last_emit_time = 0.0
    mode.index_to_event_code = {v: k for k, v in mode.label_mapping.items()}
    mode._current_label = -1
    mode._active_label = -1
    mode._labeling_samples_remaining = 0
    mode._cached_metrics = {}

    # Buffer
    mode.buffer = Buffer(
        modalities=[modality], buffer_size=mode.epoch_length_samples, logger=mode.logger,
    )

    return mode, main_q, pred_q


class TestAsynchronousProcessing:
    """Test asynchronous mode sliding window logic."""

    def test_buffer_fills_without_decoder(self):
        """Without a decoder, async mode should fill buffer but produce no predictions."""
        mode, main_q, _ = _make_async_mode()
        total = mode.epoch_length_samples + mode.samples_per_prediction_step * 3

        for i in range(total):
            mode._process_data(_make_sample(timestamp=i / 250.0))

        assert mode.prediction_count == 0  # No decoder → no predictions

    def test_with_mock_decoder_produces_predictions(self):
        """With a fitted decoder, async mode should produce predictions at step intervals."""
        mode, main_q, pred_q = _make_async_mode(n_channels=4, sample_rate=250.0)

        # Inject a mock decoder
        decoder = MagicMock()
        decoder.is_fitted = True
        decoder.predict_sample.return_value = (0, 0.85)  # (class_idx, confidence)
        mode.decoder = decoder

        total = mode.epoch_length_samples + mode.samples_per_prediction_step * 5

        for i in range(total):
            mode._process_data(_make_sample(n_channels=4, timestamp=i / 250.0))
            # Replicate _run_main_loop step check (not called by _process_data)
            if mode.buffer.is_ready_for_step(mode.samples_per_prediction_step):
                mode._trigger_prediction()

        assert mode.prediction_count >= 1
        assert decoder.predict_sample.call_count >= 1

        # Predictions go to both main_q (via FanOutQueue) and pred_q
        all_outputs = _drain(main_q) + _drain(pred_q)
        preds = [o for o in all_outputs if isinstance(o, dict) and o.get("type") == "prediction"]
        assert len(preds) >= 1
        assert preds[0]["data"]["confidence"] == 0.85
