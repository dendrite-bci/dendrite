"""Direct mode processing tests — feed synthetic data, verify output.

Tests the core processing pipeline of each mode without ring buffers or
multiprocessing. Creates mode instances via __new__ (bypassing Process.__init__),
configures them minimally, and feeds samples through _process_data().
"""

import logging
import multiprocessing
import queue
from unittest.mock import MagicMock

import numpy as np
import pytest

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

    # IAF disabled by default; tests that need it construct an IAFCalibrator.
    mode.iaf = None
    mode.iaf_baseline_sec = 5.0
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

    def test_resolve_selected_labels_maps_indices_to_labels(self):
        """channel_selection indices should pick the matching labels from the full list."""
        from dendrite.processing.modes.neurofeedback_mode import NeurofeedbackMode

        full = [f"E{i}" for i in range(60)]
        full[7], full[23], full[24] = "C3", "Cz", "C4"

        obj = MagicMock()
        obj.modality_name = "eeg"
        obj.modality_labels = {"eeg": full}
        obj.channel_selection = {"eeg": [7, 23, 24]}

        result = NeurofeedbackMode._resolve_selected_labels(obj)
        assert result == ["C3", "Cz", "C4"]

    def test_resolve_selected_labels_no_selection_returns_full(self):
        """Empty channel_selection should fall back to full modality labels."""
        from dendrite.processing.modes.neurofeedback_mode import NeurofeedbackMode

        obj = MagicMock()
        obj.modality_name = "eeg"
        obj.modality_labels = {"eeg": ["C3", "Cz", "C4"]}
        obj.channel_selection = {}

        result = NeurofeedbackMode._resolve_selected_labels(obj)
        assert result == ["C3", "Cz", "C4"]

    def test_output_keys_match_selected_labels_in_non_contiguous_stream(self):
        """End-to-end: output channel_powers keys must equal the selected EEG labels.

        Regression: a stream where EEG channels are not stream-contiguous
        (e.g., interleaved with EOG) used to mismatch labels and data because
        channel_selection and channel_labels lived in different coordinate
        spaces. With per-modality local_index, both are modality-relative and
        the resolution lines up.
        """
        # 60-channel EEG modality with C3/Cz/C4 at modality-relative positions 5/21/22
        full_labels = [f"E{i}" for i in range(60)]
        full_labels[5], full_labels[21], full_labels[22] = "C3", "Cz", "C4"

        n_selected = 3
        mode, _, pred_q = _make_nfb_mode(n_channels=n_selected, sample_rate=250.0)

        # Apply the production resolution path
        mode.modality_labels = {"eeg": full_labels}
        mode.channel_selection = {"eeg": [5, 21, 22]}
        mode.channel_labels = mode._resolve_selected_labels()

        _feed_nfb(mode, n_selected, 250.0)

        outputs = _drain(pred_q)
        nfb = [o for o in outputs if isinstance(o, dict) and o.get("type") == "neurofeedback"]
        assert len(nfb) >= 1
        assert set(nfb[0]["data"]["channel_powers"].keys()) == {"C3", "Cz", "C4"}

    def test_resolve_selected_labels_empty_full_returns_empty(self):
        """No labels available means caller will fall back to ch{i} naming."""
        from dendrite.processing.modes.neurofeedback_mode import NeurofeedbackMode

        obj = MagicMock()
        obj.modality_name = "eeg"
        obj.modality_labels = {}
        obj.channel_selection = {"eeg": [0, 1, 2]}

        result = NeurofeedbackMode._resolve_selected_labels(obj)
        assert result == []


# ---------------------------------------------------------------------------
# IAF Detection (pure functions + integration)
# ---------------------------------------------------------------------------


class TestIAFDetection:
    """Test IAF computation and band shifting."""

    def test_compute_iaf_known_peak(self):
        """PAF on a clean 10 Hz sinusoid should return ~10 Hz with finite CoG diagnostic."""
        from dendrite.ml.features.iaf import compute_iaf

        fs = 250.0
        n_channels, n_samples = 4, int(5 * fs)
        t = np.arange(n_samples) / fs
        data = np.sin(2 * np.pi * 10 * t)[np.newaxis, :].repeat(n_channels, axis=0)
        data += np.random.default_rng(0).standard_normal(data.shape) * 0.1

        est = compute_iaf(data, fs, (7.0, 14.0))
        assert 9.5 < est.paf_hz < 10.5, f"Expected ~10 Hz, got {est.paf_hz:.2f}"
        assert np.isfinite(est.cog_hz)

    def test_compute_iaf_shifted_peak(self):
        """PAF on a clean 9.5 Hz sinusoid should return ~9.5 Hz."""
        from dendrite.ml.features.iaf import compute_iaf

        fs = 250.0
        n_channels, n_samples = 2, int(5 * fs)
        t = np.arange(n_samples) / fs
        data = np.sin(2 * np.pi * 9.5 * t)[np.newaxis, :].repeat(n_channels, axis=0)

        est = compute_iaf(data, fs, (7.0, 14.0))
        assert 9.0 < est.paf_hz < 10.0, f"Expected ~9.5 Hz, got {est.paf_hz:.2f}"
        assert np.isfinite(est.cog_hz)

    def test_compute_iaf_emits_both_on_success(self):
        """Philistine's success branch always emits both PAF and CoG together."""
        from dendrite.ml.features.iaf import compute_iaf

        fs = 250.0
        n_channels, n_samples = 4, int(5 * fs)
        t = np.arange(n_samples) / fs
        data = np.sin(2 * np.pi * 10 * t)[np.newaxis, :].repeat(n_channels, axis=0)
        data += np.random.default_rng(1).standard_normal(data.shape) * 0.1

        est = compute_iaf(data, fs, (7.0, 14.0))
        assert np.isfinite(est.paf_hz)
        assert np.isfinite(est.cog_hz)
        # On a clean peak the two estimators should agree within ~1 Hz.
        assert abs(est.paf_hz - est.cog_hz) < 1.0, (
            f"PAF={est.paf_hz:.2f} and CoG={est.cog_hz:.2f} diverge unexpectedly"
        )

    def test_shift_bands_alpha_only(self):
        """Only the band named "alpha" should be shifted; others stay canonical."""
        from dendrite.ml.features.iaf import shift_bands

        bands = {"alpha": [8.0, 12.0], "beta": [15.0, 30.0]}
        shifted = shift_bands(bands, iaf=9.5)
        assert shifted["alpha"] == [7.5, 11.5], f"Got {shifted['alpha']}"
        assert shifted["beta"] == [15.0, 30.0], "Beta should not shift"

    def test_shift_bands_smr_and_theta_unchanged(self):
        """SMR and theta have separate generators — must not track IAF."""
        from dendrite.ml.features.iaf import shift_bands

        bands = {"smr": [12.0, 15.0], "theta": [4.0, 7.0]}
        shifted = shift_bands(bands, iaf=11.5)
        assert shifted["smr"] == [12.0, 15.0], "SMR is mu-rhythm-anchored, not alpha"
        assert shifted["theta"] == [4.0, 7.0], "Theta should not shift"

    def test_shift_bands_non_alpha_names_unchanged(self):
        """Only the literal name 'alpha' is recognized — sub-band names don't shift."""
        from dendrite.ml.features.iaf import shift_bands

        bands = {"alpha1": [8.0, 10.0], "low_alpha": [6.0, 9.0]}
        shifted = shift_bands(bands, iaf=9.0)
        assert shifted["alpha1"] == [8.0, 10.0]
        assert shifted["low_alpha"] == [6.0, 9.0]

    def test_shift_bands_clamp_zero(self):
        """Shifted alpha band should not go below 0 Hz."""
        from dendrite.ml.features.iaf import shift_bands

        bands = {"alpha": [3.0, 7.0]}
        shifted = shift_bands(bands, iaf=6.0)
        assert shifted["alpha"][0] >= 0, "Clamped to 0"
        assert shifted["alpha"] == [0.0, 3.0]

    def test_iaf_integration(self):
        """Full integration: trigger event → collect baseline → verify IAF."""
        from dendrite.ml.features.iaf import IAFCalibrator

        iaf_event_id = 99
        baseline_sec = 2.0  # Corcoran needs enough samples for Sav-Gol PSD smoothing
        fs = 250.0
        n_channels = 2
        target_freq = 9.0

        mode, main_q, _ = _make_nfb_mode(n_channels=n_channels, sample_rate=fs)
        mode.iaf = IAFCalibrator(
            event_id=iaf_event_id,
            baseline_samples=int(baseline_sec * fs),
            iaf_range=(7.0, 14.0),
        )
        original_bands_before = {k: list(v) for k, v in mode.target_bands.items()}

        rng = np.random.default_rng(42)

        total_fill = mode.window_length_samples + mode.window_step_samples
        for i in range(total_fill):
            sample = _make_sample(n_channels=n_channels, modality="eeg", timestamp=i / fs)
            sample["eeg"] = rng.standard_normal((n_channels, 1)).astype(np.float32)
            processed = mode._preprocess_sample(sample)
            if processed is None:
                continue
            mode.last_lsl_timestamp = processed.get("lsl_timestamp", 0.0)
            mode.buffer.add_sample(processed)

        trigger = _make_sample(n_channels=n_channels, modality="eeg", timestamp=total_fill / fs)
        trigger["markers"] = np.array([[iaf_event_id]])
        trigger["eeg"] = rng.standard_normal((n_channels, 1)).astype(np.float32)
        processed = mode._preprocess_sample(trigger)
        mode.last_lsl_timestamp = processed.get("lsl_timestamp", 0.0)
        if extract_event_code(processed) == mode.iaf.event_id:
            data = processed.get(mode.modality_name)
            mode.iaf.trigger(data.shape[0])
        mode.buffer.add_sample(processed)

        assert mode.iaf.state == "collecting"

        baseline_samples = mode.iaf.baseline_samples
        for i in range(baseline_samples):
            t = (total_fill + 1 + i) / fs
            sample = _make_sample(n_channels=n_channels, modality="eeg", timestamp=t)
            tone = np.sin(2 * np.pi * target_freq * t).astype(np.float32)
            sample["eeg"] = np.full((n_channels, 1), tone, dtype=np.float32)
            processed = mode._preprocess_sample(sample)
            if processed is None:
                continue
            mode.last_lsl_timestamp = processed.get("lsl_timestamp", 0.0)
            data = processed.get(mode.modality_name)
            if data is not None and mode.iaf.accumulate(data):
                mode._on_iaf_complete()
            mode.buffer.add_sample(processed)

        assert mode.iaf.state == "done"
        assert mode.target_bands != original_bands_before

        outputs = _drain(main_q)
        iaf_outputs = [o for o in outputs if isinstance(o, dict) and o.get("type") == "iaf_result"]
        assert len(iaf_outputs) == 1
        iaf_hz = iaf_outputs[0]["data"]["iaf_hz"]
        assert 8.0 < iaf_hz < 10.5, f"Expected ~{target_freq}, got {iaf_hz:.2f}"
        assert np.isfinite(iaf_outputs[0]["data"]["cog_hz"])

    def test_compute_iaf_raises_on_no_peak(self):
        """Pink-ish noise trips philistine's pink-noise R² guard → compute_iaf raises."""
        from dendrite.ml.features.iaf import compute_iaf

        fs = 250.0
        rng = np.random.default_rng(0)
        # Pink-ish noise via cumsum of white noise; no embedded peak.
        data = np.cumsum(rng.standard_normal((4, int(8 * fs))), axis=1).astype(np.float32)
        with pytest.raises(RuntimeError, match=r"alpha|savgol_iaf failed"):
            compute_iaf(data, fs, (7.0, 14.0))

    def test_on_iaf_complete_skips_on_failure(self):
        """When the calibrator's finalize() returns None, the mode skips cleanly."""
        from dendrite.ml.features.iaf import IAFCalibrator

        fs = 250.0
        n_channels = 2
        mode, main_q, _ = _make_nfb_mode(n_channels=n_channels, sample_rate=fs)
        original_bands = {k: list(v) for k, v in mode.target_bands.items()}

        mode.iaf = IAFCalibrator(
            event_id=99,
            baseline_samples=int(2.0 * fs),
            iaf_range=(7.0, 14.0),
        )
        mode.iaf.trigger(n_channels)
        # Fill with zeros → no peak detectable, finalize() returns None.
        mode.iaf.accumulate(
            np.zeros((n_channels, mode.iaf.baseline_samples), dtype=np.float32)
        )

        mode._on_iaf_complete()

        assert mode.iaf.state == "done", "Must mark done so the mode doesn't loop"
        assert mode.target_bands == original_bands, "Bands must be untouched on failure"
        outputs = _drain(main_q)
        iaf_outputs = [o for o in outputs if isinstance(o, dict) and o.get("type") == "iaf_result"]
        assert iaf_outputs == [], "No IAFPayload should be sent on failure"

    def test_iaf_disabled_ignores_markers(self):
        """Without an IAFCalibrator, markers are ignored."""
        mode, main_q, _ = _make_nfb_mode()
        assert mode.iaf is None

        _feed_nfb(mode, 4, 250.0)
        assert mode.iaf is None
        outputs = _drain(main_q)
        iaf_outputs = [o for o in outputs if isinstance(o, dict) and o.get("type") == "iaf_result"]
        assert iaf_outputs == []


# ---------------------------------------------------------------------------
# IAFCalibrator (state machine + finalize, unit-tested without a mode)
# ---------------------------------------------------------------------------


class TestIAFCalibrator:
    """Direct tests of the IAFCalibrator state machine."""

    def _calibrator(self, baseline_samples=500, iaf_range=(7.0, 14.0)):
        from dendrite.ml.features.iaf import IAFCalibrator
        return IAFCalibrator(
            event_id=99, baseline_samples=baseline_samples, iaf_range=iaf_range
        )

    def test_trigger_from_idle(self):
        c = self._calibrator()
        assert c.state == "idle"
        assert c.trigger(n_channels=4) is True
        assert c.state == "collecting"

    def test_reject_re_trigger_from_done(self):
        c = self._calibrator(baseline_samples=10)
        c.trigger(n_channels=2)
        c.accumulate(np.zeros((2, 10), dtype=np.float32))
        c.finalize(250.0, {"alpha": [8.0, 12.0]})
        assert c.state == "done"
        assert c.trigger(n_channels=2) is False
        assert c.state == "done"

    def test_accumulate_partial_then_full(self):
        c = self._calibrator(baseline_samples=10)
        c.trigger(n_channels=2)
        assert c.accumulate(np.zeros((2, 3), dtype=np.float32)) is False
        assert c.accumulate(np.zeros((2, 3), dtype=np.float32)) is False
        assert c.accumulate(np.zeros((2, 3), dtype=np.float32)) is False
        assert c.accumulate(np.zeros((2, 1), dtype=np.float32)) is True

    def test_finalize_returns_payload(self):
        fs = 250.0
        n_samples = int(5 * fs)
        n_channels = 2
        c = self._calibrator(baseline_samples=n_samples)
        c.trigger(n_channels)
        t = np.arange(n_samples) / fs
        tone = np.sin(2 * np.pi * 10 * t).astype(np.float32)
        c.accumulate(np.tile(tone, (n_channels, 1)))
        result = c.finalize(fs, {"alpha": [8.0, 12.0]})
        assert result is not None
        assert 9.5 < result.iaf_hz < 10.5

    def test_finalize_returns_none_on_no_peak(self):
        fs = 250.0
        n_samples = int(5 * fs)
        c = self._calibrator(baseline_samples=n_samples)
        c.trigger(n_channels=4)
        c.accumulate(np.zeros((4, n_samples), dtype=np.float32))
        result = c.finalize(fs, {"alpha": [8.0, 12.0]})
        assert result is None
        assert c.state == "done"

    def test_accumulate_multi_sample_chunks(self):
        """Multi-column chunks must be fully accumulated.

        Regression: accumulate previously copied only data[:, 0] per call,
        dropping k-1 of every k samples when feeding width-k chunks.
        """
        fs = 250.0
        n_channels = 2
        target_freq = 10.0
        chunk_width = 4
        n_samples = int(1.0 * fs)

        c = self._calibrator(baseline_samples=n_samples)
        c.trigger(n_channels)
        t = np.arange(n_samples) / fs
        tone = np.sin(2 * np.pi * target_freq * t).astype(np.float32)
        full = np.tile(tone, (n_channels, 1))

        pos = 0
        filled = False
        while pos < n_samples:
            end = pos + chunk_width
            if end > n_samples:
                # Pad past the tone end — must be clamped by accumulate().
                chunk = np.zeros((n_channels, chunk_width), dtype=np.float32)
                chunk[:, : n_samples - pos] = full[:, pos:n_samples]
            else:
                chunk = full[:, pos:end]
            filled = c.accumulate(chunk) or filled
            pos = end

        assert filled
        result = c.finalize(fs, {"alpha": [8.0, 12.0]})
        assert result is not None
        assert 9.5 < result.iaf_hz < 10.5, (
            f"Expected ~{target_freq} Hz, got {result.iaf_hz:.2f}"
        )


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
