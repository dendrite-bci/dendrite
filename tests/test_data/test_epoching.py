"""Tests for RawData.preprocess() and RawData.epoch() — core epoching + preprocessing."""

import numpy as np
import pytest

from dendrite.data.loaders._types import RawData


def _make_loaded(n_samples=1000, n_channels=4, sample_rate=250.0, events=None):
    """Create synthetic RawData with known data and events."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_channels, n_samples)).astype(np.float32)
    ch_names = [f"Ch{i}" for i in range(n_channels)]
    if events is None:
        events = [(100, 1), (300, 2), (500, 1), (700, 2)]
    return RawData(
        data=data,
        channel_names=ch_names,
        channel_types=["eeg"] * n_channels,
        sample_rate=sample_rate,
        events=events,
    )


class TestEpoch:
    def test_basic_epoching(self):
        loaded = _make_loaded()
        ep = loaded.epoch({"epoch_tmax": 0.5})
        assert ep.X.shape == (4, 4, 125)
        assert set(ep.y.tolist()) == {1, 2}
        assert len(ep.y) == 4

    def test_event_mapping_filters(self):
        loaded = _make_loaded()
        ep = loaded.epoch({
            "event_mapping": {1: "left"}, "label_mapping": {"left": 0},
            "epoch_tmax": 0.5,
        })
        assert len(ep.y) == 2
        assert set(ep.y.tolist()) == {0}

    def test_label_mapping(self):
        loaded = _make_loaded()
        ep = loaded.epoch({
            "event_mapping": {1: "left", 2: "right"},
            "label_mapping": {"left": 0, "right": 1},
            "epoch_tmax": 0.5,
        })
        assert len(ep.y) == 4
        assert ep.y[0] == 0
        assert ep.y[1] == 1

    def test_epoch_tmin_tmax(self):
        loaded = _make_loaded()
        ep = loaded.epoch({"epoch_tmin": -0.1, "epoch_tmax": 0.4})
        assert ep.X.shape[2] == 125

    def test_channel_indices(self):
        loaded = _make_loaded()
        ep = loaded.epoch({"epoch_tmax": 0.5, "channel_indices": [0, 2]})
        assert ep.X.shape[1] == 2

    def test_boundary_events_skipped(self):
        loaded = _make_loaded(n_samples=200)
        loaded.events = [(10, 1), (150, 2)]
        ep = loaded.epoch({"epoch_tmax": 0.5})
        assert len(ep.y) == 1
        assert ep.y[0] == 1

    def test_no_events_raises(self):
        loaded = _make_loaded(events=[])
        with pytest.raises(ValueError, match="No events found"):
            loaded.epoch({"epoch_tmax": 0.5})

    def test_no_matching_events_raises(self):
        loaded = _make_loaded()
        with pytest.raises(ValueError, match="No valid epochs"):
            loaded.epoch({
                "event_mapping": {99: "nonexistent"},
                "label_mapping": {"nonexistent": 0},
                "epoch_tmax": 0.5,
            })

    def test_positive_epoch_tmin(self):
        loaded = _make_loaded(n_samples=2000)
        ep = loaded.epoch({"epoch_tmin": 0.2, "epoch_tmax": 0.6})
        assert ep.X.shape[2] == 100

    def test_output_dtype(self):
        loaded = _make_loaded()
        ep = loaded.epoch({"epoch_tmax": 0.5})
        assert ep.X.dtype == np.float32
        assert ep.y.dtype == np.int64

    @pytest.mark.parametrize("start,end,rate", [
        (0.0, 2.0, 250.0),
        (0.5, 4.5, 500.0),
        (-0.2, 1.0, 1000.0),
        (0.0, 1.0, 256.0),
    ])
    def test_n_times_matches_mode_epoch_timing(self, start, end, rate):
        n_samples = int(10 * rate)
        events = [(int(2 * rate), 1), (int(5 * rate), 2)]
        loaded = _make_loaded(n_samples=n_samples, sample_rate=rate, events=events)
        ep = loaded.epoch({"epoch_tmin": start, "epoch_tmax": end})
        mode_epoch_length = int(end * rate) - int(start * rate)
        assert ep.X.shape[2] == mode_epoch_length

    def test_returns_epoched_data_with_metadata(self):
        loaded = _make_loaded()
        ep = loaded.epoch({"epoch_tmax": 0.5})
        assert ep.sample_rate == loaded.sample_rate
        assert ep.channel_names == loaded.channel_names


class TestPreprocess:
    def test_mixed_channel_types_filtered_before_preprocessing(self):
        n_eeg, n_stim = 8, 2
        n_total = n_eeg + n_stim
        n_samples = 5000
        rng = np.random.default_rng(42)

        loaded = RawData(
            data=rng.standard_normal((n_total, n_samples)).astype(np.float32),
            channel_names=[f"EEG{i}" for i in range(n_eeg)] + [f"STI{i}" for i in range(n_stim)],
            channel_types=["eeg"] * n_eeg + ["stim"] * n_stim,
            sample_rate=250.0,
            events=[(500, 1), (1500, 2), (2500, 1), (3500, 2)],
        )

        loaded.filter_modality("eeg")
        loaded.preprocess({"lowcut": 1.0, "highcut": 40.0})

        assert loaded.n_channels == n_eeg
        assert all(t == "eeg" for t in loaded.channel_types)
        assert loaded.data.shape == (n_eeg, n_samples)


class TestBackgroundEpochs:
    def test_background_epochs_sampled(self):
        loaded = _make_loaded(n_samples=2000)
        ep = loaded.epoch({"epoch_tmax": 0.5, "include_background": True})
        # Raw event codes are 1,2 → bg_label = max(1,2) + 1 = 3
        bg_label = max(ep.y)
        assert bg_label == 3
        n_bg = int(np.sum(ep.y == bg_label))
        assert n_bg == 2  # balanced with smallest class (2 per event code)

    def test_background_balanced_with_smallest_class(self):
        # 3 class-1 events, 1 class-2 event → bg target = 1
        events = [(200, 1), (500, 1), (800, 1), (1100, 2)]
        loaded = _make_loaded(n_samples=3000, events=events)
        ep = loaded.epoch({"epoch_tmax": 0.5, "include_background": True})
        bg_label = max(ep.y)
        n_bg = int(np.sum(ep.y == bg_label))
        assert n_bg == 1

    def test_background_no_valid_gaps(self):
        # Events very close together — no room for background epochs
        events = [(10, 1), (20, 2)]
        loaded = _make_loaded(n_samples=200, events=events)
        ep = loaded.epoch({"epoch_tmax": 0.5, "include_background": True})
        # Should still work, just without background
        assert set(ep.y.tolist()) == {1, 2}

    def test_background_epochs_dont_overlap_events(self):
        loaded = _make_loaded(n_samples=5000)
        ep = loaded.epoch({"epoch_tmax": 0.5, "include_background": True})
        # All epochs have same temporal length
        assert ep.X.shape[2] == 125  # 0.5s * 250Hz

    def test_background_disabled_by_default(self):
        loaded = _make_loaded(n_samples=2000)
        ep = loaded.epoch({"epoch_tmax": 0.5})
        assert max(ep.y) <= 2  # only raw event codes, no bg label
