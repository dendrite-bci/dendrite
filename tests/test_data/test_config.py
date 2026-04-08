"""Tests for MoabbConfig."""

from dendrite.data.loaders.moabb_loader import MoabbConfig


class TestMoabbConfig:
    def test_defaults(self):
        cfg = MoabbConfig(dataset="BNCI2014_001", paradigm="MotorImagery")
        assert cfg.dataset == "BNCI2014_001"
        assert cfg.paradigm == "MotorImagery"
        assert cfg.channels == "eeg"
        assert cfg.events == {}
        assert cfg.n_classes is None

    def test_with_events(self):
        cfg = MoabbConfig(
            dataset="test", paradigm="P300",
            events={"Target": 1, "NonTarget": 0},
        )
        assert cfg.events == {"Target": 1, "NonTarget": 0}
