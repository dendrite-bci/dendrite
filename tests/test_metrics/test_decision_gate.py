"""Tests for DecisionGate."""

from dendrite.ml.decision_gate import DecisionGate


class TestDecisionGate:
    def test_filter_prediction_passes_above_threshold(self):
        gate = DecisionGate(confidence_threshold=0.5)
        assert gate.filter_prediction(1, 0.8) == 1

    def test_filter_prediction_abstains_below_threshold(self):
        gate = DecisionGate(confidence_threshold=0.5)
        assert gate.filter_prediction(1, 0.3) == -1

    def test_filter_prediction_no_threshold(self):
        gate = DecisionGate(confidence_threshold=0.0)
        assert gate.filter_prediction(1, 0.01) == 1

    def test_filter_predictions_batch(self):
        gate = DecisionGate(confidence_threshold=0.5)
        result = gate.filter_predictions([0, 1, 2], [0.8, 0.3, 0.6])
        assert result == [0, -1, 2]

    def test_filter_predictions_no_threshold(self):
        gate = DecisionGate(confidence_threshold=0.0)
        result = gate.filter_predictions([0, 1, 2], [0.01, 0.01, 0.01])
        assert result == [0, 1, 2]

    def test_round_trip_dict(self):
        gate = DecisionGate(strategy="majority", dwell_n=5, confidence_threshold=0.7)
        rebuilt = DecisionGate.from_dict(gate.to_dict())
        assert rebuilt == gate

    def test_from_config(self):
        gate = DecisionGate.from_config({
            "detection_strategy": "dwell",
            "dwell_n": 5,
            "confidence_threshold": 0.8,
        })
        assert gate.strategy == "dwell"
        assert gate.dwell_n == 5
        assert gate.confidence_threshold == 0.8

    def test_from_config_strategy_key(self):
        gate = DecisionGate.from_config({
            "strategy": "majority",
            "confidence_threshold": 0.6,
        })
        assert gate.strategy == "majority"
        assert gate.confidence_threshold == 0.6

    def test_from_config_defaults(self):
        gate = DecisionGate.from_config({})
        assert gate.strategy == "dwell"
        assert gate.dwell_n == 3
        assert gate.confidence_threshold == 0.0

    def test_use_dwell_property(self):
        assert DecisionGate(strategy="dwell").use_dwell is True
        assert DecisionGate(strategy="majority").use_dwell is False
