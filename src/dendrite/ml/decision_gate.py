"""Unified decision gating for BCI prediction evaluation.

A ``DecisionGate`` holds two orthogonal concerns:

1. **Confidence threshold** — pre-filter that replaces low-confidence
   predictions with ``-1`` (abstain).
2. **Strategy** — how to aggregate filtered predictions into a decision
   (``"dwell"`` = N-consecutive, ``"majority"`` = majority vote).

Abstained predictions break dwell streaks and are excluded from
majority votes.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass
class DecisionGate:
    """Strategy + optional confidence threshold for BCI decision logic."""

    strategy: str = "dwell"
    dwell_n: int = 3
    confidence_threshold: float = 0.0

    @property
    def use_dwell(self) -> bool:
        return self.strategy == "dwell"

    def decide(
        self,
        predictions: list[int],
        confidences: list[float] | None = None,
    ) -> tuple[int | None, int | None]:
        """Apply confidence filter + strategy → (vote, decision_step).

        Returns (decided_class, step_index).  ``step_index`` is the step
        where dwell triggered (None for majority or if no dwell fires).
        """
        preds = self.filter_predictions(predictions, confidences)

        if self.use_dwell:
            return _dwell_decide(preds, self.dwell_n)
        return _majority_vote(preds), None

    def filter_prediction(self, prediction: int, confidence: float) -> int:
        """Return *prediction* or ``-1`` (abstain) if below threshold."""
        if self.confidence_threshold > 0.0 and confidence < self.confidence_threshold:
            return -1
        return prediction

    def filter_predictions(
        self, predictions: list[int], confidences: list[float] | None,
    ) -> list[int]:
        """Batch filter. Returns new list with low-confidence entries as -1."""
        if not confidences or self.confidence_threshold <= 0.0:
            return list(predictions)
        thresh = self.confidence_threshold
        return [-1 if c < thresh else p for p, c in zip(predictions, confidences, strict=True)]

    # -- Serialisation -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "dwell_n": self.dwell_n,
            "confidence_threshold": self.confidence_threshold,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DecisionGate":
        return cls(
            strategy=d.get("strategy", "dwell"),
            dwell_n=d.get("dwell_n", 3),
            confidence_threshold=d.get("confidence_threshold", 0.0),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DecisionGate":
        """Construct from an evaluation config dict."""
        return cls(
            strategy=config.get("strategy", config.get("detection_strategy", "dwell")),
            dwell_n=int(config.get("dwell_n", 3)),
            confidence_threshold=float(config.get("confidence_threshold", 0.0)),
        )


# -- Standalone helpers (used by metrics_utils for FAR counting) -----------

def _dwell_decide(preds: list[int], dwell_n: int) -> tuple[int | None, int | None]:
    """First class to reach ``dwell_n`` consecutive predictions → (class, step)."""
    streak = 0
    current: int | None = None
    for i, p in enumerate(preds):
        if p < 0:
            streak, current = 0, None
            continue
        if p == current:
            streak += 1
        else:
            current, streak = p, 1
        if streak >= dwell_n:
            return current, i
    return None, None


def _majority_vote(predictions: list[int]) -> int | None:
    """Majority class excluding abstentions (-1). None if tied or empty."""
    filtered = [p for p in predictions if p >= 0]
    if not filtered:
        return None
    counts = Counter(filtered)
    ((winner, top_count),) = counts.most_common(1)
    if sum(1 for cnt in counts.values() if cnt == top_count) > 1:
        return None
    return winner


def dwell_detect_any(preds: list[int], dwell_n: int) -> int:
    """Count dwell streaks of any single class in background predictions.

    Each completed streak counts as one false detection; the counter resets
    after each detection so overlapping streaks are not double-counted.
    """
    if dwell_n < 1 or not preds:
        return 0
    detections = 0
    streak = 1
    for i in range(1, len(preds)):
        if preds[i] == preds[i - 1] and preds[i] >= 0:
            streak += 1
            if streak >= dwell_n:
                detections += 1
                streak = 0
        else:
            streak = 1
    return detections
