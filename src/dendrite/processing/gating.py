"""Standalone prediction gating for output smoothing.

Consumers can use this utility if they need gated predictions.
The mode outputs raw (prediction, confidence), and consumers apply gating as needed.
"""

from collections import deque


class PredictionGating:
    """Standalone prediction gating for output smoothing.

    Example:
        gating = PredictionGating(confidence_threshold=0.7, use_confidence=True)
        gated_pred = gating.apply(prediction=1, confidence=0.5)  # Returns background_class
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        dwell_window_size: int = 5,
        background_class: int = 0,
        use_confidence: bool = True,
        use_dwell: bool = False,
    ):
        """Initialize gating.

        Args:
            confidence_threshold: Minimum confidence to accept prediction
            dwell_window_size: Number of consistent predictions required for dwell gating
            background_class: Class to output when gating rejects prediction
            use_confidence: Enable confidence-based gating
            use_dwell: Enable dwell time (consistency) gating
        """
        self.confidence_threshold = confidence_threshold
        self.background_class = background_class
        self.use_confidence = use_confidence
        self.use_dwell = use_dwell
        self.dwell_buffer: deque[int] | None = (
            deque(maxlen=dwell_window_size) if use_dwell else None
        )

    def apply(self, prediction: int, confidence: float) -> int:
        """Apply gating to a prediction.

        Args:
            prediction: Raw prediction from model
            confidence: Prediction confidence (0.0-1.0)

        Returns:
            Gated prediction (either original or background_class)
        """
        if self.use_confidence and confidence < self.confidence_threshold:
            return self.background_class

        if self.use_dwell:
            self.dwell_buffer.append(prediction)
            if len(self.dwell_buffer) < self.dwell_buffer.maxlen:
                return self.background_class
            if len(set(self.dwell_buffer)) > 1:
                return self.background_class

        return prediction

    def reset(self) -> None:
        """Reset dwell buffer."""
        if self.dwell_buffer:
            self.dwell_buffer.clear()
