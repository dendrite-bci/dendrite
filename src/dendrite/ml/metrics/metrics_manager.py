from typing import Any

from dendrite.utils.logger_central import get_logger

from .asynchronous_metrics import AsynchronousMetrics
from .synchronous_metrics import SynchronousMetrics


class MetricsManager:
    """
    Mode-aware metrics router for Dendrite classification.

    Routes metric calculations to appropriate backend based on mode type:
    - Synchronous mode: Uses SynchronousMetrics (trial-based)
    - Asynchronous mode: Uses AsynchronousMetrics (trial-level: TPR, TNR, FAR, TTD)
    """

    def __init__(
        self,
        mode_type: str = "synchronous",
        sample_rate: int = 250,
        num_classes: int = 2,
        detection_window_samples: int | None = None,
        label_mapping: dict[int, str] | None = None,
        background_class: int | None = None,
    ):
        """
        Initialize the metrics manager.

        Args:
            mode_type: 'synchronous' for trial-based, 'asynchronous' for continuous
            sample_rate: Sample rate in Hz
            num_classes: Number of classes for classification
            detection_window_samples: For async mode, detection window (= decoder epoch length)
            label_mapping: Optional mapping from class index to name (async mode only)
            background_class: Class index for background/idle state (async mode only).
                              Predictions of this class outside trial windows are not
                              counted as false alarms.
        """
        self.mode_type = mode_type.lower()
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.label_mapping = label_mapping
        self.background_class = background_class
        self.logger = get_logger()

        if self.mode_type == "synchronous":
            self.forgetting_factor = 0.95
            self.sync_metrics = SynchronousMetrics(num_classes=num_classes)
            self.async_metrics = None

        elif self.mode_type == "asynchronous":
            if detection_window_samples is None:
                detection_window_samples = sample_rate  # Default: 1 second

            self.async_metrics = AsynchronousMetrics(
                detection_window_samples=detection_window_samples,
                sample_rate=sample_rate,
                label_mapping=label_mapping,
                background_class=background_class,
            )
            self.sync_metrics = None
        else:
            raise ValueError(f"Invalid mode_type '{mode_type}'")

    def register_event(self, sample_idx: int, label: int) -> None:
        """Register an event onset for async mode.

        Args:
            sample_idx: Sample index of event onset
            label: Event class label (0 to num_classes-1)
        """
        if self.mode_type == "asynchronous" and self.async_metrics:
            self.async_metrics.register_event(sample_idx, label)

    def update_metrics(
        self,
        prediction: int,
        true_label: int,
        **mode_context,
    ) -> dict[str, Any]:
        """Update metrics based on a new prediction."""
        if self.mode_type == "synchronous":
            self.sync_metrics.add_prediction(
                prediction=prediction,
                true_label=true_label,
                forgetting_factor=self.forgetting_factor,
            )
            return self.get_current_metrics()

        else:  # asynchronous
            current_sample_idx = mode_context.get("current_sample_idx", 0)

            self.async_metrics.add_prediction(
                prediction=prediction,
                current_sample_idx=current_sample_idx,
            )

            return self.get_current_metrics()

    def get_current_metrics(self) -> dict[str, Any]:
        """Get current metrics."""
        if self.mode_type == "synchronous":
            return self.sync_metrics.get_all_metrics()
        else:
            return self.async_metrics.get_metrics()

    def reset(self):
        """Reset metrics."""
        if self.mode_type == "synchronous":
            self.sync_metrics.reset()
        else:
            self.async_metrics.reset()
