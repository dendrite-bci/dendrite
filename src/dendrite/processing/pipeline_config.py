"""
Pipeline Configuration

Typed configuration dataclass for the processing pipeline, formalizing the
contract between MainWindow and run_pipeline().
"""

import multiprocessing
from dataclasses import dataclass, field
from typing import Any

from dendrite.constants import DEFAULT_RECORDING_NAME, DEFAULT_STUDY_NAME
from dendrite.data.stream_schemas import StreamMetadata
from dendrite.processing.queue_utils import FanOutQueue


@dataclass
class PipelineConfig:
    """Typed configuration for run_pipeline().

    Replaces the untyped dict that was passed between MainWindow and the pipeline,
    catching missing keys at construction time and self-documenting the pipeline contract.
    """

    # Core processing
    sample_rate: float
    mode_instances: dict[str, dict[str, Any]]
    file_identifier: str

    # Inter-process communication
    stop_event: multiprocessing.Event
    data_queue: multiprocessing.Queue
    save_queue: multiprocessing.Queue
    plot_queue: multiprocessing.Queue
    prediction_queue: multiprocessing.Queue
    mode_output_queues: dict[str, FanOutQueue]

    # Preprocessing
    preprocess_data: bool = False
    modality_preprocessing: dict[str, Any] = field(default_factory=dict)
    quality_control: dict[str, Any] = field(default_factory=dict)

    # Stream configuration
    stream_configs: list[StreamMetadata] = field(default_factory=list)

    # BIDS metadata
    study_name: str = DEFAULT_STUDY_NAME
    recording_name: str = DEFAULT_RECORDING_NAME
    subject_id: str = ""
    session_id: str = ""
    run_number: int = 1
    experiment_description: str = ""

    # Optional components
    pid_queue: "multiprocessing.Queue | None" = None
    shared_state: Any = None
    modality_data: dict[str, Any] = field(default_factory=dict)

    # Output configuration (passed through for streamer setup)
    output: dict[str, Any] = field(default_factory=dict)

    # Cross-mode sharing
    sync_to_async_sharing: dict[str, list[str]] = field(default_factory=dict)
