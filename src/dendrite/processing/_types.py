"""Sample type contract for the processing pipeline.

Annotation-only — no runtime validation.
"""

from typing import NotRequired, TypedDict

import numpy as np


class Sample(TypedDict):
    """Sample dict flowing from ring buffers through modes.

    Built by BaseMode._read_from_ring_buffer() from shared memory data.
    Contains dynamic modality keys ("eeg", "emg", etc.) as np.ndarray.
    """

    markers: np.ndarray  # type: ignore[assignment]  # (1, 1) event code
    lsl_timestamp: float
    _stream_name: str
    _receive_ns: NotRequired[int]
