"""Sample type contract for the processing pipeline.

Annotation-only — no runtime validation.
"""

from typing import Any

# Sample dict flowing from ring buffers through modes.
# Built by BaseMode._read_from_ring_buffer() from shared memory data.
# Well-known keys:
#   markers: np.ndarray — (1, 1) event code
#   lsl_timestamp: float
#   _stream_name: str
#   _receive_ns: int (optional)
# Plus dynamic modality keys ("eeg", "emg", etc.) as np.ndarray.
#
# A TypedDict can't express the dynamic keys (Python <3.13 has no
# `__extra_items__`), so we use dict[str, Any] and treat the well-known
# keys as documented convention.
Sample = dict[str, Any]
