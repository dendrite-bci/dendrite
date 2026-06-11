"""Filter-priming continuity in ModalityProcessor.

Priming initialises the causal filters to steady-state on the first chunk. The
``_primed`` flag must flip once the filters have actually been primed on a non-empty
input — NOT based on the post-downsample output length, which can be empty for a
first chunk shorter than ``downsample_factor``. If it keyed off the output length, a
tiny first chunk would leave ``_primed`` False and the next chunk would re-prime,
clobbering the filter state advanced by the first chunk → a discontinuity.

Stateful filtering is chunk-invariant once primed identically (both paths prime on
the same first sample), so splitting the input must reproduce the single-call output.
"""

import numpy as np

from dendrite.processing.preprocessing.preprocessor import ModalityProcessor

SR = 256.0


def _proc() -> ModalityProcessor:
    return ModalityProcessor({
        "num_channels": 4, "sample_rate": SR,
        "lowcut": 1.0, "highcut": 40.0, "filter_order": 4,
        "downsample_factor": 4,
    })


def _signal() -> np.ndarray:
    rng = np.random.default_rng(0)
    n = int(8 * SR)
    t = np.arange(n) / SR
    # Large DC offset so a mishandled prime produces a visible startup transient.
    return rng.normal(0, 1.0, (4, n)) + 50.0 + np.sin(2 * np.pi * 5 * t)


def test_tiny_first_chunk_does_not_break_filter_continuity():
    data = _signal()
    ref = _proc().process_chunk(data.copy())

    # First chunk (3 samples) is shorter than downsample_factor=4, so it produces an
    # empty decimated output — the exact condition that used to leave _primed False.
    split = _proc()
    out1 = split.process_chunk(data[:, :3].copy())
    out2 = split.process_chunk(data[:, 3:].copy())
    combined = np.concatenate([out1, out2], axis=1)

    assert combined.shape == ref.shape
    assert np.allclose(combined, ref, atol=1e-9)


def test_primed_set_after_tiny_first_chunk():
    p = _proc()
    p.process_chunk(_signal()[:, :3].copy())  # input present, decimated output empty
    assert p._primed is True
