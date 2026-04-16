"""Shared memory SPMC ring buffer — replaces queue-based IPC.

DAQ writes raw samples + markers to ring buffer. All consumers read directly. Zero pickle.
"""

import logging
import struct
from multiprocessing.shared_memory import SharedMemory

import numpy as np

logger = logging.getLogger(__name__)

# Header: [0:8] write_pos u64, [8:12] n_ch u32, [12:16] max_s u32, [16:24] rate f64
_HDR = struct.Struct("<QIId")
_HDR_SZ = 64


class SharedRingBuffer:
    """SPMC ring buffer: [header][data float32][lsl_ts float64][local_ts float64][receive_ns uint64].

    Last column of data is the markers channel (injected by DAQ).
    """

    __slots__ = (
        "_buf",
        "_data",
        "_local_ts",
        "_receive_ns",
        "_shm",
        "_ts",
        "max_samples",
        "n_channels",
        "name",
        "sample_rate",
    )

    def __init__(self, name: str, n_channels: int, max_samples: int,
                 sample_rate: float, shm: SharedMemory):
        self.name = name
        self.n_channels = n_channels
        self.max_samples = max_samples
        self.sample_rate = sample_rate
        self._shm = shm
        buf = shm.buf
        assert buf is not None, "SharedMemory buffer closed"
        self._buf: memoryview = buf
        d0 = _HDR_SZ
        t0 = d0 + max_samples * n_channels * 4
        l0 = t0 + max_samples * 8
        n0 = l0 + max_samples * 8
        self._data = np.ndarray((max_samples, n_channels), np.float32,
                                buffer=buf[d0:t0])
        self._ts = np.ndarray((max_samples,), np.float64,
                              buffer=buf[t0:l0])
        self._local_ts = np.ndarray((max_samples,), np.float64,
                                    buffer=buf[l0:n0])
        self._receive_ns = np.ndarray((max_samples,), np.uint64,
                                  buffer=buf[n0:n0 + max_samples * 8])

    @classmethod
    def create(cls, name: str, n_channels: int, max_samples: int,
               sample_rate: float) -> "SharedRingBuffer":
        total = (_HDR_SZ
                 + max_samples * n_channels * 4  # data float32
                 + max_samples * 8               # lsl_ts float64
                 + max_samples * 8               # local_ts float64
                 + max_samples * 8)              # daq_ns uint64
        _try_unlink(name)
        shm = SharedMemory(name=name, create=True, size=total)
        rb = cls(name, n_channels, max_samples, sample_rate, shm)
        _HDR.pack_into(rb._buf, 0, 0, n_channels, max_samples, sample_rate)
        return rb

    @classmethod
    def connect(cls, name: str) -> "SharedRingBuffer":
        shm = SharedMemory(name=name, create=False)
        buf = shm.buf
        assert buf is not None, "SharedMemory buffer closed"
        _, n_ch, max_s, sr = _HDR.unpack_from(buf, 0)
        return cls(name, n_ch, max_s, sr, shm)

    @property
    def write_pos(self) -> int:
        return struct.unpack_from("<Q", self._buf, 0)[0]

    @write_pos.setter
    def write_pos(self, value: int):
        struct.pack_into("<Q", self._buf, 0, value)

    def write(self, sample: np.ndarray, lsl_timestamp: float,
              local_timestamp: float = 0.0, receive_ns: int = 0) -> int:
        pos = self.write_pos
        idx = pos % self.max_samples
        self._data[idx] = sample.ravel()[:self.n_channels]
        self._ts[idx] = lsl_timestamp
        self._local_ts[idx] = local_timestamp
        self._receive_ns[idx] = receive_ns
        pos += 1
        self.write_pos = pos
        return pos

    def read_new(self, last_read_pos: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """Returns (data, lsl_timestamps, local_timestamps, daq_ns, new_pos)."""
        wp = self.write_pos
        if last_read_pos >= wp:
            return _EMPTY_2D, _EMPTY_F64, _EMPTY_F64, _EMPTY_U64, last_read_pos
        n = wp - last_read_pos
        if n > self.max_samples:
            raise OverrunError(f"'{self.name}': missed {n - self.max_samples} samples")
        s = last_read_pos % self.max_samples
        e = wp % self.max_samples
        if e == 0:  # wp lands on boundary — slice to end ([:0] would be empty)
            d = self._data[s:].copy()
            t = self._ts[s:].copy()
            lt = self._local_ts[s:].copy()
            ns = self._receive_ns[s:].copy()
        elif s < e:
            d = self._data[s:e].copy()
            t = self._ts[s:e].copy()
            lt = self._local_ts[s:e].copy()
            ns = self._receive_ns[s:e].copy()
        else:
            d = np.concatenate([self._data[s:], self._data[:e]])
            t = np.concatenate([self._ts[s:], self._ts[:e]])
            lt = np.concatenate([self._local_ts[s:], self._local_ts[:e]])
            ns = np.concatenate([self._receive_ns[s:], self._receive_ns[:e]])
        return d, t, lt, ns, wp

    @property
    def is_valid(self) -> bool:
        """Check if shared memory is still accessible."""
        try:
            if self._shm is None or self._shm.buf is None:
                return False
            struct.unpack_from("<Q", self._shm.buf, 0)
            return True
        except Exception:
            return False

    def close(self):
        try:
            self._shm.close()
        except Exception as e:
            logger.debug("shm close '%s': %s", self.name, e)

    def unlink(self):
        try:
            self._shm.unlink()
        except Exception as e:
            logger.debug("shm unlink '%s': %s", self.name, e)

    def __repr__(self) -> str:
        return f"SharedRingBuffer('{self.name}', {self.n_channels}ch, pos={self.write_pos})"


class OverrunError(Exception):
    """Reader too slow — writer overwrote unread data."""


_EMPTY_2D = np.empty((0, 0), np.float32)
_EMPTY_F64 = np.empty(0, np.float64)
_EMPTY_U64 = np.empty(0, np.uint64)

DEFAULT_BUFFER_DURATION_S = 30


def compute_max_samples(sample_rate: float,
                        duration_s: float = DEFAULT_BUFFER_DURATION_S) -> int:
    return int(sample_rate * duration_s)


def _try_unlink(name: str):
    try:
        old = SharedMemory(name=name, create=False)
        old.close()
        old.unlink()
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(f"Failed to clean up shared memory '{name}': {e}")
