import json
import time

import multiprocessing as mp
import numpy as np
import pytest
from threading import Thread

from dendrite.data.acquisition import DataAcquisition
from dendrite.data.storage.data_saver import DataSaver
from dendrite.data.stream_schemas import StreamMetadata
from tests.conftest import cleanup_process


class FakeLiveInfo:
    """Lightweight stand-in for pylsl.StreamInfo used by inlet.info()."""

    def __init__(self, name, stream_type, nominal_srate, channel_count, source_id, uid):
        self._name = name
        self._type = stream_type
        self._srate = nominal_srate
        self._ch = channel_count
        self._source_id = source_id
        self._uid = uid

    def name(self):
        return self._name

    def type(self):
        return self._type

    def nominal_srate(self):
        return self._srate

    def channel_count(self):
        return self._ch

    def source_id(self):
        return self._source_id

    def uid(self):
        return self._uid

    def created_at(self):
        return time.time()

    def version(self):
        return 1.0


class FakeInlet:
    """Simulated LSL inlet that produces samples at a target rate without external deps."""

    def __init__(self, config: StreamMetadata):
        self.name = config.name
        self.stream_type = config.type
        self.rate = float(config.sample_rate)
        self.channels = int(config.channel_count)
        self.channel_format = config.channel_format
        self._last_ts = 0.0
        self._period = 0.0 if self.rate <= 0 else 1.0 / self.rate
        # Pre-create a deterministic sample for numeric streams to reduce CPU
        if self.stream_type == 'Events':
            # Events stream: JSON-encoded event dict
            event = {"event_id": 1, "event_type": "stress"}
            self._base_sample = [json.dumps(event)]
        elif self.channel_format == 'string':
            # Generic string stream
            self._base_sample = ["text"] * self.channels
        else:
            # Numeric stream (EEG, EMG, etc.)
            rng = np.random.default_rng(0)
            self._base_sample = rng.standard_normal(self.channels).astype(np.float32).tolist()

    def info(self):
        return FakeLiveInfo(
            name=self.name,
            stream_type=self.stream_type,
            nominal_srate=self.rate,
            channel_count=self.channels,
            source_id=f"src_{self.name}",
            uid=f"uid_{self.name}"
        )

    def pull_sample(self, timeout=0.0):
        """Simulate LSL pull_sample with blocking behavior when timeout > 0."""
        start_time = time.perf_counter()
        deadline = start_time + timeout

        while True:
            now = time.perf_counter()

            # For events/irregular streams, emit roughly every 0.2s
            if self.rate <= 0.0 and (now - self._last_ts) >= 0.2:
                self._last_ts = now
                return list(self._base_sample), time.time()

            # For string streams, mimic low-rate behavior if needed
            if self.channel_format == 'string' and (now - self._last_ts) >= max(self._period, 0.05):
                self._last_ts = now
                return list(self._base_sample), time.time()

            # Numerical streams obey rate; if enough time elapsed, emit sample
            if self.rate > 0.0:
                if self._last_ts == 0.0:
                    self._last_ts = now
                    return list(self._base_sample), time.time()
                if (now - self._last_ts) >= self._period:
                    self._last_ts = now
                    return list(self._base_sample), time.time()

            # If no timeout or timeout expired, return None
            if timeout <= 0 or now >= deadline:
                return None, None

            # Sleep briefly before retrying (avoid busy-wait)
            sleep_time = min(self._period * 0.5 if self._period > 0 else 0.001, deadline - now)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def pull_chunk(self, timeout=0.0, max_samples=1024):
        """Simulate LSL pull_chunk with realistic batching.

        Real LSL inlets buffer samples between calls. This simulates that by
        requiring a minimum accumulation period before returning samples.
        """
        samples = []
        timestamps = []
        now = time.perf_counter()

        # Initialize last_ts if not set
        if self._last_ts == 0.0:
            self._last_ts = now

        # For events/irregular streams
        if self.rate <= 0.0:
            if (now - self._last_ts) >= 0.2:
                self._last_ts = now
                return [list(self._base_sample)], [time.time()]
            return [], []

        # For string streams
        if self.channel_format == 'string':
            if (now - self._last_ts) >= max(self._period, 0.05):
                self._last_ts = now
                return [list(self._base_sample)], [time.time()]
            return [], []

        # For numerical streams: simulate LSL buffering with minimum accumulation
        MIN_BATCH_PERIOD = 0.01  # 10ms - ensures proper batching like real LSL

        elapsed = now - self._last_ts
        if elapsed < MIN_BATCH_PERIOD:
            return [], []  # Wait for more samples to accumulate

        n_samples = int(elapsed / self._period)
        n_samples = min(n_samples, max_samples)

        if n_samples > 0:
            base_time = time.time()
            for i in range(n_samples):
                samples.append(list(self._base_sample))
                timestamps.append(base_time - (n_samples - 1 - i) * self._period)
            self._last_ts += n_samples * self._period  # Advance, don't reset

        return samples, timestamps

    def close_stream(self):
        return


def build_eeg_config(num_eeg_channels: int, sample_rate: float, include_markers: bool = False) -> StreamMetadata:
    labels = [f'EEG_{i+1}' for i in range(num_eeg_channels)]
    channel_types = ['EEG'] * num_eeg_channels
    if include_markers:
        labels.append('Markers')
        channel_types.append('Markers')
    return StreamMetadata(
        type='EEG',
        name='EEG',
        channel_count=len(labels),
        sample_rate=float(sample_rate),
        labels=labels,
        channel_types=channel_types,
        channel_units=['µV'] * len(labels),
        channel_format='float32'
    )


def build_aux_numeric_config(name: str, channels: int, sample_rate: float) -> StreamMetadata:
    return StreamMetadata(
        type=name,
        name=name,
        channel_count=int(channels),
        sample_rate=float(sample_rate),
        labels=[f'{name}_{i+1}' for i in range(channels)],
        channel_types=[name] * channels,
        channel_units=['a.u.'] * channels,
        channel_format='float32'
    )


def build_events_config() -> StreamMetadata:
    return StreamMetadata(
        type='Events',
        name='Events',
        channel_count=1,
        sample_rate=0.0,
        labels=['Event'],
        channel_types=['Event'],
        channel_units=['string'],
        channel_format='string'
    )


def send_all_metadata_to_queue(configs, save_queue):
    """Send metadata for all streams to save queue before starting DAQ.

    This bypasses DAQ's metadata sending to ensure all streams have datasets created
    before data collection begins. The performance test focuses on data throughput,
    not metadata handling.
    """
    from dendrite.data.acquisition import DataRecord

    for config in configs:
        stream_type = config.type
        metadata_record = DataRecord(
            modality=f'{stream_type}_Metadata',
            sample=json.dumps(config.model_dump()),
            timestamp=time.time(),
            local_timestamp=time.time()
        )
        save_queue.put(metadata_record)


def run_daq_in_thread(daq: DataAcquisition, duration_s: float):
    """Run DAQ with data collection (_collect_data handles stream connection)."""
    # Run data collection in thread (_collect_data calls _connect_streams internally)
    thread = Thread(target=daq._collect_data, daemon=True)
    thread.start()
    time.sleep(duration_s)
    daq.stop_event.set()
    thread.join(timeout=5)

    try:
        daq._cleanup()
    except Exception:
        pass


def drain_queue(q, stop_event):
    """Simple consumer that drains a queue (simulates processor)."""
    import queue
    while not stop_event.is_set():
        try:
            q.get(timeout=0.01)
        except queue.Empty:
            pass


@pytest.mark.slow
@pytest.mark.parametrize(
    "eeg_ch, num_aux, aux_ch, aux_rate, include_events, duration",
    [
        (32, 1, 8, 1000.0, True, 1.0),
        (64, 3, 16, 1000.0, True, 1.5),
        (96, 2, 32, 2000.0, False, 1.5),
        (128, 2, 16, 2000.0, True, 2.0),
    ],
)
def test_daq_saver_stress_matrix(eeg_ch, num_aux, aux_ch, aux_rate, include_events, duration, tmp_path):
    """
    Stress DAQ + Saver by scaling number of streams, channel counts, and rates.
    Runs DAQ main loop in-thread with FakeInlet backends and a real saver process.
    """
    # Queues and stop event
    data_queue = mp.Queue(maxsize=5000)
    save_queue = mp.Queue(maxsize=20000)
    stop_event = mp.Event()

    # Build stream configs
    stream_configs = []
    stream_configs.append(build_eeg_config(num_eeg_channels=eeg_ch, sample_rate=500.0, include_markers=False))
    for i in range(num_aux):
        name = f"EMG{i+1}"
        stream_configs.append(build_aux_numeric_config(name=name, channels=aux_ch, sample_rate=aux_rate))
    if include_events:
        stream_configs.append(build_events_config())

    # Prepare mapping of name->FakeInlet
    inlet_map = {cfg.name: FakeInlet(cfg) for cfg in stream_configs}

    # Create DAQ instance (do not start as a separate process)
    daq = DataAcquisition(
        data_queue=data_queue,
        save_queue=save_queue,
        stop_event=stop_event,
        stream_configs=stream_configs
    )

    # Monkeypatch the instance method to return our FakeInlet
    def _resolve_stream_stub(name, stream_type):
        return inlet_map.get(name)

    daq._resolve_stream = _resolve_stream_stub  # type: ignore[attr-defined]
    # Prevent DAQ from re-sending metadata to avoid duplicates and conflicts
    daq._send_stream_metadata = lambda *args, **kwargs: None  # type: ignore[attr-defined]

    # Send metadata for all streams upfront (clean separation of concerns)
    send_all_metadata_to_queue(stream_configs, save_queue)

    # Start saver process
    out_file = tmp_path / "stress_output.h5"
    saver = DataSaver(filename=str(out_file), save_queue=save_queue, stop_event=stop_event, chunk_size=50)
    saver.start()

    # Start data_queue consumer (simulates processor)
    consumer_thread = Thread(target=drain_queue, args=(data_queue, stop_event), daemon=True)
    consumer_thread.start()

    # Give saver time to process metadata before data collection starts
    time.sleep(0.2)

    try:
        start = time.time()
        run_daq_in_thread(daq, duration_s=duration)
        elapsed = time.time() - start

        # Allow saver to drain remaining queue
        time.sleep(0.5)
    finally:
        stop_event.set()
        cleanup_process(saver, timeout=5, queues_to_drain=[data_queue, save_queue])

    # Basic validations
    assert out_file.exists(), "Saver should have created an HDF5 file"

    # Open file and validate datasets exist and have some samples
    import h5py
    with h5py.File(out_file, 'r') as f:
        # EEG must exist
        assert 'EEG' in f, "EEG dataset missing in saved file"
        assert f['EEG'].shape[0] >= 1, "EEG dataset should have at least 1 sample"

        # Each auxiliary numeric stream should exist
        for i in range(num_aux):
            name = f"EMG{i+1}"
            assert name in f, f"{name} dataset missing"
            assert f[name].shape[0] >= 1, f"{name} should have samples"

        # Events dataset if included
        if include_events:
            assert 'Event' in f, "Event dataset missing"
            assert f['Event'].shape[0] >= 1, "Event dataset should have at least 1 entry"

    # Ensure queues are not catastrophically backed up
    # (Not strict pass/fail, but a sanity bound.)
    try:
        dq_size = data_queue.qsize()
    except (NotImplementedError, AttributeError):
        dq_size = 0
    try:
        sq_size = save_queue.qsize()
    except (NotImplementedError, AttributeError):
        sq_size = 0

    # Heuristic bounds depending on duration and configuration
    assert dq_size < 4000, f"data_queue too backed up: {dq_size}"
    assert sq_size < 18000, f"save_queue too backed up: {sq_size}"

    # Print minimal runtime stats for manual inspection in CI logs
    print(
        f"Stress run completed: eeg_ch={eeg_ch}, aux_streams={num_aux}x{aux_ch}@{aux_rate}Hz, "
        f"events={include_events}, duration={duration:.2f}s, elapsed={elapsed:.2f}s, "
        f"dq_size={dq_size}, sq_size={sq_size}"
    )
