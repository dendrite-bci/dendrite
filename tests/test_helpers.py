"""
Shared test utilities for LSL stream mocking.

Provides two approaches for mocking LSL streams in tests:

1. **FakeInlet/FakeLiveInfo** - Lightweight mocks without LSL dependencies
   - Used in performance tests for deterministic, fast testing
   - No actual LSL protocol involved
   - Monkeypatch DAQ's _resolve_stream() method

2. **MockEEGStreamer/MockEventStreamer** - Real LSL outlets for integration tests
   - Uses actual pylsl library to create streams
   - Tests real LSL discovery and connection
   - Requires pylsl to be installed

Usage:
    # Lightweight approach (performance tests)
    from tests.test_helpers import FakeInlet, FakeLiveInfo
    inlet_map = {'EEG': FakeInlet(config)}
    daq._resolve_stream = lambda name, type: inlet_map.get(name)

    # Real LSL approach (integration tests)
    from tests.test_helpers import MockEEGStreamer, MockEventStreamer
    mock_stream = MockEEGStreamer(stream_name="TestEEG", sample_rate=500, n_channels=32)
    mock_stream.start()
"""

import time
import json
import numpy as np
import multiprocessing as mp


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

    def __init__(self, config):
        self.name = config.get('name', config.get('type', 'Unknown'))
        self.stream_type = config.get('type', 'Unknown')
        self.rate = float(config.get('sample_rate', 0.0))
        self.channels = int(config.get('channels', 1))
        self.channel_format = config.get('channel_format', 'float32')
        self._last_ts = 0.0
        self._period = 0.0 if self.rate <= 0 else 1.0 / self.rate
        # Pre-create a deterministic sample for numeric streams to reduce CPU
        if self.stream_type == 'Events' or config.get('type') == 'Events':
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
        # For events/irregular streams, emit roughly every 0.2s
        if self.rate <= 0.0 and (time.perf_counter() - self._last_ts) >= 0.2:
            self._last_ts = time.perf_counter()
            return list(self._base_sample), time.time()

        # For string streams, mimic low-rate behavior if needed
        if self.channel_format == 'string' and (time.perf_counter() - self._last_ts) >= max(self._period, 0.05):
            self._last_ts = time.perf_counter()
            return list(self._base_sample), time.time()

        # Numerical streams obey rate; if enough time elapsed, emit sample
        if self.rate > 0.0:
            now = time.perf_counter()
            if self._last_ts == 0.0:
                self._last_ts = now
                return list(self._base_sample), time.time()
            if (now - self._last_ts) >= self._period:
                # Collapse to single sample per call to avoid over-produce
                self._last_ts = now
                return list(self._base_sample), time.time()
        # No sample ready
        return None, None

    def pull_chunk(self, timeout=0.0, max_samples=1024):
        """Simulate LSL pull_chunk - return all available samples immediately."""
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

        # For numerical streams: return all samples that have accumulated
        if self.rate > 0.0 and self._period > 0:
            elapsed = now - self._last_ts
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


try:
    import pylsl

    class MockEEGStreamer(mp.Process):
        """
        Mock EEG stream using real pylsl outlet.

        Creates an actual LSL stream that can be discovered and connected to
        by DataAcquisition, allowing integration testing of LSL protocol.
        """

        def __init__(self, stream_name="MockEEG", sample_rate=500, n_channels=32, stop_event=None):
            super().__init__(daemon=True)
            self.stream_name = stream_name
            self.sample_rate = sample_rate
            self.n_channels = n_channels
            self.stop_event = stop_event if stop_event is not None else mp.Event()

        def run(self):
            """Run the mock EEG streamer."""
            outlet = None
            try:
                # Create stream info
                info = pylsl.StreamInfo(
                    name=self.stream_name,
                    type='EEG',
                    channel_count=self.n_channels,
                    nominal_srate=self.sample_rate,
                    channel_format='float32',
                    source_id='mock_eeg_123'
                )

                # Create outlet
                outlet = pylsl.StreamOutlet(info)

                # Generate and push samples
                period = 1.0 / self.sample_rate
                next_sample_time = time.time()

                while not self.stop_event.is_set():
                    # Generate random EEG data
                    sample = np.random.randn(self.n_channels).astype(np.float32).tolist()

                    # Push sample
                    outlet.push_sample(sample)

                    # Sleep until next sample time
                    next_sample_time += period
                    sleep_time = next_sample_time - time.time()
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            finally:
                # Cleanup LSL outlet
                if outlet is not None:
                    del outlet


    class MockEventStreamer(mp.Process):
        """
        Mock Events stream using real pylsl outlet.

        Creates an actual LSL events stream for integration testing.
        """

        def __init__(self, stream_name="MockEvents", stop_event=None, event_interval=0.5):
            super().__init__(daemon=True)
            self.stream_name = stream_name
            self.stop_event = stop_event if stop_event is not None else mp.Event()
            self.event_interval = event_interval  # Seconds between events

        def run(self):
            """Run the mock event streamer."""
            outlet = None
            try:
                # Create stream info for string-based events
                info = pylsl.StreamInfo(
                    name=self.stream_name,
                    type='Events',
                    channel_count=1,
                    nominal_srate=0,  # Irregular rate
                    channel_format='string',
                    source_id='mock_events_456'
                )

                # Create outlet
                outlet = pylsl.StreamOutlet(info)

                event_id = 1
                while not self.stop_event.is_set():
                    # Create event JSON
                    event = {
                        "event_id": event_id,
                        "event_type": "test_event"
                    }

                    # Push event as JSON string
                    outlet.push_sample([json.dumps(event)])

                    event_id += 1

                    # Wait before next event
                    time.sleep(self.event_interval)

            finally:
                # Cleanup LSL outlet
                if outlet is not None:
                    del outlet


    class MockEMGStreamer(mp.Process):
        """
        Mock EMG stream using real pylsl outlet.

        Similar to MockEEGStreamer but for EMG data with configurable sample rate.
        Useful for testing multi-rate scenarios (e.g., 500Hz EEG + 2000Hz EMG).
        """

        def __init__(self, stream_name="MockEMG", sample_rate=1000, n_channels=8, stop_event=None):
            super().__init__(daemon=True)
            self.stream_name = stream_name
            self.sample_rate = sample_rate
            self.n_channels = n_channels
            self.stop_event = stop_event if stop_event is not None else mp.Event()

        def run(self):
            """Run the mock EMG streamer."""
            outlet = None
            try:
                # Create stream info
                info = pylsl.StreamInfo(
                    name=self.stream_name,
                    type='EMG',
                    channel_count=self.n_channels,
                    nominal_srate=self.sample_rate,
                    channel_format='float32',
                    source_id='mock_emg_789'
                )

                # Create outlet
                outlet = pylsl.StreamOutlet(info)

                # Generate and push samples
                period = 1.0 / self.sample_rate
                next_sample_time = time.time()

                while not self.stop_event.is_set():
                    # Generate random EMG data
                    sample = np.random.randn(self.n_channels).astype(np.float32).tolist()

                    # Push sample
                    outlet.push_sample(sample)

                    # Sleep until next sample time
                    next_sample_time += period
                    sleep_time = next_sample_time - time.time()
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            finally:
                # Cleanup LSL outlet
                if outlet is not None:
                    del outlet

except ImportError:
    # pylsl not available - mock streamers won't work
    MockEEGStreamer = None
    MockEventStreamer = None
    MockEMGStreamer = None
