#!/usr/bin/env python3
"""
Offline Data Streamer V2
Streamlined version that maintains functionality while removing bloat.
"""

import logging
import time
from multiprocessing import Process
from multiprocessing.queues import Queue

import numpy as np
from pylsl import local_clock

from dendrite.data import EventOutlet, MOAABLoader, get_moabb_dataset_info
from dendrite.data.lsl_helpers import LSLOutlet
from dendrite.data.stream_schemas import StreamConfig
from dendrite.utils.format_loaders import SUPPORTED_FORMATS, load_file


class OfflineDataStreamer(Process):
    """
    Streamlined data streamer that supports multi-modal data files (EEG, EMG, ECG, EOG, etc.),
    synthetic data generation, and multiple data types with significantly reduced complexity
    compared to the original.

    Key improvements:
    - Detects and streams ALL channel types present in files (not just EEG)
    - Maintains proper channel labels and types for multi-modal data
    - Supports both single-modal and multi-modal streaming
    """

    def __init__(
        self,
        sample_rate: float,
        stop_event,
        data_type: str = "EEG",
        channels: int = 64,
        data_file_path: str = "",
        stream_name_prefix: str | None = None,
        moabb_preset: str | None = None,
        moabb_subject: int | None = None,
        moabb_session: str | None = None,
        enable_event_stream: bool = False,
        info_queue: Queue | None = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.stop_event = stop_event
        self.data_type = data_type
        self.channels = channels
        self.data_file_path = data_file_path
        self.stream_name_prefix = stream_name_prefix
        self.moabb_preset = moabb_preset
        self.moabb_subject = moabb_subject
        self.moabb_session = moabb_session
        self.enable_event_stream = enable_event_stream
        self.info_queue = info_queue
        self.logger = logging.getLogger(f"StreamlinedStreamer_{data_type}")

    def _get_stream_name(self, base_name: str) -> str:
        """Generate LSL stream name with optional prefix"""
        if self.stream_name_prefix:
            return f"{base_name}_{self.stream_name_prefix}"
        return base_name

    def _create_stream_config(
        self, name: str, channels: int, has_markers: bool = False
    ) -> StreamConfig:
        """Create minimal stream configuration"""
        total_channels = channels + (1 if has_markers else 0)

        # Generate channel labels
        labels = [f"{self.data_type}{i + 1:02d}" for i in range(channels)]
        if has_markers:
            labels.append("Markers")

        # Set appropriate units
        units_map = {
            "EEG": "microvolts",
            "EMG": "microvolts",
            "ECG": "microvolts",
            "GSR": "microsiemens",
            "Position": "mm",
            "Force": "N",
            "Acceleration": "m/s^2",
            "Temperature": "celsius",
        }
        unit = units_map.get(self.data_type, "a.u.")
        units = [unit] * channels + (["unknown"] if has_markers else [])

        return StreamConfig(
            name=self._get_stream_name(name),
            type=self.data_type,
            channel_count=total_channels,
            sample_rate=self.sample_rate,
            channel_format="float32",
            labels=labels,
            channel_types=[self.data_type] * channels + (["Markers"] if has_markers else []),
            channel_units=units,
            source_id=f"streamlined_{self.data_type.lower()}_id",
        )

    def _create_file_stream_config(
        self, name: str, channel_names: list[str], channel_types_list: list[str], sample_rate: float
    ) -> StreamConfig:
        """Create stream configuration for file-based data with actual channel names and types"""
        # Check if markers already present (H5 format has embedded Markers)
        has_markers = "MARKERS" in [t.upper() for t in channel_types_list]

        if has_markers:
            all_labels = channel_names
            all_channel_types = channel_types_list
        else:
            all_labels = channel_names + ["Markers"]
            all_channel_types = channel_types_list + ["Markers"]
        total_channels = len(all_labels)

        # Create channel units based on channel types
        channel_units = []
        for ch_type in all_channel_types:
            ch_lower = ch_type.lower()
            if ch_lower in ["eeg", "emg", "ecg", "eog", "seeg", "ecog"]:
                channel_units.append("microvolts")
            elif ch_lower == "meg":
                channel_units.append("tesla")
            elif ch_lower in ["stim", "stimulus"]:
                channel_units.append("volts")
            elif ch_lower in ["resp", "gsr"]:
                channel_units.append("a.u.")
            elif ch_lower == "temperature":
                channel_units.append("celsius")
            elif ch_lower == "markers":
                channel_units.append("unknown")
            else:
                channel_units.append("a.u.")

        # Always use 'EEG' as stream type for neural data streams
        # This maintains compatibility with existing LSL receivers expecting EEG streams
        unique_types = list(set(channel_types_list))
        if "eeg" in [t.lower() for t in unique_types]:
            primary_type = "EEG"
        elif len(unique_types) == 1:
            primary_type = unique_types[0].upper()
        else:
            primary_type = "EEG"  # Default to EEG for multi-modal neural data

        return StreamConfig(
            name=self._get_stream_name(name),
            type=primary_type,
            channel_count=total_channels,
            sample_rate=sample_rate,
            channel_format="float32",
            labels=all_labels,
            channel_types=all_channel_types,
            channel_units=channel_units,
            source_id=f"streamlined_file_{primary_type.lower()}_id",
        )

    def run(self):
        """Main entry point for streaming"""
        try:
            if self.moabb_preset:
                self._stream_data_from_moabb()
            elif self.data_file_path and self._is_supported_file(self.data_file_path):
                self._stream_data_from_file()
            elif self.data_type == "Events":
                self._stream_events()
            else:
                self._stream_synthetic_data()
        except Exception as e:
            self.logger.error(f"Streaming error: {e}")

    def _is_supported_file(self, file_path: str) -> bool:
        """Check if file format is supported."""
        from pathlib import Path

        ext = Path(file_path).suffix.lower()
        return ext in SUPPORTED_FORMATS

    def _convert_units(self, data: np.ndarray, channel_types: list[str]) -> np.ndarray:
        """Convert data from volts to appropriate units based on channel types.

        MNE returns data in volts, but BMI uses physiologically correct units:
        - EEG/EOG: µV (microvolts)
        - EMG/ECG: mV (millivolts)
        """
        converted = data.copy()
        for i, ch_type in enumerate(channel_types):
            ch_type_upper = ch_type.upper()
            if ch_type_upper in ["EEG", "EOG", "VEOG", "HEOG", "SEEG", "ECOG"]:
                converted[:, i] = data[:, i] * 1e6  # V → µV
            elif ch_type_upper in ["EMG", "ECG"]:
                converted[:, i] = data[:, i] * 1e3  # V → mV
        return converted

    def _stream_data_from_file(self):
        """Stream data from file using format-agnostic loader.

        Supports .set (EEGLAB), .fif (MNE), .h5/.hdf5 (internal BMI format).
        """
        try:
            # Load via format abstraction
            loaded = load_file(self.data_file_path)

            # Apply unit conversion (MNE data is in volts)
            from pathlib import Path

            ext = Path(self.data_file_path).suffix.lower()
            if ext in [".set", ".fif"]:
                # MNE formats need unit conversion
                data = self._convert_units(loaded.data, loaded.channel_types)
            else:
                # H5 data is already in correct units
                data = loaded.data

            n_samples, n_channels = data.shape
            channel_types_upper = [t.upper() for t in loaded.channel_types]

            # Check if data already has markers (H5 format)
            has_markers_column = "MARKERS" in channel_types_upper

            # Log channel type summary
            type_counts = {}
            for ch_type in channel_types_upper:
                type_counts[ch_type] = type_counts.get(ch_type, 0) + 1
            type_summary = ", ".join(
                [f"{count} {ch_type}" for ch_type, count in type_counts.items()]
            )
            markers_info = "" if has_markers_column else " + markers"
            self.logger.info(f"Streaming {type_summary}{markers_info} @ {loaded.sample_rate} Hz")

            # Create stream config
            config = self._create_file_stream_config(
                "FileData", loaded.channel_names, channel_types_upper, loaded.sample_rate
            )
            streamer = LSLOutlet(config=config)

            # Create separate event outlet if enabled
            event_outlet = None
            if self.enable_event_stream and loaded.events:
                # Build event mapping from event_id or generate from unique codes
                if loaded.event_id:
                    event_mapping = loaded.event_id
                else:
                    unique_codes = sorted(set(code for _, code in loaded.events))
                    event_mapping = {f"event_{code}": code for code in unique_codes}

                stream_name = self._get_stream_name("File_Events")
                event_outlet = EventOutlet(
                    stream_name=stream_name, events=event_mapping, stream_id="file_events_id"
                )
                self.logger.info(
                    f"Event stream enabled: {len(loaded.events)} events, mapping: {event_mapping}"
                )

            # Build event lookup for fast access (sample_idx -> event_code)
            event_dict = {sample_idx: code for sample_idx, code in loaded.events}

            # Send duration to GUI
            total_duration = n_samples / loaded.sample_rate
            if self.info_queue:
                self.info_queue.put({"type": "duration", "value": total_duration})

            # Stream data with timing control
            start_time = time.perf_counter()
            sample_interval = 1.0 / loaded.sample_rate
            progress_interval = int(loaded.sample_rate // 2)  # Report progress every 0.5s

            for sample_idx in range(n_samples):
                if self.stop_event.is_set():
                    break

                # Timing control
                expected_time = start_time + (sample_idx * sample_interval)
                sleep_time = expected_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)

                sample = data[sample_idx].tolist()

                # Only add marker if data doesn't already have markers column
                marker = 0
                if not has_markers_column:
                    marker = event_dict.get(sample_idx, 0)

                    # Send event on separate stream if enabled (before zeroing marker)
                    if event_outlet and marker != 0:
                        event_mapping = loaded.event_id or {}
                        event_name = next(
                            (k for k, v in event_mapping.items() if v == marker), f"event_{marker}"
                        )
                        event_outlet.send_event(
                            event_type=event_name, additional_data={"sample_idx": sample_idx}
                        )

                    # Zero out marker in main stream if separate events enabled and remove option is on
                    if self.enable_event_stream:
                        marker = 0

                    sample.append(marker)

                streamer.push_sample(sample, local_clock())

                # Report progress to GUI
                if self.info_queue and sample_idx % progress_interval == 0:
                    progress = sample_idx / n_samples
                    self.info_queue.put({"type": "progress", "value": progress})

                if sample_idx % 10000 == 0:
                    self.logger.debug(f"Streamed {sample_idx}/{n_samples} samples")

            # Send final progress
            if self.info_queue:
                self.info_queue.put({"type": "progress", "value": 1.0})

            streamer.close()
            if event_outlet:
                event_outlet.close()
            self.logger.info("File streaming completed")

        except Exception as e:
            self.logger.error(f"File streaming error: {e}")

    def _stream_data_from_moabb(self):
        """Stream continuous data from MOABB dataset.

        Loads data via MOAABLoader and streams sample-by-sample via LSL.
        Optionally creates a separate event stream.
        """
        try:
            # Get preset config and create loader
            info = get_moabb_dataset_info(self.moabb_preset)
            if not info:
                raise ValueError(f"Unknown MOABB dataset: {self.moabb_preset}")
            config = info["config"]
            loader = MOAABLoader(config)

            self.logger.info(
                f"Loading MOABB data: {self.moabb_preset}, "
                f"subject {self.moabb_subject}, session {self.moabb_session}"
            )

            # Load continuous data (returns 4 values including event_mapping)
            data, event_times, event_labels, event_mapping = loader.load_continuous(
                self.moabb_subject, self.moabb_session
            )

            # Get channel info from MNE
            channel_names = loader.get_channel_names(self.moabb_subject)
            channel_types = loader.get_channel_types(self.moabb_subject)
            sample_rate = loader.get_sample_rate()

            # Data is (n_channels, n_samples), transpose to (n_samples, n_channels)
            data = data.T

            # Data is already in microvolts (preprocessing applies 1e6 scaling)
            n_samples, n_channels = data.shape
            total_duration = n_samples / sample_rate
            self.logger.info(
                f"Loaded {n_samples} samples, {n_channels} channels @ {sample_rate} Hz, "
                f"{len(event_times)} events, duration: {total_duration:.1f}s"
            )

            # Send duration to GUI
            if self.info_queue:
                self.info_queue.put({"type": "duration", "value": total_duration})

            # Create stream config (always adds Markers channel - MOABB events come from annotations)
            stream_config = self._create_file_stream_config(
                "MOABB_EEG", channel_names, channel_types, sample_rate
            )
            streamer = LSLOutlet(config=stream_config)

            # Create separate event outlet if enabled
            event_outlet = None
            # Build reverse lookup: label -> event name
            label_to_name = {label: name for name, label in event_mapping.items()}
            if self.enable_event_stream and len(event_times) > 0:
                # Use proper event names with +1 offset for marker codes (0 = no event)
                stream_event_mapping = {name: label + 1 for name, label in event_mapping.items()}

                stream_name = self._get_stream_name("MOABB_Events")
                event_outlet = EventOutlet(
                    stream_name=stream_name,
                    events=stream_event_mapping,
                    stream_id=f"moabb_events_{self.moabb_preset}",
                )
                self.logger.info(f"Event stream enabled: {stream_event_mapping}")

            # Create event lookup for fast access: sample_idx -> (marker_code, event_name)
            # Offset event codes by +1 so that 0 = "no event"
            event_dict = {}
            for evt_time, evt_label in zip(event_times, event_labels, strict=False):
                event_name = label_to_name.get(evt_label, f"class_{evt_label}")
                event_dict[int(evt_time)] = (int(evt_label) + 1, event_name)  # +1 offset

            # Stream data with timing control
            start_time = time.perf_counter()
            sample_interval = 1.0 / sample_rate
            progress_interval = int(sample_rate // 2)  # Report progress every 0.5s

            for sample_idx in range(n_samples):
                if self.stop_event.is_set():
                    break

                # Timing control
                expected_time = start_time + (sample_idx * sample_interval)
                sleep_time = expected_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Get sample and add marker from annotations
                # MOABB events come from annotations (event_dict), not embedded in data
                sample = data[sample_idx].tolist()
                event_info = event_dict.get(sample_idx)
                marker = event_info[0] if event_info else 0

                # Send event on separate stream if enabled (before zeroing marker)
                if event_outlet and event_info:
                    event_outlet.send_event(
                        event_type=event_info[1], additional_data={"sample_idx": sample_idx}
                    )

                # Zero out marker in main stream if separate events enabled and remove option is on
                if self.enable_event_stream:
                    marker = 0

                sample.append(marker)
                streamer.push_sample(sample, local_clock())

                # Report progress to GUI
                if self.info_queue and sample_idx % progress_interval == 0:
                    progress = sample_idx / n_samples
                    self.info_queue.put({"type": "progress", "value": progress})

                if sample_idx % 10000 == 0:
                    self.logger.debug(f"Streamed {sample_idx}/{n_samples} samples")

            # Send final progress
            if self.info_queue:
                self.info_queue.put({"type": "progress", "value": 1.0})

            streamer.close()
            if event_outlet:
                event_outlet.close()
            self.logger.info("MOABB streaming completed")

        except Exception as e:
            self.logger.error(f"MOABB streaming error: {e}")

    def _stream_events(self):
        """Stream discrete events using EventOutlet"""
        try:
            # Define event mapping for motor imagery or other events
            event_mapping = {"left_hand": 1, "right_hand": 2, "rest": 0}

            # Create EventOutlet with stream name prefix if available
            stream_name = self._get_stream_name("Events")
            event_outlet = EventOutlet(
                stream_name=stream_name, events=event_mapping, stream_id="streamlined_events_id"
            )

            self.logger.info("Streaming events every 5 seconds using EventOutlet")

            last_event_time = 0
            event_interval = 5.0  # seconds

            while not self.stop_event.is_set():
                current_time = time.perf_counter()

                if current_time - last_event_time >= event_interval:
                    # Send random event
                    event_type = np.random.choice(["left_hand", "right_hand", "rest"])
                    additional_data = {
                        "trial_number": int((current_time - last_event_time) / event_interval),
                        "source": "synthetic_generator",
                    }

                    event_outlet.send_event(event_type=event_type, additional_data=additional_data)
                    self.logger.info(f"Sent event: {event_type}")
                    last_event_time = current_time

                time.sleep(0.1)  # Check every 100ms

            event_outlet.close()

        except Exception as e:
            self.logger.error(f"Event streaming error: {e}")

    def _stream_synthetic_data(self):
        """Stream synthetic data - simplified generation"""
        try:
            has_markers = self.data_type in ["EEG", "EMG"]

            config = self._create_stream_config(
                f"Synthetic{self.data_type}", self.channels, has_markers
            )
            streamer = LSLOutlet(config=config)

            self.logger.info(
                f"Streaming synthetic {self.data_type}: {self.channels} channels at {self.sample_rate} Hz"
            )

            # Simple signal parameters
            signal_params = self._get_signal_params(self.data_type)
            channel_phases = np.random.uniform(0, 2 * np.pi, self.channels)

            # Timing control
            start_time = time.perf_counter()
            sample_interval = 1.0 / self.sample_rate
            sample_count = 0
            last_event_time = -10  # Force first event

            while not self.stop_event.is_set():
                # Timing control
                expected_time = start_time + (sample_count * sample_interval)
                sleep_time = expected_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)

                elapsed_time = time.perf_counter() - start_time
                sample_count += 1

                # Generate sample
                sample = []
                for ch in range(self.channels):
                    signal = self._generate_simple_signal(
                        self.data_type, elapsed_time, channel_phases[ch], signal_params
                    )
                    sample.append(signal)

                # Add event marker if needed
                if has_markers:
                    marker = 0
                    if elapsed_time - last_event_time >= 5.0:  # Event every 5 seconds
                        marker = np.random.choice([1, 2])
                        last_event_time = elapsed_time
                    sample.append(marker)

                streamer.push_sample(sample, local_clock())

                # Occasional logging
                if sample_count % 10000 == 0:
                    rate = sample_count / elapsed_time if elapsed_time > 0 else 0
                    self.logger.debug(f"Sample {sample_count}, rate: {rate:.1f} Hz")

            streamer.close()
            self.logger.info("Synthetic streaming completed")

        except Exception as e:
            self.logger.error(f"Synthetic streaming error: {e}")

    def _get_signal_params(self, data_type: str) -> dict:
        """Get simplified signal parameters for each data type"""
        params = {
            "EEG": {"freq": [10, 20], "amp": [10, 30], "noise": 5},
            "MEG": {"freq": [10, 40], "amp": [1e-13, 5e-13], "noise": 1e-14},  # Tesla units
            "EMG": {"freq": [50, 100], "amp": [0.1, 0.5], "noise": 0.05},
            "ECG": {"freq": [1, 2], "amp": [100, 200], "noise": 10},
            "EOG": {"freq": [0.5, 5], "amp": [50, 200], "noise": 10},
            "STIM": {"freq": [0.1, 1], "amp": [0, 5], "noise": 0.1},  # Trigger signals
            "RESP": {"freq": [0.1, 0.5], "amp": [1, 5], "noise": 0.2},  # Respiration
            "GSR": {"freq": [0.1, 0.5], "amp": [1, 3], "noise": 0.2},
            "Position": {"freq": [0.1, 1], "amp": [10, 50], "noise": 2},
            "Force": {"freq": [0.5, 2], "amp": [1, 10], "noise": 0.5},
            "MISC": {"freq": [1, 10], "amp": [1, 10], "noise": 1},
            "UNCLASSIFIED": {"freq": [1, 10], "amp": [1, 10], "noise": 1},
            "ContinuousEvents": {"freq": [0.1, 1], "amp": [10, 50], "noise": 2},
        }
        return params.get(data_type, {"freq": [1], "amp": [1], "noise": 0.1})

    def _generate_simple_signal(
        self, data_type: str, time: float, phase: float, params: dict
    ) -> float:
        """Generate simple synthetic signal"""
        signal = 0.0

        # Add frequency components
        for freq in params["freq"]:
            amp = np.random.uniform(params["amp"][0], params["amp"][1])
            signal += amp * np.sin(2 * np.pi * freq * time + phase)

        signal += np.random.normal(0, params["noise"])

        # Special case for specific data types
        if data_type == "ECG":
            # Simple ECG-like pattern
            heart_rate = 70
            period = 60.0 / heart_rate
            cycle_phase = (time % period) / period
            if 0.1 < cycle_phase < 0.3:  # QRS complex
                signal += 300 * np.sin((cycle_phase - 0.1) * np.pi / 0.2)

        return signal
