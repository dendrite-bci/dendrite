from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy import signal

from dendrite.utils.logger_central import get_logger


class BaseModalityProcessor(ABC):
    """
    Abstract base class for modality-specific preprocessing.
    Each modality (EEG, EMG, etc.) should inherit from this class.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the processor with modality-specific configuration.

        Args:
            config: Dictionary containing modality-specific parameters
        """
        self.config = config
        self.logger = get_logger()
        self.modality_name = self.__class__.__name__.replace("Processor", "")

        self.num_channels = config.get("num_channels")
        self.sample_rate = config.get("sample_rate")
        self.downsample_factor = config.get("downsample_factor", 1)

        # Downsampling state (shared by all processors)
        self._downsample_state = {
            "buffer": np.zeros((self.num_channels, 0)),
        }

        # Anti-aliasing filter for downsampling (if needed)
        if self.downsample_factor > 1:
            # Cutoff at 80% of output Nyquist (same margin as scipy.signal.decimate)
            output_nyquist = (self.sample_rate / self.downsample_factor) / 2
            normalized_cutoff = 0.8 * output_nyquist / (self.sample_rate / 2)
            # Chebyshev Type I, order 8, 0.05 dB ripple (matches scipy.signal.decimate default)
            self._aa_b, self._aa_a = signal.cheby1(8, 0.05, normalized_cutoff)
            self._aa_zi = np.zeros((self.num_channels, max(len(self._aa_a), len(self._aa_b)) - 1))

        self._initialize_state()

    @abstractmethod
    def _initialize_state(self):
        """Initialize any internal state needed for processing."""
        pass

    def _downsample_with_state(self, data: np.ndarray) -> np.ndarray:
        """Stateful downsampling with proper anti-aliasing filter.

        Uses Chebyshev Type I lowpass filter (matching scipy.signal.decimate)
        followed by stride decimation. The filter prevents aliasing by removing
        frequencies above the output Nyquist before decimation.
        """
        # Apply anti-aliasing lowpass filter
        data, self._aa_zi = signal.lfilter(self._aa_b, self._aa_a, data, axis=1, zi=self._aa_zi)

        # Accumulate samples
        self._downsample_state["buffer"] = np.concatenate(
            [self._downsample_state["buffer"], data], axis=1
        )

        total_samples = self._downsample_state["buffer"].shape[1]
        num_output = total_samples // self.downsample_factor

        if num_output == 0:
            return np.zeros((self.num_channels, 0))

        # Stride decimation (now safe - high frequencies removed by AA filter)
        output_data = self._downsample_state["buffer"][:, :: self.downsample_factor][:, :num_output]

        # Keep remaining samples for next chunk
        consumed = num_output * self.downsample_factor
        self._downsample_state["buffer"] = self._downsample_state["buffer"][:, consumed:]

        return output_data

    def _create_filter_state(self, b: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Create filter state vector for lfilter."""
        return np.zeros((self.num_channels, max(len(a), len(b)) - 1))

    @abstractmethod
    def process_chunk(self, data: np.ndarray) -> np.ndarray:
        """
        Process a chunk of data for this modality.

        Args:
            data: Input data (num_channels, num_samples)

        Returns:
            Processed data (num_channels, num_samples_out)
        """
        pass

    def reset_state(self):
        """Reset the processor state. Called when starting a new session."""
        self._downsample_state["buffer"] = np.zeros((self.num_channels, 0))
        if self.downsample_factor > 1:
            self._aa_zi = np.zeros(
                (self.num_channels, max(len(self._aa_a), len(self._aa_b)) - 1)
            )
        self._initialize_state()
        self.logger.info(f"{self.modality_name} processor state reset")


class EEGProcessor(BaseModalityProcessor):
    """EEG-specific preprocessing: filtering, re-referencing, downsampling."""

    def _initialize_state(self):
        """Initialize EEG filtering state."""
        # Bandpass filter parameters
        self.lowcut = self.config.get("lowcut", 0.5)
        self.highcut = self.config.get("highcut", 50.0)
        self.apply_rereferencing = self.config.get("apply_rereferencing", True)

        nyquist = 0.5 * self.sample_rate

        # Validate frequency range
        if self.lowcut <= 0 or self.lowcut >= nyquist:
            self.lowcut = max(0.1, nyquist * 0.01)
            self.logger.warning(f"Adjusted EEG lowcut to {self.lowcut:.2f} Hz")

        if self.highcut <= 0 or self.highcut >= nyquist:
            self.highcut = nyquist * 0.99
            self.logger.warning(f"Adjusted EEG highcut to {self.highcut:.2f} Hz")

        if self.lowcut >= self.highcut:
            raise ValueError(f"Invalid EEG frequency range: {self.lowcut}-{self.highcut} Hz")

        self.b, self.a = signal.butter(
            4, [self.lowcut / nyquist, self.highcut / nyquist], btype="band"
        )
        self.zi = self._create_filter_state(self.b, self.a)

        self.logger.info(
            f"EEG processor initialized: {self.lowcut}-{self.highcut} Hz, "
            f"{self.num_channels} channels, downsample={self.downsample_factor}"
        )

    def process_chunk(self, data: np.ndarray, bad_channels: list[int] | None = None) -> np.ndarray:
        """
        Process EEG chunk: re-reference -> filter -> downsample.

        Args:
            data: EEG data (channels, samples)
            bad_channels: Optional list of bad channel indices to exclude from re-referencing

        Returns:
            Processed EEG data
        """
        data = data.astype(np.float64)

        # Re-referencing (common average reference, excluding bad channels)
        if self.apply_rereferencing and data.shape[0] > 1:
            if bad_channels:
                good_mask = np.ones(data.shape[0], dtype=bool)
                good_mask[bad_channels] = False

                # Compute reference using only good channels
                if np.any(good_mask):
                    reference = np.mean(data[good_mask, :], axis=0, keepdims=True)
                    data = data - reference
            else:
                # Standard common average reference
                data = data - np.mean(data, axis=0, keepdims=True)

        data, self.zi = signal.lfilter(self.b, self.a, data, axis=1, zi=self.zi)

        if self.downsample_factor > 1:
            data = self._downsample_with_state(data)

        return data


class EMGProcessor(BaseModalityProcessor):
    """EMG-specific preprocessing: bandpass filtering, notch filter, downsampling."""

    def _initialize_state(self):
        """Initialize EMG filtering state."""
        # EMG-specific frequency ranges
        self.lowcut = max(20, self.config.get("lowcut", 20))
        self.highcut = min(450, self.config.get("highcut", 450))
        self.line_freq = self.config.get("line_freq", 50)  # 50 or 60 Hz
        self.notch_width = self.config.get("notch_width", 4)

        nyquist = 0.5 * self.sample_rate

        self.b_bp, self.a_bp = signal.butter(
            4, [self.lowcut / nyquist, self.highcut / nyquist], btype="band"
        )
        self.zi_bp = self._create_filter_state(self.b_bp, self.a_bp)

        # Notch filter for power line noise
        notch_low = (self.line_freq - self.notch_width / 2) / nyquist
        notch_high = (self.line_freq + self.notch_width / 2) / nyquist

        if 0 < notch_low < notch_high < 1:
            self.b_notch, self.a_notch = signal.butter(4, [notch_low, notch_high], btype="bandstop")
            self.zi_notch = self._create_filter_state(self.b_notch, self.a_notch)
            self.has_notch = True
        else:
            self.has_notch = False
            self.logger.warning(f"Invalid notch filter range for {self.line_freq}Hz")

        self.logger.info(
            f"EMG processor initialized: {self.lowcut}-{self.highcut} Hz, "
            f"notch at {self.line_freq}Hz, {self.num_channels} channels"
        )

    def process_chunk(self, data: np.ndarray) -> np.ndarray:
        """Process EMG chunk: notch filter -> bandpass filter -> downsample."""
        data = data.astype(np.float64)

        # Notch filter (remove power line noise)
        if self.has_notch:
            data, self.zi_notch = signal.lfilter(
                self.b_notch, self.a_notch, data, axis=1, zi=self.zi_notch
            )

        data, self.zi_bp = signal.lfilter(self.b_bp, self.a_bp, data, axis=1, zi=self.zi_bp)

        if self.downsample_factor > 1:
            data = self._downsample_with_state(data)

        return data


class EOGProcessor(BaseModalityProcessor):
    """EOG-specific preprocessing: minimal filtering, used mainly as reference."""

    def _initialize_state(self):
        """Initialize EOG processing state."""
        # Minimal filtering for EOG channels (used as references)
        # EOG artifacts (blinks/saccades) are primarily 0.1-5 Hz
        # Research shows 0.1-10 Hz is optimal for artifact removal
        self.lowcut = self.config.get("lowcut", 0.1)
        self.highcut = self.config.get("highcut", 10.0)

        nyquist = 0.5 * self.sample_rate
        self.b, self.a = signal.butter(
            2, [self.lowcut / nyquist, self.highcut / nyquist], btype="band"
        )
        self.zi = self._create_filter_state(self.b, self.a)

        self.logger.info(
            f"EOG processor initialized: {self.lowcut}-{self.highcut} Hz, "
            f"{self.num_channels} channels"
        )

    def process_chunk(self, data: np.ndarray) -> np.ndarray:
        """Process EOG chunk: light filtering -> downsample."""
        data = data.astype(np.float64)

        data, self.zi = signal.lfilter(self.b, self.a, data, axis=1, zi=self.zi)

        if self.downsample_factor > 1:
            data = self._downsample_with_state(data)

        return data


class PassthroughProcessor(BaseModalityProcessor):
    """Simple passthrough processor for modalities that don't need preprocessing."""

    def _initialize_state(self):
        """Initialize passthrough processor (downsampling handled by base class)."""
        self.logger.info(
            f"Passthrough processor initialized for {self.num_channels} channels, downsample={self.downsample_factor}"
        )

    def process_chunk(self, data: np.ndarray) -> np.ndarray:
        """Process data with optional downsampling."""
        data = data.astype(np.float64)

        if self.downsample_factor > 1:
            data = self._downsample_with_state(data)

        return data


class OnlinePreprocessor:
    """
    Modular preprocessor that uses modality-specific processing blocks.

    Features:
    - Modality-specific preprocessing using dedicated processor classes
    - Easy to extend with new modalities
    - Configurable processing pipeline per modality
    """

    # Registry of available processors (lowercase keys)
    PROCESSOR_REGISTRY = {
        "eeg": EEGProcessor,
        "emg": EMGProcessor,
        "eog": EOGProcessor,
        "veog": EOGProcessor,
        "heog": EOGProcessor,
        # Note: markers/events not preprocessed - handled directly in DataProcessor
    }

    def __init__(self, modality_preprocessing: dict[str, dict]) -> None:
        """
        Initialize the modular preprocessor.

        Args:
            modality_preprocessing: Dict where keys are modality names and values are preprocessing config dicts
        """
        # Normalize modality keys to lowercase (boundary: loaded configs may have uppercase)
        self.modality_preprocessing = {k.lower(): v for k, v in modality_preprocessing.items()}
        self.processors = {}
        self.logger = get_logger()
        self._create_processors()

    def _create_processors(self):
        """Create processor instances for each configured modality."""
        for modality, config in self.modality_preprocessing.items():
            try:
                processor_class = self.PROCESSOR_REGISTRY.get(modality)

                if processor_class is None:
                    processor_class = PassthroughProcessor
                    self.logger.info(
                        f"Using passthrough processor for unknown modality: {modality}"
                    )

                self.processors[modality] = processor_class(config)
                self.logger.info(f"Created {processor_class.__name__} for '{modality}'")

            except Exception as e:
                self.logger.error(f"Failed to create processor for '{modality}': {e}")

    def process(
        self, data_dict: dict[str, np.ndarray], bad_channels_eeg: list[int] | None = None
    ) -> dict[str, np.ndarray]:
        """
        Process data using pre-created modality-specific processors.

        Args:
            data_dict: Dict where keys are modality names and values are data arrays
            bad_channels_eeg: Optional list of bad EEG channel indices to exclude from re-referencing

        Returns:
            Dict of processed data for each modality
        """
        processed_data = {}

        for modality, data in data_dict.items():
            processor = self.processors.get(modality)

            if processor is None:
                # No processor for this modality - pass through as float64
                processed_data[modality] = (
                    data.astype(np.float64) if isinstance(data, np.ndarray) else data
                )
                continue

            try:
                if modality == "eeg":
                    processed_data[modality] = processor.process_chunk(
                        data, bad_channels=bad_channels_eeg
                    )
                else:
                    processed_data[modality] = processor.process_chunk(data)
            except Exception as e:
                self.logger.error(f"Processing failed for {modality}: {e}")
                processed_data[modality] = (
                    data.astype(np.float64) if isinstance(data, np.ndarray) else data
                )

        return processed_data

    def reset_all_states(self):
        """Reset all processor states."""
        for processor in self.processors.values():
            processor.reset_state()
        self.logger.info("All processor states reset")
