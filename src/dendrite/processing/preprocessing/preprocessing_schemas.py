"""
Pydantic configuration schemas for preprocessing.

Defines validated configuration models for the preprocessing pipeline.
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModalityConfig(BaseModel):
    """Configuration for a single modality (EEG, EMG, ECG, etc)."""

    model_config = ConfigDict(extra="allow")

    num_channels: int = Field(..., ge=1, description="Number of channels")
    sample_rate: float = Field(500.0, gt=0, description="Sampling rate in Hz")
    # Note: downsample_factor is injected at runtime by processing_pipeline.py (extra='allow')

    lowcut: float | None = Field(None, ge=0, description="Lowcut frequency in Hz")
    highcut: float | None = Field(None, gt=0, description="Highcut frequency in Hz")
    apply_rereferencing: bool = Field(False, description="Apply common average reference")
    line_freq: float | None = Field(None, ge=0, description="Line frequency for notch filter")

    @model_validator(mode="after")
    def validate_frequency_range(self):
        """Validate filter frequency constraints."""
        # Check highcut > lowcut if both are provided
        if self.lowcut is not None and self.highcut is not None:
            if self.highcut <= self.lowcut:
                raise ValueError(
                    f"Highcut frequency ({self.highcut}Hz) must be greater than lowcut frequency ({self.lowcut}Hz)"
                )

        # Check frequencies are below Nyquist frequency
        nyquist = self.sample_rate / 2.0
        if self.highcut is not None and self.highcut >= nyquist:
            raise ValueError(
                f"Highcut frequency ({self.highcut}Hz) must be less than Nyquist frequency ({nyquist}Hz for {self.sample_rate}Hz sampling)"
            )

        if self.lowcut is not None and self.lowcut >= nyquist:
            raise ValueError(
                f"Lowcut frequency ({self.lowcut}Hz) must be less than Nyquist frequency ({nyquist}Hz for {self.sample_rate}Hz sampling)"
            )

        return self


# Default configurations for each modality type
# These serve as the single source of truth for GUI defaults
DEFAULT_EEG_CONFIG = {
    "lowcut": 0.5,  # High-pass at 0.5Hz removes slow drifts
    "highcut": 50.0,  # Low-pass at 50Hz captures main EEG bands (delta to gamma)
    "apply_rereferencing": True,  # Common average reference is standard for EEG
}

DEFAULT_EMG_CONFIG = {
    "lowcut": 20.0,  # EMG signals start above 20Hz
    "highcut": 200.0,  # Safe value below Nyquist for 500Hz sampling
    "line_freq": 50.0,  # European standard (use 60.0 for North America)
}

DEFAULT_EOG_CONFIG = {
    "lowcut": 0.1,  # Very low frequency to capture slow eye drifts
    "highcut": 10.0,  # Eye movements and blinks are below 10Hz
}


class QualityControlConfig(BaseModel):
    """Channel quality monitoring configuration."""

    enabled: bool = Field(False, description="Enable channel quality monitoring")

    # All other parameters use hardcoded defaults in ChannelQualityMonitor
    # This keeps the GUI simple (just enable/disable checkbox)


class PreprocessingConfig(BaseModel):
    """Complete preprocessing configuration for the Dendrite system."""

    model_config = ConfigDict(extra="forbid")

    preprocess_data: bool = Field(True, description="Enable preprocessing")
    target_sample_rate: int | None = Field(
        None, gt=0, description="Target sample rate in Hz (None = no resampling)"
    )
    downsample_factor: int = Field(
        1, ge=1, le=16, description="Global downsampling factor (legacy, prefer target_sample_rate)"
    )

    modality_preprocessing: dict[str, ModalityConfig] = Field(
        default_factory=dict, description="Preprocessing parameters per modality (EEG, EMG, etc)"
    )

    quality_control: QualityControlConfig = Field(
        default_factory=QualityControlConfig, description="Channel quality monitoring configuration"
    )
