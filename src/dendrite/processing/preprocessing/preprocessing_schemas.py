"""
Pydantic configuration schemas for preprocessing.

ModalityPreprocessing: per-modality preprocessing config (filters, resampling, runtime context).
PreprocessingConfig: stored in decoder metadata to record training conditions.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModalityPreprocessing(BaseModel):
    """Per-modality preprocessing config (filters, resampling, runtime context).

    Runtime fields (num_channels, sample_rate) are optional for storage
    but required when actually preprocessing.
    """

    model_config = ConfigDict(extra="forbid")

    # Filter params (user-configurable)
    lowcut: float | None = Field(None, ge=0)
    highcut: float | None = Field(None, gt=0)
    filter_order: int = Field(4, ge=1, le=10)
    apply_rereferencing: bool = False
    apply_eog_correction: bool = False  # regress EOG reference channels out of EEG
    eog_crossover_hz: float | None = Field(None, gt=0)  # ocular/preserved band split; default 6 Hz
    line_freq: float | None = Field(None, ge=0)
    target_sample_rate: float | None = Field(None, gt=0)
    downsample_factor: int | None = Field(None, ge=1)
    notch_width: float | None = Field(None, gt=0)
    channel_labels: list[str] | None = None

    # Runtime context (injected from stream/file metadata)
    num_channels: int | None = Field(None, ge=1)
    sample_rate: float | None = Field(None, gt=0)

    @model_validator(mode="after")
    def validate_frequencies(self):
        """Validate highcut > lowcut, and Nyquist check when sample_rate is present."""
        if self.lowcut is not None and self.highcut is not None:
            if self.highcut <= self.lowcut:
                raise ValueError(
                    f"Highcut ({self.highcut}Hz) must be greater than lowcut ({self.lowcut}Hz)"
                )
        if self.sample_rate is not None:
            nyquist = self.sample_rate / 2.0
            if self.highcut is not None and self.highcut >= nyquist:
                raise ValueError(
                    f"Highcut ({self.highcut}Hz) must be less than Nyquist "
                    f"({nyquist}Hz for {self.sample_rate}Hz sampling)"
                )
            if self.lowcut is not None and self.lowcut >= nyquist:
                raise ValueError(
                    f"Lowcut ({self.lowcut}Hz) must be less than Nyquist "
                    f"({nyquist}Hz for {self.sample_rate}Hz sampling)"
                )
            if self.eog_crossover_hz is not None and self.eog_crossover_hz >= nyquist:
                raise ValueError(
                    f"eog_crossover_hz ({self.eog_crossover_hz}Hz) must be less than Nyquist "
                    f"({nyquist}Hz for {self.sample_rate}Hz sampling)"
                )
        # The crossover splits [lowcut, crossover] | [crossover, highcut]; it must sit
        # strictly inside the passband or the high-band filter becomes degenerate.
        if self.eog_crossover_hz is not None:
            if self.lowcut is not None and self.eog_crossover_hz <= self.lowcut:
                raise ValueError(
                    f"eog_crossover_hz ({self.eog_crossover_hz}Hz) must be above "
                    f"lowcut ({self.lowcut}Hz)"
                )
            if self.highcut is not None and self.eog_crossover_hz >= self.highcut:
                raise ValueError(
                    f"eog_crossover_hz ({self.eog_crossover_hz}Hz) must be below "
                    f"highcut ({self.highcut}Hz)"
                )
        return self

    @classmethod
    def default_for(cls, modality: str) -> "ModalityPreprocessing":
        """Return default preprocessing config for a modality."""
        return cls(**MODALITY_DEFAULTS.get(modality.lower(), {}))

    @classmethod
    def from_user_config(
        cls, config: dict[str, Any], n_channels: int, sample_rate: float,
    ) -> "ModalityPreprocessing":
        """Build from user-facing dict (computes downsample_factor from target_rate)."""
        fields = {**config}
        target = fields.pop("target_sample_rate", None)
        if target and sample_rate > target and sample_rate % target == 0:
            fields["downsample_factor"] = int(sample_rate // target)
        fields["num_channels"] = n_channels
        fields["sample_rate"] = sample_rate
        return cls(**fields)


# Default configs per modality — single source of truth for GUI defaults
MODALITY_DEFAULTS: dict[str, dict[str, Any]] = {
    "eeg": {"lowcut": 0.5, "highcut": 50.0, "apply_rereferencing": True, "filter_order": 4},
    "emg": {"lowcut": 20.0, "highcut": 200.0, "line_freq": 50.0, "filter_order": 4},
    "eog": {"lowcut": 0.1, "highcut": 10.0, "filter_order": 2},
}

class PreprocessingConfig(BaseModel):
    """Preprocessing config stored in decoder metadata.

    Records what preprocessing was applied during training so inference
    can reproduce the same pipeline.
    """

    model_config = ConfigDict(extra="ignore")

    modality_preprocessing: dict[str, ModalityPreprocessing] = Field(
        default_factory=dict, description="Preprocessing parameters per modality (EEG, EMG, etc)"
    )
