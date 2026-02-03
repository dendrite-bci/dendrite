"""
Dendrite Decoder Module

This module contains the main decoder implementation for brain-machine interface applications.

The Decoder class supports both neural network and classical ML pipelines for
EEG signal classification with an sklearn-compatible interface.
"""

import json
import os
from typing import Any

import joblib
import torch
from pydantic import ValidationError

from dendrite.ml.decoders.decoder import Decoder
from dendrite.ml.decoders.decoder_schemas import DecoderConfig
from dendrite.ml.decoders.registry import (
    check_decoder_compatibility,
    get_available_decoders,
    get_decoder_capabilities,
)
from dendrite.utils.logger_central import get_logger

logger = get_logger()

__all__ = [
    "create_decoder",
    "load_decoder",
    "get_available_decoders",
    "get_decoder_capabilities",
    "check_decoder_compatibility",
    "validate_decoder_file",
    "get_decoder_metadata",
    "Decoder",
    "DecoderConfig",
]


def create_decoder(model_type: str = "EEGNet", **kwargs):
    """Create a Decoder instance using the complete decoder configuration.

    Args:
        model_type: Model architecture name ('EEGNet', 'TransformerEEG', 'LinearEEG', etc.).
        **kwargs: Additional decoder config including num_classes, input_shapes,
            event_mapping, label_mapping, epochs, learning_rate, etc.

    Returns:
        Configured Decoder instance ready for training.

    Example:
        ```python
        decoder = create_decoder(
            model_type='EEGNet',
            num_classes=3,
            input_shapes={'eeg': (32, 250)},
            event_mapping={1: 'left', 2: 'right'},
            label_mapping={'left': 0, 'right': 1}
        )
        ```
    """
    config = DecoderConfig(model_type=model_type, **kwargs)
    logger.info(f"Creating decoder with model_type={model_type}")
    return Decoder(config)


def load_decoder(decoder_path: str) -> Decoder:
    """Load a pre-trained decoder from a saved decoder file.

    Args:
        decoder_path: Path to the saved decoder JSON metadata file.

    Returns:
        Decoder instance with loaded weights/parameters ready for inference.

    Raises:
        FileNotFoundError: If decoder file doesn't exist.
        RuntimeError: If loading fails.
    """
    if not decoder_path.endswith(".json"):
        decoder_path = f"{decoder_path}.json"

    if not os.path.exists(decoder_path):
        raise FileNotFoundError(f"Decoder file not found: {decoder_path}")

    try:
        logger.info(f"Loading decoder from: {decoder_path}")

        with open(decoder_path) as f:
            metadata = json.load(f)

        config = DecoderConfig(**metadata)
        decoder = Decoder(config)

        # Restore essential attributes from config
        decoder.num_classes = config.num_classes or 2
        decoder.input_shapes = config.input_shapes
        decoder.event_mapping = config.event_mapping or {}
        decoder.label_mapping = config.label_mapping or {}
        decoder.sample_rate = config.sample_rate or 500.0
        decoder.target_sample_rate = config.target_sample_rate  # None = no resampling

        base_path = decoder_path.rsplit(".json", 1)[0]
        pt_path = f"{base_path}.pt"
        joblib_path = f"{base_path}.joblib"

        if os.path.exists(pt_path):
            # Neural: load pipeline and model from .pt
            data = torch.load(pt_path, map_location="cpu", weights_only=False)
            decoder.pipeline = data["pipeline"]
            classifier = decoder.pipeline.named_steps["classifier"]

            # Restore model
            classifier.model = classifier._create_model(data["input_shape"])
            classifier.model.load_state_dict(data["model_state_dict"])
            classifier.model.to(classifier.device)
            classifier.model.eval()
            classifier.input_shape = data["input_shape"]
            classifier.classes_ = data["classes_"]
            classifier.is_fitted = True

        elif os.path.exists(joblib_path):
            # Classical: load from .joblib
            decoder.pipeline = joblib.load(joblib_path)
        else:
            raise RuntimeError(f"No pipeline file found: {pt_path} or {joblib_path}")

        decoder._is_fitted = True
        logger.info(f"Loaded decoder successfully: {config.model_type}")
        return decoder

    except (OSError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load decoder from {decoder_path}: {e}")
        raise RuntimeError(f"Failed to load decoder: {e}") from e


def validate_decoder_file(
    filepath: str,
    expected_shapes: dict[str, tuple[int, int]] | None = None,
    expected_sample_rate: float | None = None,
    expected_labels: dict[str, list[str]] | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate decoder file with detailed error reporting.

    Args:
        filepath: Path to decoder JSON file or base path
        expected_shapes: Expected input shapes per modality: {'eeg': (channels, samples), ...}
        expected_sample_rate: Expected sampling rate in Hz
        expected_labels: Expected channel labels per modality: {'EEG': ['Fp1', 'Fp2', ...], ...}

    Returns:
        Tuple of (metadata_dict or None, list of validation issues)
    """
    json_path = filepath if filepath.endswith(".json") else f"{filepath}.json"

    if not os.path.exists(json_path):
        return None, [f"Decoder file not found: {json_path}"]

    issues = []

    # Load and parse JSON
    try:
        with open(json_path) as f:
            metadata_dict = json.load(f)
    except json.JSONDecodeError as e:
        return None, [f"Invalid JSON format: {e!s}"]
    except OSError as e:
        return None, [f"Error reading file: {e!s}"]

    # Validate with Pydantic schema
    try:
        validated_config = DecoderConfig(**metadata_dict)
    except ValidationError as e:
        for error in e.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            issues.append(f"{field}: {error['msg']}")
        return None, issues
    except (TypeError, ValueError) as e:
        return None, [f"Validation error: {e!s}"]

    # Check compatibility with system configuration
    if expected_shapes:
        compatibility_issues = validated_config.check_compatibility(
            system_shapes=expected_shapes,
            system_labels=expected_labels,
            system_sample_rate=expected_sample_rate,
        )
        # Filter out sample rate warnings (not errors, just info)
        for issue in compatibility_issues:
            if "resampling may be needed" not in issue:
                issues.append(issue)

    # Return validated metadata as dict with resampling info
    decoder_rate = validated_config.sample_rate or 500.0
    result = validated_config.model_dump(exclude_none=True)
    result["_system_sample_rate"] = expected_sample_rate
    result["_needs_resampling"] = (
        expected_sample_rate and decoder_rate and abs(decoder_rate - expected_sample_rate) > 0.1
    )
    return result, issues


def get_decoder_metadata(filepath: str) -> dict[str, Any]:
    """Get metadata from saved decoder file for inspection without loading.

    Use this to inspect decoder properties (model type, input shapes, class
    mappings, training info) without loading the full model weights.

    Args:
        filepath: Path to saved decoder JSON file.

    Returns:
        Decoder metadata including model_type, input_shapes, num_classes,
        event_mapping, label_mapping, and training configuration.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If metadata validation fails.
    """
    metadata, issues = validate_decoder_file(filepath)

    if metadata is None:
        raise FileNotFoundError(issues[0] if issues else "File not found")

    if issues:
        raise ValueError(f"Metadata validation failed: {issues[0]}")

    return metadata
