import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from dendrite.data.quality import EpochQualityChecker
from dendrite.utils.logger_central import get_logger


class Dataset:
    """
    Multimodal dataset for Dendrite with source tracking and quality checking.

    Core responsibilities:
    - Store and manage multimodal samples
    - Provide training data with optional augmentation
    - Track data sources and quality metrics
    - Handle HDF5 serialization
    """

    def __init__(
        self,
        name: str = "Dendrite_Dataset",
        max_samples: int | None = None,
        sample_rate: float | None = None,
    ) -> None:
        """
        Initialize dataset.

        Args:
            name: Dataset name
            max_samples: Maximum samples to keep (FIFO if exceeded)
            sample_rate: Sampling rate in Hz (needed for quality checking)
        """
        self.name = name
        self.max_samples = max_samples
        self.sample_rate = sample_rate

        # Core data storage
        self.samples = []  # [{'eeg': array, 'label': int, 'source': str}, ...]
        self.sources = {}  # {'source_id': {'info': dict, 'samples': count}}
        self.reference_shapes = {}  # Expected shapes per modality
        self.classes = set()  # Available class labels

        # Quality checking
        self.quality_checker = EpochQualityChecker()

        self.logger = get_logger()
        self.logger.info(f"Dataset '{name}' initialized")

    def add_sample(
        self,
        data_dict: dict[str, Any],
        label: int,
        source_id: str = "default",
        source_info: dict[str, Any] | None = None,
        input_shape: str = "(n_channels, n_times)",
    ) -> bool:
        """
        Add a single sample to the dataset.

        Args:
            data_dict: Data by modality, e.g., {'eeg': array, 'emg': array}
            label: Class label
            source_id: Source identifier
            source_info: Additional source information
            input_shape: Data format specification using tensor notation

        Returns:
            True if sample was added successfully, False otherwise
        """
        try:
            # Validate data (type/shape only)
            if not self._validate_sample(data_dict):
                return False

            # Auto-flag bad epochs via quality checker
            is_bad, reason = self.quality_checker.check(data_dict)
            if is_bad:
                self.logger.debug(f"Epoch flagged as bad: {reason} (label={label})")

            # Create sample with converted data
            sample = {
                "label": int(label),
                "source": source_id,
                "is_bad": is_bad,
                **{
                    k: v.astype(np.float32) if isinstance(v, np.ndarray) else v
                    for k, v in data_dict.items()
                },
            }

            self.samples.append(sample)
            self.classes.add(int(label))

            # Update source tracking with input_shape information
            enhanced_source_info = source_info.copy() if source_info else {}
            enhanced_source_info["input_shape"] = input_shape
            self._update_source_tracking(source_id, enhanced_source_info)

            # Enforce size limit (FIFO)
            self._enforce_size_limit()

            return True

        except Exception as e:
            self.logger.error(f"Error adding sample: {e}")
            return False

    def get_training_data(self, modality: str = "eeg", exclude_bad: bool = False) -> dict[str, Any]:
        """
        Get training data for model training.

        Note: Augmentation is now handled at the classifier level.

        Args:
            modality: Modality to extract (e.g., 'eeg', 'emg')
            exclude_bad: Whether to exclude bad quality samples

        Returns:
            Training data package with X (array), y, and metadata
        """
        # Get filtered samples
        filtered_samples = [
            sample for sample in self.samples if not exclude_bad or not sample.get("is_bad", False)
        ]

        if not filtered_samples:
            return self._empty_result(modality)

        # Filter samples with requested modality
        valid_samples = [s for s in filtered_samples if modality in s]
        if not valid_samples:
            return self._empty_result(modality)

        # Return array directly (not dict)
        X = np.stack([s[modality] for s in valid_samples])
        y = np.array([s["label"] for s in valid_samples])
        input_shape = self.reference_shapes.get(modality)

        result = {"X": X, "y": y, "num_classes": len(self.classes), "input_shape": input_shape}

        # Log quality summary
        if exclude_bad:
            self.logger.info(f"Training data: {self.quality_checker.get_stats_summary()}")

        return result

    def _empty_result(self, modality: str) -> dict[str, Any]:
        """Return empty result structure."""
        return {
            "X": np.array([]),
            "y": np.array([]),
            "num_classes": len(self.classes),
            "input_shape": self.reference_shapes.get(modality),
        }

    def _validate_sample(self, data_dict: dict[str, Any]) -> bool:
        """Validate sample type and shape (quality checks done separately)."""
        for modality, data in data_dict.items():
            if not isinstance(data, np.ndarray):
                self.logger.error(f"Data for '{modality}' must be numpy array")
                return False

            # Set/check reference shape
            if modality in self.reference_shapes:
                if data.shape != self.reference_shapes[modality]:
                    self.logger.error(
                        f"Shape mismatch for '{modality}': {data.shape} vs {self.reference_shapes[modality]}"
                    )
                    return False
            else:
                self.reference_shapes[modality] = data.shape

        return True

    def _update_source_tracking(self, source_id: str, source_info: dict[str, Any] | None) -> None:
        """Update source tracking information."""
        if source_id not in self.sources:
            self.sources[source_id] = {"info": {}, "samples": 0}

        if source_info:
            self.sources[source_id]["info"].update(source_info)

        self.sources[source_id]["samples"] += 1

    def _enforce_size_limit(self) -> None:
        """Enforce maximum samples limit (FIFO)."""
        if self.max_samples and len(self.samples) > self.max_samples:
            removed = self.samples.pop(0)
            self.sources[removed["source"]]["samples"] -= 1
            # Recalculate classes
            self.classes = set(s["label"] for s in self.samples)

    def get_data(self, modality: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Get raw data arrays for a specific modality.

        Args:
            modality: The modality to extract data for

        Returns:
            Tuple of (X data array, y labels array)
        """
        # Filter samples that contain the requested modality
        valid_samples = [s for s in self.samples if modality in s]

        if not valid_samples:
            return np.array([]), np.array([])

        # Extract data and labels
        X = np.stack([s[modality] for s in valid_samples])
        y = np.array([s["label"] for s in valid_samples])

        return X, y

    def get_info(self) -> dict[str, Any]:
        """Get dataset summary."""
        return {
            "name": self.name,
            "total_samples": len(self.samples),
            "sources": len(self.sources),
            "modalities": list(self.reference_shapes.keys()),
            "classes": sorted(list(self.classes)),
            "max_samples": self.max_samples,
            "sample_rate": self.sample_rate,
        }

    def save_dataset(self, filepath: str | Path) -> None:
        """
        Save dataset to HDF5 format.

        Args:
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if not filepath.suffix:
            filepath = filepath.with_suffix(".h5")

        with h5py.File(filepath, "w") as f:
            self._save_metadata(f)
            self._save_data(f)

        self.logger.info(f"Dataset saved to {filepath}")

    def _save_metadata(self, h5file: h5py.File) -> None:
        """Save metadata to HDF5 file."""
        meta = h5file.create_group("metadata")
        meta.attrs["name"] = self.name
        meta.attrs["total_samples"] = len(self.samples)
        meta.attrs["classes"] = list(self.classes)
        meta.attrs["max_samples"] = self.max_samples or -1
        meta.attrs["sample_rate"] = self.sample_rate or -1

        # Reference shapes
        shapes = meta.create_group("reference_shapes")
        for modality, shape in self.reference_shapes.items():
            shapes.attrs[modality] = shape

        # Sources info
        sources = meta.create_group("sources")
        for source_id, source_info in self.sources.items():
            src = sources.create_group(source_id)
            src.attrs["samples"] = source_info["samples"]
            for key, value in source_info["info"].items():
                try:
                    # Handle special complex types
                    if key in [
                        "event_mapping",
                        "label_mapping",
                        "reverse_label_mapping",
                        "channel_selection",
                    ]:
                        if isinstance(value, dict):
                            # Save dict as JSON string for complex structures
                            src.attrs[key] = json.dumps(value)
                        else:
                            src.attrs[key] = str(value)
                    else:
                        src.attrs[key] = value
                except TypeError:
                    src.attrs[key] = str(value)

    def _save_data(self, h5file: h5py.File) -> None:
        """Save data to HDF5 file."""
        data_group = h5file.create_group("data")

        for modality in self.reference_shapes.keys():
            X, y = self.get_data(modality)
            if len(X) > 0:
                data_group.create_dataset(f"{modality}_X", data=X, compression="gzip")
                data_group.create_dataset(f"{modality}_y", data=y, compression="gzip")

                # Store source and quality info
                sources_data = [s["source"] for s in self.samples if modality in s]
                data_group.create_dataset(
                    f"{modality}_sources",
                    data=[s.encode() for s in sources_data],
                    compression="gzip",
                )

                quality_data = [s.get("is_bad", False) for s in self.samples if modality in s]
                data_group.create_dataset(
                    f"{modality}_quality_bad", data=quality_data, compression="gzip"
                )

    @classmethod
    def load_dataset(cls, filepath: str | Path) -> "Dataset":
        """
        Load dataset from HDF5 file.

        Args:
            filepath: Path to dataset file

        Returns:
            Loaded dataset instance
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        with h5py.File(filepath, "r") as f:
            dataset = cls._load_metadata(f)
            cls._load_data(f, dataset)

        return dataset

    @classmethod
    def _load_metadata(cls, h5file: h5py.File) -> "Dataset":
        """Load metadata from HDF5 file."""
        meta = h5file["metadata"]

        # Create dataset instance
        max_samples = meta.attrs.get("max_samples", -1)
        max_samples = None if max_samples == -1 else max_samples
        sample_rate = meta.attrs.get("sample_rate", -1)
        sample_rate = None if sample_rate == -1 else sample_rate

        dataset = cls(name=meta.attrs["name"], max_samples=max_samples, sample_rate=sample_rate)
        dataset.classes = set(meta.attrs["classes"])

        shapes = meta["reference_shapes"]
        for modality in shapes.attrs.keys():
            dataset.reference_shapes[modality] = tuple(shapes.attrs[modality])

        sources = meta["sources"]
        for source_id in sources.keys():
            src = sources[source_id]
            source_info = {}

            for k, v in src.attrs.items():
                if k == "samples":
                    continue

                # Handle JSON-encoded dict fields
                if k in [
                    "event_mapping",
                    "label_mapping",
                    "reverse_label_mapping",
                    "channel_selection",
                ] and isinstance(v, str):
                    try:
                        source_info[k] = json.loads(v)
                    except (json.JSONDecodeError, TypeError):
                        source_info[k] = v
                else:
                    source_info[k] = v

            dataset.sources[source_id] = {"info": source_info, "samples": src.attrs["samples"]}

        return dataset

    @classmethod
    def _load_data(cls, h5file: h5py.File, dataset: "Dataset") -> None:
        """Load data from HDF5 file."""
        data_group = h5file["data"]
        if not dataset.reference_shapes:
            return

        # Get first modality to determine sample count
        first_modality = list(dataset.reference_shapes.keys())[0]
        if f"{first_modality}_X" not in data_group:
            return

        y = data_group[f"{first_modality}_y"][:]
        sources_data = [s.decode() for s in data_group[f"{first_modality}_sources"][:]]
        quality_flags = data_group.get(f"{first_modality}_quality_bad", [])

        # Reconstruct samples
        for i in range(len(y)):
            sample_data = {}
            for modality in dataset.reference_shapes.keys():
                if f"{modality}_X" in data_group:
                    X = data_group[f"{modality}_X"][:]
                    sample_data[modality] = X[i]

            if sample_data:
                dataset.add_sample(sample_data, y[i], sources_data[i])
                # Add quality flag if available
                if i < len(quality_flags):
                    dataset.samples[-1]["is_bad"] = bool(quality_flags[i])

    def clear(self) -> None:
        """Clear all data."""
        self.samples.clear()
        self.sources.clear()
        self.reference_shapes.clear()
        self.classes.clear()
        self.logger.info(f"Dataset '{self.name}' cleared")

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        return (
            f"Dataset(name='{self.name}', samples={len(self.samples)}, sources={len(self.sources)})"
        )
