"""Decoder storage module - save/load trained decoders.

Ported from v1 trainer/backend.py with simplified API.
"""

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any

import joblib
import numpy as np

from .config import TrainResult

logger = logging.getLogger(__name__)


class DecoderStorage:
    """Stores and retrieves trained decoders."""

    def __init__(self, storage_dir: str):
        """Initialize storage.

        Args:
            storage_dir: Directory for storing decoders
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

        # In-memory cache of loaded decoders
        self._cache: dict[str, TrainResult] = {}

    def save(self, result: TrainResult, name: str) -> str:
        """Save a trained decoder.

        Args:
            result: TrainResult containing decoder and metadata
            name: Human-readable name for the decoder

        Returns:
            Unique decoder ID
        """
        # Generate unique ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_suffix = str(uuid.uuid4())[:8]
        decoder_id = f"{name}_{timestamp}_{unique_suffix}"

        # Save model (joblib - matches decoder.py pattern)
        model_path = os.path.join(self.storage_dir, f"{decoder_id}.pkl")
        joblib.dump(result.decoder, model_path)

        # Save metadata (JSON)
        metadata = self._build_metadata(result, name, decoder_id, model_path)
        metadata_path = os.path.join(self.storage_dir, f"{decoder_id}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=self._json_serializer)

        # Cache in memory
        self._cache[decoder_id] = result

        logger.info(f"Saved decoder: {decoder_id}")
        return decoder_id

    def load(self, decoder_id: str) -> TrainResult:
        """Load a decoder by ID.

        Args:
            decoder_id: Unique decoder ID

        Returns:
            TrainResult with loaded decoder

        Raises:
            KeyError: If decoder not found
        """
        # Check cache first
        if decoder_id in self._cache:
            return self._cache[decoder_id]

        # Load from disk
        model_path = os.path.join(self.storage_dir, f"{decoder_id}.pkl")
        metadata_path = os.path.join(self.storage_dir, f"{decoder_id}_metadata.json")

        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            raise KeyError(f"Decoder not found: {decoder_id}")

        decoder = joblib.load(model_path)

        with open(metadata_path) as f:
            metadata = json.load(f)

        # Reconstruct TrainResult
        result = TrainResult(
            decoder=decoder,
            accuracy=metadata.get("accuracy", 0.0),
            val_accuracy=metadata.get("val_accuracy", 0.0),
            confusion_matrix=np.array(metadata.get("confusion_matrix", [])),
            train_history=metadata.get("train_history", {}),
            cv_results=metadata.get("cv_results"),
        )

        self._cache[decoder_id] = result
        return result

    def list_decoders(self) -> list[dict[str, Any]]:
        """List all saved decoders with metadata.

        Returns:
            List of decoder info dictionaries
        """
        decoders = []

        for filename in os.listdir(self.storage_dir):
            if filename.endswith("_metadata.json"):
                filepath = os.path.join(self.storage_dir, filename)
                try:
                    with open(filepath) as f:
                        metadata = json.load(f)
                    decoders.append(
                        {
                            "id": metadata.get("decoder_id"),
                            "name": metadata.get("name"),
                            "accuracy": metadata.get("accuracy"),
                            "val_accuracy": metadata.get("val_accuracy"),
                            "model_type": metadata.get("model_type"),
                            "timestamp": metadata.get("timestamp"),
                            "cv_mean_accuracy": metadata.get("cv_mean_accuracy"),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to read metadata {filename}: {e}")

        return decoders

    def delete(self, decoder_id: str) -> bool:
        """Delete a decoder.

        Args:
            decoder_id: Unique decoder ID

        Returns:
            True if deleted, False if not found
        """
        model_path = os.path.join(self.storage_dir, f"{decoder_id}.pkl")
        metadata_path = os.path.join(self.storage_dir, f"{decoder_id}_metadata.json")

        deleted = False

        if os.path.exists(model_path):
            os.remove(model_path)
            deleted = True

        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            deleted = True

        # Remove from cache
        self._cache.pop(decoder_id, None)

        if deleted:
            logger.info(f"Deleted decoder: {decoder_id}")

        return deleted

    def _build_metadata(
        self, result: TrainResult, name: str, decoder_id: str, model_path: str
    ) -> dict[str, Any]:
        """Build metadata dictionary for saving."""
        return {
            "decoder_id": decoder_id,
            "name": name,
            "model_path": model_path,
            "accuracy": result.accuracy,
            "val_accuracy": result.val_accuracy,
            "confusion_matrix": result.confusion_matrix.tolist()
            if isinstance(result.confusion_matrix, np.ndarray)
            else result.confusion_matrix,
            "train_history": result.train_history,
            "cv_results": result.cv_results,
            "cv_mean_accuracy": result.cv_results.get("mean_accuracy")
            if result.cv_results
            else None,
            "timestamp": datetime.now().isoformat(),
        }

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
