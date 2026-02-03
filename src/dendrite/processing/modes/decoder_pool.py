"""Decoder pool for multi-decoder training and search."""

import time
from typing import Any

from dendrite.ml.decoders import create_decoder


class DecoderPool:
    """Manages multiple decoders for parallel training and comparison."""

    def __init__(self, base_config: dict[str, Any], search_configs: list[dict[str, Any]], logger):
        self.logger = logger
        self.base_config = base_config
        self.search_configs = search_configs

        # Storage for decoders and their performances
        self.decoders: dict[str, Any] = {}  # {decoder_id: decoder_instance}
        self.performances: dict[str, float] = {}  # {decoder_id: accuracy}
        self.best_decoder_id: str | None = None

        # Training statistics
        self.training_count = 0

        self.logger.info(
            f"DecoderPool initialized with {len(search_configs)} decoder configurations"
        )

    def _create_decoder_id(self, index: int, config: dict[str, Any]) -> str:
        """Generate unique decoder ID from config parameters."""
        model_type = config.get("model_type", "unknown")
        lr = config.get("learning_rate", 0.0)
        return f"{model_type}_lr{lr:.0e}_#{index}"

    def _merge_configs(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Merge override config into base config (deep merge for nested dicts)."""
        merged = base.copy()

        # Handle nested model_config if present
        if "model_config" in override:
            if "model_config" not in merged:
                merged["model_config"] = {}
            merged["model_config"].update(override["model_config"])

        # Update top-level parameters
        for key, value in override.items():
            if key != "model_config":
                merged[key] = value

        return merged

    def initialize_decoders(
        self,
        num_classes: int,
        input_shapes: dict[str, tuple[int, ...]],
        event_mapping: dict[int, str],
        label_mapping: dict[str, int],
    ) -> None:
        """Create all decoder instances with detected input shapes."""
        self.decoders = {}
        self.performances = {}

        for idx, search_config in enumerate(self.search_configs):
            # Merge search config into base config
            decoder_config = self._merge_configs(self.base_config, search_config)

            # Add runtime parameters
            decoder_config["num_classes"] = num_classes
            decoder_config["input_shapes"] = input_shapes
            decoder_config["event_mapping"] = event_mapping
            decoder_config["label_mapping"] = label_mapping

            try:
                decoder = create_decoder(**decoder_config)
                decoder_id = self._create_decoder_id(idx, decoder_config)

                self.decoders[decoder_id] = decoder
                self.performances[decoder_id] = 0.0

                self.logger.info(f"Created decoder: {decoder_id}")
            except Exception as e:
                self.logger.error(f"Failed to create decoder {idx}: {e}")

        self.logger.info(f"DecoderPool: {len(self.decoders)} decoders ready")

    def train_all(self, X, y: Any, lock: Any) -> dict[str, float]:
        """Train all decoders and return {decoder_id: accuracy}."""
        self.training_count += 1
        start_time = time.time()

        # Extract array from dict if needed (decoder expects array directly)
        if isinstance(X, dict):
            primary_key = next(iter(X))
            X_array = X[primary_key]
        else:
            X_array = X

        for decoder_id, decoder in self.decoders.items():
            try:
                # Train decoder with lock for thread safety (fit returns self)
                with lock:
                    decoder.fit(X_array, y)

                # Extract validation accuracy (prefer val over train)
                training_metrics = decoder.get_training_metrics() or {}
                accuracy = next(
                    (
                        m.get("final_val_acc", m.get("final_train_acc", 0.0))
                        for m in training_metrics.values()
                        if isinstance(m, dict)
                    ),
                    0.0,
                )
                self.performances[decoder_id] = accuracy

            except (ValueError, RuntimeError) as e:
                self.logger.error(f"Training failed for {decoder_id}: {e}")
                self.performances[decoder_id] = 0.0
            except Exception as e:
                self.logger.error(f"Unexpected error training {decoder_id}: {e}", exc_info=True)
                self.performances[decoder_id] = 0.0

        # Update best decoder
        if self.performances:
            self.best_decoder_id = max(self.performances, key=self.performances.get)

        elapsed = time.time() - start_time
        self.logger.info(
            f"DecoderPool training #{self.training_count} completed in {elapsed:.1f}s | "
            f"Best: {self.best_decoder_id} ({self.performances.get(self.best_decoder_id, 0.0):.3f})"
        )

        return self.performances.copy()

    def get_best_decoder(self) -> tuple[str | None, Any | None]:
        """Get best decoder as (decoder_id, decoder_instance) or (None, None)."""
        if not self.best_decoder_id or self.best_decoder_id not in self.decoders:
            return None, None

        return self.best_decoder_id, self.decoders[self.best_decoder_id]
