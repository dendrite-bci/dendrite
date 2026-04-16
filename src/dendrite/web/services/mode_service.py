"""
Mode Service

Mode instance CRUD management.
"""

import copy
from typing import Any

from dendrite.processing.modes.mode_schemas import validate_mode_config
from dendrite.utils.logger_central import get_logger


class ModeService:
    """Manages mode instance configurations."""

    def __init__(self, stream_service=None):
        self.logger = get_logger("ModeService")
        self._stream_service = stream_service
        self._instances: dict[str, dict[str, Any]] = {}

    def validate_instance(self, config: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize a mode config via Pydantic schemas.

        Returns the validated (normalized) dict with defaults filled in.
        Raises ValueError if validation fails.
        """
        is_valid, errors, validated = validate_mode_config(config)
        if not is_valid or validated is None:
            raise ValueError("; ".join(errors))
        return validated

    def add_instance(self, instance_name: str, config: dict[str, Any]) -> bool:
        if instance_name in self._instances:
            self.logger.warning(f"Instance '{instance_name}' already exists")
            return False

        validated = self.validate_instance(config)
        self._auto_disable_on_decoder_error(instance_name, validated)
        self.logger.info(f"Adding mode instance: {instance_name}")
        self._instances[instance_name] = validated
        return True

    def update_instance(self, instance_name: str, config: dict[str, Any]) -> bool:
        if instance_name not in self._instances:
            self.logger.warning(f"Instance '{instance_name}' not found")
            return False

        validated = self.validate_instance(config)
        self._auto_disable_on_decoder_error(instance_name, validated)
        self.logger.info(f"Updating mode instance: {instance_name}")
        self._instances[instance_name] = validated
        return True

    def remove_instance(self, instance_name: str) -> bool:
        if instance_name not in self._instances:
            self.logger.warning(f"Instance '{instance_name}' not found")
            return False

        self.logger.info(f"Removing mode instance: {instance_name}")
        del self._instances[instance_name]
        return True

    def rename_instance(self, old_name: str, new_name: str) -> bool:
        if old_name not in self._instances:
            self.logger.warning(f"Instance '{old_name}' not found")
            return False
        if new_name in self._instances:
            self.logger.warning(f"Instance '{new_name}' already exists")
            return False

        self.logger.info(f"Renaming mode instance: {old_name} -> {new_name}")
        config = self._instances.pop(old_name)
        config["name"] = new_name
        self._instances[new_name] = config
        return True

    def get_instance(self, instance_name: str) -> dict[str, Any] | None:
        config = self._instances.get(instance_name)
        return copy.deepcopy(config) if config else None

    def has_instance(self, instance_name: str) -> bool:
        return instance_name in self._instances

    def get_all_instances(self) -> dict[str, dict[str, Any]]:
        return copy.deepcopy(self._instances)

    def get_all_instance_names(self) -> list[str]:
        return list(self._instances.keys())

    def generate_unique_name(
        self, base_name: str, exclude_name: str | None = None, sanitize: bool = False
    ) -> str:
        if sanitize:
            base_name = base_name.replace("_", " ").title().replace(" ", "")

        existing = set(self._instances.keys())
        if exclude_name:
            existing.discard(exclude_name)

        if base_name not in existing:
            return base_name

        counter = 1
        while f"{base_name}_{counter}" in existing:
            counter += 1
        return f"{base_name}_{counter}"

    def _auto_disable_on_decoder_error(
        self, instance_name: str, config: dict[str, Any]
    ) -> None:
        """Auto-disable async modes with incompatible decoders.

        Reuses validate_mode_config() with stream_context for decoder checks.
        """
        if config.get("mode") != "asynchronous":
            return
        if config.get("decoder_source") not in ("database", "online"):
            return
        if not config.get("decoder_config", {}).get("decoder_path"):
            return

        stream_context = self._build_stream_context(config)
        if not stream_context:
            return

        _, errors, _ = validate_mode_config(config, stream_context)
        if errors:
            config["enabled"] = False
            config["_disable_reason"] = "; ".join(errors)
            self.logger.warning(
                f"Auto-disabled '{instance_name}': {'; '.join(errors)}"
            )

    def _build_stream_context(self, mode_config: dict[str, Any]) -> dict[str, Any] | None:
        """Build stream context for decoder validation using mode's configured modality."""
        if not self._stream_service or not self._stream_service.has_streams():
            return None
        # Match stream by mode's selected modality
        required = list(mode_config.get("channel_selection", {}).keys())
        if not required:
            return None
        for stream in self._stream_service.get_streams().values():
            if stream.type.lower() in required and stream.sample_rate:
                return {"sample_rate": stream.sample_rate}
        return None

    def clear_all(self) -> None:
        self.logger.info("Clearing all mode instances")
        self._instances.clear()
