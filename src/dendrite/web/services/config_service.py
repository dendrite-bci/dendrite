"""
Config Service

Holds all configuration state and aggregates it for pipeline startup.
"""

import json
import os
from typing import Any

from pydantic import ValidationError

from dendrite.constants import DEFAULT_STUDY_NAME, STUDIES_DIR, get_study_paths

DEFAULT_RECORDING_NAME = "recording"
from dendrite.processing.pipeline_schemas import PipelineConfig
from dendrite.utils.logger_central import get_logger


def _validation_error_to_str(e: ValidationError, prefix: str = "") -> str:
    """Format a Pydantic ValidationError as a semicolon-joined string."""
    return "; ".join(
        f"{prefix}{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
        for err in e.errors()
    )


class ConfigService:
    """Holds configuration state and builds pipeline config dicts."""

    def __init__(self, stream_service, mode_service):
        self.logger = get_logger("ConfigService")
        self._stream_service = stream_service
        self._mode_service = mode_service

        # General params (BIDS)
        self.study_name: str = DEFAULT_STUDY_NAME
        self.subject_id: str = ""
        self.session_id: str = ""
        self.recording_name: str = DEFAULT_RECORDING_NAME

        # Output protocols (LSL on by default)
        from dendrite.data.streaming.output_schemas import DEFAULT_LSL_CONFIG

        self.output: dict[str, Any] = {
            "lsl": {"enabled": True, "config": {**DEFAULT_LSL_CONFIG}},
        }

    # ------------------------------------------------------------------
    # Build full config for pipeline
    # ------------------------------------------------------------------

    def build_configuration(self) -> PipelineConfig:
        """Build the full validated config needed by PipelineService.start()."""
        sm = self._stream_service
        all_modes = self._mode_service.get_all_instances()

        return PipelineConfig(
            study_name=self.study_name,
            subject_id=self.subject_id,
            session_id=self.session_id,
            recording_name=self.recording_name,
            stream_configs=list(sm.get_streams().values()),
            modalities_by_stream=sm.get_modalities_by_stream(),
            mode_instances={
                name: cfg for name, cfg in all_modes.items()
                if cfg.get("enabled", True)
            },
            output=self.output,
        )

    # ------------------------------------------------------------------
    # General params
    # ------------------------------------------------------------------

    def get_general_config(self) -> dict[str, str]:
        return {
            "study_name": self.study_name,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "recording_name": self.recording_name,
        }

    def set_general_config(self, config: dict[str, str]) -> None:
        from dendrite.web.schemas import StudyConfig

        try:
            validated = StudyConfig(**config)
        except ValidationError as e:
            raise ValueError(_validation_error_to_str(e)) from e

        for field in StudyConfig.model_fields:
            if field in config:
                setattr(self, field, getattr(validated, field))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load_configuration(self, file_path: str) -> dict[str, Any]:
        """Load configuration from JSON and apply to services.

        Returns:
            Dict with 'config' (raw loaded data) and 'warnings' (list of
            issues encountered during restore).

        Raises:
            FileNotFoundError, json.JSONDecodeError.
        """
        from dendrite.web.schemas import StudyConfig

        with open(file_path) as f:
            cfg = json.load(f)

        warnings: list[str] = []

        # General
        general = {k: cfg[k] for k in StudyConfig.model_fields if k in cfg}
        if general:
            try:
                self.set_general_config(general)
            except ValueError as e:
                warnings.append(f"General config: {e}")

        # Output
        if "output" in cfg:
            self.output = cfg["output"]

        # Stream configs — restore directly (liveness will check availability)
        if "stream_configs" in cfg:
            self._stream_service.restore_from_config(cfg["stream_configs"])

        # Mode instances — validate all before clearing existing
        new_modes = cfg.get("mode_instances", {})
        valid_modes: list[tuple[str, dict]] = []
        for instance_name, instance_info in new_modes.items():
            config = {"name": instance_name, **instance_info}
            try:
                self._mode_service.validate_instance(config)
                valid_modes.append((instance_name, config))
            except ValueError as e:
                warnings.append(f"Mode '{instance_name}': {e}")

        if valid_modes or new_modes:
            self._mode_service.clear_all()
            for instance_name, config in valid_modes:
                self._mode_service.add_instance(instance_name, config)

        for w in warnings:
            self.logger.warning(f"Config load: {w}")
        self.logger.info(f"Configuration loaded from {file_path}")
        return {"config": cfg, "warnings": warnings}

    def save_configuration(self, file_path: str | None = None) -> str:
        """Save current configuration to JSON.

        Args:
            file_path: Path to save. If None, auto-generates in study config dir.

        Returns:
            Path where config was saved.
        """
        config = self.build_configuration()

        if file_path is None:
            config_dir = get_study_paths(self.study_name)["config"]
            os.makedirs(config_dir, exist_ok=True)
            file_path = str(config_dir / "config.json")

        with open(file_path, "w") as f:
            json.dump(config.model_dump(mode="json"), f, indent=2, default=str)

        self.logger.info(f"Configuration saved to {file_path}")
        return file_path

    def list_study_names(self) -> list[str]:
        """List all known study directory names on disk."""
        if not STUDIES_DIR.exists():
            return []
        return sorted(
            d.name for d in STUDIES_DIR.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    def list_configs(self) -> list[dict[str, Any]]:
        """List all saved config files across studies."""
        configs: list[dict[str, Any]] = []
        if not STUDIES_DIR.exists():
            return configs
        for study_dir in sorted(STUDIES_DIR.iterdir()):
            if not study_dir.is_dir():
                continue
            config_dir = study_dir / "config"
            if not config_dir.is_dir():
                continue
            for f in sorted(config_dir.glob("*.json")):
                try:
                    stat = f.stat()
                    configs.append({
                        "file_path": str(f),
                        "study_name": study_dir.name,
                        "file_name": f.name,
                        "modified": stat.st_mtime,
                        "size": stat.st_size,
                    })
                except OSError:
                    continue
        return configs
