"""
System Configuration Manager for Dendrite GUI

Aggregates configuration from GUI widgets into dictionaries for persistence and runtime use.
Acts as a translator between GUI state and JSON configuration files.
"""

import json
import logging
import os

from PyQt6 import QtWidgets

from dendrite.constants import (
    DEFAULT_RECORDING_NAME,
    DEFAULT_STUDY_NAME,
    MODE_ASYNCHRONOUS,
    MODE_SYNCHRONOUS,
    get_study_paths,
)

logger = logging.getLogger(__name__)


class SystemConfigurationManager:
    """Aggregates GUI state into configuration dictionaries and handles persistence."""

    def __init__(self, main_window):
        self.main_window = main_window

    def load_configuration(self, file_path: str | None = None) -> bool:
        """
        Load configuration from JSON file and apply to GUI widgets.

        Args:
            file_path: Path to configuration JSON file. If None, prompts user.

        Returns:
            True if configuration loaded successfully, False otherwise.
        """
        if not file_path:
            try:
                general_config = self.main_window.general_params_widget.get_general_config()
                study_name = general_config.get("study_name", DEFAULT_STUDY_NAME)
            except AttributeError:
                study_name = DEFAULT_STUDY_NAME

            default_dir = str(get_study_paths(study_name)["config"])
            os.makedirs(default_dir, exist_ok=True)

            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.main_window,
                "Load Configuration",
                default_dir,
                "JSON Files (*.json);;All Files (*)",
            )
            if not file_path:
                return False

        try:
            with open(file_path) as f:
                cfg = json.load(f)

            # Apply to widgets (optional widgets may not exist)
            try:
                if general := {
                    k: cfg[k]
                    for k in ["study_name", "subject_id", "session_id", "recording_name"]
                    if k in cfg
                }:
                    self.main_window.general_params_widget.set_general_config(general)
            except AttributeError:
                pass

            try:
                if preprocessing := {
                    k: cfg[k]
                    for k in [
                        "preprocess_data",
                        "modality_preprocessing",
                        "quality_control",
                    ]
                    if k in cfg
                }:
                    self.main_window.preprocessing_widget.set_preprocessing_config(preprocessing)
            except AttributeError:
                pass

            if "output" in cfg:
                self.main_window.output_widget.load_output_configuration(cfg["output"])

            if "stream_configs" in cfg:
                self.main_window.stream_config_manager.loaded_stream_configs = cfg["stream_configs"]
                logger.info(f"Loaded {len(cfg['stream_configs'])} stream configurations")

            # Clear and reload mode instances via manager
            self.main_window.mode_config_manager.clear_all()

            mode_instances = cfg.get("mode_instances", {})
            for instance_name, instance_info in mode_instances.items():
                config = {"name": instance_name, **instance_info}
                self.main_window.mode_config_manager.add_instance(instance_name, config)

            logger.info(
                f"Configuration loaded from {file_path} - {len(mode_instances)} mode instance(s)"
            )
            return True

        except (json.JSONDecodeError, FileNotFoundError, OSError, KeyError) as e:
            logger.error(f"Failed to load config: {e}", exc_info=True)
            QtWidgets.QMessageBox.warning(
                self.main_window, "Configuration Error", f"Failed to load configuration:\n{e!s}"
            )
            return False

    def build_configuration(self) -> dict:
        """
        Build serializable configuration dictionary from current GUI state.

        Returns:
            Dictionary with serializable configuration for persistence.
            Runtime objects (queues, events) are NOT included - they should be
            added at the call site when starting the pipeline.
        """
        mw = self.main_window
        sm = mw.stream_config_manager

        config = {}

        # General settings (sample_rate may be None if no EEG streams configured)
        config["sample_rate"] = sm.get_system_sample_rate() if sm.has_streams() else None

        try:
            general = mw.general_params_widget.get_general_config()
            config["study_name"] = general.get("study_name", DEFAULT_STUDY_NAME)
            config["subject_id"] = general.get("subject_id", "")
            config["session_id"] = general.get("session_id", "")
            config["recording_name"] = general.get("recording_name", DEFAULT_RECORDING_NAME)
        except AttributeError:
            config["study_name"] = DEFAULT_STUDY_NAME
            config["subject_id"] = ""
            config["session_id"] = ""
            config["recording_name"] = DEFAULT_RECORDING_NAME

        # Stream configuration
        streams = sm.get_streams()
        modality_data = sm.get_modality_data()

        config["stream_configs"] = list(streams.values())
        config["modality_data"] = modality_data  # Used by GUI for channel selection

        # Preprocessing settings
        try:
            preprocessing = mw.preprocessing_widget.get_preprocessing_config()
            config["preprocess_data"] = preprocessing.get("preprocess_data", False)
            config["modality_preprocessing"] = preprocessing.get("modality_preprocessing", {})
            config["quality_control"] = preprocessing.get("quality_control", {"enabled": False})
        except AttributeError:
            config["preprocess_data"] = False
            config["modality_preprocessing"] = {}
            config["quality_control"] = {"enabled": False}

        # Mode instances and model sharing
        self._add_mode_instances(config)
        self._add_model_sharing(config)

        # Output configuration
        config["output"] = mw.output_widget.get_output_configuration()

        return config

    def _add_mode_instances(self, config: dict) -> None:
        """Add mode instance configurations from enabled badges."""
        mode_instances = {}

        for badge in self.main_window.mode_instance_badges:
            if not badge.is_enabled():
                continue

            instance_name = badge.instance_data.get("name", "").strip()
            if not instance_name:
                logger.warning("Skipping unnamed mode instance badge")
                continue

            instance = badge.instance_data.copy()
            instance["mode"] = instance.get("mode", "Synchronous")
            mode_instances[instance_name] = instance

        config["mode_instances"] = mode_instances
        logger.debug(f"Added {len(mode_instances)} mode instance(s) to configuration")

    def _add_model_sharing(self, config: dict) -> None:
        """Add model sharing configuration between sync and async modes."""
        modes = config["mode_instances"]
        sync_modes = [n for n, c in modes.items() if c.get("mode", "").lower() == MODE_SYNCHRONOUS]
        sharing = {}

        for name, instance in modes.items():
            if instance.get("mode", "").lower() != MODE_ASYNCHRONOUS:
                continue
            if instance.get("decoder_source") != "sync_mode":
                continue

            source = instance.get("source_sync_mode", "")
            if source == "Any Synchronous Mode":
                sync_mode = sync_modes[0] if sync_modes else None
            elif source.startswith("Instance: "):
                actual = source.replace("Instance: ", "")
                sync_mode = (
                    actual
                    if actual in modes and modes[actual].get("mode", "").lower() == MODE_SYNCHRONOUS
                    else None
                )
            else:
                sync_mode = (
                    source
                    if source in modes and modes[source].get("mode", "").lower() == MODE_SYNCHRONOUS
                    else None
                )

            if sync_mode:
                instance["source_sync_mode"] = sync_mode
                sharing.setdefault(sync_mode, []).append(name)

        config["sync_to_async_sharing"] = sharing
