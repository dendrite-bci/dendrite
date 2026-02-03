"""
Central Mode Configuration Manager

This service acts as the single source of truth for all mode instance configurations
in the Dendrite GUI. It manages mode instance configurations and provides reactive updates
to all dependent components.

Key Features:
- Single source of truth for mode instance configurations
- Reactive updates via Qt signals
- Centralized mode instance management
- Consistent API for all mode-related consumers
"""

import copy
from contextlib import contextmanager
from typing import Any

from PyQt6 import QtCore

from dendrite.utils.logger_central import get_logger


class ModeConfigManager(QtCore.QObject):
    """
    Central manager for mode instance configurations.

    Stores mode instance configurations and provides reactive updates.
    """

    # Signals for reactive state management
    instance_updated = QtCore.pyqtSignal(str, dict)  # instance_name, config
    instance_added = QtCore.pyqtSignal(str, dict)  # instance_name, config
    instance_removed = QtCore.pyqtSignal(str)  # instance_name
    instance_renamed = QtCore.pyqtSignal(str, str)  # old_name, new_name
    instances_cleared = QtCore.pyqtSignal()  # for bulk clear

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = get_logger("ModeConfigManager")

        # Core data storage - mode instance configurations
        self._instances: dict[str, dict[str, Any]] = {}
        self._updating: bool = False  # Guard against recursion

        self.logger.info("ModeConfigManager initialized")

    @contextmanager
    def _update_context(self):
        """Context manager for update operations to prevent recursion."""
        if self._updating:
            yield
            return

        self._updating = True
        try:
            yield
        finally:
            self._updating = False

    def _check_instance_exists(self, instance_name: str) -> bool:
        """Check if instance exists, log warning if not."""
        if instance_name not in self._instances:
            self.logger.warning(f"Instance '{instance_name}' not found")
            return False
        return True

    def add_instance(self, instance_name: str, config: dict[str, Any]) -> bool:
        """Add a new mode instance configuration."""
        if self._updating or instance_name in self._instances:
            if instance_name in self._instances:
                self.logger.warning(f"Instance '{instance_name}' already exists")
            return False

        with self._update_context():
            self.logger.info(f"Adding mode instance: {instance_name}")
            self._instances[instance_name] = config
            self.instance_added.emit(instance_name, copy.deepcopy(config))
            return True

    def update_instance(self, instance_name: str, config: dict[str, Any]) -> bool:
        """Update an existing mode instance configuration."""
        if not self._check_instance_exists(instance_name):
            return False

        with self._update_context():
            self.logger.info(f"Updating mode instance: {instance_name}")
            self._instances[instance_name] = config
            self.instance_updated.emit(instance_name, copy.deepcopy(config))
            return True

    def remove_instance(self, instance_name: str) -> bool:
        """
        Remove a mode instance configuration.

        Args:
            instance_name: Name of the instance to remove

        Returns:
            bool: True if successfully removed, False if instance doesn't exist
        """
        if not self._check_instance_exists(instance_name):
            return False

        with self._update_context():
            self.logger.info(f"Removing mode instance: {instance_name}")
            del self._instances[instance_name]
            self.instance_removed.emit(instance_name)
            return True

    def rename_instance(self, old_name: str, new_name: str) -> bool:
        """
        Rename a mode instance.

        Args:
            old_name: Current instance name
            new_name: New instance name

        Returns:
            bool: True if successfully renamed, False otherwise
        """
        if not self._check_instance_exists(old_name):
            return False

        if new_name in self._instances:
            self.logger.warning(f"Instance '{new_name}' already exists")
            return False

        with self._update_context():
            self.logger.info(f"Renaming mode instance: {old_name} -> {new_name}")
            config = self._instances.pop(old_name)
            config["name"] = new_name
            self._instances[new_name] = config
            self.instance_renamed.emit(old_name, new_name)
            return True

    def get_instance(self, instance_name: str) -> dict[str, Any] | None:
        """Get a specific instance configuration by name."""
        config = self._instances.get(instance_name)
        return copy.deepcopy(config) if config else None

    def has_instance(self, instance_name: str) -> bool:
        """Check if an instance exists."""
        return instance_name in self._instances

    def generate_unique_name(
        self, base_name: str, exclude_name: str | None = None, sanitize: bool = False
    ) -> str:
        """Generate a unique instance name from a base name.

        Args:
            base_name: Base name to generate unique name from
            exclude_name: Name to exclude from collision check (for edit mode)
            sanitize: If True, converts base_name to TitleCase format
        """
        if sanitize:
            base_name = base_name.replace("_", " ").title().replace(" ", "")

        existing = set(self._instances.keys())
        if exclude_name:
            existing.discard(exclude_name)

        # Try base name first (without suffix)
        if base_name not in existing:
            return base_name

        # Find next available suffix
        counter = 1
        while f"{base_name}_{counter}" in existing:
            counter += 1
        return f"{base_name}_{counter}"

    def get_all_instance_names(self) -> list[str]:
        """Get list of all instance names."""
        return list(self._instances.keys())

    def get_all_instances(self) -> dict[str, dict[str, Any]]:
        """Get deep copy of all instances."""
        return copy.deepcopy(self._instances)

    def update_channel_selection(
        self, instance_name: str, channel_selection: dict[str, list[int]]
    ) -> bool:
        """Update channel selection for a specific instance."""
        if not self._check_instance_exists(instance_name):
            return False

        with self._update_context():
            self._instances[instance_name]["channel_selection"] = channel_selection
            self.instance_updated.emit(instance_name, copy.deepcopy(self._instances[instance_name]))
            return True

    def clear_all(self) -> None:
        """Clear all mode instance configurations."""
        with self._update_context():
            self.logger.info("Clearing all mode instances")
            self._instances.clear()
            self.instances_cleared.emit()


# Global instance
_manager = None


def get_mode_config_manager() -> ModeConfigManager:
    """Get the global mode configuration manager instance."""
    global _manager
    if _manager is None:
        _manager = ModeConfigManager()
    return _manager


def initialize_mode_config_manager(parent=None) -> ModeConfigManager:
    """Initialize the global mode configuration manager."""
    global _manager
    if _manager is not None:
        _manager.deleteLater()
    _manager = ModeConfigManager(parent)
    return _manager
