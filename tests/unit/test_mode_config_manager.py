"""
Tests for ModeConfigManager instance management behavior.

Tests ensure mode instances are properly managed with correct validation
for duplicates, existence checks, renaming, and unique name generation.
"""

import pytest
from dendrite.gui.config.mode_config_manager import ModeConfigManager


class TestModeInstanceManagement:
    """Test basic CRUD operations for mode instances."""

    def test_add_instance_success(self):
        """Adding a new instance should succeed and store config."""
        manager = ModeConfigManager()
        config = {'mode': 'Synchronous', 'decoder': 'EEGNet'}

        result = manager.add_instance('MyMode', config)

        assert result is True
        assert manager.has_instance('MyMode')
        stored = manager.get_instance('MyMode')
        assert stored['mode'] == 'Synchronous'
        assert stored['decoder'] == 'EEGNet'

    def test_add_instance_duplicate_rejected(self):
        """Adding an instance with existing name should fail."""
        manager = ModeConfigManager()
        manager.add_instance('MyMode', {'mode': 'Synchronous'})

        result = manager.add_instance('MyMode', {'mode': 'Asynchronous'})

        assert result is False
        # Original config unchanged
        assert manager.get_instance('MyMode')['mode'] == 'Synchronous'

    def test_update_instance_success(self):
        """Updating an existing instance should succeed."""
        manager = ModeConfigManager()
        manager.add_instance('MyMode', {'mode': 'Synchronous'})

        result = manager.update_instance('MyMode', {'mode': 'Asynchronous'})

        assert result is True
        assert manager.get_instance('MyMode')['mode'] == 'Asynchronous'

    def test_update_nonexistent_instance_fails(self):
        """Updating a non-existent instance should fail."""
        manager = ModeConfigManager()

        result = manager.update_instance('NonExistent', {'mode': 'Sync'})

        assert result is False

    def test_remove_instance_success(self):
        """Removing an existing instance should succeed."""
        manager = ModeConfigManager()
        manager.add_instance('MyMode', {'mode': 'Synchronous'})

        result = manager.remove_instance('MyMode')

        assert result is True
        assert not manager.has_instance('MyMode')

    def test_remove_nonexistent_instance_fails(self):
        """Removing a non-existent instance should fail."""
        manager = ModeConfigManager()

        result = manager.remove_instance('NonExistent')

        assert result is False

    def test_get_instance_returns_deep_copy(self):
        """get_instance should return a deep copy, not a reference."""
        manager = ModeConfigManager()
        manager.add_instance('MyMode', {'mode': 'Synchronous', 'nested': {'key': 'value'}})

        retrieved = manager.get_instance('MyMode')
        retrieved['mode'] = 'Modified'
        retrieved['nested']['key'] = 'modified'

        # Original should be unchanged
        original = manager.get_instance('MyMode')
        assert original['mode'] == 'Synchronous'
        assert original['nested']['key'] == 'value'

    def test_get_all_instances_returns_deep_copy(self):
        """get_all_instances should return deep copies."""
        manager = ModeConfigManager()
        manager.add_instance('Mode1', {'mode': 'Synchronous'})
        manager.add_instance('Mode2', {'mode': 'Asynchronous'})

        all_instances = manager.get_all_instances()
        all_instances['Mode1']['mode'] = 'Modified'

        # Original should be unchanged
        assert manager.get_instance('Mode1')['mode'] == 'Synchronous'

    def test_clear_all_removes_all_instances(self):
        """clear_all should remove all instances."""
        manager = ModeConfigManager()
        manager.add_instance('Mode1', {'mode': 'Synchronous'})
        manager.add_instance('Mode2', {'mode': 'Asynchronous'})

        manager.clear_all()

        assert len(manager.get_all_instance_names()) == 0


class TestRenaming:
    """Test instance renaming behavior."""

    def test_rename_instance_success(self):
        """Renaming an instance should move config to new name."""
        manager = ModeConfigManager()
        manager.add_instance('OldName', {'mode': 'Synchronous'})

        result = manager.rename_instance('OldName', 'NewName')

        assert result is True
        assert not manager.has_instance('OldName')
        assert manager.has_instance('NewName')
        assert manager.get_instance('NewName')['mode'] == 'Synchronous'

    def test_rename_to_existing_name_fails(self):
        """Renaming to an existing name should fail."""
        manager = ModeConfigManager()
        manager.add_instance('Mode1', {'mode': 'Synchronous'})
        manager.add_instance('Mode2', {'mode': 'Asynchronous'})

        result = manager.rename_instance('Mode1', 'Mode2')

        assert result is False
        # Both instances should still exist with original names
        assert manager.has_instance('Mode1')
        assert manager.has_instance('Mode2')

    def test_rename_updates_config_name_field(self):
        """Rename should update the 'name' field in config."""
        manager = ModeConfigManager()
        manager.add_instance('OldName', {'name': 'OldName', 'mode': 'Synchronous'})

        manager.rename_instance('OldName', 'NewName')

        config = manager.get_instance('NewName')
        assert config['name'] == 'NewName'

    def test_rename_nonexistent_instance_fails(self):
        """Renaming a non-existent instance should fail."""
        manager = ModeConfigManager()

        result = manager.rename_instance('NonExistent', 'NewName')

        assert result is False


class TestUniqueNameGeneration:
    """Test unique name generation behavior."""

    def test_generate_unique_name_no_collision(self):
        """When no collision, return base name as-is."""
        manager = ModeConfigManager()

        name = manager.generate_unique_name('MyMode')

        assert name == 'MyMode'

    def test_generate_unique_name_with_collision(self):
        """When name exists, append _1, _2, etc."""
        manager = ModeConfigManager()
        manager.add_instance('MyMode', {})

        name = manager.generate_unique_name('MyMode')

        assert name == 'MyMode_1'

    def test_generate_unique_name_multiple_collisions(self):
        """When multiple names exist, find next available suffix."""
        manager = ModeConfigManager()
        manager.add_instance('MyMode', {})
        manager.add_instance('MyMode_1', {})
        manager.add_instance('MyMode_2', {})

        name = manager.generate_unique_name('MyMode')

        assert name == 'MyMode_3'

    def test_generate_unique_name_with_exclude(self):
        """exclude_name should be ignored in collision check (for edit mode)."""
        manager = ModeConfigManager()
        manager.add_instance('MyMode', {})

        # When editing 'MyMode', exclude it from collision check
        name = manager.generate_unique_name('MyMode', exclude_name='MyMode')

        assert name == 'MyMode'  # No collision since we exclude it

    def test_generate_unique_name_sanitize(self):
        """sanitize=True should convert to TitleCase format."""
        manager = ModeConfigManager()

        name = manager.generate_unique_name('synchronous_mode', sanitize=True)

        assert name == 'SynchronousMode'

    def test_generate_unique_name_sanitize_with_collision(self):
        """Sanitize should work with collision detection."""
        manager = ModeConfigManager()
        manager.add_instance('SynchronousMode', {})

        name = manager.generate_unique_name('synchronous_mode', sanitize=True)

        assert name == 'SynchronousMode_1'


class TestChannelSelection:
    """Test channel selection updates."""

    def test_update_channel_selection_success(self):
        """Updating channel selection for existing instance should succeed."""
        manager = ModeConfigManager()
        manager.add_instance('MyMode', {'mode': 'Synchronous'})

        channel_selection = {'eeg': [0, 1, 2], 'emg': [0, 1]}
        result = manager.update_channel_selection('MyMode', channel_selection)

        assert result is True
        stored = manager.get_instance('MyMode')
        assert stored['channel_selection'] == channel_selection

    def test_update_channel_selection_nonexistent_fails(self):
        """Updating channel selection for non-existent instance should fail."""
        manager = ModeConfigManager()

        result = manager.update_channel_selection('NonExistent', {'eeg': [0, 1]})

        assert result is False
