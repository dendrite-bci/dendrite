"""Tests for mode factory functions in processing/modes/__init__.py."""

import pytest

from dendrite.processing.modes import create_mode, get_available_modes


class TestGetAvailableModes:
    def test_returns_list(self):
        result = get_available_modes()
        assert isinstance(result, list)

    def test_contains_all_modes(self):
        modes = get_available_modes()
        assert "synchronous" in modes
        assert "asynchronous" in modes
        assert "neurofeedback" in modes

    def test_length_is_three(self):
        assert len(get_available_modes()) == 3


class TestCreateMode:
    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode type"):
            create_mode("bogus")

    def test_case_insensitive(self):
        # create_mode lowercases the input before lookup — just verify no KeyError
        # We can't fully instantiate without queue args, so we test the ValueError path
        modes = get_available_modes()
        for mode in modes:
            # Uppercase should not raise ValueError about unknown mode
            with pytest.raises(TypeError):
                # Will raise TypeError for missing __init__ args, not ValueError
                create_mode(mode.upper())
