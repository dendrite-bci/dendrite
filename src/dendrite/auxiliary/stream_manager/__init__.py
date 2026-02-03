"""
Dendrite Stream Manager

Launches LSL streams from files or synthetic generators.
Enables development and testing without hardware.
"""

from .app import StreamManager, main

__all__ = ["StreamManager", "main"]
