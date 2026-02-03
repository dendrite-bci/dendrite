"""
Dendrite Visualization Dashboard

Real-time visualization and monitoring for Dendrite system.
Subscribes to LSL visualization streams and displays mode outputs.
"""

from .app import MainWindow, main

__all__ = ["MainWindow", "main"]
