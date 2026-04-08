"""
Dendrite - Open-source platform for real-time neural signal processing and brain-computer interfaces.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dendrite")
except PackageNotFoundError:
    __version__ = "unknown"
