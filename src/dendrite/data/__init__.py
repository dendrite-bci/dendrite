"""
Dendrite Data Module

Import classes directly from submodules:
- `from dendrite.data.acquisition import DataAcquisition`
- `from dendrite.data.storage.data_saver import DataSaver`
- `from dendrite.data.stream_schemas import StreamConfig, StreamMetadata`
"""

from .event_outlet import EventOutlet

__all__ = ["EventOutlet"]
