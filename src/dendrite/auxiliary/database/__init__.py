"""
Dendrite Database Explorer

GUI for browsing and managing Dendrite recording database.
Provides search, filtering, and data lineage tracking.
"""

from .app import DBExplorer, main

__all__ = ["DBExplorer", "main"]
