"""Study item representation for dataset management.

Provides capability detection for registered studies.
"""

from typing import TYPE_CHECKING

from dendrite.utils.logger_central import get_logger

if TYPE_CHECKING:
    from .config import DatasetConfig

logger = get_logger(__name__)


class StudyItem:
    """Represents a dataset for offline ML.

    Wraps a DatasetConfig and provides lazy loading of the appropriate
    data loader based on source type, plus capability detection.
    """

    CAPABILITY_FULL = "full"
    CAPABILITY_EPOCHS = "epochs_only"
    CAPABILITY_UNAVAILABLE = "unavailable"

    def __init__(self, config: "DatasetConfig", is_preset: bool = False, is_moabb: bool = False):
        self.config = config
        self.is_preset = is_preset
        self.is_moabb = is_moabb or config.source_type == "moabb"
        self._capability: str | None = None
        self._loader = None

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def loader(self):
        """Get appropriate loader based on source type."""
        if self._loader is None:
            if self.is_moabb:
                from .moabb_loader import MOAABLoader

                self._loader = MOAABLoader(self.config)
            else:
                raise NotImplementedError("Only MOABB datasets are supported")
        return self._loader

    def detect_capability(self) -> str:
        """Detect what functionality is available for this study.

        Returns:
            One of CAPABILITY_FULL, CAPABILITY_EPOCHS, or CAPABILITY_UNAVAILABLE
        """
        if self._capability is not None:
            return self._capability

        # MOABB datasets: assume full capability without loading data
        if self.is_moabb:
            self._capability = self.CAPABILITY_FULL
            return self._capability

        # Try continuous data first (full capability)
        try:
            if self.config.subjects:
                subject = self.config.subjects[0]
                block = None
                if self.config.blocks:
                    block_name = list(self.config.blocks.keys())[0]
                    block = (
                        self.config.blocks[block_name][0]
                        if self.config.blocks[block_name]
                        else None
                    )

                data, times, labels, _ = self.loader.load_continuous(subject, block=block)
                if len(times) > 0:
                    self._capability = self.CAPABILITY_FULL
                    return self._capability
        except Exception as e:
            logger.debug(f"Could not load continuous data for {self.name}: {e}")

        # Fall back to epochs-only
        try:
            if self.config.subjects:
                epochs = self.loader.load_epochs(self.config.subjects[0])
                if len(epochs) > 0:
                    self._capability = self.CAPABILITY_EPOCHS
                    return self._capability
        except Exception as e:
            logger.debug(f"Could not load epochs for {self.name}: {e}")

        self._capability = self.CAPABILITY_UNAVAILABLE
        return self._capability

    @property
    def capability(self) -> str:
        return self.detect_capability()
