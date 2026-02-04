"""
Modality-Agnostic Data Augmentation for Dendrite/Neural Signals

This module provides simple, general augmentation techniques that work with
any 2D signal data in (channels, times) format. Supports both single arrays
and dictionaries of arrays for multimodal data.
"""

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class BaseAugmentationStrategy(ABC):
    """Base class for all augmentation strategies with automatic dict/array handling."""

    def __init__(self, name: str):
        self.name = name

    def apply(self, data: np.ndarray | dict[str, np.ndarray]) -> np.ndarray | dict[str, np.ndarray]:
        """
        Apply augmentation to array or dictionary of arrays.

        Parameters:
        -----------
        data : np.ndarray or dict
            Single array (channels, times) or dict of arrays

        Returns:
        --------
        Augmented data in same format as input
        """
        if isinstance(data, dict):
            # Apply to each modality in dictionary
            return {key: self._apply_to_array(value) for key, value in data.items()}
        else:
            # Apply to single array
            return self._apply_to_array(data)

    @abstractmethod
    def _apply_to_array(self, sample: np.ndarray) -> np.ndarray:
        """Apply augmentation to a single array in (channels, times) format."""
        pass

    def __call__(
        self, data: np.ndarray | dict[str, np.ndarray]
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Make the strategy callable."""
        return self.apply(data)


class NoiseAugmentation(BaseAugmentationStrategy):
    """Add gaussian or uniform noise to signals."""

    def __init__(self, noise_type: str = "gaussian", intensity: float = 0.05):
        """
        Initialize noise augmentation.

        Parameters:
        -----------
        noise_type : str
            Type of noise ('gaussian' or 'uniform')
        intensity : float
            Noise intensity as fraction of signal std
        """
        super().__init__(f"noise_{noise_type}_{intensity}")
        self.noise_type = noise_type.lower()
        self.intensity = intensity

    def _apply_to_array(self, sample: np.ndarray) -> np.ndarray:
        """Add noise to array. Works with any dimensionality."""
        augmented = sample.copy().astype(np.float32)

        # Calculate noise level based on signal statistics
        noise_level = self.intensity * np.std(augmented)

        if self.noise_type == "gaussian":
            noise = np.random.normal(0, noise_level, augmented.shape)
        elif self.noise_type == "uniform":
            noise = np.random.uniform(-noise_level, noise_level, augmented.shape)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

        return augmented + noise


class AmplitudeAugmentation(BaseAugmentationStrategy):
    """Scale signal amplitude uniformly."""

    def __init__(self, scale_range: tuple[float, float] = (0.8, 1.2)):
        """
        Initialize amplitude scaling.

        Parameters:
        -----------
        scale_range : tuple
            Min and max scaling factors
        """
        super().__init__(f"amplitude_{scale_range[0]}_{scale_range[1]}")
        self.scale_range = scale_range

    def _apply_to_array(self, sample: np.ndarray) -> np.ndarray:
        """Apply amplitude scaling."""
        scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
        return sample * scale_factor


class MaskingAugmentation(BaseAugmentationStrategy):
    """Mask (zero out) random segments of the signal."""

    def __init__(self, mask_type: str = "time", mask_ratio: float = 0.1):
        """
        Initialize masking augmentation.

        Parameters:
        -----------
        mask_type : str
            Type of masking ('time', 'channel', or 'random')
        mask_ratio : float
            Fraction of data to mask
        """
        super().__init__(f"masking_{mask_type}_{mask_ratio}")
        self.mask_type = mask_type.lower()
        self.mask_ratio = mask_ratio

    def _apply_to_array(self, sample: np.ndarray) -> np.ndarray:
        """Apply masking to array."""
        augmented = sample.copy()

        if len(sample.shape) == 1:
            # 1D array - only time masking makes sense
            mask_length = int(len(sample) * self.mask_ratio)
            if mask_length > 0:
                mask_start = np.random.randint(0, max(1, len(sample) - mask_length))
                augmented[mask_start : mask_start + mask_length] = 0

        elif len(sample.shape) == 2:
            # 2D array (channels, times)
            n_channels, n_times = sample.shape

            if self.mask_type == "time":
                # Mask consecutive time points
                mask_length = int(n_times * self.mask_ratio)
                if mask_length > 0:
                    mask_start = np.random.randint(0, max(1, n_times - mask_length))
                    augmented[:, mask_start : mask_start + mask_length] = 0

            elif self.mask_type == "channel":
                # Mask entire channels
                n_mask = max(1, int(n_channels * self.mask_ratio))
                mask_indices = np.random.choice(n_channels, n_mask, replace=False)
                augmented[mask_indices, :] = 0

            elif self.mask_type == "random":
                # Random element masking
                mask = np.random.random(sample.shape) < self.mask_ratio
                augmented[mask] = 0

        return augmented


class DropoutAugmentation(BaseAugmentationStrategy):
    """Random dropout of signal values."""

    def __init__(self, dropout_rate: float = 0.1):
        """
        Initialize dropout augmentation.

        Parameters:
        -----------
        dropout_rate : float
            Fraction of values to drop
        """
        super().__init__(f"dropout_{dropout_rate}")
        self.dropout_rate = dropout_rate

    def _apply_to_array(self, sample: np.ndarray) -> np.ndarray:
        """Apply random dropout."""
        augmented = sample.copy()

        # Create dropout mask
        mask = np.random.random(sample.shape) < self.dropout_rate

        # Zero out dropped values
        augmented[mask] = 0

        # Scale remaining values to maintain expected value
        if self.dropout_rate < 1.0:
            augmented = augmented / (1.0 - self.dropout_rate)

        return augmented


class CompositeAugmentation(BaseAugmentationStrategy):
    """Combine multiple augmentation strategies."""

    def __init__(
        self,
        strategies: list[BaseAugmentationStrategy],
        apply_prob: float = 0.5,
        max_strategies: int = 2,
    ):
        """
        Initialize composite augmentation.

        Parameters:
        -----------
        strategies : list
            List of augmentation strategies to combine
        apply_prob : float
            Probability of applying each strategy
        max_strategies : int
            Maximum number of strategies to apply
        """
        super().__init__(f"composite_{len(strategies)}")
        self.strategies = strategies
        self.apply_prob = apply_prob
        self.max_strategies = max_strategies

    def _apply_to_array(self, sample: np.ndarray) -> np.ndarray:
        """Apply random combination of strategies."""
        augmented = sample.copy()

        # Randomly select which strategies to apply
        applied_count = 0
        for strategy in np.random.permutation(self.strategies):
            if applied_count >= self.max_strategies:
                break

            if np.random.random() < self.apply_prob:
                augmented = strategy._apply_to_array(augmented)
                applied_count += 1

        return augmented


class AugmentationManager:
    """Manager for organizing and applying augmentation strategies."""

    def __init__(self):
        self.strategies = {}
        self.presets = {}
        self._setup_default_strategies()

    def register_strategy(self, name: str, strategy: BaseAugmentationStrategy):
        """Register a new augmentation strategy."""
        self.strategies[name] = strategy
        logger.debug(f"Registered augmentation strategy: {name}")

    def get_strategy(self, name: str) -> BaseAugmentationStrategy | None:
        """Get a strategy by name."""
        return self.strategies.get(name) or self.presets.get(name)

    def _setup_default_strategies(self):
        """Setup default augmentation strategies and presets."""
        # Register individual strategies
        self.register_strategy("noise_light", NoiseAugmentation("gaussian", 0.03))
        self.register_strategy("noise_medium", NoiseAugmentation("gaussian", 0.05))
        self.register_strategy("noise_heavy", NoiseAugmentation("gaussian", 0.1))
        self.register_strategy("amplitude", AmplitudeAugmentation((0.8, 1.2)))
        self.register_strategy("masking_time", MaskingAugmentation("time", 0.1))
        self.register_strategy("masking_channel", MaskingAugmentation("channel", 0.1))
        self.register_strategy("dropout", DropoutAugmentation(0.1))

        # Create presets (3 levels of augmentation intensity)
        self.presets["light"] = CompositeAugmentation(
            [self.strategies["noise_light"], self.strategies["amplitude"]],
            apply_prob=0.5,
            max_strategies=1,
        )

        self.presets["moderate"] = CompositeAugmentation(
            [
                self.strategies["noise_medium"],
                self.strategies["amplitude"],
                self.strategies["masking_channel"],
            ],
            apply_prob=0.6,
            max_strategies=2,
        )

        self.presets["aggressive"] = CompositeAugmentation(
            [
                self.strategies["noise_heavy"],
                self.strategies["amplitude"],
                self.strategies["masking_time"],
                self.strategies["masking_channel"],
                self.strategies["dropout"],
            ],
            apply_prob=0.7,
            max_strategies=3,
        )

        # Backwards compatibility alias
        self.presets["conservative"] = self.presets["light"]

        logger.debug("Default augmentation strategies and presets initialized")

    def transform_batch(
        self, batch_data: dict[str, np.ndarray], strategy_name: str = "moderate", prob: float = 0.5
    ) -> dict[str, np.ndarray]:
        """
        Apply augmentation to a batch of data online (in-place transformation).

        This method applies augmentation stochastically to batches during training,
        providing better memory usage and variability compared to offline augmentation.

        Parameters:
        -----------
        batch_data : Dict[str, np.ndarray]
            Batch data with modality keys (e.g., {'eeg': batch_array})
            Each array has shape (batch_size, n_channels, n_times)
        strategy_name : str, default='moderate'
            Name of augmentation strategy or preset to apply
        prob : float, default=0.5
            Probability of applying augmentation to the batch

        Returns:
        --------
        Dict[str, np.ndarray]
            Transformed batch data (same structure as input)
        """
        # Stochastic application - only augment with given probability
        if np.random.random() > prob:
            return batch_data

        strategy = self.get_strategy(strategy_name)
        if strategy is None:
            logger.warning(
                f"Augmentation strategy '{strategy_name}' not found, returning original batch"
            )
            return batch_data

        # Apply augmentation to each modality in the batch
        transformed_batch = {}
        for modality, data in batch_data.items():
            if len(data) == 0:
                transformed_batch[modality] = data
                continue

            # Apply augmentation to each sample in the batch
            transformed_batch[modality] = np.array(
                [strategy._apply_to_array(sample) for sample in data]
            )

        return transformed_batch

    def transform_array(
        self, data: np.ndarray, strategy_name: str = "moderate", prob: float = 0.5
    ) -> np.ndarray:
        """
        Apply augmentation to a batch array directly.

        Parameters:
        -----------
        data : np.ndarray
            Batch data with shape (batch_size, n_channels, n_times)
        strategy_name : str, default='moderate'
            Name of augmentation strategy or preset to apply
        prob : float, default=0.5
            Probability of applying augmentation to the batch

        Returns:
        --------
        np.ndarray
            Transformed batch data (same shape as input)
        """
        # Stochastic application - only augment with given probability
        if np.random.random() > prob:
            return data

        strategy = self.get_strategy(strategy_name)
        if strategy is None:
            logger.warning(
                f"Augmentation strategy '{strategy_name}' not found, returning original data"
            )
            return data

        if len(data) == 0:
            return data

        return np.array([strategy._apply_to_array(sample) for sample in data])

    def list_available(self) -> dict[str, list[str]]:
        """List all available strategies and presets."""
        return {"strategies": list(self.strategies.keys()), "presets": list(self.presets.keys())}


# Global augmentation manager instance
augmentation_manager = AugmentationManager()


def get_augmentation_manager() -> AugmentationManager:
    """Get the global augmentation manager instance."""
    return augmentation_manager


def apply_mixup(
    x: np.ndarray, y: np.ndarray, alpha: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Apply mixup augmentation: blend pairs of samples.

    Returns mixed inputs, original targets, shuffled targets, and lambda.
    Loss = lam * loss(pred, y1) + (1-lam) * loss(pred, y2)
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.shape[0]
    indices = np.random.permutation(batch_size)

    x_mixed = lam * x + (1 - lam) * x[indices]
    return x_mixed, y, y[indices], lam


def apply_cutmix(
    x: np.ndarray, y: np.ndarray, alpha: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Apply cutmix augmentation: paste rectangular region from one sample to another.

    Better than mixup for preserving local temporal structure in EEG.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.shape[0]
    indices = np.random.permutation(batch_size)

    # Calculate cut boundaries (along time axis, last dimension)
    time_len = x.shape[-1]
    cut_len = int(time_len * (1 - lam))
    cut_start = np.random.randint(0, time_len - cut_len + 1) if cut_len < time_len else 0

    x_mixed = x.copy()
    x_mixed[..., cut_start : cut_start + cut_len] = x[indices, ..., cut_start : cut_start + cut_len]

    # Adjust lambda to actual mixed ratio
    lam_adj = 1 - cut_len / time_len
    return x_mixed, y, y[indices], lam_adj
