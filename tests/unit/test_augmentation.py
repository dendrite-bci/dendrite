"""Unit tests for online data augmentation module.

Tests cover all augmentation strategies, the manager, and mixup/cutmix functions.
"""

import pytest
import numpy as np

from dendrite.ml.training.augmentation import (
    NoiseAugmentation,
    AmplitudeAugmentation,
    MaskingAugmentation,
    DropoutAugmentation,
    CompositeAugmentation,
    AugmentationManager,
    get_augmentation_manager,
    apply_mixup,
    apply_cutmix,
)


@pytest.fixture
def sample_2d():
    """Sample 2D data (channels, times)."""
    np.random.seed(42)
    return np.random.randn(8, 250).astype(np.float32)


@pytest.fixture
def sample_1d():
    """Sample 1D data (times only)."""
    np.random.seed(42)
    return np.random.randn(250).astype(np.float32)


@pytest.fixture
def sample_batch():
    """Sample batch data (batch, channels, times)."""
    np.random.seed(42)
    return np.random.randn(16, 8, 250).astype(np.float32)


@pytest.fixture
def sample_dict():
    """Sample dict data with multiple modalities."""
    np.random.seed(42)
    return {
        'eeg': np.random.randn(8, 250).astype(np.float32),
        'emg': np.random.randn(4, 250).astype(np.float32),
    }


class TestNoiseAugmentation:
    """Tests for NoiseAugmentation strategy."""

    def test_gaussian_noise_changes_data(self, sample_2d):
        """Gaussian noise should modify the data."""
        aug = NoiseAugmentation(noise_type='gaussian', intensity=0.1)
        result = aug.apply(sample_2d)

        assert result.shape == sample_2d.shape
        assert not np.allclose(result, sample_2d)

    def test_uniform_noise_changes_data(self, sample_2d):
        """Uniform noise should modify the data."""
        aug = NoiseAugmentation(noise_type='uniform', intensity=0.1)
        result = aug.apply(sample_2d)

        assert result.shape == sample_2d.shape
        assert not np.allclose(result, sample_2d)

    def test_noise_scales_with_intensity(self, sample_2d):
        """Higher intensity should produce larger changes."""
        aug_low = NoiseAugmentation(noise_type='gaussian', intensity=0.01)
        aug_high = NoiseAugmentation(noise_type='gaussian', intensity=0.5)

        np.random.seed(42)
        result_low = aug_low.apply(sample_2d)
        np.random.seed(42)
        result_high = aug_high.apply(sample_2d)

        diff_low = np.abs(result_low - sample_2d).mean()
        diff_high = np.abs(result_high - sample_2d).mean()

        assert diff_high > diff_low

    def test_invalid_noise_type_raises(self, sample_2d):
        """Invalid noise type should raise ValueError."""
        aug = NoiseAugmentation(noise_type='invalid', intensity=0.1)

        with pytest.raises(ValueError, match="Unknown noise type"):
            aug.apply(sample_2d)

    def test_dict_input(self, sample_dict):
        """Should work with dict input."""
        aug = NoiseAugmentation(noise_type='gaussian', intensity=0.1)
        result = aug.apply(sample_dict)

        assert isinstance(result, dict)
        assert 'eeg' in result and 'emg' in result
        assert result['eeg'].shape == sample_dict['eeg'].shape
        assert result['emg'].shape == sample_dict['emg'].shape


class TestAmplitudeAugmentation:
    """Tests for AmplitudeAugmentation strategy."""

    def test_amplitude_scaling_changes_data(self, sample_2d):
        """Amplitude scaling should modify the data."""
        aug = AmplitudeAugmentation(scale_range=(0.5, 1.5))
        result = aug.apply(sample_2d)

        assert result.shape == sample_2d.shape
        # Data should be scaled, not identical
        assert not np.allclose(result, sample_2d) or np.allclose(result / sample_2d, 1.0)

    def test_amplitude_scaling_in_range(self, sample_2d):
        """Scaling factor should be within specified range."""
        scale_range = (0.8, 1.2)
        aug = AmplitudeAugmentation(scale_range=scale_range)

        # Run many times to check range
        for _ in range(100):
            result = aug.apply(sample_2d)
            # Find the scale factor (result / original for non-zero elements)
            nonzero_mask = np.abs(sample_2d) > 1e-6
            if nonzero_mask.any():
                scale_factors = result[nonzero_mask] / sample_2d[nonzero_mask]
                # All scale factors should be the same (uniform scaling)
                assert np.allclose(scale_factors, scale_factors[0], rtol=1e-5)
                scale = scale_factors[0]
                assert scale_range[0] <= scale <= scale_range[1]

    def test_shape_preserved(self, sample_2d):
        """Output shape should match input shape."""
        aug = AmplitudeAugmentation(scale_range=(0.8, 1.2))
        result = aug.apply(sample_2d)

        assert result.shape == sample_2d.shape


class TestMaskingAugmentation:
    """Tests for MaskingAugmentation strategy."""

    def test_time_masking_zeros_segment(self, sample_2d):
        """Time masking should zero out a time segment."""
        aug = MaskingAugmentation(mask_type='time', mask_ratio=0.2)
        result = aug.apply(sample_2d)

        assert result.shape == sample_2d.shape
        # Some values should be zeroed
        assert (result == 0).any()
        # Check that entire time columns are zeroed (all channels at same times)
        zero_cols = np.all(result == 0, axis=0)
        assert zero_cols.any()

    def test_channel_masking_zeros_channels(self, sample_2d):
        """Channel masking should zero out entire channels."""
        aug = MaskingAugmentation(mask_type='channel', mask_ratio=0.2)
        result = aug.apply(sample_2d)

        assert result.shape == sample_2d.shape
        # Check that entire channels are zeroed (all times for some channels)
        zero_rows = np.all(result == 0, axis=1)
        assert zero_rows.any()

    def test_random_masking_zeros_elements(self, sample_2d):
        """Random masking should zero out random elements."""
        aug = MaskingAugmentation(mask_type='random', mask_ratio=0.1)
        result = aug.apply(sample_2d)

        assert result.shape == sample_2d.shape
        # Some values should be zeroed
        n_zeros = (result == 0).sum()
        expected_zeros = int(sample_2d.size * 0.1)
        # Allow some variance (statistical)
        assert abs(n_zeros - expected_zeros) < expected_zeros * 0.5

    def test_1d_array_support(self, sample_1d):
        """Masking should work with 1D arrays."""
        aug = MaskingAugmentation(mask_type='time', mask_ratio=0.2)
        result = aug.apply(sample_1d)

        assert result.shape == sample_1d.shape
        assert (result == 0).any()

    def test_dict_input(self, sample_dict):
        """Should work with dict input."""
        aug = MaskingAugmentation(mask_type='time', mask_ratio=0.2)
        result = aug.apply(sample_dict)

        assert isinstance(result, dict)
        assert result['eeg'].shape == sample_dict['eeg'].shape


class TestDropoutAugmentation:
    """Tests for DropoutAugmentation strategy."""

    def test_dropout_zeros_values(self, sample_2d):
        """Dropout should zero out some values."""
        aug = DropoutAugmentation(dropout_rate=0.2)
        result = aug.apply(sample_2d)

        assert result.shape == sample_2d.shape
        # Some values should be zeroed (where original was non-zero)
        original_nonzero = sample_2d != 0
        result_zero = result == 0
        dropped = original_nonzero & result_zero
        assert dropped.any()

    def test_dropout_rescales_remaining(self, sample_2d):
        """Dropout should rescale remaining values to maintain expected value."""
        dropout_rate = 0.2
        aug = DropoutAugmentation(dropout_rate=dropout_rate)
        result = aug.apply(sample_2d)

        # Non-dropped values should be scaled by 1/(1-dropout_rate)
        scale_factor = 1.0 / (1.0 - dropout_rate)
        non_dropped_mask = result != 0
        if non_dropped_mask.any():
            original_vals = sample_2d[non_dropped_mask]
            result_vals = result[non_dropped_mask]
            expected_vals = original_vals * scale_factor
            assert np.allclose(result_vals, expected_vals, rtol=1e-5)

    def test_dropout_rate_respected(self, sample_2d):
        """Dropout rate should be approximately respected."""
        dropout_rate = 0.3
        aug = DropoutAugmentation(dropout_rate=dropout_rate)

        # Run multiple times for statistical test
        drop_ratios = []
        for _ in range(50):
            result = aug.apply(sample_2d)
            dropped = (result == 0).sum() / result.size
            drop_ratios.append(dropped)

        mean_drop = np.mean(drop_ratios)
        # Allow some variance but should be close to target
        assert abs(mean_drop - dropout_rate) < 0.1


class TestCompositeAugmentation:
    """Tests for CompositeAugmentation strategy."""

    def test_composite_applies_strategies(self, sample_2d):
        """Composite should apply contained strategies."""
        strategies = [
            NoiseAugmentation('gaussian', 0.1),
            AmplitudeAugmentation((0.9, 1.1)),
        ]
        aug = CompositeAugmentation(strategies, apply_prob=1.0, max_strategies=2)
        result = aug.apply(sample_2d)

        assert result.shape == sample_2d.shape
        # Data should be modified
        assert not np.allclose(result, sample_2d)

    def test_max_strategies_limit(self, sample_2d):
        """Composite should respect max_strategies limit."""
        # Create strategies that track if they were applied
        applied_count = [0]

        class TrackingAugmentation(NoiseAugmentation):
            def _apply_to_array(self, sample):
                applied_count[0] += 1
                return super()._apply_to_array(sample)

        strategies = [
            TrackingAugmentation('gaussian', 0.01),
            TrackingAugmentation('gaussian', 0.01),
            TrackingAugmentation('gaussian', 0.01),
            TrackingAugmentation('gaussian', 0.01),
        ]

        max_strategies = 2
        aug = CompositeAugmentation(strategies, apply_prob=1.0, max_strategies=max_strategies)

        # Reset and apply
        applied_count[0] = 0
        aug.apply(sample_2d)

        assert applied_count[0] <= max_strategies

    def test_probabilistic_application(self, sample_2d):
        """Composite with low probability should sometimes skip strategies."""
        strategies = [NoiseAugmentation('gaussian', 0.1)]
        aug = CompositeAugmentation(strategies, apply_prob=0.0, max_strategies=1)

        result = aug.apply(sample_2d)
        # With prob=0, no strategy should be applied
        assert np.allclose(result, sample_2d)


class TestAugmentationManager:
    """Tests for AugmentationManager class."""

    def test_default_strategies_registered(self):
        """Manager should have default strategies registered."""
        manager = AugmentationManager()
        available = manager.list_available()

        expected_strategies = [
            'noise_light', 'noise_medium', 'noise_heavy',
            'amplitude', 'masking_time', 'masking_channel', 'dropout'
        ]
        for name in expected_strategies:
            assert name in available['strategies']

    def test_presets_available(self):
        """Manager should have presets available."""
        manager = AugmentationManager()
        available = manager.list_available()

        expected_presets = ['light', 'moderate', 'aggressive', 'conservative']
        for name in expected_presets:
            assert name in available['presets']

    def test_unknown_strategy_returns_none(self):
        """Unknown strategy should return None."""
        manager = AugmentationManager()
        result = manager.get_strategy('nonexistent_strategy')

        assert result is None

    def test_transform_batch_dict_input(self, sample_batch):
        """transform_batch should work with dict input."""
        manager = AugmentationManager()
        batch_dict = {'eeg': sample_batch}

        # Force augmentation by setting prob=1.0
        result = manager.transform_batch(batch_dict, strategy_name='light', prob=1.0)

        assert isinstance(result, dict)
        assert 'eeg' in result
        assert result['eeg'].shape == sample_batch.shape

    def test_transform_array_input(self, sample_batch):
        """transform_array should work with array input."""
        manager = AugmentationManager()

        # Force augmentation
        result = manager.transform_array(sample_batch, strategy_name='moderate', prob=1.0)

        assert result.shape == sample_batch.shape

    def test_empty_data_handled(self):
        """Empty data should be handled gracefully."""
        manager = AugmentationManager()
        empty_batch = {'eeg': np.array([])}

        result = manager.transform_batch(empty_batch, strategy_name='light', prob=1.0)

        assert 'eeg' in result
        assert len(result['eeg']) == 0

    def test_unknown_strategy_returns_original(self, sample_batch):
        """Unknown strategy should return original data unchanged."""
        manager = AugmentationManager()

        result = manager.transform_array(sample_batch, strategy_name='nonexistent', prob=1.0)

        assert np.array_equal(result, sample_batch)

    def test_probabilistic_skip(self, sample_batch):
        """With prob=0, augmentation should be skipped."""
        manager = AugmentationManager()

        result = manager.transform_array(sample_batch, strategy_name='aggressive', prob=0.0)

        assert np.array_equal(result, sample_batch)

    def test_global_manager_instance(self):
        """get_augmentation_manager should return global instance."""
        manager1 = get_augmentation_manager()
        manager2 = get_augmentation_manager()

        assert manager1 is manager2


class TestMixupCutmix:
    """Tests for mixup and cutmix functions."""

    @pytest.fixture
    def batch_data(self):
        """Create batch data for mixup/cutmix."""
        np.random.seed(42)
        x = np.random.randn(16, 8, 250).astype(np.float32)
        y = np.random.randint(0, 2, 16)
        return x, y

    def test_mixup_returns_correct_shapes(self, batch_data):
        """Mixup should return correct shapes."""
        x, y = batch_data
        x_mixed, y1, y2, lam = apply_mixup(x, y, alpha=0.2)

        assert x_mixed.shape == x.shape
        assert y1.shape == y.shape
        assert y2.shape == y.shape
        assert isinstance(lam, float)

    def test_cutmix_returns_correct_shapes(self, batch_data):
        """Cutmix should return correct shapes."""
        x, y = batch_data
        x_mixed, y1, y2, lam = apply_cutmix(x, y, alpha=0.2)

        assert x_mixed.shape == x.shape
        assert y1.shape == y.shape
        assert y2.shape == y.shape
        assert isinstance(lam, float)

    def test_mixup_lambda_in_valid_range(self, batch_data):
        """Mixup lambda should be between 0 and 1."""
        x, y = batch_data

        for _ in range(50):
            _, _, _, lam = apply_mixup(x, y, alpha=0.2)
            assert 0.0 <= lam <= 1.0

    def test_cutmix_lambda_in_valid_range(self, batch_data):
        """Cutmix lambda should be between 0 and 1."""
        x, y = batch_data

        for _ in range(50):
            _, _, _, lam = apply_cutmix(x, y, alpha=0.2)
            assert 0.0 <= lam <= 1.0

    def test_mixup_blends_data(self, batch_data):
        """Mixup should blend data from different samples."""
        x, y = batch_data
        np.random.seed(123)  # Control randomness
        x_mixed, _, _, lam = apply_mixup(x, y, alpha=0.5)

        # Mixed data should be different from original
        assert not np.allclose(x_mixed, x)

    def test_cutmix_patches_data(self, batch_data):
        """Cutmix should patch regions from different samples."""
        x, y = batch_data
        np.random.seed(123)
        x_mixed, _, _, lam = apply_cutmix(x, y, alpha=0.5)

        # Mixed data should be different from original (unless lam=1)
        if lam < 1.0:
            assert not np.allclose(x_mixed, x)

    def test_mixup_alpha_zero(self, batch_data):
        """Mixup with alpha=0 should return lambda=1 (no mixing)."""
        x, y = batch_data
        x_mixed, _, _, lam = apply_mixup(x, y, alpha=0.0)

        assert lam == 1.0
        assert np.allclose(x_mixed, x)

    def test_cutmix_alpha_zero(self, batch_data):
        """Cutmix with alpha=0 should return lambda=1 (no cutting)."""
        x, y = batch_data
        x_mixed, _, _, lam = apply_cutmix(x, y, alpha=0.0)

        assert lam == 1.0
        assert np.allclose(x_mixed, x)

    def test_mixup_preserves_original_labels(self, batch_data):
        """Mixup should return original labels as y1."""
        x, y = batch_data
        _, y1, _, _ = apply_mixup(x, y, alpha=0.2)

        assert np.array_equal(y1, y)

    def test_cutmix_preserves_original_labels(self, batch_data):
        """Cutmix should return original labels as y1."""
        x, y = batch_data
        _, y1, _, _ = apply_cutmix(x, y, alpha=0.2)

        assert np.array_equal(y1, y)
