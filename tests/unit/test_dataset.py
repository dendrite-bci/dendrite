import pytest
import numpy as np
import tempfile
from pathlib import Path
from dendrite.data.dataset import Dataset


class TestDataset:
    """Test suite for Dataset class."""
    
    def setup_method(self):
        """Setup for each test."""
        self.dataset = Dataset(name="test_dataset")

        # Create sample data with realistic amplitude (scaled to ~10µV = 1e-5 V)
        self.eeg_data = (np.random.randn(100, 32) * 1e-5).astype(np.float32)  # 100 timepoints, 32 channels
        self.emg_data = (np.random.randn(100, 4) * 1e-4).astype(np.float32)   # 100 timepoints, 4 channels
        self.label = 1
        self.source_id = "test_source"
    
    def test_init(self):
        """Test dataset initialization."""
        assert self.dataset.name == "test_dataset"
        assert self.dataset.max_samples is None
        assert len(self.dataset.samples) == 0
        assert len(self.dataset.sources) == 0
        assert len(self.dataset.reference_shapes) == 0
        assert len(self.dataset.classes) == 0
    
    def test_init_with_max_samples(self):
        """Test initialization with max_samples limit."""
        dataset = Dataset(name="limited", max_samples=10)
        assert dataset.max_samples == 10
    
    def test_add_sample_single_modality(self):
        """Test adding a single sample with one modality."""
        data_dict = {'eeg': self.eeg_data}
        result = self.dataset.add_sample(data_dict, self.label, self.source_id)
        
        assert result is True
        assert len(self.dataset.samples) == 1
        assert self.dataset.samples[0]['label'] == self.label
        assert self.dataset.samples[0]['source'] == self.source_id
        assert np.array_equal(self.dataset.samples[0]['eeg'], self.eeg_data)
        assert 'eeg' in self.dataset.reference_shapes
        assert self.dataset.reference_shapes['eeg'] == self.eeg_data.shape
        assert self.label in self.dataset.classes
    
    def test_add_sample_multimodal(self):
        """Test adding a sample with multiple modalities."""
        data_dict = {'eeg': self.eeg_data, 'emg': self.emg_data}
        result = self.dataset.add_sample(data_dict, self.label, self.source_id)
        
        assert result is True
        assert len(self.dataset.samples) == 1
        assert 'eeg' in self.dataset.samples[0]
        assert 'emg' in self.dataset.samples[0]
        assert len(self.dataset.reference_shapes) == 2
        assert self.dataset.reference_shapes['eeg'] == self.eeg_data.shape
        assert self.dataset.reference_shapes['emg'] == self.emg_data.shape
    
    def test_add_multiple_samples(self):
        """Test adding multiple samples."""
        # Add first sample
        data_dict1 = {'eeg': self.eeg_data}
        self.dataset.add_sample(data_dict1, 0, "source1")
        
        # Add second sample with different label
        data_dict2 = {'eeg': self.eeg_data + 0.1}
        self.dataset.add_sample(data_dict2, 1, "source2")
        
        assert len(self.dataset.samples) == 2
        assert len(self.dataset.classes) == 2
        assert 0 in self.dataset.classes
        assert 1 in self.dataset.classes
        assert len(self.dataset.sources) == 2
        assert self.dataset.sources["source1"]["samples"] == 1
        assert self.dataset.sources["source2"]["samples"] == 1
    
    def test_max_samples_limit(self):
        """Test FIFO behavior with max_samples limit."""
        dataset = Dataset(name="limited", max_samples=2)
        
        # Add 3 samples, first should be removed
        data_dict = {'eeg': self.eeg_data}
        dataset.add_sample(data_dict, 0, "source1")
        dataset.add_sample(data_dict, 1, "source2")
        dataset.add_sample(data_dict, 2, "source3")
        
        assert len(dataset.samples) == 2
        assert dataset.samples[0]['label'] == 1  # First sample (label=0) was removed
        assert dataset.samples[1]['label'] == 2
        assert len(dataset.classes) == 2  # Classes recalculated after removal
        assert 1 in dataset.classes
        assert 2 in dataset.classes
        assert 0 not in dataset.classes
    
    def test_shape_validation(self):
        """Test shape validation for consistent modalities."""
        # Add first sample
        data_dict1 = {'eeg': self.eeg_data}
        result1 = self.dataset.add_sample(data_dict1, 0, "source1")
        assert result1 is True
        
        # Try to add sample with different shape - should fail
        wrong_shape_data = np.random.randn(50, 16).astype(np.float32)
        data_dict2 = {'eeg': wrong_shape_data}
        result2 = self.dataset.add_sample(data_dict2, 1, "source2")
        assert result2 is False
        assert len(self.dataset.samples) == 1  # Only first sample added
    
    def test_nan_inf_flagged_not_rejected(self):
        """Test that NaN/Inf values are accepted but flagged as bad."""
        # Test NaN values - accepted but flagged
        nan_data = self.eeg_data.copy()
        nan_data[0, 0] = np.nan
        result1 = self.dataset.add_sample({'eeg': nan_data}, 0, "source1")
        assert result1 is True
        assert self.dataset.samples[-1]['is_bad'] is True

        # Test infinite values - accepted but flagged
        inf_data = self.eeg_data.copy()
        inf_data[0, 0] = np.inf
        result2 = self.dataset.add_sample({'eeg': inf_data}, 0, "source2")
        assert result2 is True
        assert self.dataset.samples[-1]['is_bad'] is True

        assert len(self.dataset.samples) == 2
    
    def test_invalid_data_type(self):
        """Test validation of data types."""
        # Non-numpy array should fail
        data_dict = {'eeg': [[1, 2, 3], [4, 5, 6]]}
        result = self.dataset.add_sample(data_dict, 0, "source1")
        assert result is False
        assert len(self.dataset.samples) == 0
    
    def test_get_data_empty(self):
        """Test getting data from empty dataset."""
        X, y = self.dataset.get_data('eeg')
        assert len(X) == 0
        assert len(y) == 0
    
    def test_get_data_single_modality(self):
        """Test getting data for single modality."""
        # Add some samples
        data_dict = {'eeg': self.eeg_data}
        self.dataset.add_sample(data_dict, 0, "source1")
        self.dataset.add_sample(data_dict, 1, "source2")
        
        X, y = self.dataset.get_data('eeg')
        assert X.shape == (2, 100, 32)
        assert len(y) == 2
        assert y[0] == 0
        assert y[1] == 1
    
    def test_get_data_multimodal(self):
        """Test getting data with multiple modalities."""
        data_dict = {'eeg': self.eeg_data, 'emg': self.emg_data}
        self.dataset.add_sample(data_dict, 0, "source1")
        
        X_eeg, y_eeg = self.dataset.get_data('eeg')
        X_emg, y_emg = self.dataset.get_data('emg')
        
        assert X_eeg.shape == (1, 100, 32)
        assert X_emg.shape == (1, 100, 4)
        assert np.array_equal(y_eeg, y_emg)
    
    def test_get_training_data_basic(self):
        """Test get_training_data basic functionality."""
        data_dict = {'eeg': self.eeg_data}
        self.dataset.add_sample(data_dict, 0, "source1")
        self.dataset.add_sample(data_dict, 1, "source2")

        result = self.dataset.get_training_data('eeg')

        # X is now an array directly
        assert result['X'].shape == (2, 100, 32)
        assert len(result['y']) == 2
        assert result['num_classes'] == 2
        assert 'input_shape' in result
    
    def test_get_info(self):
        """Test dataset info retrieval."""
        # Empty dataset
        info = self.dataset.get_info()
        assert info['name'] == "test_dataset"
        assert info['total_samples'] == 0
        assert info['sources'] == 0
        assert len(info['modalities']) == 0
        assert len(info['classes']) == 0
        
        # Add some data
        data_dict = {'eeg': self.eeg_data, 'emg': self.emg_data}
        self.dataset.add_sample(data_dict, 0, "source1")
        self.dataset.add_sample(data_dict, 1, "source2")
        
        info = self.dataset.get_info()
        assert info['total_samples'] == 2
        assert info['sources'] == 2
        assert len(info['modalities']) == 2
        assert 'eeg' in info['modalities']
        assert 'emg' in info['modalities']
        assert info['classes'] == [0, 1]
    
    def test_clear(self):
        """Test dataset clearing."""
        # Add some data
        data_dict = {'eeg': self.eeg_data}
        self.dataset.add_sample(data_dict, 0, "source1")
        
        assert len(self.dataset.samples) == 1
        
        # Clear dataset
        self.dataset.clear()
        
        assert len(self.dataset.samples) == 0
        assert len(self.dataset.sources) == 0
        assert len(self.dataset.reference_shapes) == 0
        assert len(self.dataset.classes) == 0
    
    def test_len_and_repr(self):
        """Test __len__ and __repr__ methods."""
        assert len(self.dataset) == 0
        assert "test_dataset" in repr(self.dataset)
        assert "samples=0" in repr(self.dataset)
        
        # Add sample
        data_dict = {'eeg': self.eeg_data}
        self.dataset.add_sample(data_dict, 0, "source1")
        
        assert len(self.dataset) == 1
        assert "samples=1" in repr(self.dataset)
        assert "sources=1" in repr(self.dataset)
    
    def test_save_and_load_dataset(self):
        """Test saving and loading dataset."""
        # Add some multimodal data
        data_dict = {'eeg': self.eeg_data, 'emg': self.emg_data}
        self.dataset.add_sample(data_dict, 0, "source1")
        self.dataset.add_sample(data_dict, 1, "source2")
        
        # Save to temporary file
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "test_dataset.h5"
            
            # Save dataset
            self.dataset.save_dataset(filepath)
            assert filepath.exists()
            
            # Load dataset
            loaded_dataset = Dataset.load_dataset(filepath)
            
            # Verify loaded dataset
            assert loaded_dataset.name == self.dataset.name
            assert len(loaded_dataset.samples) == len(self.dataset.samples)
            assert loaded_dataset.classes == self.dataset.classes
            assert loaded_dataset.reference_shapes == self.dataset.reference_shapes
            assert len(loaded_dataset.sources) == len(self.dataset.sources)
            
            # Check data integrity
            X_orig, y_orig = self.dataset.get_data('eeg')
            X_loaded, y_loaded = loaded_dataset.get_data('eeg')
            
            assert np.array_equal(X_orig, X_loaded)
            assert np.array_equal(y_orig, y_loaded)
    
    def test_load_nonexistent_file(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            Dataset.load_dataset("nonexistent_file.h5")
    
    def test_save_with_max_samples(self):
        """Test saving dataset with max_samples configuration."""
        dataset = Dataset(name="limited", max_samples=5)
        data_dict = {'eeg': self.eeg_data}
        dataset.add_sample(data_dict, 0, "source1")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "limited_dataset.h5"
            dataset.save_dataset(filepath)
            
            loaded_dataset = Dataset.load_dataset(filepath)
            assert loaded_dataset.max_samples == 5
    
    def test_source_tracking(self):
        """Test detailed source tracking functionality."""
        data_dict = {'eeg': self.eeg_data}

        # Add samples from different sources
        self.dataset.add_sample(data_dict, 0, "subject_01")
        self.dataset.add_sample(data_dict, 1, "subject_01")
        self.dataset.add_sample(data_dict, 0, "subject_02")

        # Check source tracking
        assert len(self.dataset.sources) == 2
        assert self.dataset.sources["subject_01"]["samples"] == 2
        assert self.dataset.sources["subject_02"]["samples"] == 1


class TestEpochRejection:
    """Test automatic epoch quality flagging."""

    def test_good_epoch_not_flagged(self):
        """Normal epoch should not be flagged."""
        dataset = Dataset()
        # Normal EEG range (channels, times) with typical µV range
        good_data = {'eeg': np.random.randn(32, 500) * 1e-5}
        dataset.add_sample(good_data, label=0)
        assert dataset.samples[-1]['is_bad'] is False

    def test_nan_epoch_flagged(self):
        """Epoch with NaN should be flagged."""
        dataset = Dataset()
        bad_data = {'eeg': np.array([[np.nan, 1, 2], [3, 4, 5]])}
        dataset.add_sample(bad_data, label=0)
        assert dataset.samples[-1]['is_bad'] is True

    def test_inf_epoch_flagged(self):
        """Epoch with Inf should be flagged."""
        dataset = Dataset()
        bad_data = {'eeg': np.array([[np.inf, 1, 2], [3, 4, 5]])}
        dataset.add_sample(bad_data, label=0)
        assert dataset.samples[-1]['is_bad'] is True

    def test_extreme_outlier_epoch_flagged(self):
        """Epoch with extreme amplitude outlier (>10 sigma) should be flagged."""
        dataset = Dataset()
        # Create data with extreme outlier
        normal_data = np.random.randn(32, 500) * 1e-5
        normal_data[0, 0] = 1.0  # Extreme outlier (orders of magnitude larger)
        dataset.add_sample({'eeg': normal_data}, label=0)
        assert dataset.samples[-1]['is_bad'] is True

    def test_high_amplitude_consistent_not_flagged(self):
        """High but consistent amplitude should NOT be flagged (not an outlier)."""
        dataset = Dataset()
        # Large values but consistent (not outliers within the epoch)
        high_amp = np.random.randn(32, 500) * 1e-3  # 1mV range - all consistent
        dataset.add_sample({'eeg': high_amp}, label=0)
        assert dataset.samples[-1]['is_bad'] is False

    def test_flat_epoch_flagged(self):
        """Flat signal epoch should be flagged."""
        dataset = Dataset()
        flat = {'eeg': np.zeros((32, 500))}
        dataset.add_sample(flat, label=0)
        assert dataset.samples[-1]['is_bad'] is True

    def test_training_data_excludes_bad_by_default(self):
        """get_training_data should exclude bad epochs by default."""
        dataset = Dataset()
        # Add good epoch
        dataset.add_sample({'eeg': np.random.randn(32, 500) * 1e-5}, label=0)
        # Add bad epoch (flat)
        dataset.add_sample({'eeg': np.zeros((32, 500))}, label=1)

        result = dataset.get_training_data('eeg', exclude_bad=True)
        assert len(result['y']) == 1  # Only good epoch
        assert result['y'][0] == 0

    def test_training_data_includes_bad_when_requested(self):
        """get_training_data can include bad epochs if requested."""
        dataset = Dataset()
        dataset.add_sample({'eeg': np.random.randn(32, 500) * 1e-5}, label=0)
        dataset.add_sample({'eeg': np.zeros((32, 500))}, label=1)

        result = dataset.get_training_data('eeg', exclude_bad=False)
        assert len(result['y']) == 2  # Both epochs

    def test_multimodal_epoch_flagged_if_any_bad(self):
        """Multimodal epoch should be flagged if ANY modality is bad."""
        dataset = Dataset()
        # Good EEG but bad EMG (flat)
        data = {
            'eeg': np.random.randn(32, 500) * 1e-5,
            'emg': np.zeros((4, 500))
        }
        dataset.add_sample(data, label=0)
        assert dataset.samples[-1]['is_bad'] is True


class TestDatasetIntegration:
    """Integration tests for Dataset with real-world scenarios."""

    def test_typical_synchronous_mode_usage(self):
        """Test typical usage pattern from SynchronousMode."""
        dataset = Dataset(name="SyncMode_test")

        # Simulate epoch-by-epoch addition (like SynchronousMode)
        # Data scaled to realistic ~10µV range (1e-5 V)
        for epoch in range(20):
            eeg_epoch = (np.random.randn(512, 32) * 1e-5).astype(np.float32)  # 2 seconds at 256Hz
            label = epoch % 3  # 3 classes
            source_id = f"epoch_{epoch}"

            result = dataset.add_sample({'eeg': eeg_epoch}, label, source_id)
            assert result is True

        # Test training data format (like training would use)
        training_data = dataset.get_training_data('eeg')
        assert training_data['X'].shape == (20, 512, 32)
        assert len(training_data['y']) == 20
        assert training_data['num_classes'] == 3

        # Verify info
        info = dataset.get_info()
        assert info['total_samples'] == 20
        assert info['sources'] == 20
        assert len(info['classes']) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])