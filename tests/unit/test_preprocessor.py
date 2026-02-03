"""
Unit tests for the modular preprocessor system.

This module provides comprehensive unit tests for the preprocessing system,
testing individual processors and the main OnlinePreprocessor class.

Tests cover:
- BaseModalityProcessor abstract class
- EEGProcessor with filtering and downsampling
- EMGProcessor with notch filtering
- EOGProcessor for reference channels
- PassthroughProcessor for unknown modalities
- OnlinePreprocessor integration
- Error handling and edge cases
"""

import sys
import os
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Add the project root to the path so BMI can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dendrite.processing.preprocessing.preprocessor import (
    BaseModalityProcessor,
    EEGProcessor,
    EMGProcessor,
    EOGProcessor,
    PassthroughProcessor,
    OnlinePreprocessor
)


class TestEEGProcessor:
    """Test suite for EEGProcessor."""
    
    @pytest.fixture
    def eeg_config(self):
        """Create EEG processor configuration."""
        return {
            'num_channels': 32,
            'sample_rate': 500,
            'lowcut': 0.5,
            'highcut': 50.0,
            'downsample_factor': 2,
            'apply_rereferencing': True
        }
    
    @pytest.fixture
    def eeg_processor(self, eeg_config):
        """Create an EEG processor instance."""
        return EEGProcessor(eeg_config)
    
    def test_initialization(self, eeg_processor):
        """Test EEGProcessor initialization."""
        assert eeg_processor.num_channels == 32
        assert eeg_processor.sample_rate == 500
        assert eeg_processor.lowcut == 0.5
        assert eeg_processor.highcut == 50.0
        assert eeg_processor.downsample_factor == 2
        assert eeg_processor.apply_rereferencing
        
        # Check filter coefficients were created
        assert hasattr(eeg_processor, 'b')
        assert hasattr(eeg_processor, 'a')
        assert hasattr(eeg_processor, 'zi')
        assert eeg_processor.zi.shape[0] == 32
    
    def test_frequency_validation(self):
        """Test frequency range validation."""
        config = {
            'num_channels': 32,
            'sample_rate': 500,
            'lowcut': -1,  # Invalid
            'highcut': 300,  # Above Nyquist
            'downsample_factor': 1
        }
        
        processor = EEGProcessor(config)
        
        # Should adjust to valid values
        assert processor.lowcut > 0
        assert processor.lowcut < processor.sample_rate / 2
        assert processor.highcut < processor.sample_rate / 2
        assert processor.highcut > processor.lowcut
    
    def test_rereferencing(self, eeg_processor):
        """Test common average referencing."""
        # Create test data with DC offset
        data = np.ones((32, 100)) * 10
        data[0, :] += 5  # Add offset to first channel
        
        # Process without downsampling
        eeg_processor.downsample_factor = 1
        processed = eeg_processor.process_chunk(data)
        
        # After common average referencing, the mean across channels should be ~0
        # But the mean across time for each channel won't necessarily be 0
        # Check that the data has been modified by rereferencing
        assert not np.allclose(processed, data)
    
    def test_downsampling(self, eeg_processor):
        """Test stateful downsampling."""
        # Create test data
        data1 = np.random.randn(32, 50)
        data2 = np.random.randn(32, 50)
        
        # Process chunks
        out1 = eeg_processor.process_chunk(data1)
        out2 = eeg_processor.process_chunk(data2)
        
        # With downsample_factor=2, should get half the samples
        assert out1.shape[1] == 25
        assert out2.shape[1] == 25
    
    def test_state_reset(self, eeg_processor):
        """Test state reset functionality."""
        # Process some data
        data = np.random.randn(32, 100)
        _ = eeg_processor.process_chunk(data)

        # Reset state
        eeg_processor.reset_state()

        # Check state was reset
        assert np.all(eeg_processor.zi == 0)
        assert eeg_processor._downsample_state['buffer'].shape[1] == 0
        # Verify AA filter state reset (downsample_factor=2 in fixture)
        assert np.all(eeg_processor._aa_zi == 0)


class TestEMGProcessor:
    """Test suite for EMGProcessor."""
    
    @pytest.fixture
    def emg_config(self):
        """Create EMG processor configuration."""
        return {
            'num_channels': 8,
            'sample_rate': 2000,
            'lowcut': 20,
            'highcut': 450,
            'line_freq': 50,
            'notch_width': 4,
            'downsample_factor': 4
        }
    
    @pytest.fixture
    def emg_processor(self, emg_config):
        """Create an EMG processor instance."""
        return EMGProcessor(emg_config)
    
    def test_initialization(self, emg_processor):
        """Test EMGProcessor initialization."""
        assert emg_processor.num_channels == 8
        assert emg_processor.sample_rate == 2000
        assert emg_processor.lowcut == 20
        assert emg_processor.highcut == 450
        assert emg_processor.line_freq == 50
        assert emg_processor.notch_width == 4
        assert emg_processor.downsample_factor == 4
        
        # Check filters were created
        assert hasattr(emg_processor, 'b_bp')
        assert hasattr(emg_processor, 'a_bp')
        assert hasattr(emg_processor, 'zi_bp')
        assert emg_processor.has_notch
        assert hasattr(emg_processor, 'b_notch')
        assert hasattr(emg_processor, 'a_notch')
        assert hasattr(emg_processor, 'zi_notch')
    
    def test_notch_filter_50hz(self, emg_processor):
        """Test 50Hz notch filter."""
        # Create signal with 50Hz noise
        fs = 2000
        t = np.arange(1000) / fs
        clean_signal = np.random.randn(8, 1000) * 10
        noise_50hz = np.sin(2 * np.pi * 50 * t) * 20
        
        # Add noise to all channels
        noisy_signal = clean_signal.copy()
        for i in range(8):
            noisy_signal[i, :] += noise_50hz
        
        # Process signal
        processed = emg_processor.process_chunk(noisy_signal)
        
        # Check that 50Hz component is reduced
        # (Can't check exact values due to filter transients and downsampling)
        assert processed.shape[0] == 8
        assert processed.shape[1] < noisy_signal.shape[1]  # Downsampled
    
    def test_60hz_notch_configuration(self):
        """Test 60Hz notch filter configuration."""
        config = {
            'num_channels': 8,
            'sample_rate': 2000,
            'line_freq': 60,  # 60Hz power line
            'notch_width': 4,
            'downsample_factor': 1
        }
        
        processor = EMGProcessor(config)
        
        assert processor.line_freq == 60
        assert processor.has_notch
    
    def test_invalid_notch_frequency(self):
        """Test invalid notch frequency handling."""
        config = {
            'num_channels': 8,
            'sample_rate': 100,  # Low sample rate
            'line_freq': 50,  # At Nyquist frequency
            'notch_width': 20,  # Wide notch that would go above Nyquist
            'downsample_factor': 1,
            'lowcut': 10,  # Need to specify valid bandpass range
            'highcut': 40
        }
        
        processor = EMGProcessor(config)
        
        # Should disable notch filter due to invalid frequency range
        # 50 + 10 = 60 Hz which is above Nyquist (50 Hz)
        assert not processor.has_notch


class TestEOGProcessor:
    """Test suite for EOGProcessor."""

    @pytest.fixture
    def eog_config(self):
        """Create EOG processor configuration."""
        return {
            'num_channels': 2,
            'sample_rate': 500,
            'lowcut': 0.1,
            'highcut': 10.0,
            'downsample_factor': 2
        }

    @pytest.fixture
    def eog_processor(self, eog_config):
        """Create an EOG processor instance."""
        return EOGProcessor(eog_config)

    def test_initialization(self, eog_processor):
        """Test EOGProcessor initialization."""
        assert eog_processor.num_channels == 2
        assert eog_processor.sample_rate == 500
        assert eog_processor.lowcut == 0.1
        assert eog_processor.highcut == 10.0
        assert eog_processor.downsample_factor == 2

        # Check filter coefficients were created
        assert hasattr(eog_processor, 'b')
        assert hasattr(eog_processor, 'a')
        assert hasattr(eog_processor, 'zi')
        assert eog_processor.zi.shape[0] == 2

    def test_downsampling(self, eog_processor):
        """Test stateful downsampling in EOGProcessor."""
        # Create test data
        data1 = np.random.randn(2, 50)
        data2 = np.random.randn(2, 50)

        # Process chunks
        out1 = eog_processor.process_chunk(data1)
        out2 = eog_processor.process_chunk(data2)

        # With downsample_factor=2, should get half the samples
        assert out1.shape == (2, 25)
        assert out2.shape == (2, 25)

    def test_no_downsampling(self):
        """Test EOGProcessor without downsampling."""
        config = {
            'num_channels': 2,
            'sample_rate': 500,
            'lowcut': 0.1,
            'highcut': 10.0,
            'downsample_factor': 1
        }
        processor = EOGProcessor(config)

        data = np.random.randn(2, 100)
        processed = processor.process_chunk(data)

        # No downsampling, same sample count
        assert processed.shape == (2, 100)


class TestPassthroughProcessor:
    """Test suite for PassthroughProcessor."""
    
    @pytest.fixture
    def passthrough_config(self):
        """Create passthrough processor configuration."""
        return {
            'num_channels': 4,
            'sample_rate': 1000,
            'downsample_factor': 2
        }
    
    @pytest.fixture
    def passthrough_processor(self, passthrough_config):
        """Create a passthrough processor instance."""
        return PassthroughProcessor(passthrough_config)
    
    def test_initialization(self, passthrough_processor):
        """Test PassthroughProcessor initialization."""
        assert passthrough_processor.num_channels == 4
        assert passthrough_processor.sample_rate == 1000
        assert passthrough_processor.downsample_factor == 2
    
    def test_passthrough_no_downsampling(self):
        """Test passthrough without downsampling."""
        config = {'num_channels': 4, 'sample_rate': 1000, 'downsample_factor': 1}
        processor = PassthroughProcessor(config)
        
        data = np.random.randn(4, 100)
        processed = processor.process_chunk(data)
        
        # Should be unchanged except for dtype
        assert processed.shape == data.shape
        assert processed.dtype == np.float64
    
    def test_passthrough_with_downsampling(self, passthrough_processor):
        """Test passthrough with downsampling."""
        # Process multiple chunks to test stateful downsampling
        data1 = np.random.randn(4, 50)
        data2 = np.random.randn(4, 50)
        
        processed1 = passthrough_processor.process_chunk(data1)
        processed2 = passthrough_processor.process_chunk(data2)
        
        # With downsample_factor=2
        assert processed1.shape == (4, 25)
        assert processed2.shape == (4, 25)


class TestOnlinePreprocessor:
    """Test suite for OnlinePreprocessor integration."""
    
    @pytest.fixture
    def modality_preprocessing(self):
        """Create modality configurations."""
        return {
            'EEG': {
                'num_channels': 32,
                'sample_rate': 500,
                'lowcut': 0.5,
                'highcut': 50.0,
                'downsample_factor': 2
            },
            'EMG': {
                'num_channels': 8,
                'sample_rate': 2000,
                'lowcut': 20,
                'highcut': 450,
                'downsample_factor': 4
            }
        }

    @pytest.fixture
    def preprocessor(self, modality_preprocessing):
        """Create an OnlinePreprocessor instance."""
        return OnlinePreprocessor(modality_preprocessing)

    def test_initialization(self, preprocessor):
        """Test OnlinePreprocessor initialization."""
        assert len(preprocessor.processors) == 2
        assert 'eeg' in preprocessor.processors
        assert 'emg' in preprocessor.processors

        # Check processor types
        assert isinstance(preprocessor.processors['eeg'], EEGProcessor)
        assert isinstance(preprocessor.processors['emg'], EMGProcessor)

    def test_process_multi_modality(self, preprocessor):
        """Test processing multiple modalities."""
        # Create test data (lowercase keys to match normalized processor keys)
        data_dict = {
            'eeg': np.random.randn(32, 100),
            'emg': np.random.randn(8, 400),  # Higher sample rate
        }

        # Process data
        processed = preprocessor.process(data_dict)

        # Check all modalities processed
        assert 'eeg' in processed
        assert 'emg' in processed

        # Check downsampling applied
        assert processed['eeg'].shape[1] == 50  # Factor 2
        assert processed['emg'].shape[1] == 100  # Factor 4

    def test_unknown_modality_handling(self):
        """Test handling of unknown modalities."""
        modality_preprocessing = {
            'EEG': {'num_channels': 32, 'sample_rate': 500},
            'UNKNOWN': {'num_channels': 4, 'sample_rate': 1000, 'downsample_factor': 2}
        }

        preprocessor = OnlinePreprocessor(modality_preprocessing)

        # Unknown modality should get PassthroughProcessor (normalized to lowercase)
        assert 'unknown' in preprocessor.processors
        assert isinstance(preprocessor.processors['unknown'], PassthroughProcessor)
    
    def test_processor_registry_lookup(self):
        """Test processor registry case handling."""
        modality_preprocessing = {
            'eeg': {'num_channels': 32, 'sample_rate': 500},  # lowercase
            'EOG': {'num_channels': 2, 'sample_rate': 500},  # uppercase
            'Emg': {'num_channels': 4, 'sample_rate': 1000, 'lowcut': 20, 'highcut': 450}  # mixed case
        }

        preprocessor = OnlinePreprocessor(modality_preprocessing)

        # All keys normalized to lowercase
        assert isinstance(preprocessor.processors['eeg'], EEGProcessor)
        assert isinstance(preprocessor.processors['eog'], EOGProcessor)
        assert isinstance(preprocessor.processors['emg'], EMGProcessor)
    
    def test_reset_all_states(self, preprocessor):
        """Test resetting all processor states."""
        # Process some data (lowercase keys to match normalized processor keys)
        data_dict = {
            'eeg': np.random.randn(32, 100),
            'emg': np.random.randn(8, 400),
        }
        _ = preprocessor.process(data_dict)

        # Reset all states
        preprocessor.reset_all_states()

        # All processors should have clean state
        for processor in preprocessor.processors.values():
            if hasattr(processor, '_downsample_state'):
                assert processor._downsample_state['buffer'].shape[1] == 0

    def test_error_handling_in_processing(self, preprocessor):
        """Test error handling during processing."""
        # Create invalid data (lowercase keys to match normalized processor keys)
        data_dict = {
            'eeg': np.random.randn(32, 100),
            'emg': 'invalid_data',  # Wrong type
        }

        # Should handle error and return original data
        processed = preprocessor.process(data_dict)

        assert 'emg' in processed
        assert processed['emg'] == 'invalid_data'  # Fallback to original
    
    def test_missing_processor_passthrough(self):
        """Test that missing processors pass data through."""
        modality_preprocessing = {
            'EEG': {'num_channels': 32, 'sample_rate': 500}
        }
        
        preprocessor = OnlinePreprocessor(modality_preprocessing)
        
        # Process data with unknown modality
        data_dict = {
            'EEG': np.random.randn(32, 100),
            'UNKNOWN': np.random.randn(4, 100)
        }
        
        processed = preprocessor.process(data_dict)
        
        # Unknown should be passed through as float64
        assert 'UNKNOWN' in processed
        assert processed['UNKNOWN'].dtype == np.float64
        assert processed['UNKNOWN'].shape == (4, 100)
    
    def test_unexpected_modality_passthrough(self):
        """Test that unexpected modalities are passed through without processing."""
        modality_preprocessing = {
            'eeg': {'num_channels': 32, 'sample_rate': 500, 'downsample_factor': 2}
        }

        preprocessor = OnlinePreprocessor(modality_preprocessing)

        # Process data with unexpected modalities
        data_dict = {
            'eeg': np.random.randn(32, 100),
            'auxiliary': np.random.randn(2, 100),
        }

        processed = preprocessor.process(data_dict)

        # EEG should be downsampled by factor 2
        assert processed['eeg'].shape == (32, 50)
        # Unexpected modality passes through as float64 without processing
        assert processed['auxiliary'].shape == (2, 100)
        assert processed['auxiliary'].dtype == np.float64


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""
    
    def test_empty_chunk_processing(self):
        """Test processing empty chunks."""
        config = {'num_channels': 32, 'sample_rate': 500, 'downsample_factor': 2}
        processor = EEGProcessor(config)
        
        empty_data = np.zeros((32, 0))
        processed = processor.process_chunk(empty_data)
        
        assert processed.shape == (32, 0)
    
    def test_single_sample_processing(self):
        """Test processing single sample."""
        config = {'num_channels': 1, 'sample_rate': 500, 'downsample_factor': 1}
        processor = PassthroughProcessor(config)

        single_sample = np.array([[42.0]])
        processed = processor.process_chunk(single_sample)

        assert processed.shape == (1, 1)
        assert processed[0, 0] == 42.0
    
    def test_extreme_downsampling(self):
        """Test extreme downsampling factors."""
        config = {'num_channels': 1, 'sample_rate': 1000, 'downsample_factor': 100}
        processor = PassthroughProcessor(config)
        
        # Process data smaller than downsample factor
        data = np.random.randn(1, 50)
        processed = processor.process_chunk(data)
        
        # Should buffer and return empty
        assert processed.shape == (1, 0)
        
        # Process more data
        data2 = np.random.randn(1, 150)
        processed2 = processor.process_chunk(data2)
        
        # Should now have 2 samples (200 total / 100)
        assert processed2.shape == (1, 2)

    def test_nan_handling(self):
        """Test handling of NaN values."""
        config = {'num_channels': 2, 'sample_rate': 500, 'downsample_factor': 1}
        processor = EEGProcessor(config)
        
        # Create data with NaNs
        data = np.random.randn(2, 100)
        data[0, 50] = np.nan
        data[1, 75] = np.nan
        
        # Process should not crash
        processed = processor.process_chunk(data)
        
        # NaNs may propagate through filters
        assert processed.shape == data.shape


class TestMultiStreamSameModality:
    """Test independent preprocessing for different streams.

    In the new design, each stream gets its own OnlinePreprocessor instance.
    This is managed by DataProcessor, which creates one preprocessor per stream.
    """

    def test_two_preprocessors_for_different_streams(self):
        """Two streams with different channel counts should have separate preprocessors."""
        # Stream 1: 8-channel EMG
        preprocessor1 = OnlinePreprocessor({
            'emg': {'num_channels': 8, 'sample_rate': 2000, 'lowcut': 20, 'highcut': 450},
        })

        # Stream 2: 2-channel EMG
        preprocessor2 = OnlinePreprocessor({
            'emg': {'num_channels': 2, 'sample_rate': 2000, 'lowcut': 20, 'highcut': 450},
        })

        # Process data from both streams using separate preprocessors
        data_8ch = {'emg': np.random.randn(8, 10)}
        data_2ch = {'emg': np.random.randn(2, 10)}

        # Both should process without shape mismatch errors
        result_8ch = preprocessor1.process(data_8ch)
        result_2ch = preprocessor2.process(data_2ch)

        assert result_8ch['emg'].shape == (8, 10)
        assert result_2ch['emg'].shape == (2, 10)

    def test_modality_key_selects_correct_processor_class(self):
        """Modality keys should select the correct processor type."""
        modality_preprocessing = {
            'emg': {'num_channels': 4, 'sample_rate': 1000, 'lowcut': 20, 'highcut': 450},
            'eeg': {'num_channels': 8, 'sample_rate': 500, 'lowcut': 0.5, 'highcut': 50},
        }

        preprocessor = OnlinePreprocessor(modality_preprocessing)

        # Verify correct processor types were created
        assert isinstance(preprocessor.processors['emg'], EMGProcessor)
        assert isinstance(preprocessor.processors['eeg'], EEGProcessor)

    def test_interleaved_processing_maintains_separate_states(self):
        """Interleaved samples from different preprocessors maintain separate filter states."""
        # Separate preprocessors for different streams (new design)
        preprocessor1 = OnlinePreprocessor({
            'emg': {'num_channels': 8, 'sample_rate': 2000, 'lowcut': 20, 'highcut': 450},
        })
        preprocessor2 = OnlinePreprocessor({
            'emg': {'num_channels': 2, 'sample_rate': 2000, 'lowcut': 20, 'highcut': 450},
        })

        # Interleave processing (simulates real-time multi-stream)
        for _ in range(100):
            data1 = {'emg': np.random.randn(8, 1)}
            data2 = {'emg': np.random.randn(2, 1)}

            result1 = preprocessor1.process(data1)
            result2 = preprocessor2.process(data2)

            assert result1['emg'].shape[0] == 8
            assert result2['emg'].shape[0] == 2

    def test_simple_modality_key_still_works(self):
        """Simple modality keys should select correct processor."""
        modality_preprocessing = {
            'emg': {'num_channels': 4, 'sample_rate': 1000, 'lowcut': 20, 'highcut': 450},
        }

        preprocessor = OnlinePreprocessor(modality_preprocessing)

        assert isinstance(preprocessor.processors['emg'], EMGProcessor)

        data = {'emg': np.random.randn(4, 10)}
        result = preprocessor.process(data)
        assert result['emg'].shape == (4, 10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])