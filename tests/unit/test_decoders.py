"""
Unit tests for BMI decoder classes.

This module provides comprehensive unit tests for the decoder system,
focusing on testing the Decoder class.

Tests cover:
- Decoder initialization and configuration
- Pipeline building and component creation
- Training (fit)
- Prediction and probability calibration
- Model saving and loading
- Error handling and edge cases
"""

import sys
import os
import pytest
import numpy as np
import tempfile
from unittest.mock import Mock, patch

# Add the project root to the path so BMI can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dendrite.ml.decoders import Decoder, load_decoder
from dendrite.ml.decoders.decoder_schemas import NeuralNetConfig, DecoderConfig


class TestDecoder:
    """Test suite for Decoder."""
    
    @pytest.fixture
    def decoder_config(self):
        """Basic decoder configuration using DecoderConfig."""
        return DecoderConfig(
            model_type='EEGNet',
            num_classes=2,
            input_shapes={'eeg': (32, 100)},  # (channels, timepoints)
            learning_rate=0.001,
            batch_size=32,
            epochs=10,
            event_mapping={1: 'class1', 2: 'class2'},
            label_mapping={'class1': 0, 'class2': 1}
        )
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample EEG data for testing."""
        n_samples = 50
        n_channels = 32
        n_timepoints = 100

        # Use array format with (samples, channels, timepoints) standardization
        X = np.random.randn(n_samples, n_channels, n_timepoints)
        y = np.random.randint(0, 2, n_samples)

        return X, y
    
    def test_initialization(self, decoder_config):
        """Test Decoder initialization."""
        decoder = Decoder(decoder_config)

        # Check core attributes
        assert decoder.num_classes == 2
        # input_shapes may be stored as lists or tuples
        assert 'eeg' in decoder.input_shapes
        assert list(decoder.input_shapes['eeg']) == [32, 100]
        assert decoder.config == decoder_config
        assert decoder.pipeline is not None
    
    def test_initialization_without_shapes(self):
        """Test initialization without input shapes."""
        config = DecoderConfig(
            model_type='EEGNet',
            num_classes=2,
            input_shapes=None
        )
        
        decoder = Decoder(config)
        assert decoder.pipeline is None  # Pipeline not built yet
    
    def test_classifier_types(self, sample_data):
        """Test neural network classifier type."""
        X, y = sample_data

        config = DecoderConfig(
            model_type='EEGNet',
            num_classes=2,
            input_shapes={'eeg': X.shape[1:]},  # (n_channels, n_timepoints)
        )

        decoder = Decoder(config)
        assert decoder.pipeline is not None

    def test_fit_method(self, decoder_config, sample_data):
        """Test complete training with fit method."""
        X, y = sample_data
        decoder = Decoder(decoder_config)

        # Test with array input
        decoder.fit(X, y)
        assert decoder.is_fitted
    
    def test_predict_methods(self, decoder_config, sample_data):
        """Test prediction methods."""
        X, y = sample_data
        decoder = Decoder(decoder_config)
        X_train = X[:40]
        decoder.fit(X_train, y[:40])

        # Test single sample prediction
        X_test = X[40:41]

        # predict_proba
        probs = decoder.predict_proba(X_test)
        assert probs.shape == (1, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)

        # predict
        pred = decoder.predict(X_test)
        assert isinstance(pred, (int, np.integer))
        assert pred in [0, 1]

        # predict_sample
        pred, conf = decoder.predict_sample(X_test[0])
        assert isinstance(pred, (int, np.integer))
        assert isinstance(conf, (float, np.floating))
        assert 0 <= conf <= 1

    def test_batch_prediction(self, decoder_config, sample_data):
        """Test batch prediction."""
        X, y = sample_data
        decoder = Decoder(decoder_config)
        X_train = X[:30]
        decoder.fit(X_train, y[:30])

        # Test batch prediction
        X_test = X[30:40]

        probs = decoder.predict_proba(X_test)
        assert probs.shape == (10, 2)

        preds = decoder.predict(X_test)
        assert preds.shape == (10,)
        assert all(p in [0, 1] for p in preds)

    def test_temperature_calibration(self, sample_data):
        """Test that predictions are well-calibrated probabilities."""
        X, y = sample_data

        config = DecoderConfig(
            model_type='EEGNet',
            num_classes=2,
            input_shapes={'eeg': X.shape[1:]},
            epochs=1  # Fast training for testing
        )
        decoder = Decoder(config)
        decoder.fit(X, y)

        # Mock the pipeline prediction to return high confidence
        with patch.object(decoder.pipeline, 'predict_proba', return_value=np.array([[0.95, 0.05]])):
            X_test = X[:1]
            probs = decoder.predict_proba(X_test)
            # Check probabilities are valid
            assert probs.shape == (1, 2)
            assert np.allclose(probs.sum(), 1.0)
            assert probs[0, 0] == 0.95
    
    def test_model_save_load(self, decoder_config, sample_data):
        """Test model saving and loading."""
        X, y = sample_data
        decoder = Decoder(decoder_config)
        decoder.fit(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model
            model_path = os.path.join(tmpdir, 'test_model')
            saved_path = decoder.save(file_identifier=model_path)

            # Check file exists
            assert os.path.exists(saved_path)
            
            # Load model
            decoder2 = load_decoder(f"{model_path}.json")
            assert decoder2.is_fitted

            # Test predictions work after loading
            X_test = X[:1]
            probs = decoder2.predict_proba(X_test)
            assert probs.shape == (1, 2)

    def test_model_save_load_preserves_metadata(self, sample_data):
        """Test all metadata is preserved through save/load cycle with braindecode model."""
        X, y = sample_data
        config = DecoderConfig(
            model_type='BDEEGNet',  # Test braindecode model (uses parametrize)
            num_classes=2,
            input_shapes={'eeg': (32, 100)},
            epochs=1,
            event_mapping={1: 'left', 2: 'right'},
            label_mapping={'left': 0, 'right': 1},
            sample_rate=250.0,
        )
        decoder = Decoder(config)
        decoder.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_bd')
            decoder.save(path)

            # Check correct files created for neural model (2 files only)
            assert os.path.exists(f"{path}.json")
            assert os.path.exists(f"{path}.pt")  # Pipeline + weights
            assert not os.path.exists(f"{path}.joblib")  # No separate joblib for neural

            # Load into fresh decoder
            decoder2 = load_decoder(f"{path}.json")
            assert decoder2.is_fitted

            # Verify all metadata preserved (JSON converts tuples to lists)
            assert decoder2.event_mapping == {1: 'left', 2: 'right'}
            assert decoder2.label_mapping == {'left': 0, 'right': 1}
            assert decoder2.input_shapes == {'eeg': [32, 100]}  # JSON: tuple -> list
            assert decoder2.sample_rate == 250.0
            assert decoder2.num_classes == 2

            # Verify predictions work
            X_test = X[:5]
            preds = decoder2.predict(X_test)
            assert len(preds) == 5
            probs = decoder2.predict_proba(X_test)
            assert probs.shape == (5, 2)

    def test_training_metrics(self, decoder_config, sample_data):
        """Test training metrics extraction."""
        X, y = sample_data
        decoder = Decoder(decoder_config)
        
        # Mock training results
        with patch.object(decoder, '_extract_training_metrics', return_value={
            'neural_net': {
                'final_train_acc': 0.95,
                'final_val_acc': 0.88,
                'best_val_acc': 0.90,
                'epochs_completed': 10
            }
        }):
            decoder.fit(X, y)
            
            metrics = decoder.get_training_metrics()
            assert metrics is not None
            assert 'neural_net' in metrics
            assert metrics['neural_net']['final_train_acc'] == 0.95
    
    def test_error_handling(self, decoder_config):
        """Test error handling in various scenarios."""
        # Test prediction without fitting raises RuntimeError
        decoder = Decoder(decoder_config)
        X = np.random.randn(1, 32, 100)

        with pytest.raises(RuntimeError, match="Model not fitted"):
            decoder.predict_proba(X)
    
    def test_pipeline_info(self, decoder_config, sample_data):
        """Test pipeline information retrieval."""
        X, y = sample_data
        decoder = Decoder(decoder_config)
        decoder.fit(X, y)

        # Pipeline exists after training
        assert decoder.pipeline is not None
        assert decoder.is_fitted
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test with invalid parameters should raise validation error during config creation
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            config = DecoderConfig(
                model_type='EEGNet',
                num_classes=0,  # Invalid - must be >= 2
                input_shapes={'eeg': (32, 100)}
            )
        
        with pytest.raises(ValidationError):
            config = DecoderConfig(
                model_type='EEGNet',
                num_classes=2,
                learning_rate=-0.5,  # Invalid - must be > 0
                input_shapes={'eeg': (32, 100)}
            )


class TestCalibrationCorrectness:
    """Test suite specifically for calibration correctness."""

    def test_probability_passthrough(self):
        """Test that probabilities are returned correctly from pipeline."""
        config = DecoderConfig(
            model_type='EEGNet',
            num_classes=3,
            input_shapes={'eeg': (32, 100)},
            epochs=1
        )

        decoder = Decoder(config)
        decoder._is_fitted = True

        # Mock pipeline to return known probabilities
        raw_probs = np.array([[0.7, 0.2, 0.1]])
        decoder.pipeline = Mock()
        decoder.pipeline.predict_proba = Mock(return_value=raw_probs)

        # Get probabilities
        X = np.random.randn(1, 32, 100)
        result = decoder.predict_proba(X)

        # Verify probabilities are returned correctly
        assert np.allclose(result, raw_probs)
        assert result[0, 0] == 0.7
        assert result[0, 1] == 0.2
        assert result[0, 2] == 0.1
        assert np.allclose(result.sum(axis=1), 1.0)

    def test_calibration_preserves_ordering(self):
        """Test that probability ordering is preserved."""
        config = DecoderConfig(
            model_type='EEGNet',
            num_classes=4,
            input_shapes={'eeg': (32, 100)},
            epochs=1
        )

        decoder = Decoder(config)
        decoder._is_fitted = True

        # Test with various probability distributions
        test_probs = [
            np.array([[0.9, 0.05, 0.03, 0.02]]),
            np.array([[0.4, 0.3, 0.2, 0.1]]),
            np.array([[0.25, 0.25, 0.25, 0.25]])
        ]

        for raw_probs in test_probs:
            decoder.pipeline = Mock()
            decoder.pipeline.predict_proba = Mock(return_value=raw_probs)

            result = decoder.predict_proba(np.random.randn(1, 32, 100))

            # Check ordering is preserved
            raw_order = np.argsort(raw_probs[0])[::-1]
            result_order = np.argsort(result[0])[::-1]
            assert np.array_equal(raw_order, result_order)

            # Check probabilities sum to 1
            assert np.allclose(result.sum(axis=1), 1.0)


class TestPipelineScaler:
    """Test suite for scaler integration in decoder pipeline."""

    def test_pipeline_includes_scaler_by_default(self):
        """Pipeline has scaler step when use_scaler=True (default)."""
        config = DecoderConfig(
            model_type='EEGNet',
            num_classes=2,
            input_shapes={'eeg': (4, 50)},
            use_scaler=True  # Explicit, but True is default
        )
        decoder = Decoder(config)

        steps = list(decoder.pipeline.named_steps.keys())
        assert 'scaler' in steps
        assert 'classifier' in steps
        assert steps.index('scaler') < steps.index('classifier')

    def test_pipeline_excludes_scaler_when_disabled(self):
        """Pipeline has no scaler when use_scaler=False."""
        config = DecoderConfig(
            model_type='EEGNet',
            num_classes=2,
            input_shapes={'eeg': (4, 50)},
            use_scaler=False
        )
        decoder = Decoder(config)

        steps = list(decoder.pipeline.named_steps.keys())
        assert 'scaler' not in steps
        assert 'classifier' in steps

    def test_scaler_persists_through_save_load(self):
        """Scaler config preserved after save/load cycle."""
        # Create and train decoder with scaler
        X = np.random.randn(20, 32, 100)
        y = np.random.randint(0, 2, 20)

        config = DecoderConfig(
            model_type='EEGNet',
            num_classes=2,
            input_shapes={'eeg': (32, 100)},
            use_scaler=True,
            epochs=1
        )
        decoder = Decoder(config)
        decoder.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_scaler')
            decoder.save(path)

            # Load into fresh decoder
            decoder2 = load_decoder(f"{path}.json")

            # Check scaler is in loaded pipeline
            steps = list(decoder2.pipeline.named_steps.keys())
            assert 'scaler' in steps

            # Verify predictions work
            X_test = np.random.randn(5, 32, 100)
            probs = decoder2.predict_proba(X_test)
            assert probs.shape == (5, 2)


class TestDecoderCompatibility:
    """Test suite for DecoderConfig.check_compatibility() method."""

    @pytest.fixture
    def decoder_config(self):
        """Decoder config with known shapes and labels."""
        return DecoderConfig(
            model_type='EEGNet',
            num_classes=2,
            input_shapes={'eeg': [32, 250]},
            channel_labels={'eeg': ['Fp1', 'Fp2', 'F3', 'F4'] + [f'Ch{i}' for i in range(5, 33)]},
            sample_rate=500.0,
        )

    def test_check_compatibility_all_match(self, decoder_config):
        """No issues when everything matches."""
        issues = decoder_config.check_compatibility(
            system_shapes={'eeg': [32, 250]},
            system_labels={'eeg': decoder_config.channel_labels['eeg']},
            system_sample_rate=500.0,
        )
        assert issues == []

    def test_check_compatibility_missing_modality(self, decoder_config):
        """Decoder missing a modality required by system."""
        issues = decoder_config.check_compatibility(
            system_shapes={'eeg': [32, 250], 'emg': [8, 250]},
        )
        assert len(issues) == 1
        assert 'missing modality' in issues[0].lower()
        assert 'EMG' in issues[0]

    def test_check_compatibility_channel_mismatch(self, decoder_config):
        """Decoder channel count differs from system."""
        issues = decoder_config.check_compatibility(
            system_shapes={'eeg': [64, 250]},  # 64 channels vs decoder's 32
        )
        assert len(issues) == 1
        assert 'channels' in issues[0].lower()
        assert '32' in issues[0]
        assert '64' in issues[0]

    def test_check_compatibility_time_samples_ignored(self, decoder_config):
        """Time samples dimension is not validated (freely configurable)."""
        issues = decoder_config.check_compatibility(
            system_shapes={'eeg': [32, 500]},  # 500 samples vs decoder's 250 - should be OK
        )
        assert len(issues) == 0  # No issues - time dimension is flexible

    def test_check_compatibility_label_mismatch(self, decoder_config):
        """Channel labels differ between decoder and system."""
        system_labels = decoder_config.channel_labels['eeg'].copy()
        system_labels[0] = 'FP1'  # Different case/spelling

        issues = decoder_config.check_compatibility(
            system_shapes={'eeg': [32, 250]},
            system_labels={'eeg': system_labels},
        )
        assert len(issues) == 1
        assert 'label mismatch' in issues[0].lower()
        assert 'idx 0' in issues[0]

    def test_check_compatibility_labels_match(self, decoder_config):
        """No issues when labels match exactly."""
        issues = decoder_config.check_compatibility(
            system_shapes={'eeg': [32, 250]},
            system_labels={'eeg': decoder_config.channel_labels['eeg']},
        )
        assert issues == []

    def test_check_compatibility_no_decoder_labels(self):
        """No label issues when decoder has no stored labels."""
        config = DecoderConfig(
            model_type='EEGNet',
            num_classes=2,
            input_shapes={'eeg': [32, 250]},
            channel_labels=None,  # No labels stored
            sample_rate=500.0,
        )
        issues = config.check_compatibility(
            system_shapes={'eeg': [32, 250]},
            system_labels={'eeg': ['Ch1', 'Ch2']},  # System has labels
        )
        # Should not report label mismatch when decoder has no labels
        assert not any('label' in issue.lower() for issue in issues)

    def test_check_compatibility_sample_rate_mismatch(self, decoder_config):
        """Sample rate differs significantly."""
        issues = decoder_config.check_compatibility(
            system_shapes={'eeg': [32, 250]},
            system_sample_rate=250.0,  # 250 Hz vs decoder's 500 Hz
        )
        assert len(issues) == 1
        assert 'sample rate' in issues[0].lower()
        assert '500' in issues[0]
        assert '250' in issues[0]

    def test_check_compatibility_sample_rate_close(self, decoder_config):
        """Sample rate within tolerance (0.1 Hz) passes."""
        issues = decoder_config.check_compatibility(
            system_shapes={'eeg': [32, 250]},
            system_sample_rate=500.05,  # Within 0.1 Hz tolerance
        )
        assert not any('sample rate' in issue.lower() for issue in issues)

    def test_check_compatibility_case_insensitive_modality(self, decoder_config):
        """Modality keys are case-insensitive."""
        issues = decoder_config.check_compatibility(
            system_shapes={'EEG': [32, 250]},  # Uppercase
        )
        assert issues == []

    def test_check_compatibility_multiple_issues(self):
        """Multiple issues reported together."""
        config = DecoderConfig(
            model_type='EEGNet',
            num_classes=2,
            input_shapes={'eeg': [32, 250]},
            sample_rate=500.0,
        )
        issues = config.check_compatibility(
            system_shapes={'eeg': [64, 500]},  # Wrong channels (samples ignored)
            system_sample_rate=250.0,  # Wrong rate
        )
        assert len(issues) == 2  # channels and rate (samples not validated)

    def test_check_compatibility_empty_system_shapes(self, decoder_config):
        """Empty system shapes returns no issues."""
        issues = decoder_config.check_compatibility(
            system_shapes={},
        )
        assert issues == []

    def test_check_compatibility_effective_sample_rate(self):
        """Uses target_sample_rate when available (resampled decoder)."""
        config = DecoderConfig(
            model_type='EEGNet',
            num_classes=2,
            input_shapes={'eeg': [32, 125]},
            sample_rate=500.0,  # Original rate
            target_sample_rate=250.0,  # Resampled to 250 Hz
        )
        # System at 250 Hz should match (uses effective rate)
        issues = config.check_compatibility(
            system_shapes={'eeg': [32, 125]},
            system_sample_rate=250.0,
        )
        assert not any('sample rate' in issue.lower() for issue in issues)

        # System at 500 Hz should NOT match
        issues = config.check_compatibility(
            system_shapes={'eeg': [32, 125]},
            system_sample_rate=500.0,
        )
        assert any('sample rate' in issue.lower() for issue in issues)

    def test_check_compatibility_multiple_label_mismatches_all_reported(self):
        """All channel label mismatches should be reported, not just the first."""
        config = DecoderConfig(
            model_type='EEGNet',
            num_classes=2,
            input_shapes={'eeg': [4, 250]},
            channel_labels={'eeg': ['Fp1', 'Fp2', 'F3', 'F4']},
        )
        # All 4 labels differ
        system_labels = ['C3', 'C4', 'P3', 'P4']

        issues = config.check_compatibility(
            system_shapes={'eeg': [4, 250]},
            system_labels={'eeg': system_labels},
        )

        label_issues = [i for i in issues if 'label mismatch' in i.lower()]
        assert len(label_issues) == 1  # Single summary message

        # Message should mention the count of mismatches
        assert '4' in label_issues[0], (
            f"Expected mismatch count in message, got: {label_issues[0]}"
        )

    def test_check_compatibility_two_label_mismatches_both_shown(self):
        """Two mismatched labels should both appear in the message."""
        config = DecoderConfig(
            model_type='EEGNet',
            num_classes=2,
            input_shapes={'eeg': [4, 250]},
            channel_labels={'eeg': ['Fp1', 'Fp2', 'F3', 'F4']},
        )
        # First and third labels differ
        system_labels = ['C3', 'Fp2', 'P3', 'F4']

        issues = config.check_compatibility(
            system_shapes={'eeg': [4, 250]},
            system_labels={'eeg': system_labels},
        )

        label_issues = [i for i in issues if 'label mismatch' in i.lower()]
        assert len(label_issues) == 1

        # Both mismatched channels should be mentioned
        msg = label_issues[0]
        assert 'Fp1' in msg or 'C3' in msg, f"First mismatch not in message: {msg}"
        assert 'F3' in msg or 'P3' in msg, f"Second mismatch not in message: {msg}"
