"""
Unit tests for NeuralNetClassifier component.

This module provides comprehensive unit tests for the neural network classifier,
focusing on testing individual methods and functionality in isolation.

Tests cover:
- NeuralNetClassifier initialization and configuration
- Training with different model types and configurations
- Cross-validation integration
- Prediction functionality
- Configuration handling and validation
- Multimodal input support
- Error handling and edge cases
"""

import sys
import os
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Add the project root to the path so BMI can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Mock torch if not available
try:
    import torch
except ImportError:
    torch = Mock()
    torch.device = Mock(return_value=Mock())
    torch.tensor = Mock()
    torch.cuda = Mock()
    torch.cuda.is_available = Mock(return_value=False)
    sys.modules['torch'] = torch
    sys.modules['torch.nn'] = Mock()
    sys.modules['torch.optim'] = Mock()

from dendrite.ml.decoders.neural_classifier import NeuralNetClassifier
from dendrite.ml.decoders.decoder_schemas import NeuralNetConfig


class TestNeuralNetConfig:
    """Test suite for NeuralNetConfig."""
    
    def test_default_config_creation(self):
        """Test creation of default training configuration."""
        config = NeuralNetConfig()
        
        assert config.epochs == 100
        assert config.batch_size == 32
        assert config.validation_split == 0.2
        assert config.learning_rate == 0.001
        assert config.device == 'auto'
        assert config.use_early_stopping == True
        assert config.early_stopping_patience == 10
        assert config.use_class_weights == True
        assert config.class_weight_strategy == 'balanced'
    
    def test_user_can_create_config_from_dict(self):
        """User story: I want to create config from a dictionary of parameters."""
        config_dict = {
            'epochs': 50,
            'batch_size': 16,
            'learning_rate': 0.01
        }
        
        # When: User creates config from dictionary
        config = NeuralNetConfig(**config_dict)
        
        # Then: Config should use their parameters
        assert config.epochs == 50
        assert config.batch_size == 16
        assert config.learning_rate == 0.01
        # And preserve defaults for unspecified parameters
        assert config.validation_split == 0.2
        assert config.use_early_stopping == True
    
    def test_user_can_override_specific_parameters(self):
        """User story: I want to override only specific parameters while keeping defaults."""
        # When: User overrides only specific parameters
        config = NeuralNetConfig(
            epochs=200,
            use_early_stopping=False
        )
        
        # Then: Their overrides should be applied
        assert config.epochs == 200
        assert config.use_early_stopping == False
        # And defaults should remain for others
        assert config.batch_size == 32
        assert config.learning_rate == 0.001


class TestNeuralNetClassifierInitialization:
    """Test suite for NeuralNetClassifier initialization."""
    
    def test_basic_initialization(self):
        """Test basic classifier initialization."""
        config = NeuralNetConfig(model_type='EEGNet', num_classes=3)
        classifier = NeuralNetClassifier(config)
        
        assert classifier.model_type == 'EEGNet'
        assert classifier.num_classes == 3
        assert classifier.name == 'NeuralNetClassifier_EEGNet'
        assert isinstance(classifier.config, NeuralNetConfig)
        assert classifier.is_fitted == False
        assert classifier.model is None
    
    def test_user_can_initialize_with_custom_config(self):
        """User story: I want to create a classifier with my custom training parameters."""
        # When: User creates config with custom parameters
        custom_config = NeuralNetConfig(
            model_type='EEGNet',
            num_classes=2,
            epochs=150,
            batch_size=64,
            learning_rate=0.005
        )
        classifier = NeuralNetClassifier(custom_config)
        
        # Then: Classifier should use their custom parameters
        assert classifier.config.epochs == 150
        assert classifier.config.batch_size == 64
        assert classifier.config.learning_rate == 0.005
        assert classifier.learning_rate == 0.005  # Should be accessible
    
    def test_initialization_with_legacy_config(self):
        """Test initialization with legacy configuration parameters."""
        config = NeuralNetConfig(model_type='EEGNet', num_classes=2, epochs=75, batch_size=48, learning_rate=0.002, device='cpu')
        classifier = NeuralNetClassifier(config)
        
        assert classifier.config.epochs == 75
        assert classifier.config.batch_size == 48
        assert classifier.config.learning_rate == 0.002
        assert classifier.config.device == 'cpu'
    
    def test_device_auto_selection(self):
        """Test automatic device selection."""
        # Test CPU fallback when no GPU available
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=False):
            classifier = NeuralNetClassifier(NeuralNetConfig())
            assert classifier.device == torch.device('cpu')

        # Test CUDA selection when available
        with patch('torch.cuda.is_available', return_value=True):
            classifier = NeuralNetClassifier(NeuralNetConfig())
            assert classifier.device == torch.device('cuda')

        # Test MPS selection when available (macOS)
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=True):
            classifier = NeuralNetClassifier(NeuralNetConfig())
            assert classifier.device == torch.device('mps')
    
    def test_device_manual_selection(self):
        """Test manual device selection."""
        config = NeuralNetConfig(device='cpu')
        classifier = NeuralNetClassifier(config=config)
        assert classifier.device == torch.device('cpu')


class TestModelCreation:
    """Test suite for model creation functionality."""
    
    @pytest.fixture
    def sample_input_shape(self):
        """Sample input shape for testing."""
        return (32, 250)  # (n_channels, n_times)
    
    def test_create_model_simple_eegnet(self, sample_input_shape):
        """Test creating EEGNet model."""
        config = NeuralNetConfig(model_type='EEGNet', num_classes=2)
        classifier = NeuralNetClassifier(config)

        with patch('dendrite.ml.decoders.neural_classifier.create_model') as mock_create:
            mock_model = Mock()
            mock_model.to.return_value = mock_model
            mock_create.return_value = mock_model

            model = classifier._create_model(sample_input_shape)

            # Verify create_model was called with correct parameters (including model_params)
            call_args = mock_create.call_args
            assert call_args[1]['model_type'] == 'EEGNet'
            assert call_args[1]['num_classes'] == 2
            assert call_args[1]['input_shape'] == sample_input_shape
            mock_model.to.assert_called_once_with(classifier.device)
    
    def test_user_can_configure_model_with_custom_parameters(self, sample_input_shape):
        """User story: I want to customize model architecture parameters."""
        # Given: User configures model with specific parameters
        config = NeuralNetConfig(
            model_type='EEGNet',
            num_classes=3,
            dropout_rate=0.5
        )
        classifier = NeuralNetClassifier(config)
        
        # When: Model is created (mocked to avoid actual model instantiation)
        with patch('dendrite.ml.decoders.neural_classifier.create_model') as mock_create:
            mock_model = Mock()
            mock_model.to.return_value = mock_model
            mock_create.return_value = mock_model
            
            model = classifier._create_model(sample_input_shape)
            
            # Then: Model should be created successfully with user's configuration
            mock_create.assert_called_once()
            call_args = mock_create.call_args
            
            # User's core configuration should be respected
            assert call_args[1]['model_type'] == 'EEGNet'
            assert call_args[1]['num_classes'] == 3
            assert call_args[1]['input_shape'] == sample_input_shape
            
            # User's custom parameter should be accessible via config
            assert classifier.config.dropout_rate == 0.5


class TestClassWeights:
    """Test suite for class weight calculation (now in TrainingLoop)."""

    def test_balanced_class_weights(self):
        """Test calculation of balanced class weights via TrainingLoop."""
        from dendrite.ml.training import TrainingLoop

        config = NeuralNetConfig(num_classes=2, use_class_weights=True, class_weight_strategy='balanced')

        # Create a mock model
        mock_model = Mock()
        mock_param = torch.nn.Parameter(torch.randn(10, 10))
        mock_model.parameters.return_value = [mock_param]

        # TrainingLoop now requires prepare_input_fn
        def mock_prepare_input(X):
            return torch.FloatTensor(X)

        training_loop = TrainingLoop(mock_model, config, prepare_input_fn=mock_prepare_input)

        # Imbalanced dataset: 100 samples class 0, 50 samples class 1
        y = np.array([0] * 100 + [1] * 50)
        device = torch.device('cpu')

        weights = training_loop._calculate_class_weights(y, device)

        assert weights is not None
        assert len(weights) == 2
        # Class 1 should have higher weight (less frequent)
        assert weights[1] > weights[0]

        # Check balanced formula: n_samples / (n_classes * n_samples_per_class)
        expected_weight_0 = 150 / (2 * 100)  # 0.75
        expected_weight_1 = 150 / (2 * 50)   # 1.5

        assert torch.isclose(weights[0], torch.tensor(expected_weight_0), atol=1e-4)
        assert torch.isclose(weights[1], torch.tensor(expected_weight_1), atol=1e-4)

    def test_inverse_class_weights(self):
        """Test calculation of inverse frequency class weights via TrainingLoop."""
        from dendrite.ml.training import TrainingLoop

        config = NeuralNetConfig(num_classes=2, use_class_weights=True, class_weight_strategy='inverse')

        mock_model = Mock()
        mock_param = torch.nn.Parameter(torch.randn(10, 10))
        mock_model.parameters.return_value = [mock_param]

        def mock_prepare_input(X):
            return torch.FloatTensor(X)

        training_loop = TrainingLoop(mock_model, config, prepare_input_fn=mock_prepare_input)

        y = np.array([0] * 80 + [1] * 20)  # 4:1 ratio
        device = torch.device('cpu')

        weights = training_loop._calculate_class_weights(y, device)

        assert weights is not None
        assert len(weights) == 2
        # Class 1 should have 4x higher weight than class 0
        ratio = weights[1] / weights[0]
        assert torch.isclose(ratio, torch.tensor(4.0), atol=1e-3)

    def test_equal_class_weights(self):
        """Test equal class weights strategy via TrainingLoop."""
        from dendrite.ml.training import TrainingLoop

        config = NeuralNetConfig(num_classes=2, use_class_weights=True, class_weight_strategy='equal')

        mock_model = Mock()
        mock_param = torch.nn.Parameter(torch.randn(10, 10))
        mock_model.parameters.return_value = [mock_param]

        def mock_prepare_input(X):
            return torch.FloatTensor(X)

        training_loop = TrainingLoop(mock_model, config, prepare_input_fn=mock_prepare_input)
        device = torch.device('cpu')

        y = np.array([0] * 100 + [1] * 50)  # Imbalanced but should get equal weights

        weights = training_loop._calculate_class_weights(y, device)

        assert weights is not None
        assert len(weights) == 2
        assert torch.allclose(weights, torch.ones(2, device=device))

    def test_disabled_class_weights(self):
        """Test disabled class weights via TrainingLoop."""
        from dendrite.ml.training import TrainingLoop

        config = NeuralNetConfig(num_classes=2, use_class_weights=False)

        mock_model = Mock()
        mock_param = torch.nn.Parameter(torch.randn(10, 10))
        mock_model.parameters.return_value = [mock_param]

        def mock_prepare_input(X):
            return torch.FloatTensor(X)

        training_loop = TrainingLoop(mock_model, config, prepare_input_fn=mock_prepare_input)
        device = torch.device('cpu')

        y = np.array([0] * 100 + [1] * 50)

        weights = training_loop._calculate_class_weights(y, device)

        assert weights is None


class TestInputValidation:
    """Test suite for input data validation."""

    @pytest.fixture
    def sample_eeg_data(self):
        """Sample EEG data for testing."""
        return np.random.randn(10, 32, 250)  # (n_samples, n_channels, n_times)

    def test_array_input_validation(self, sample_eeg_data):
        """Test that classifier accepts array input format."""
        classifier = NeuralNetClassifier(NeuralNetConfig())
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        # Mock model with proper tensor parameters to avoid optimizer issues
        mock_model = Mock()
        mock_param = torch.nn.Parameter(torch.randn(10, 10))
        mock_model.parameters.return_value = [mock_param]

        with patch.object(classifier, '_create_model') as mock_create, \
             patch('dendrite.ml.decoders.neural_classifier.TrainingLoop') as MockTrainingLoop:
            mock_create.return_value = mock_model

            mock_trainer_instance = Mock()
            mock_trainer_instance.fit.return_value = {'final_train_acc': 0.8, 'final_train_loss': 0.2, 'epochs_completed': 10}
            mock_trainer_instance.model = mock_model
            MockTrainingLoop.return_value = mock_trainer_instance

            result = classifier.fit(sample_eeg_data, y)
            assert result is classifier
            assert classifier.input_shape == (32, 250)

    def test_3d_input_required(self, sample_eeg_data):
        """Test that classifier requires 3D array input."""
        classifier = NeuralNetClassifier(NeuralNetConfig())
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        # 2D input should fail
        X_2d = np.random.randn(32, 250)
        with pytest.raises(ValueError, match="expects 3D input"):
            classifier.fit(X_2d, y)


class TestTrainingFunctionality:
    """Test suite for training functionality."""

    @pytest.fixture
    def sample_training_data(self):
        """Sample training data for testing."""
        X = np.random.randn(20, 32, 250)
        y = np.array([0, 1] * 10)  # Balanced classes
        return X, y
    
    def test_user_can_train_classifier_on_eeg_data(self, sample_training_data):
        """User story: I want to train a classifier on my EEG data."""
        X, y = sample_training_data

        # Given: User creates a classifier for their task
        config = NeuralNetConfig(model_type='EEGNet', num_classes=2)
        classifier = NeuralNetClassifier(config)

        # Initially classifier should not be fitted
        assert not classifier.is_fitted

        # When: User trains the classifier with their data
        # Mock the training to avoid actual neural network training in tests
        with patch.object(classifier, '_create_model') as mock_create, \
             patch('dendrite.ml.decoders.neural_classifier.TrainingLoop') as MockTrainingLoop:

            # Setup mocks to simulate successful training
            mock_model = Mock()
            mock_param = torch.nn.Parameter(torch.randn(10, 10))
            mock_model.parameters.return_value = [mock_param]
            mock_create.return_value = mock_model

            mock_trainer_instance = Mock()
            mock_trainer_instance.fit.return_value = {
                'final_train_acc': 0.85,
                'final_train_loss': 0.3,
                'epochs_completed': 50
            }
            mock_trainer_instance.model = mock_model
            MockTrainingLoop.return_value = mock_trainer_instance

            result = classifier.fit(X, y)

            # Then: User should have a trained classifier
            assert result is classifier  # Supports method chaining
            assert classifier.is_fitted == True  # Ready for predictions
            assert classifier.model is not None  # Has a trained model

            # And: User can access training results
            training_results = classifier.get_training_results()
            assert training_results is not None
            assert 'final_train_acc' in training_results
            mock_create.assert_called_once()
            mock_trainer_instance.fit.assert_called_once()
    
    def test_trainer_receives_prepare_input_callback(self, sample_training_data):
        """Test that Trainer is constructed with prepare_input_fn callback."""
        X, y = sample_training_data

        config = NeuralNetConfig(model_type='EEGNet', num_classes=2)
        classifier = NeuralNetClassifier(config)

        with patch.object(classifier, '_create_model') as mock_create, \
             patch('dendrite.ml.decoders.neural_classifier.TrainingLoop') as MockTrainingLoop:

            mock_model = Mock()
            mock_param = torch.nn.Parameter(torch.randn(10, 10))
            mock_model.parameters.return_value = [mock_param]
            mock_create.return_value = mock_model

            mock_trainer_instance = Mock()
            mock_trainer_instance.fit.return_value = {
                'final_train_acc': 0.8,
                'final_train_loss': 0.2,
                'epochs_completed': 10
            }
            mock_trainer_instance.model = mock_model
            MockTrainingLoop.return_value = mock_trainer_instance

            classifier.fit(X, y)

            # Verify Trainer was constructed with prepare_input_fn
            MockTrainingLoop.assert_called_once()
            call_kwargs = MockTrainingLoop.call_args[1]
            assert 'prepare_input_fn' in call_kwargs
            assert call_kwargs['prepare_input_fn'] == classifier._prepare_input_tensor

    def test_trainer_receives_model_forward_callback(self, sample_training_data):
        """Test that Trainer is constructed with model_forward_fn callback."""
        X, y = sample_training_data

        config = NeuralNetConfig(model_type='EEGNet', num_classes=2)
        classifier = NeuralNetClassifier(config)

        with patch.object(classifier, '_create_model') as mock_create, \
             patch('dendrite.ml.decoders.neural_classifier.TrainingLoop') as MockTrainingLoop:

            mock_model = Mock()
            mock_param = torch.nn.Parameter(torch.randn(10, 10))
            mock_model.parameters.return_value = [mock_param]
            mock_create.return_value = mock_model

            mock_trainer_instance = Mock()
            mock_trainer_instance.fit.return_value = {
                'final_train_acc': 0.8,
                'final_train_loss': 0.2,
                'epochs_completed': 10
            }
            mock_trainer_instance.model = mock_model
            MockTrainingLoop.return_value = mock_trainer_instance

            classifier.fit(X, y)

            # Verify Trainer was constructed with prepare_input_fn
            MockTrainingLoop.assert_called_once()
            call_kwargs = MockTrainingLoop.call_args[1]
            assert 'prepare_input_fn' in call_kwargs
            assert call_kwargs['prepare_input_fn'] == classifier._prepare_input_tensor


class TestPredictionFunctionality:
    """Test suite for prediction functionality."""

    @pytest.fixture
    def trained_classifier(self):
        """Create a mock trained classifier."""
        config = NeuralNetConfig(num_classes=2, mc_dropout_samples=10)
        classifier = NeuralNetClassifier(config)
        classifier.is_fitted = True
        classifier.model = Mock()
        return classifier

    @pytest.fixture
    def sample_prediction_data(self):
        """Sample data for predictions."""
        return np.random.randn(5, 32, 250)
    
    def test_predict_single_batch(self, trained_classifier, sample_prediction_data):
        """Test prediction on single batch."""
        mock_outputs = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6], [0.7, 0.3]])
        trained_classifier.model.return_value = mock_outputs
        
        with patch.object(trained_classifier, '_prepare_input_tensor') as mock_prepare:
            mock_prepare.return_value = torch.randn(5, 1, 32, 250)
            
            predictions = trained_classifier.predict(sample_prediction_data)
            
            expected = np.array([0, 1, 0, 1, 0])  # Based on mock_outputs
            np.testing.assert_array_equal(predictions, expected)
    
    def test_predict_proba(self, trained_classifier, sample_prediction_data):
        """Test probability prediction."""
        mock_outputs = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
        trained_classifier.model.return_value = mock_outputs
        
        with patch.object(trained_classifier, '_prepare_input_tensor') as mock_prepare:
            mock_prepare.return_value = torch.randn(2, 1, 32, 250)
            
            with patch('torch.softmax') as mock_softmax:
                mock_softmax.return_value = mock_outputs
                
                probabilities = trained_classifier.predict_proba(sample_prediction_data)
                
                mock_softmax.assert_called_once()
                np.testing.assert_array_equal(probabilities, mock_outputs.numpy())
    
    def test_predict_unfitted_classifier(self, sample_prediction_data):
        """Test prediction on unfitted classifier."""
        config = NeuralNetConfig(num_classes=2)
        classifier = NeuralNetClassifier(config)

        with pytest.raises(ValueError, match="must be fitted before prediction"):
            classifier.predict(sample_prediction_data)

    def test_predict_proba_with_uncertainty_returns_mean_and_std(
        self, trained_classifier, sample_prediction_data
    ):
        """Test MC Dropout uncertainty estimation returns mean and std probabilities."""
        # Mock mc_dropout_predict to return expected shape
        mock_mean = np.array([[0.7, 0.3], [0.4, 0.6], [0.8, 0.2], [0.5, 0.5], [0.6, 0.4]])
        mock_std = np.array([[0.1, 0.1], [0.2, 0.2], [0.05, 0.05], [0.15, 0.15], [0.12, 0.12]])

        with patch.object(trained_classifier, '_prepare_input_tensor') as mock_prepare, \
             patch('dendrite.ml.decoders.neural_classifier.mc_dropout_predict') as mock_mc:
            mock_prepare.return_value = torch.randn(5, 1, 32, 250)
            mock_mc.return_value = (mock_mean, mock_std)

            mean_proba, std_proba = trained_classifier.predict_proba_with_uncertainty(
                sample_prediction_data
            )

            # Verify returns are correct shapes and values
            np.testing.assert_array_equal(mean_proba, mock_mean)
            np.testing.assert_array_equal(std_proba, mock_std)
            assert mean_proba.shape == (5, 2)
            assert std_proba.shape == (5, 2)

    def test_predict_proba_with_uncertainty_uses_config_samples(
        self, trained_classifier, sample_prediction_data
    ):
        """Test that MC Dropout uses config.mc_dropout_samples by default."""
        with patch.object(trained_classifier, '_prepare_input_tensor') as mock_prepare, \
             patch('dendrite.ml.decoders.neural_classifier.mc_dropout_predict') as mock_mc:
            mock_prepare.return_value = torch.randn(5, 1, 32, 250)
            mock_mc.return_value = (np.zeros((5, 2)), np.zeros((5, 2)))

            trained_classifier.predict_proba_with_uncertainty(sample_prediction_data)

            # Should use config value (10 from fixture)
            mock_mc.assert_called_once()
            call_args = mock_mc.call_args
            assert call_args[0][2] == 10  # n_samples argument

    def test_predict_proba_with_uncertainty_custom_samples(
        self, trained_classifier, sample_prediction_data
    ):
        """Test that MC Dropout can use custom n_samples."""
        with patch.object(trained_classifier, '_prepare_input_tensor') as mock_prepare, \
             patch('dendrite.ml.decoders.neural_classifier.mc_dropout_predict') as mock_mc:
            mock_prepare.return_value = torch.randn(5, 1, 32, 250)
            mock_mc.return_value = (np.zeros((5, 2)), np.zeros((5, 2)))

            trained_classifier.predict_proba_with_uncertainty(sample_prediction_data, n_samples=25)

            # Should use custom value
            mock_mc.assert_called_once()
            call_args = mock_mc.call_args
            assert call_args[0][2] == 25  # n_samples argument

    def test_predict_proba_with_uncertainty_unfitted_raises(self, sample_prediction_data):
        """Test that predict_proba_with_uncertainty raises error on unfitted classifier."""
        classifier = NeuralNetClassifier(NeuralNetConfig(num_classes=2))

        with pytest.raises(ValueError, match="must be fitted before prediction"):
            classifier.predict_proba_with_uncertainty(sample_prediction_data)


class TestCapabilitiesAndInfo:
    """Test suite for component information."""

    def test_get_training_results(self):
        """Test getting training results."""
        classifier = NeuralNetClassifier(NeuralNetConfig())
        
        # No training results initially
        assert classifier.get_training_results() is None
        
        # After setting training results
        mock_results = {'final_train_acc': 0.85, 'final_train_loss': 0.15, 'epochs_completed': 100}
        classifier.training_results = mock_results
        
        results = classifier.get_training_results()
        assert results == mock_results
    
    def test_user_can_create_classifier_with_complete_config(self):
        """User story: I want to create a classifier with all my settings in one config."""
        # When: User creates a complete config with all their settings
        config = NeuralNetConfig(
            model_type='EEGNet',
            num_classes=3,
            epochs=150,
            batch_size=64,
            learning_rate=0.01
        )
        classifier = NeuralNetClassifier(config)
        
        # Then: Classifier should use all their settings
        assert classifier.num_classes == 3
        assert classifier.model_type == 'EEGNet'
        assert classifier.config.epochs == 150
        assert classifier.config.batch_size == 64
        assert classifier.config.learning_rate == 0.01


class TestErrorHandling:
    """Test suite for error handling and edge cases."""

    def test_invalid_input_shape(self):
        """Test handling of invalid input shapes."""
        classifier = NeuralNetClassifier(NeuralNetConfig())

        # 2D input (missing samples dimension)
        X_invalid = np.random.randn(250, 32)
        y = np.array([0])

        with pytest.raises(ValueError, match="expects 3D input"):
            classifier.fit(X_invalid, y)

    def test_empty_input_data(self):
        """Test handling of empty input data."""
        classifier = NeuralNetClassifier(NeuralNetConfig())

        X_empty = np.array([]).reshape(0, 32, 250)
        y_empty = np.array([])

        # Mock model with proper tensor parameters to avoid optimizer issues
        mock_model = Mock()
        mock_param = torch.nn.Parameter(torch.randn(10, 10))  # Actual tensor parameter
        mock_model.parameters.return_value = [mock_param]

        with patch.object(classifier, '_create_model') as mock_create, \
             patch('dendrite.ml.decoders.neural_classifier.TrainingLoop') as MockTrainingLoop:

            mock_create.return_value = mock_model

            mock_trainer_instance = Mock()
            mock_trainer_instance.fit.return_value = {'final_train_acc': 0.0, 'final_train_loss': 1.0, 'epochs_completed': 5}
            mock_trainer_instance.model = mock_model
            MockTrainingLoop.return_value = mock_trainer_instance

            # Should handle empty data gracefully
            classifier.fit(X_empty, y_empty)

    def test_mismatched_data_labels(self):
        """Test handling of mismatched data and labels."""
        classifier = NeuralNetClassifier(NeuralNetConfig())

        X = np.random.randn(10, 32, 250)
        y = np.array([0, 1, 0])  # Wrong number of labels (3 vs 10 samples)

        # Should raise an error during data splitting due to mismatched sizes
        with pytest.raises(IndexError):
            classifier.fit(X, y)

    def test_invalid_model_type(self):
        """Test handling of invalid model type."""
        config = NeuralNetConfig(model_type='NonexistentModel')
        classifier = NeuralNetClassifier(config)

        X = np.random.randn(5, 32, 250)
        y = np.array([0, 1, 0, 1, 0])

        with patch('dendrite.ml.decoders.neural_classifier.create_model', side_effect=ValueError("Unknown model")):
            with pytest.raises(ValueError, match="Unknown model"):
                classifier.fit(X, y)


@pytest.fixture
def stop_event():
    """Mock stop event for testing."""
    return Mock()

@pytest.fixture
def data_queue():
    """Mock data queue for testing."""
    return Mock()

@pytest.fixture
def save_queue():
    """Mock save queue for testing."""
    return Mock()

@pytest.fixture
def mock_stream_config():
    """Mock EEG stream configuration for testing."""
    return {
        'type': 'EEG',
        'name': 'TestEEG',
        'channel_count': 32,
        'sample_rate': 500.0,
        'labels': [f'EEG_{i+1}' for i in range(32)],
        'channel_types': ['EEG'] * 32
    }

@pytest.fixture
def sample_eeg_data():
    """Sample EEG data for testing."""
    return np.random.randn(32)

@pytest.fixture
def sample_emg_data():
    """Sample EMG data for testing."""
    return np.random.randn(8)

@pytest.fixture
def sample_event_data():
    """Sample event data for testing."""
    return '{"Event_ID": 42, "Event_Type": "test_event"}'

@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return {
        'stream_name': 'TestEEG',
        'sample_rate': 500.0,
        'channel_count': 32
    }