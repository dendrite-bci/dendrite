"""Tests for offline ML storage module - TDD approach."""

import os
import tempfile
import pytest
import numpy as np

from dendrite.auxiliary.ml_workbench import TrainResult


# Module-level mock class (can be pickled)
class MockDecoder:
    """Simple mock decoder for testing (picklable)."""
    def predict(self, X):
        return np.zeros(len(X) if hasattr(X, '__len__') else 1)


class TestDecoderStorage:
    """Tests for DecoderStorage class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for storage tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_result(self):
        """Create sample TrainResult for testing."""
        return TrainResult(
            decoder=MockDecoder(),
            accuracy=0.85,
            val_accuracy=0.82,
            confusion_matrix=np.array([[10, 2], [1, 12]]),
            train_history={"loss": [1.0, 0.5, 0.2], "accuracy": [0.5, 0.8, 0.85]},
            cv_results={"mean_accuracy": 0.83, "std_accuracy": 0.02},
        )

    def test_storage_creation(self, temp_dir):
        """Storage should initialize with directory."""
        from dendrite.auxiliary.ml_workbench import DecoderStorage

        storage = DecoderStorage(temp_dir)
        assert storage is not None
        assert os.path.exists(temp_dir)

    def test_save_creates_files(self, temp_dir, sample_result):
        """Save should create model and metadata files."""
        from dendrite.auxiliary.ml_workbench import DecoderStorage

        storage = DecoderStorage(temp_dir)
        decoder_id = storage.save(sample_result, "test_decoder")

        assert decoder_id is not None
        assert len(os.listdir(temp_dir)) > 0

    def test_save_returns_id(self, temp_dir, sample_result):
        """Save should return unique decoder ID."""
        from dendrite.auxiliary.ml_workbench import DecoderStorage

        storage = DecoderStorage(temp_dir)
        id1 = storage.save(sample_result, "decoder1")
        id2 = storage.save(sample_result, "decoder2")

        assert id1 is not None
        assert id2 is not None
        assert id1 != id2

    def test_load_returns_result(self, temp_dir, sample_result):
        """Load should return saved decoder."""
        from dendrite.auxiliary.ml_workbench import DecoderStorage

        storage = DecoderStorage(temp_dir)
        decoder_id = storage.save(sample_result, "test_decoder")

        loaded = storage.load(decoder_id)

        assert loaded is not None
        assert loaded.accuracy == sample_result.accuracy
        assert loaded.val_accuracy == sample_result.val_accuracy

    def test_load_decoder_works(self, temp_dir, sample_result):
        """Loaded decoder should be usable."""
        from dendrite.auxiliary.ml_workbench import DecoderStorage

        storage = DecoderStorage(temp_dir)
        decoder_id = storage.save(sample_result, "test_decoder")

        loaded = storage.load(decoder_id)

        # Decoder should be able to predict
        assert hasattr(loaded.decoder, "predict")

    def test_list_decoders_empty(self, temp_dir):
        """List should return empty when no decoders saved."""
        from dendrite.auxiliary.ml_workbench import DecoderStorage

        storage = DecoderStorage(temp_dir)
        decoders = storage.list_decoders()

        assert decoders == []

    def test_list_decoders_with_items(self, temp_dir, sample_result):
        """List should return saved decoder info."""
        from dendrite.auxiliary.ml_workbench import DecoderStorage

        storage = DecoderStorage(temp_dir)
        storage.save(sample_result, "decoder1")
        storage.save(sample_result, "decoder2")

        decoders = storage.list_decoders()

        assert len(decoders) == 2
        assert all("name" in d for d in decoders)
        assert all("accuracy" in d for d in decoders)

    def test_delete_decoder(self, temp_dir, sample_result):
        """Delete should remove decoder."""
        from dendrite.auxiliary.ml_workbench import DecoderStorage

        storage = DecoderStorage(temp_dir)
        decoder_id = storage.save(sample_result, "test_decoder")

        assert len(storage.list_decoders()) == 1

        deleted = storage.delete(decoder_id)

        assert deleted is True
        assert len(storage.list_decoders()) == 0

    def test_load_nonexistent_raises(self, temp_dir):
        """Load nonexistent decoder should raise."""
        from dendrite.auxiliary.ml_workbench import DecoderStorage

        storage = DecoderStorage(temp_dir)

        with pytest.raises(KeyError):
            storage.load("nonexistent_id")

    def test_metadata_persistence(self, temp_dir, sample_result):
        """Metadata should persist after reload."""
        from dendrite.auxiliary.ml_workbench import DecoderStorage

        storage = DecoderStorage(temp_dir)
        decoder_id = storage.save(sample_result, "test_decoder")

        # Create new storage instance (simulates restart)
        storage2 = DecoderStorage(temp_dir)
        decoders = storage2.list_decoders()

        assert len(decoders) == 1
        assert decoders[0]["accuracy"] == 0.85
