"""
Clean unit tests for BMI Database classes.

This module provides comprehensive unit tests for the database system,
focusing on testing individual components and methods in isolation.

Tests cover:
- Database initialization and configuration
- RecordingRepository CRUD operations
- DecoderRepository CRUD operations
- Database schema creation and validation
- Error handling and edge cases
- Database connection management
"""

import sys
import os
import pytest
import sqlite3
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add the project root to the path so BMI can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dendrite.data.storage.database import Database, RecordingRepository, DecoderRepository, StudyRepository


@pytest.fixture
def temp_db_path():
    """Create a temporary database file path for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        temp_path = tmp.name
    yield temp_path
    try:
        os.unlink(temp_path)
    except FileNotFoundError:
        pass


@pytest.fixture
def bmi_database(temp_db_path):
    """Create a Database instance with temporary database."""
    db = Database(db_path=temp_db_path)
    db.init_db()
    return db


@pytest.fixture
def study_repo(bmi_database):
    """Create a StudyRepository instance."""
    return StudyRepository(bmi_database)


@pytest.fixture
def recording_repo(bmi_database):
    """Create a RecordingRepository instance."""
    return RecordingRepository(bmi_database)


@pytest.fixture
def decoder_repo(bmi_database):
    """Create a DecoderRepository instance."""
    return DecoderRepository(bmi_database)


@pytest.fixture
def test_study(study_repo):
    """Create a test study and return its data."""
    return study_repo.get_or_create('Study_Alpha', 'Test study for unit tests')


@pytest.fixture
def sample_recording_data(test_study):
    """Sample recording data for testing."""
    return {
        'study_id': test_study['study_id'],
        'recording_name': 'MotorImagery_Test',
        'session_timestamp': '20231201_120000',
        'hdf5_file_path': '/data/recordings/test_recording.h5',
        'subject_id': '01',
        'session_id': '01'
    }


@pytest.fixture
def sample_decoder_data(test_study):
    """Sample decoder data for testing."""
    return {
        'study_id': test_study['study_id'],
        'decoder_name': 'Motor_Imagery_Decoder_v1',
        'decoder_path': '/models/test_decoder.pkl',
        'model_type': 'EEGNet',
        'num_classes': 2,
        'num_channels': 64,
        'sampling_freq': 500.0,
        'source': 'Training data from Study_Alpha',
        'description': 'Test decoder for motor imagery classification'
    }


class TestDatabaseInitialization:
    """Test suite for Database initialization."""
    
    def test_default_initialization(self):
        """Test Database initialization with default path."""
        db = Database()
        
        # Check that default path is constructed correctly
        assert db.db_path.endswith('dendrite.db')
        assert 'data' in db.db_path
        
        # Check that it's a proper Database instance
        assert isinstance(db, Database)
    
    def test_custom_path_initialization(self, temp_db_path):
        """Test Database initialization with custom path."""
        db = Database(db_path=temp_db_path)
        
        assert db.db_path == temp_db_path
        assert isinstance(db, Database)
    
    def test_database_connection_context_manager(self, bmi_database):
        """Test database connection context manager."""
        with bmi_database.get_connection() as conn:
            assert isinstance(conn, sqlite3.Connection)
            assert conn.row_factory == sqlite3.Row
            
            # Test that foreign keys are enabled
            cursor = conn.execute('PRAGMA foreign_keys')
            result = cursor.fetchone()
            assert result[0] == 1
    
    def test_database_directory_creation(self, temp_db_path):
        """Test that database directory is created if it doesn't exist."""
        # Use a path in a non-existent directory
        non_existent_dir = os.path.join(os.path.dirname(temp_db_path), 'non_existent')
        db_path = os.path.join(non_existent_dir, 'test.db')
        
        db = Database(db_path=db_path)
        
        with db.get_connection() as conn:
            assert isinstance(conn, sqlite3.Connection)
        
        # Verify directory was created
        assert os.path.exists(non_existent_dir)
        
        import shutil
        shutil.rmtree(non_existent_dir)


class TestDatabaseSchemaCreation:
    """Test suite for database schema creation."""
    
    def test_init_db_creates_tables(self, temp_db_path):
        """Test that init_db creates all required tables."""
        db = Database(db_path=temp_db_path)
        db.init_db()
        
        with db.get_connection() as conn:
            # Check recordings table exists
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='recordings'
            """)
            assert cursor.fetchone() is not None
            
            # Check decoders table exists
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='decoders'
            """)
            assert cursor.fetchone() is not None
    
    def test_recordings_table_schema(self, bmi_database):
        """Test recordings table schema."""
        with bmi_database.get_connection() as conn:
            cursor = conn.execute('PRAGMA table_info(recordings)')
            columns = {row['name']: row for row in cursor.fetchall()}

            # Check required columns exist (schema uses study_id FK)
            required_columns = [
                'recording_id', 'recording_name', 'study_id',
                'subject_id', 'session_id', 'session_timestamp',
                'hdf5_file_path', 'created_at'
            ]

            for col in required_columns:
                assert col in columns, f"Missing column: {col}"

            # Check primary key
            assert columns['recording_id']['pk'] == 1

            # Check NOT NULL constraints
            assert columns['recording_name']['notnull'] == 1
            assert columns['study_id']['notnull'] == 1
            assert columns['session_timestamp']['notnull'] == 1
            assert columns['hdf5_file_path']['notnull'] == 1

    def test_decoders_table_schema(self, bmi_database):
        """Test decoders table schema."""
        with bmi_database.get_connection() as conn:
            cursor = conn.execute('PRAGMA table_info(decoders)')
            columns = {row['name']: row for row in cursor.fetchall()}

            # Check required columns exist (new simplified schema)
            required_columns = [
                'decoder_id', 'decoder_name', 'decoder_path', 'model_type',
                'num_classes', 'num_channels', 'sampling_freq',
                'source', 'description', 'created_at'
            ]

            for col in required_columns:
                assert col in columns, f"Missing column: {col}"

            # Check primary key
            assert columns['decoder_id']['pk'] == 1

            # Check NOT NULL constraints
            assert columns['decoder_name']['notnull'] == 1
            assert columns['decoder_path']['notnull'] == 1
            assert columns['model_type']['notnull'] == 1

    def test_indexes_creation(self, bmi_database):
        """Test that indexes are created properly."""
        with bmi_database.get_connection() as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='index' AND name LIKE 'idx_%'
            """)
            indexes = [row['name'] for row in cursor.fetchall()]

            # Check indexes (new schema)
            expected_indexes = [
                'idx_recordings_name',
                'idx_recordings_study',
                'idx_recordings_subject',
                'idx_recordings_session',
                'idx_decoders_name',
                'idx_decoders_model_type',
                'idx_decoders_study'
            ]

            for idx in expected_indexes:
                assert idx in indexes, f"Missing index: {idx}"


class TestRecordingRepositoryBasicOperations:
    """Test suite for RecordingRepository basic CRUD operations."""
    
    def test_add_recording_success(self, recording_repo, sample_recording_data):
        """Test successful recording addition."""
        recording_id = recording_repo.add_recording(**sample_recording_data)
        
        assert recording_id is not None
        assert isinstance(recording_id, int)
        assert recording_id > 0
    
    def test_add_recording_minimal_data(self, recording_repo, test_study):
        """Test recording addition with minimal required data."""
        recording_id = recording_repo.add_recording(
            study_id=test_study['study_id'],
            recording_name='MinimalTest',
            session_timestamp='20231201_130000',
            hdf5_file_path='/data/minimal.h5'
        )

        assert recording_id is not None
        assert isinstance(recording_id, int)
    
    def test_add_recording_duplicate_timestamp(self, recording_repo, sample_recording_data):
        """Test handling of duplicate session timestamps."""
        # Add first recording
        first_id = recording_repo.add_recording(**sample_recording_data)
        
        # Try to add duplicate
        duplicate_id = recording_repo.add_recording(**sample_recording_data)
        
        # Should return existing ID
        assert duplicate_id == first_id
    
    def test_get_by_id_success(self, recording_repo, sample_recording_data):
        """Test successful retrieval by ID."""
        recording_id = recording_repo.add_recording(**sample_recording_data)

        retrieved = recording_repo.get_by_id(recording_id)

        assert retrieved is not None
        assert retrieved['recording_id'] == recording_id
        assert retrieved['session_timestamp'] == sample_recording_data['session_timestamp']
        assert retrieved['recording_name'] == sample_recording_data['recording_name']
    
    def test_get_by_id_not_found(self, recording_repo):
        """Test retrieval of non-existent recording."""
        retrieved = recording_repo.get_by_id(999999)
        
        assert retrieved is None
    
    def test_get_by_timestamp_success(self, recording_repo, sample_recording_data):
        """Test successful retrieval by timestamp."""
        recording_repo.add_recording(**sample_recording_data)
        
        retrieved = recording_repo.get_by_timestamp(sample_recording_data['session_timestamp'])
        
        assert retrieved is not None
        assert retrieved['session_timestamp'] == sample_recording_data['session_timestamp']
    
    def test_get_by_timestamp_not_found(self, recording_repo):
        """Test retrieval of non-existent timestamp."""
        retrieved = recording_repo.get_by_timestamp('20991231_235959')
        
        assert retrieved is None
    
    def test_get_all_recordings_empty(self, recording_repo):
        """Test getting all recordings when database is empty."""
        recordings = recording_repo.get_all_recordings()
        
        assert isinstance(recordings, list)
        assert len(recordings) == 0
    
    def test_get_all_recordings_with_data(self, recording_repo, sample_recording_data):
        """Test getting all recordings with data."""
        # Add multiple recordings with unique file paths
        sample_recording_data['session_timestamp'] = '20231201_120000'
        sample_recording_data['hdf5_file_path'] = '/data/recordings/test_recording1.h5'
        recording_repo.add_recording(**sample_recording_data)
        
        sample_recording_data['session_timestamp'] = '20231201_130000'
        sample_recording_data['hdf5_file_path'] = '/data/recordings/test_recording2.h5'
        recording_repo.add_recording(**sample_recording_data)
        
        recordings = recording_repo.get_all_recordings()
        
        assert isinstance(recordings, list)
        assert len(recordings) == 2
        
        # Check ordering (should be DESC by timestamp)
        assert recordings[0]['session_timestamp'] >= recordings[1]['session_timestamp']


class TestRecordingRepositorySearchOperations:
    """Test suite for RecordingRepository search operations."""

    def test_search_recordings_by_experiment(self, recording_repo, sample_recording_data):
        """Test searching recordings by recording name."""
        recording_repo.add_recording(**sample_recording_data)

        results = recording_repo.search_recordings('MotorImagery')

        assert len(results) == 1
        assert results[0]['recording_name'] == sample_recording_data['recording_name']

    def test_search_recordings_by_study(self, recording_repo, sample_recording_data, test_study):
        """Test searching recordings by study name."""
        recording_repo.add_recording(**sample_recording_data)

        results = recording_repo.search_recordings('Alpha')

        assert len(results) == 1
        assert results[0]['study_name'] == test_study['study_name']

    def test_search_recordings_by_timestamp(self, recording_repo, sample_recording_data):
        """Test searching recordings by timestamp."""
        recording_repo.add_recording(**sample_recording_data)

        results = recording_repo.search_recordings('20231201')

        assert len(results) == 1
        assert '20231201' in results[0]['session_timestamp']

    def test_search_recordings_by_file_path(self, recording_repo, sample_recording_data):
        """Test searching recordings by file path."""
        recording_repo.add_recording(**sample_recording_data)

        results = recording_repo.search_recordings('test_recording')

        assert len(results) == 1
        assert 'test_recording' in results[0]['hdf5_file_path']

    def test_search_recordings_case_insensitive(self, recording_repo, sample_recording_data):
        """Test that search is case insensitive."""
        recording_repo.add_recording(**sample_recording_data)

        results = recording_repo.search_recordings('motorimagery')

        assert len(results) == 1
        assert results[0]['recording_name'] == sample_recording_data['recording_name']

    def test_search_recordings_no_results(self, recording_repo, sample_recording_data):
        """Test search with no matching results."""
        recording_repo.add_recording(**sample_recording_data)

        results = recording_repo.search_recordings('NonExistentTerm')

        assert isinstance(results, list)
        assert len(results) == 0

    def test_search_recordings_multiple_matches(self, recording_repo, sample_recording_data):
        """Test search returning multiple matches."""
        # Add multiple recordings with similar recording names and unique file paths
        sample_recording_data['session_timestamp'] = '20231201_120000'
        sample_recording_data['recording_name'] = 'MotorImagery_V1'
        sample_recording_data['hdf5_file_path'] = '/data/recordings/motorimagery_v1.h5'
        recording_repo.add_recording(**sample_recording_data)

        sample_recording_data['session_timestamp'] = '20231201_130000'
        sample_recording_data['recording_name'] = 'MotorImagery_V2'
        sample_recording_data['hdf5_file_path'] = '/data/recordings/motorimagery_v2.h5'
        recording_repo.add_recording(**sample_recording_data)

        results = recording_repo.search_recordings('MotorImagery')

        assert len(results) == 2
        for result in results:
            assert 'MotorImagery' in result['recording_name']


class TestDecoderRepositoryBasicOperations:
    """Test suite for DecoderRepository basic CRUD operations."""

    def test_add_decoder_success(self, decoder_repo, sample_decoder_data):
        """Test successful decoder addition."""
        decoder_id = decoder_repo.add_decoder(**sample_decoder_data)

        assert decoder_id is not None
        assert isinstance(decoder_id, int)
        assert decoder_id > 0

    def test_add_decoder_minimal_data(self, decoder_repo, test_study):
        """Test decoder addition with minimal required data."""
        decoder_id = decoder_repo.add_decoder(
            study_id=test_study['study_id'],
            decoder_name='Minimal_Decoder',
            decoder_path='/models/minimal.pkl',
            model_type='LDA'
        )

        assert decoder_id is not None
        assert isinstance(decoder_id, int)

    def test_add_decoder_duplicate_path(self, decoder_repo, sample_decoder_data):
        """Test handling of duplicate decoder paths."""
        # Add first decoder
        first_id = decoder_repo.add_decoder(**sample_decoder_data)

        # Try to add duplicate path
        duplicate_id = decoder_repo.add_decoder(**sample_decoder_data)

        # Should return None due to unique constraint
        assert duplicate_id is None

    def test_get_decoder_by_id_success(self, decoder_repo, sample_decoder_data):
        """Test successful decoder retrieval by ID."""
        decoder_id = decoder_repo.add_decoder(**sample_decoder_data)

        retrieved = decoder_repo.get_decoder_by_id(decoder_id)

        assert retrieved is not None
        assert retrieved['decoder_id'] == decoder_id
        assert retrieved['decoder_name'] == sample_decoder_data['decoder_name']
        assert retrieved['model_type'] == sample_decoder_data['model_type']
        assert retrieved['sampling_freq'] == sample_decoder_data['sampling_freq']

    def test_get_decoder_by_id_not_found(self, decoder_repo):
        """Test retrieval of non-existent decoder."""
        retrieved = decoder_repo.get_decoder_by_id(999999)

        assert retrieved is None
    
    def test_get_all_decoders_empty(self, decoder_repo):
        """Test getting all decoders when database is empty."""
        decoders = decoder_repo.get_all_decoders()
        
        assert isinstance(decoders, list)
        assert len(decoders) == 0
    
    def test_get_all_decoders_with_data(self, decoder_repo, sample_decoder_data):
        """Test getting all decoders with data."""
        # Add multiple decoders
        sample_decoder_data['decoder_path'] = '/models/decoder1.pkl'
        decoder_repo.add_decoder(**sample_decoder_data)
        
        sample_decoder_data['decoder_path'] = '/models/decoder2.pkl'
        decoder_repo.add_decoder(**sample_decoder_data)
        
        decoders = decoder_repo.get_all_decoders()
        
        assert isinstance(decoders, list)
        assert len(decoders) == 2
        
        # Check ordering (should be DESC by created_at)
        assert decoders[0]['created_at'] >= decoders[1]['created_at']


class TestDecoderRepositoryAdvancedOperations:
    """Test suite for DecoderRepository advanced operations."""
    
    def test_search_decoders_by_name(self, decoder_repo, sample_decoder_data):
        """Test searching decoders by name."""
        decoder_repo.add_decoder(**sample_decoder_data)
        
        results = decoder_repo.search_decoders('Motor_Imagery')
        
        assert len(results) == 1
        assert results[0]['decoder_name'] == sample_decoder_data['decoder_name']
    
    def test_search_decoders_by_type(self, decoder_repo, sample_decoder_data):
        """Test searching decoders by model type."""
        decoder_repo.add_decoder(**sample_decoder_data)

        results = decoder_repo.search_decoders('EEGNet')

        assert len(results) == 1
        assert results[0]['model_type'] == sample_decoder_data['model_type']
    
    def test_search_decoders_by_source(self, decoder_repo, sample_decoder_data):
        """Test searching decoders by source."""
        decoder_repo.add_decoder(**sample_decoder_data)
        
        results = decoder_repo.search_decoders('Study_Alpha')
        
        assert len(results) == 1
        assert 'Study_Alpha' in results[0]['source']
    
    def test_search_decoders_by_description(self, decoder_repo, sample_decoder_data):
        """Test searching decoders by description."""
        decoder_repo.add_decoder(**sample_decoder_data)
        
        results = decoder_repo.search_decoders('motor imagery')
        
        assert len(results) == 1
        assert 'motor imagery' in results[0]['description'].lower()
    


    
    def test_delete_decoder_success(self, decoder_repo, sample_decoder_data):
        """Test successful decoder deletion."""
        decoder_id = decoder_repo.add_decoder(**sample_decoder_data)
        
        deleted = decoder_repo.delete_decoder(decoder_id)
        
        assert deleted is True
        
        # Verify deletion
        retrieved = decoder_repo.get_decoder_by_id(decoder_id)
        assert retrieved is None
    
    def test_delete_decoder_not_found(self, decoder_repo):
        """Test deletion of non-existent decoder."""
        deleted = decoder_repo.delete_decoder(999999)
        
        assert deleted is False


class TestDecoderCompatibilityFields:
    """Test suite for decoder compatibility fields (channels and sampling frequency)."""

    def test_decoder_shape_storage_and_retrieval(self, decoder_repo, test_study):
        """Test that decoder channel info is stored and retrieved correctly."""
        decoder_id = decoder_repo.add_decoder(
            study_id=test_study['study_id'],
            decoder_name='Shape_Test_Decoder',
            decoder_path='/models/shape_test.pkl',
            model_type='EEGNet',
            num_channels=32,
            epoch_length_samples=1000,
            sampling_freq=250.0
        )

        assert decoder_id is not None

        retrieved = decoder_repo.get_decoder_by_id(decoder_id)
        assert retrieved['num_channels'] == 32
        assert retrieved['epoch_length_samples'] == 1000
        assert retrieved['sampling_freq'] == 250.0

    def test_sampling_frequency_storage_and_retrieval(self, decoder_repo, test_study):
        """Test that sampling frequency is stored and retrieved correctly."""
        test_freq = 512.0
        decoder_id = decoder_repo.add_decoder(
            study_id=test_study['study_id'],
            decoder_name='Freq_Test_Decoder',
            decoder_path='/models/freq_test.pkl',
            model_type='CSP+LDA',
            num_channels=64,
            sampling_freq=test_freq
        )

        assert decoder_id is not None

        retrieved = decoder_repo.get_decoder_by_id(decoder_id)
        assert retrieved['sampling_freq'] == test_freq
        assert isinstance(retrieved['sampling_freq'], float)

    def test_decoder_without_shape_and_freq(self, decoder_repo, test_study):
        """Test decoder creation without shape and frequency (optional fields)."""
        decoder_id = decoder_repo.add_decoder(
            study_id=test_study['study_id'],
            decoder_name='No_Metadata_Decoder',
            decoder_path='/models/no_metadata.pkl',
            model_type='SVM'
            # No num_channels or sampling_freq
        )

        assert decoder_id is not None

        retrieved = decoder_repo.get_decoder_by_id(decoder_id)
        assert retrieved['num_channels'] is None
        assert retrieved['sampling_freq'] is None
    

    def test_search_decoders_finds_shape_and_freq(self, decoder_repo, test_study):
        """Test that search results include channels and frequency fields."""
        decoder_repo.add_decoder(
            study_id=test_study['study_id'],
            decoder_name='Searchable_Decoder',
            decoder_path='/models/searchable.pkl',
            model_type='EEGNet',
            num_channels=64,
            sampling_freq=500.0,
            description='Searchable test decoder'
        )

        results = decoder_repo.search_decoders('Searchable')

        assert len(results) == 1
        result = results[0]
        assert result['num_channels'] == 64
        assert result['sampling_freq'] == 500.0

    def test_complex_decoder_shapes(self, decoder_repo, test_study):
        """Test storage of various channel configurations."""
        channel_counts = [32, 64, 128]

        decoder_ids = []
        for i, channels in enumerate(channel_counts):
            decoder_id = decoder_repo.add_decoder(
                study_id=test_study['study_id'],
                decoder_name=f'Channel_Test_Decoder_{i}',
                decoder_path=f'/models/complex_{i}.pkl',
                model_type='CNN',
                num_channels=channels,
                sampling_freq=500.0
            )
            decoder_ids.append(decoder_id)

        # Verify all channel counts stored correctly
        for i, decoder_id in enumerate(decoder_ids):
            retrieved = decoder_repo.get_decoder_by_id(decoder_id)
            assert retrieved['num_channels'] == channel_counts[i]

    def test_sampling_frequency_precision(self, decoder_repo, test_study):
        """Test that sampling frequency maintains precision."""
        precise_frequencies = [
            500.0,
            512.0,
            1024.0,
            2048.5,  # Half Hz precision
            125.125  # Multiple decimal places
        ]

        decoder_ids = []
        for i, freq in enumerate(precise_frequencies):
            decoder_id = decoder_repo.add_decoder(
                study_id=test_study['study_id'],
                decoder_name=f'Precision_Test_Decoder_{i}',
                decoder_path=f'/models/precision_{i}.pkl',
                model_type='LDA',
                sampling_freq=freq
            )
            decoder_ids.append(decoder_id)

        # Verify precision maintained
        for i, decoder_id in enumerate(decoder_ids):
            retrieved = decoder_repo.get_decoder_by_id(decoder_id)
            assert retrieved['sampling_freq'] == precise_frequencies[i]


class TestDatabaseConstraintsAndIntegrity:
    """Test suite for database constraints and data integrity."""

    def test_recording_unique_timestamp_constraint(self, bmi_database, study_repo):
        """Test unique constraint on recording session_timestamp."""
        # Create a study first
        study = study_repo.get_or_create('ConstraintTestStudy')
        study_id = study['study_id']

        with bmi_database.get_connection() as conn:
            # Insert first record
            conn.execute("""
                INSERT INTO recordings (recording_name, study_id, session_timestamp, hdf5_file_path)
                VALUES (?, ?, ?, ?)
            """, ('Test1', study_id, '20231201_120000', '/data/test1.h5'))

            # Try to insert duplicate timestamp
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute("""
                    INSERT INTO recordings (recording_name, study_id, session_timestamp, hdf5_file_path)
                    VALUES (?, ?, ?, ?)
                """, ('Test2', study_id, '20231201_120000', '/data/test2.h5'))

    def test_recording_unique_file_path_constraint(self, bmi_database, study_repo):
        """Test unique constraint on recording hdf5_file_path."""
        # Create a study first
        study = study_repo.get_or_create('FilePathTestStudy')
        study_id = study['study_id']

        with bmi_database.get_connection() as conn:
            # Insert first record
            conn.execute("""
                INSERT INTO recordings (recording_name, study_id, session_timestamp, hdf5_file_path)
                VALUES (?, ?, ?, ?)
            """, ('Test1', study_id, '20231201_120000', '/data/test.h5'))

            # Try to insert duplicate file path
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute("""
                    INSERT INTO recordings (recording_name, study_id, session_timestamp, hdf5_file_path)
                    VALUES (?, ?, ?, ?)
                """, ('Test2', study_id, '20231201_130000', '/data/test.h5'))

    def test_decoder_unique_path_constraint(self, bmi_database, study_repo):
        """Test unique constraint on decoder path."""
        # Create a study first
        study = study_repo.get_or_create('DecoderPathTestStudy')
        study_id = study['study_id']

        with bmi_database.get_connection() as conn:
            # Insert first decoder
            conn.execute("""
                INSERT INTO decoders (study_id, decoder_name, decoder_path, model_type)
                VALUES (?, ?, ?, ?)
            """, (study_id, 'Decoder1', '/models/test.pkl', 'EEGNet'))

            # Try to insert duplicate path
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute("""
                    INSERT INTO decoders (study_id, decoder_name, decoder_path, model_type)
                    VALUES (?, ?, ?, ?)
                """, (study_id, 'Decoder2', '/models/test.pkl', 'LDA'))

    def test_recording_required_fields(self, bmi_database, study_repo):
        """Test that required fields cannot be NULL."""
        # Create a valid study first
        study = study_repo.get_or_create('RequiredFieldsTestStudy')
        study_id = study['study_id']

        with bmi_database.get_connection() as conn:
            # Try to insert with missing recording_name
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute("""
                    INSERT INTO recordings (study_id, session_timestamp, hdf5_file_path)
                    VALUES (?, ?, ?)
                """, (study_id, '20231201_120000', '/data/test.h5'))

            # Try to insert with missing study_id (FK constraint)
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute("""
                    INSERT INTO recordings (recording_name, session_timestamp, hdf5_file_path)
                    VALUES (?, ?, ?)
                """, ('Test', '20231201_120000', '/data/test.h5'))

            # Try to insert with missing session_timestamp
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute("""
                    INSERT INTO recordings (recording_name, study_id, hdf5_file_path)
                    VALUES (?, ?, ?)
                """, ('Test', study_id, '/data/test.h5'))

    def test_decoder_required_fields(self, bmi_database):
        """Test that decoder required fields cannot be NULL."""
        with bmi_database.get_connection() as conn:
            # Try to insert with missing decoder_name
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute("""
                    INSERT INTO decoders (decoder_path, model_type)
                    VALUES (?, ?)
                """, ('/models/test.pkl', 'EEGNet'))

            # Try to insert with missing model_type
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute("""
                    INSERT INTO decoders (decoder_name, decoder_path)
                    VALUES (?, ?)
                """, ('TestDecoder', '/models/test.pkl'))

            # Try to insert with missing decoder_path
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute("""
                    INSERT INTO decoders (decoder_name, model_type)
                    VALUES (?, ?)
                """, ('TestDecoder', 'EEGNet'))


class TestErrorHandlingAndEdgeCases:
    """Test suite for error handling and edge cases."""
    
    def test_database_connection_error_handling(self):
        """Test handling of database connection errors."""
        # Try to connect to a path that cannot be created (e.g., on a readonly filesystem)
        # Use a path with invalid characters for Windows
        import platform
        if platform.system() == 'Windows':
            invalid_path = 'C:\\invalid<>path\\database.db'  # Invalid characters < >
        else:
            invalid_path = '/proc/invalid/database.db'  # /proc is typically readonly
        
        db = Database(db_path=invalid_path)
        
        # Should raise an error when trying to connect to invalid path
        with pytest.raises((sqlite3.OperationalError, OSError, PermissionError)):
            with db.get_connection() as conn:
                conn.execute('SELECT 1')
    
    def test_empty_string_parameters(self, recording_repo, test_study):
        """Test handling of empty string parameters."""
        recording_id = recording_repo.add_recording(
            study_id=test_study['study_id'],
            recording_name='',
            session_timestamp='',
            hdf5_file_path=''
        )

        # Should succeed (empty strings are valid)
        assert recording_id is not None

        retrieved = recording_repo.get_by_id(recording_id)
        assert retrieved['session_timestamp'] == ''
        assert retrieved['hdf5_file_path'] == ''
        assert retrieved['recording_name'] == ''

    def test_very_long_strings(self, recording_repo, test_study):
        """Test handling of very long string parameters."""
        long_string = 'x' * 10000

        recording_id = recording_repo.add_recording(
            study_id=test_study['study_id'],
            recording_name=long_string,
            session_timestamp='20231201_120000',
            hdf5_file_path=long_string
        )

        assert recording_id is not None

        retrieved = recording_repo.get_by_id(recording_id)
        assert len(retrieved['hdf5_file_path']) == 10000
        assert len(retrieved['recording_name']) == 10000

    def test_special_characters_in_strings(self, recording_repo, test_study):
        """Test handling of special characters in strings."""
        special_chars = "Test with 'quotes' and \"double quotes\" and \\ backslashes"

        recording_id = recording_repo.add_recording(
            study_id=test_study['study_id'],
            recording_name=special_chars,
            session_timestamp='20231201_120000',
            hdf5_file_path='/data/test.h5'
        )

        assert recording_id is not None

        retrieved = recording_repo.get_by_id(recording_id)
        assert retrieved['recording_name'] == special_chars

    def test_json_data_handling(self, decoder_repo, test_study):
        """Test handling of JSON data in decoder fields."""
        valid_json = '{"valid": "json", "accuracy": 0.85}'

        decoder_id = decoder_repo.add_decoder(
            study_id=test_study['study_id'],
            decoder_name='TestDecoder',
            decoder_path='/models/test.pkl',
            model_type='EEGNet',
            preprocessing_config=valid_json
        )

        assert decoder_id is not None

        retrieved = decoder_repo.get_decoder_by_id(decoder_id)
        assert retrieved['preprocessing_config'] == valid_json

        # Test that valid JSON can be parsed
        config = json.loads(retrieved['preprocessing_config'])
        assert config['accuracy'] == 0.85
    
    def test_search_with_sql_injection_attempts(self, recording_repo, sample_recording_data):
        """Test that search is protected against SQL injection."""
        recording_repo.add_recording(**sample_recording_data)
        
        # Try various SQL injection attempts
        injection_attempts = [
            "'; DROP TABLE recordings; --",
            "' OR '1'='1",
            "'; INSERT INTO recordings VALUES (...); --",
            "test'; DELETE FROM recordings WHERE '1'='1"
        ]
        
        for attempt in injection_attempts:
            results = recording_repo.search_recordings(attempt)
            # Should return empty results, not cause database corruption
            assert isinstance(results, list)
            
        # Verify original data is still there
        all_recordings = recording_repo.get_all_recordings()
        assert len(all_recordings) == 1


class TestDatabasePerformanceAndScaling:
    """Test suite for database performance and scaling considerations."""

    def test_bulk_recording_insertion(self, recording_repo, test_study):
        """Test performance with bulk recording insertions."""
        # Insert multiple recordings quickly
        recording_ids = []

        for i in range(50):
            recording_id = recording_repo.add_recording(
                study_id=test_study['study_id'],
                recording_name=f'Experiment_{i}',
                session_timestamp=f'20231201_{i:06d}',
                hdf5_file_path=f'/data/recording_{i}.h5'
            )
            recording_ids.append(recording_id)

        assert len(recording_ids) == 50
        assert all(id is not None for id in recording_ids)

        # Verify all can be retrieved
        all_recordings = recording_repo.get_all_recordings()
        assert len(all_recordings) == 50

    def test_search_performance_with_many_records(self, recording_repo, test_study):
        """Test search performance with many records."""
        # Insert records with various recording names
        recording_types = ['MotorImagery', 'P300', 'SSVEP', 'RestingState']

        for i in range(100):
            recording_type = recording_types[i % len(recording_types)]
            recording_repo.add_recording(
                study_id=test_study['study_id'],
                recording_name=f'{recording_type}_Session_{i}',
                session_timestamp=f'20231201_{i:06d}',
                hdf5_file_path=f'/data/recording_{i}.h5'
            )

        # Search should be fast even with many records
        results = recording_repo.search_recordings('MotorImagery')

        # Should find approximately 25 records (100/4)
        assert len(results) >= 20
        assert all('MotorImagery' in r['recording_name'] for r in results)

    def test_index_effectiveness(self, bmi_database, recording_repo, test_study):
        """Test that indexes improve query performance."""
        # Add test data
        for i in range(20):
            recording_repo.add_recording(
                study_id=test_study['study_id'],
                recording_name=f'Experiment_{i % 5}',
                session_timestamp=f'20231201_{i:06d}',
                hdf5_file_path=f'/data/recording_{i}.h5'
            )
        
        with bmi_database.get_connection() as conn:
            # Query that should use recording_name index
            cursor = conn.execute("""
                EXPLAIN QUERY PLAN
                SELECT * FROM recordings WHERE recording_name = 'Experiment_1'
            """)
            plan = cursor.fetchall()

            # Check that an index is being used (not a full table scan)
            plan_text = ' '.join([' '.join([str(col) for col in row]) for row in plan])
            # Should mention index usage, not SCAN TABLE
            assert 'idx_recordings_name' in plan_text or 'USING INDEX' in plan_text.upper()


class TestRepositoryIntegration:
    """Test suite for integration between repositories."""
    
    def test_recording_and_decoder_independent_operations(self, recording_repo, decoder_repo, 
                                                          sample_recording_data, sample_decoder_data):
        """Test that recording and decoder operations are independent."""
        # Add recording
        recording_id = recording_repo.add_recording(**sample_recording_data)
        
        # Add decoder
        decoder_id = decoder_repo.add_decoder(**sample_decoder_data)
        
        assert recording_id is not None
        assert decoder_id is not None
        
        # Both should be retrievable independently
        recording = recording_repo.get_by_id(recording_id)
        decoder = decoder_repo.get_decoder_by_id(decoder_id)
        
        assert recording is not None
        assert decoder is not None
    
    def test_shared_database_connection(self, bmi_database):
        """Test that repositories can share database connections."""
        study_repo = StudyRepository(bmi_database)
        recording_repo = RecordingRepository(bmi_database)
        decoder_repo = DecoderRepository(bmi_database)

        # All should use the same database
        assert recording_repo.db == decoder_repo.db
        assert study_repo.db == decoder_repo.db

        # Create a study first
        study = study_repo.get_or_create('TestStudy')
        study_id = study['study_id']

        # Both should be able to operate simultaneously
        recording_id = recording_repo.add_recording(
            study_id=study_id,
            recording_name='Test',
            session_timestamp='20231201_120000',
            hdf5_file_path='/data/test.h5'
        )

        decoder_id = decoder_repo.add_decoder(
            study_id=study_id,
            decoder_name='TestDecoder',
            decoder_path='/models/test.pkl',
            model_type='EEGNet'
        )

        assert recording_id is not None
        assert decoder_id is not None


class TestTimestampHandling:
    """Test suite for timestamp handling in database operations."""

    def test_created_at_timestamp_automatic(self, recording_repo, sample_recording_data):
        """Test that created_at timestamp is set automatically."""
        recording_id = recording_repo.add_recording(**sample_recording_data)

        retrieved = recording_repo.get_by_id(recording_id)

        assert retrieved['created_at'] is not None
        # Should be a recent timestamp



class TestStudyRepository:
    """Test suite for StudyRepository operations."""

    def test_get_or_create_creates_new_study(self, study_repo):
        """Test that get_or_create creates a new study if it doesn't exist."""
        study = study_repo.get_or_create('NewTestStudy', 'Test description')

        assert study is not None
        assert study['study_name'] == 'NewTestStudy'
        assert study['study_id'] is not None

    def test_get_or_create_returns_existing_study(self, study_repo):
        """Test that get_or_create returns existing study without duplicating."""
        study1 = study_repo.get_or_create('ExistingStudy')
        study2 = study_repo.get_or_create('ExistingStudy')

        assert study1['study_id'] == study2['study_id']

    def test_study_cascade_delete_recordings(self, study_repo, recording_repo):
        """Test that deleting a study cascades to delete all its recordings."""
        # Create study
        study = study_repo.get_or_create('CascadeTestStudy')
        study_id = study['study_id']

        # Add recordings
        recording_repo.add_recording(
            study_id=study_id,
            recording_name='CascadeRec1',
            session_timestamp='20231201_100000',
            hdf5_file_path='/data/cascade_test1.h5'
        )
        recording_repo.add_recording(
            study_id=study_id,
            recording_name='CascadeRec2',
            session_timestamp='20231201_110000',
            hdf5_file_path='/data/cascade_test2.h5'
        )

        # Verify recordings exist
        recordings = recording_repo.get_recordings_by_study(study_id)
        assert len(recordings) == 2

        # Delete study
        success = study_repo.delete_study(study_id)
        assert success is True

        # Verify recordings are deleted (cascade)
        recordings_after = recording_repo.get_recordings_by_study(study_id)
        assert len(recordings_after) == 0

    def test_study_cascade_delete_decoders(self, study_repo, decoder_repo):
        """Test that deleting a study cascades to delete all its decoders."""
        # Create study
        study = study_repo.get_or_create('DecoderCascadeStudy')
        study_id = study['study_id']

        # Add decoders
        decoder_repo.add_decoder(
            study_id=study_id,
            decoder_name='CascadeDec1',
            decoder_path='/models/cascade_test1.pkl',
            model_type='EEGNet'
        )
        decoder_repo.add_decoder(
            study_id=study_id,
            decoder_name='CascadeDec2',
            decoder_path='/models/cascade_test2.pkl',
            model_type='LDA'
        )

        # Verify decoders exist
        decoders = decoder_repo.get_decoders_by_study(study_id)
        assert len(decoders) == 2

        # Delete study
        success = study_repo.delete_study(study_id)
        assert success is True

        # Verify decoders are deleted (cascade)
        decoders_after = decoder_repo.get_decoders_by_study(study_id)
        assert len(decoders_after) == 0