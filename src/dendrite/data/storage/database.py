"""
Dendrite Database Backend

SQLite database for experiment metadata - recordings and trained decoders.
"""

import logging
import os
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from dendrite.constants import DATABASE_PATH

logger = logging.getLogger(__name__)


class Database:
    """Database backend for Dendrite experiments."""

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            self.db_path = str(DATABASE_PATH)
        else:
            self.db_path = db_path

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self) -> None:
        """Initialize the database schema."""
        logger.info(f"Initializing Dendrite database at: {self.db_path}")
        with self.get_connection() as conn:
            self._create_tables(conn)
            self._create_indexes(conn)

    def _create_tables(self, conn: sqlite3.Connection) -> None:
        """Create all database tables."""
        self._create_studies_table(conn)
        self._create_recordings_table(conn)
        self._create_datasets_table(conn)
        self._create_decoders_table(conn)

    def _create_studies_table(self, conn: sqlite3.Connection) -> None:
        """Create studies table - master table for organizing all experiment data."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS studies (
                study_id INTEGER PRIMARY KEY AUTOINCREMENT,
                study_name TEXT UNIQUE NOT NULL,
                description TEXT,
                study_config TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def _create_recordings_table(self, conn: sqlite3.Connection) -> None:
        """Create recordings table - raw EEG recording metadata (belongs to study)."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS recordings (
                recording_id INTEGER PRIMARY KEY AUTOINCREMENT,
                study_id INTEGER NOT NULL,
                recording_name TEXT NOT NULL,
                subject_id TEXT NOT NULL DEFAULT '',
                session_id TEXT NOT NULL DEFAULT '',
                run_number INTEGER NOT NULL DEFAULT 1,
                session_timestamp TEXT UNIQUE NOT NULL,
                hdf5_file_path TEXT UNIQUE NOT NULL,
                metrics_file_path TEXT,
                config_file_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (study_id) REFERENCES studies(study_id) ON DELETE CASCADE
            )
        """)

    def _create_datasets_table(self, conn: sqlite3.Connection) -> None:
        """Create datasets table - custom FIF files for offline ML training."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
                study_id INTEGER,
                name TEXT UNIQUE NOT NULL,
                file_path TEXT NOT NULL,
                events_json TEXT,
                epoch_tmin REAL DEFAULT -0.2,
                epoch_tmax REAL DEFAULT 0.8,
                sampling_rate REAL,
                target_sample_rate REAL,
                preproc_lowcut REAL DEFAULT 0.5,
                preproc_highcut REAL DEFAULT 50.0,
                preproc_rereference INTEGER DEFAULT 0,
                paradigm TEXT,
                modality TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (study_id) REFERENCES studies(study_id) ON DELETE SET NULL
            )
        """)

    def _create_decoders_table(self, conn: sqlite3.Connection) -> None:
        """Create decoders table - trained model metadata (belongs to study)."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS decoders (
                decoder_id INTEGER PRIMARY KEY AUTOINCREMENT,
                study_id INTEGER NOT NULL,
                decoder_name TEXT NOT NULL,
                decoder_path TEXT UNIQUE NOT NULL,
                model_type TEXT NOT NULL,
                classifier_type TEXT,
                num_classes INTEGER,
                num_channels INTEGER,
                sampling_freq REAL,
                epoch_length_samples INTEGER,
                start_offset REAL,
                end_offset REAL,
                training_accuracy REAL,
                validation_accuracy REAL,
                cv_mean_accuracy REAL,
                cv_std_accuracy REAL,
                cv_folds INTEGER,
                preprocessing_config TEXT,
                channel_names TEXT,
                class_labels TEXT,
                training_dataset_name TEXT,
                modality TEXT,
                source TEXT,
                description TEXT,
                training_config TEXT,
                search_result TEXT,
                training_recording_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (study_id) REFERENCES studies(study_id) ON DELETE CASCADE,
                FOREIGN KEY (training_recording_id) REFERENCES recordings(recording_id)
                    ON DELETE SET NULL
            )
        """)

    def _create_indexes(self, conn: sqlite3.Connection) -> None:
        """Create indexes for query performance."""
        conn.execute("CREATE INDEX IF NOT EXISTS idx_studies_name ON studies(study_name)")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_recordings_name ON recordings(recording_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_recordings_study ON recordings(study_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_recordings_subject ON recordings(subject_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_recordings_session ON recordings(session_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_recordings_bids ON recordings(subject_id, session_id, recording_name)"
        )

        conn.execute("CREATE INDEX IF NOT EXISTS idx_decoders_name ON decoders(decoder_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_decoders_model_type ON decoders(model_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_decoders_study ON decoders(study_id)")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_datasets_study ON datasets(study_id)")


class RecordingRepository:
    """Repository for recording operations."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def add_recording(
        self,
        study_id: int,
        recording_name: str,
        session_timestamp: str,
        hdf5_file_path: str,
        subject_id: str = "",
        session_id: str = "",
        run_number: int = 1,
        metrics_file_path: str | None = None,
        config_file_path: str | None = None,
    ) -> int | None:
        """Add a new recording."""
        with self.db.get_connection() as conn:
            try:
                cursor = conn.execute(
                    """INSERT INTO recordings (study_id, recording_name, subject_id, session_id, run_number, session_timestamp, hdf5_file_path, metrics_file_path, config_file_path)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        study_id,
                        recording_name,
                        subject_id,
                        session_id,
                        run_number,
                        session_timestamp,
                        hdf5_file_path,
                        metrics_file_path,
                        config_file_path,
                    ),
                )
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                existing = self.get_by_timestamp(session_timestamp)
                return existing["recording_id"] if existing else None

    def get_by_id(self, recording_id: int) -> dict[str, Any] | None:
        """Get recording by ID with study_name included."""
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT r.*, s.study_name
                FROM recordings r
                JOIN studies s ON r.study_id = s.study_id
                WHERE r.recording_id = ?
            """,
                (recording_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_by_timestamp(self, session_timestamp: str) -> dict[str, Any] | None:
        """Get recording by timestamp with study_name included."""
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT r.*, s.study_name
                FROM recordings r
                JOIN studies s ON r.study_id = s.study_id
                WHERE r.session_timestamp = ?
            """,
                (session_timestamp,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_recordings(self) -> list[dict[str, Any]]:
        """Get all recordings with study_name, newest first."""
        with self.db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT r.*, s.study_name
                FROM recordings r
                JOIN studies s ON r.study_id = s.study_id
                ORDER BY r.recording_id DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def search_recordings(self, search_term: str) -> list[dict[str, Any]]:
        """Search recordings by term."""
        with self.db.get_connection() as conn:
            query = """
                SELECT r.*, s.study_name
                FROM recordings r
                JOIN studies s ON r.study_id = s.study_id
                WHERE LOWER(r.recording_name) LIKE LOWER(?) OR
                      LOWER(s.study_name) LIKE LOWER(?) OR
                      LOWER(r.subject_id) LIKE LOWER(?) OR
                      LOWER(r.session_id) LIKE LOWER(?) OR
                      LOWER(r.session_timestamp) LIKE LOWER(?) OR
                      LOWER(r.hdf5_file_path) LIKE LOWER(?)
                ORDER BY r.recording_id DESC
            """
            like_term = f"%{search_term}%"
            cursor = conn.execute(query, (like_term,) * 6)
            return [dict(row) for row in cursor.fetchall()]

    def delete_recording(self, recording_id: int) -> bool:
        """Delete a recording."""
        with self.db.get_connection() as conn:
            try:
                cursor = conn.execute(
                    "DELETE FROM recordings WHERE recording_id = ?", (recording_id,)
                )
                conn.commit()
                return cursor.rowcount > 0
            except sqlite3.Error as e:
                logger.warning(f"Failed to delete recording {recording_id}: {e}")
                return False

    def get_recordings_by_study(self, study: int | str) -> list[dict[str, Any]]:
        """Get all recordings for a study by study_id (int) or study_name (str)."""
        with self.db.get_connection() as conn:
            if isinstance(study, int):
                cursor = conn.execute(
                    """
                    SELECT r.*, s.study_name
                    FROM recordings r
                    JOIN studies s ON r.study_id = s.study_id
                    WHERE r.study_id = ?
                    ORDER BY r.recording_id DESC
                """,
                    (study,),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT r.*, s.study_name
                    FROM recordings r
                    JOIN studies s ON r.study_id = s.study_id
                    WHERE s.study_name = ?
                    ORDER BY r.recording_id DESC
                """,
                    (study,),
                )
            return [dict(row) for row in cursor.fetchall()]

    def get_existing_subjects(self, study_name: str | None = None) -> list[str]:
        """Get distinct subject IDs, optionally filtered by study name."""
        with self.db.get_connection() as conn:
            if study_name:
                cursor = conn.execute(
                    """
                    SELECT DISTINCT r.subject_id
                    FROM recordings r
                    JOIN studies s ON r.study_id = s.study_id
                    WHERE s.study_name = ? AND r.subject_id != ''
                    ORDER BY r.subject_id
                """,
                    (study_name,),
                )
            else:
                cursor = conn.execute(
                    "SELECT DISTINCT subject_id FROM recordings WHERE subject_id != '' ORDER BY subject_id"
                )
            return [row["subject_id"] for row in cursor.fetchall()]

    def get_existing_sessions(self, study_name: str | None = None, subject_id: str | None = None) -> list[str]:
        """Get distinct session IDs, optionally filtered by study name and subject."""
        with self.db.get_connection() as conn:
            if study_name and subject_id:
                cursor = conn.execute(
                    """
                    SELECT DISTINCT r.session_id
                    FROM recordings r
                    JOIN studies s ON r.study_id = s.study_id
                    WHERE s.study_name = ? AND r.subject_id = ? AND r.session_id != ''
                    ORDER BY r.session_id
                """,
                    (study_name, subject_id),
                )
            elif study_name:
                cursor = conn.execute(
                    """
                    SELECT DISTINCT r.session_id
                    FROM recordings r
                    JOIN studies s ON r.study_id = s.study_id
                    WHERE s.study_name = ? AND r.session_id != ''
                    ORDER BY r.session_id
                """,
                    (study_name,),
                )
            else:
                cursor = conn.execute(
                    "SELECT DISTINCT session_id FROM recordings WHERE session_id != '' ORDER BY session_id"
                )
            return [row["session_id"] for row in cursor.fetchall()]

    def get_next_session_id(self, study_name: str, subject_id: str) -> str:
        """Suggest next session ID for a study+subject (e.g., '01', '02', '03')."""
        existing = self.get_existing_sessions(study_name, subject_id)
        if not existing:
            return "01"
        max_num = 0
        for ses in existing:
            try:
                num = int(ses.lstrip("0") or "0")
                max_num = max(max_num, num)
            except ValueError:
                continue
        return f"{max_num + 1:02d}"

    def get_next_run_number(self, subject_id: str, session_id: str, recording_name: str) -> int:
        """Get next BIDS run number for a subject+session+task combination."""
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """SELECT MAX(run_number) as max_run FROM recordings
                   WHERE subject_id = ? AND session_id = ? AND recording_name = ?""",
                (subject_id, session_id, recording_name),
            )
            row = cursor.fetchone()
            max_run = row["max_run"] if row and row["max_run"] else 0
            return max_run + 1


class DecoderRepository:
    """Repository for decoder operations."""

    DECODER_FIELDS = [
        "decoder_name",
        "decoder_path",
        "model_type",
        "classifier_type",
        "num_classes",
        "num_channels",
        "sampling_freq",
        "epoch_length_samples",
        "start_offset",
        "end_offset",
        "training_accuracy",
        "validation_accuracy",
        "cv_mean_accuracy",
        "cv_std_accuracy",
        "cv_folds",
        "preprocessing_config",
        "channel_names",
        "class_labels",
        "training_dataset_name",
        "modality",
        "source",
        "description",
        "training_config",
        "search_result",
        "training_recording_id",
    ]

    def __init__(self, db: Database) -> None:
        self.db = db

    def add_decoder(
        self, study_id: int, decoder_name: str, decoder_path: str, model_type: str, **kwargs
    ) -> int | None:
        """Add a new decoder."""
        with self.db.get_connection() as conn:
            try:
                fields = ["study_id", "decoder_name", "decoder_path", "model_type"]
                values = [study_id, decoder_name, decoder_path, model_type]

                for field in self.DECODER_FIELDS[3:]:
                    if field in kwargs:
                        fields.append(field)
                        values.append(kwargs[field])

                placeholders = ", ".join(["?" for _ in values])
                field_names = ", ".join(fields)

                cursor = conn.execute(
                    f"INSERT INTO decoders ({field_names}) VALUES ({placeholders})", values
                )
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                return None

    def get_decoder_by_id(self, decoder_id: int) -> dict[str, Any] | None:
        """Get decoder by ID with study_name included."""
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT d.*, s.study_name
                FROM decoders d
                JOIN studies s ON d.study_id = s.study_id
                WHERE d.decoder_id = ?
            """,
                (decoder_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_decoders(self) -> list[dict[str, Any]]:
        """Get all decoders with study_name, newest first."""
        with self.db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT d.*, s.study_name
                FROM decoders d
                JOIN studies s ON d.study_id = s.study_id
                ORDER BY d.decoder_id DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_decoders_by_study(self, study: int | str) -> list[dict[str, Any]]:
        """Get all decoders for a study by study_id (int) or study_name (str)."""
        with self.db.get_connection() as conn:
            if isinstance(study, int):
                cursor = conn.execute(
                    """
                    SELECT d.*, s.study_name
                    FROM decoders d
                    JOIN studies s ON d.study_id = s.study_id
                    WHERE d.study_id = ?
                    ORDER BY d.decoder_id DESC
                """,
                    (study,),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT d.*, s.study_name
                    FROM decoders d
                    JOIN studies s ON d.study_id = s.study_id
                    WHERE s.study_name = ?
                    ORDER BY d.decoder_id DESC
                """,
                    (study,),
                )
            return [dict(row) for row in cursor.fetchall()]

    def search_decoders(self, search_term: str) -> list[dict[str, Any]]:
        """Search decoders by term."""
        with self.db.get_connection() as conn:
            query = """
                SELECT d.*, s.study_name
                FROM decoders d
                JOIN studies s ON d.study_id = s.study_id
                WHERE LOWER(d.decoder_name) LIKE LOWER(?) OR
                      LOWER(d.model_type) LIKE LOWER(?) OR
                      LOWER(s.study_name) LIKE LOWER(?) OR
                      LOWER(d.source) LIKE LOWER(?) OR
                      LOWER(d.description) LIKE LOWER(?)
                ORDER BY d.decoder_id DESC
            """
            like_term = f"%{search_term}%"
            cursor = conn.execute(query, (like_term,) * 5)
            return [dict(row) for row in cursor.fetchall()]

    def delete_decoder(self, decoder_id: int) -> bool:
        """Delete a decoder."""
        with self.db.get_connection() as conn:
            try:
                cursor = conn.execute("DELETE FROM decoders WHERE decoder_id = ?", (decoder_id,))
                conn.commit()
                return cursor.rowcount > 0
            except sqlite3.Error as e:
                logger.warning(f"Failed to delete decoder {decoder_id}: {e}")
                return False

    def get_by_path(self, decoder_path: str) -> dict[str, Any] | None:
        """Get decoder by file path with study_name included."""
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT d.*, s.study_name
                FROM decoders d
                JOIN studies s ON d.study_id = s.study_id
                WHERE d.decoder_path = ?
            """,
                (decoder_path,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None


class StudyRepository:
    """Repository for study operations."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def get_or_create(self, study_name: str, description: str | None = None) -> dict[str, Any]:
        """Get study by name or create if not exists."""
        with self.db.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM studies WHERE study_name = ?", (study_name,))
            row = cursor.fetchone()
            if row:
                return dict(row)

            cursor = conn.execute(
                "INSERT INTO studies (study_name, description) VALUES (?, ?)",
                (study_name, description),
            )
            conn.commit()
            return {
                "study_id": cursor.lastrowid,
                "study_name": study_name,
                "description": description,
                "study_config": None,
            }

    def get_study_id(self, study_name: str) -> int | None:
        """Get study_id by study_name. Returns None if not found."""
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT study_id FROM studies WHERE study_name = ?", (study_name,)
            )
            row = cursor.fetchone()
            return row["study_id"] if row else None

    def get_study_name(self, study_id: int) -> str | None:
        """Get study_name by study_id. Returns None if not found."""
        with self.db.get_connection() as conn:
            cursor = conn.execute("SELECT study_name FROM studies WHERE study_id = ?", (study_id,))
            row = cursor.fetchone()
            return row["study_name"] if row else None

    def get_by_id(self, study_id: int) -> dict[str, Any] | None:
        """Get study by ID."""
        with self.db.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM studies WHERE study_id = ?", (study_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_by_name(self, study_name: str) -> dict[str, Any] | None:
        """Get study by name."""
        with self.db.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM studies WHERE study_name = ?", (study_name,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_studies(self) -> list[dict[str, Any]]:
        """Get all studies."""
        with self.db.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM studies ORDER BY study_name")
            return [dict(row) for row in cursor.fetchall()]

    def update_config(self, study_id: int, config: str) -> bool:
        """Update study configuration JSON."""
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "UPDATE studies SET study_config = ?, updated_at = CURRENT_TIMESTAMP WHERE study_id = ?",
                (config, study_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_study(self, study_id: int) -> bool:
        """Delete a study and all associated recordings/decoders (CASCADE)."""
        with self.db.get_connection() as conn:
            try:
                cursor = conn.execute("DELETE FROM studies WHERE study_id = ?", (study_id,))
                conn.commit()
                return cursor.rowcount > 0
            except sqlite3.Error as e:
                logger.warning(f"Failed to delete study {study_id}: {e}")
                return False


class DatasetRepository:
    """Repository for custom dataset operations (imported FIF files)."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def add_dataset(
        self,
        name: str,
        file_path: str,
        study_id: int | None = None,
        events_json: str | None = None,
        epoch_tmin: float = -0.2,
        epoch_tmax: float = 0.8,
        sampling_rate: float | None = None,
        target_sample_rate: float | None = None,
        preproc_lowcut: float = 0.5,
        preproc_highcut: float = 50.0,
        preproc_rereference: bool = False,
        paradigm: str | None = None,
        modality: str | None = None,
        description: str | None = None,
    ) -> int | None:
        """Add a custom dataset, optionally associated with a study."""
        with self.db.get_connection() as conn:
            try:
                cursor = conn.execute(
                    """INSERT INTO datasets (study_id, name, file_path, events_json, epoch_tmin, epoch_tmax,
                       sampling_rate, target_sample_rate, preproc_lowcut, preproc_highcut, preproc_rereference,
                       paradigm, modality, description)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        study_id,
                        name,
                        file_path,
                        events_json,
                        epoch_tmin,
                        epoch_tmax,
                        sampling_rate,
                        target_sample_rate,
                        preproc_lowcut,
                        preproc_highcut,
                        int(preproc_rereference),
                        paradigm,
                        modality,
                        description,
                    ),
                )
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                return None

    def get_by_id(self, dataset_id: int) -> dict[str, Any] | None:
        """Get dataset by ID with study_name included if associated."""
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT d.*, s.study_name
                FROM datasets d
                LEFT JOIN studies s ON d.study_id = s.study_id
                WHERE d.dataset_id = ?
            """,
                (dataset_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_by_name(self, name: str) -> dict[str, Any] | None:
        """Get dataset by name with study_name included if associated."""
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT d.*, s.study_name
                FROM datasets d
                LEFT JOIN studies s ON d.study_id = s.study_id
                WHERE d.name = ?
            """,
                (name,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_datasets(self) -> list[dict[str, Any]]:
        """Get all datasets with study_name, newest first."""
        with self.db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT d.*, s.study_name
                FROM datasets d
                LEFT JOIN studies s ON d.study_id = s.study_id
                ORDER BY d.dataset_id DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_datasets_by_study(self, study: int | str) -> list[dict[str, Any]]:
        """Get all datasets for a study by study_id (int) or study_name (str)."""
        with self.db.get_connection() as conn:
            if isinstance(study, int):
                cursor = conn.execute(
                    """
                    SELECT d.*, s.study_name
                    FROM datasets d
                    LEFT JOIN studies s ON d.study_id = s.study_id
                    WHERE d.study_id = ?
                    ORDER BY d.dataset_id DESC
                """,
                    (study,),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT d.*, s.study_name
                    FROM datasets d
                    JOIN studies s ON d.study_id = s.study_id
                    WHERE s.study_name = ?
                    ORDER BY d.dataset_id DESC
                """,
                    (study,),
                )
            return [dict(row) for row in cursor.fetchall()]

    def get_unassociated_datasets(self) -> list[dict[str, Any]]:
        """Get datasets not associated with any study."""
        with self.db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM datasets
                WHERE study_id IS NULL
                ORDER BY dataset_id DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def update_dataset(self, dataset_id: int, **kwargs) -> bool:
        """Update dataset fields."""
        allowed_fields = [
            "study_id",
            "name",
            "file_path",
            "events_json",
            "epoch_tmin",
            "epoch_tmax",
            "sampling_rate",
            "target_sample_rate",
            "preproc_lowcut",
            "preproc_highcut",
            "preproc_rereference",
            "description",
            "paradigm",
            "modality",
        ]
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if not updates:
            return False

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [dataset_id]

        with self.db.get_connection() as conn:
            try:
                cursor = conn.execute(
                    f"UPDATE datasets SET {set_clause} WHERE dataset_id = ?", values
                )
                conn.commit()
                return cursor.rowcount > 0
            except sqlite3.IntegrityError:
                return False

    def delete_dataset(self, dataset_id: int) -> bool:
        """Delete a dataset."""
        with self.db.get_connection() as conn:
            try:
                cursor = conn.execute("DELETE FROM datasets WHERE dataset_id = ?", (dataset_id,))
                conn.commit()
                return cursor.rowcount > 0
            except sqlite3.Error as e:
                logger.warning(f"Failed to delete dataset {dataset_id}: {e}")
                return False
