"""
Dendrite Database Backend

SQLite database for experiment metadata - recordings and trained decoders.
Uses WAL mode for concurrent read/write access and dict row factory for
zero-overhead JSON-ready results.
"""

import logging
import os
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from dendrite.constants import DATABASE_PATH

logger = logging.getLogger(__name__)


def _dict_factory(cursor: sqlite3.Cursor, row: tuple) -> dict[str, Any]:
    """Row factory that returns dicts directly — no dict(row) needed."""
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}


class Database:
    """Database backend for Dendrite experiments."""

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            self.db_path = str(DATABASE_PATH)
        else:
            self.db_path = db_path

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections with WAL mode."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = _dict_factory
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 10000")
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Explicit transaction — atomic multi-step operations.

        All operations within the block share one connection and are
        committed together or rolled back on error.
        """
        with self.get_connection() as conn:
            conn.execute("BEGIN")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def init_db(self) -> None:
        """Initialize the database schema."""
        logger.info(f"Initializing Dendrite database at: {self.db_path}")
        with self.get_connection() as conn:
            self._create_tables(conn)
            self._create_indexes(conn)

    def _create_tables(self, conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS studies (
                study_id INTEGER PRIMARY KEY AUTOINCREMENT,
                study_name TEXT UNIQUE NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS recordings (
                recording_id INTEGER PRIMARY KEY AUTOINCREMENT,
                study_id INTEGER NOT NULL,
                recording_name TEXT NOT NULL,
                subject_id TEXT NOT NULL DEFAULT '',
                session_id TEXT NOT NULL DEFAULT '',
                run_number INTEGER NOT NULL DEFAULT 1,
                session_timestamp TEXT NOT NULL,
                file_identifier TEXT NOT NULL DEFAULT '',
                hdf5_file_path TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (study_id) REFERENCES studies(study_id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS decoders (
                decoder_id INTEGER PRIMARY KEY AUTOINCREMENT,
                study_id INTEGER,
                decoder_name TEXT NOT NULL,
                decoder_path TEXT UNIQUE NOT NULL,
                model_type TEXT NOT NULL,
                description TEXT,
                num_classes INTEGER,
                training_accuracy REAL,
                validation_accuracy REAL,
                training_recording_ids TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (study_id) REFERENCES studies(study_id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_jobs (
                job_id INTEGER PRIMARY KEY AUTOINCREMENT,
                study_id INTEGER,
                model_type TEXT NOT NULL,
                job_type TEXT NOT NULL DEFAULT 'training',
                status TEXT NOT NULL DEFAULT 'pending',
                config_json TEXT NOT NULL,
                result_json TEXT,
                decoder_id INTEGER,
                error_message TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (study_id) REFERENCES studies(study_id) ON DELETE CASCADE,
                FOREIGN KEY (decoder_id) REFERENCES decoders(decoder_id) ON DELETE SET NULL
            )
        """)

    def _create_indexes(self, conn: sqlite3.Connection) -> None:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_recordings_name ON recordings(recording_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_recordings_study ON recordings(study_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_decoders_name ON decoders(decoder_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_decoders_study ON decoders(study_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_jobs_study ON training_jobs(study_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status)"
        )


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
        file_identifier: str = "",
        _conn: sqlite3.Connection | None = None,
    ) -> int | None:
        """Add a new recording. Returns None if hdf5_file_path already exists.

        Pass _conn from a transaction() block for atomic batch inserts.
        """
        def _insert(conn: sqlite3.Connection) -> int | None:
            try:
                cursor = conn.execute(
                    """INSERT INTO recordings
                       (study_id, recording_name, subject_id, session_id,
                        run_number, session_timestamp, file_identifier, hdf5_file_path)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        study_id, recording_name, subject_id, session_id,
                        run_number, session_timestamp, file_identifier, hdf5_file_path,
                    ),
                )
                if _conn is None:
                    conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                return None

        if _conn is not None:
            return _insert(_conn)
        with self.db.get_connection() as conn:
            return _insert(conn)

    def get_by_id(self, recording_id: int) -> dict[str, Any] | None:
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """SELECT r.*, s.study_name FROM recordings r
                   JOIN studies s ON r.study_id = s.study_id
                   WHERE r.recording_id = ?""",
                (recording_id,),
            )
            return cursor.fetchone()

    def get_all_recordings(self) -> list[dict[str, Any]]:
        with self.db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT r.*, s.study_name FROM recordings r
                JOIN studies s ON r.study_id = s.study_id
                ORDER BY r.recording_id DESC
            """)
            return cursor.fetchall()

    def search_recordings(self, search_term: str) -> list[dict[str, Any]]:
        with self.db.get_connection() as conn:
            like_term = f"%{search_term}%"
            cursor = conn.execute(
                """SELECT r.*, s.study_name FROM recordings r
                   JOIN studies s ON r.study_id = s.study_id
                   WHERE LOWER(r.recording_name) LIKE LOWER(?) OR
                         LOWER(s.study_name) LIKE LOWER(?) OR
                         LOWER(r.subject_id) LIKE LOWER(?) OR
                         LOWER(r.session_id) LIKE LOWER(?) OR
                         LOWER(r.hdf5_file_path) LIKE LOWER(?)
                   ORDER BY r.recording_id DESC""",
                (like_term,) * 5,
            )
            return cursor.fetchall()

    def delete_recording(self, recording_id: int) -> bool:
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
        with self.db.get_connection() as conn:
            if isinstance(study, int):
                cursor = conn.execute(
                    """SELECT r.*, s.study_name FROM recordings r
                       JOIN studies s ON r.study_id = s.study_id
                       WHERE r.study_id = ?
                       ORDER BY r.session_timestamp DESC, r.recording_name ASC""",
                    (study,),
                )
            else:
                cursor = conn.execute(
                    """SELECT r.*, s.study_name FROM recordings r
                       JOIN studies s ON r.study_id = s.study_id
                       WHERE s.study_name = ?
                       ORDER BY r.session_timestamp DESC, r.recording_name ASC""",
                    (study,),
                )
            return cursor.fetchall()

    def get_next_run_number(
        self, subject_id: str, session_id: str, recording_name: str
    ) -> int:
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

    def __init__(self, db: Database) -> None:
        self.db = db

    def add_decoder(
        self,
        study_id: int | None,
        decoder_name: str,
        decoder_path: str,
        model_type: str,
        num_classes: int | None = None,
        training_accuracy: float | None = None,
        validation_accuracy: float | None = None,
        description: str | None = None,
        training_recording_ids: list[int] | None = None,
    ) -> int | None:
        import json as _json

        with self.db.get_connection() as conn:
            try:
                cursor = conn.execute(
                    """INSERT INTO decoders
                       (study_id, decoder_name, decoder_path, model_type,
                        num_classes, training_accuracy, validation_accuracy,
                        description, training_recording_ids)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        study_id, decoder_name, decoder_path, model_type,
                        num_classes, training_accuracy, validation_accuracy,
                        description,
                        _json.dumps(training_recording_ids) if training_recording_ids else None,
                    ),
                )
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                return None

    def get_decoder_by_id(self, decoder_id: int) -> dict[str, Any] | None:
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """SELECT d.*, s.study_name FROM decoders d
                   LEFT JOIN studies s ON d.study_id = s.study_id
                   WHERE d.decoder_id = ?""",
                (decoder_id,),
            )
            return cursor.fetchone()

    def get_all_decoders(self) -> list[dict[str, Any]]:
        with self.db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT d.*, s.study_name FROM decoders d
                LEFT JOIN studies s ON d.study_id = s.study_id
                ORDER BY d.decoder_id DESC
            """)
            return cursor.fetchall()

    def get_decoders_by_study(self, study: int | str) -> list[dict[str, Any]]:
        with self.db.get_connection() as conn:
            if isinstance(study, int):
                cursor = conn.execute(
                    """SELECT d.*, s.study_name FROM decoders d
                       JOIN studies s ON d.study_id = s.study_id
                       WHERE d.study_id = ? ORDER BY d.decoder_id DESC""",
                    (study,),
                )
            else:
                cursor = conn.execute(
                    """SELECT d.*, s.study_name FROM decoders d
                       JOIN studies s ON d.study_id = s.study_id
                       WHERE s.study_name = ? ORDER BY d.decoder_id DESC""",
                    (study,),
                )
            return cursor.fetchall()

    def search_decoders(self, search_term: str) -> list[dict[str, Any]]:
        with self.db.get_connection() as conn:
            like_term = f"%{search_term}%"
            cursor = conn.execute(
                """SELECT d.*, s.study_name FROM decoders d
                   LEFT JOIN studies s ON d.study_id = s.study_id
                   WHERE LOWER(d.decoder_name) LIKE LOWER(?) OR
                         LOWER(d.model_type) LIKE LOWER(?) OR
                         LOWER(s.study_name) LIKE LOWER(?) OR
                         LOWER(d.description) LIKE LOWER(?)
                   ORDER BY d.decoder_id DESC""",
                (like_term,) * 4,
            )
            return cursor.fetchall()

    def delete_decoder(self, decoder_id: int) -> bool:
        with self.db.get_connection() as conn:
            try:
                cursor = conn.execute(
                    "DELETE FROM decoders WHERE decoder_id = ?", (decoder_id,)
                )
                conn.commit()
                return cursor.rowcount > 0
            except sqlite3.Error as e:
                logger.warning(f"Failed to delete decoder {decoder_id}: {e}")
                return False


class StudyRepository:
    """Repository for study operations."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def get_or_create(
        self,
        study_name: str,
        description: str | None = None,
        _conn: sqlite3.Connection | None = None,
    ) -> dict[str, Any]:
        """Get study by name or create if not exists.

        Pass _conn from a transaction() block for atomic operations.
        """
        def _do(conn: sqlite3.Connection) -> dict[str, Any]:
            cursor = conn.execute(
                "SELECT * FROM studies WHERE study_name = ?", (study_name,)
            )
            row = cursor.fetchone()
            if row:
                return row
            cursor = conn.execute(
                "INSERT INTO studies (study_name, description) VALUES (?, ?)",
                (study_name, description),
            )
            if _conn is None:
                conn.commit()
            return {
                "study_id": cursor.lastrowid,
                "study_name": study_name,
                "description": description,
            }

        if _conn is not None:
            return _do(_conn)
        with self.db.get_connection() as conn:
            return _do(conn)

    def get_by_id(self, study_id: int) -> dict[str, Any] | None:
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM studies WHERE study_id = ?", (study_id,)
            )
            return cursor.fetchone()

    def get_all_studies(self) -> list[dict[str, Any]]:
        with self.db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT s.*,
                    (SELECT COUNT(*) FROM recordings r WHERE r.study_id = s.study_id)
                        AS recording_count,
                    (SELECT COUNT(*) FROM decoders d WHERE d.study_id = s.study_id)
                        AS decoder_count
                FROM studies s ORDER BY s.study_name
            """)
            return cursor.fetchall()

    def update_study(self, study_id: int, description: str | None = None) -> bool:
        if description is None:
            return False
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "UPDATE studies SET description = ? WHERE study_id = ?",
                (description, study_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_study(self, study_id: int) -> bool:
        with self.db.get_connection() as conn:
            try:
                cursor = conn.execute(
                    "DELETE FROM studies WHERE study_id = ?", (study_id,)
                )
                conn.commit()
                return cursor.rowcount > 0
            except sqlite3.Error as e:
                logger.warning(f"Failed to delete study {study_id}: {e}")
                return False


class TrainingJobRepository:
    """Repository for ML training job operations."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def create_job(
        self,
        study_id: int | None,
        model_type: str,
        config_json: str,
        job_type: str = "training",
    ) -> int:
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO training_jobs
                   (study_id, model_type, job_type, status, config_json)
                   VALUES (?, ?, ?, 'pending', ?)""",
                (study_id, model_type, job_type, config_json),
            )
            conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    def get_by_id(self, job_id: int) -> dict[str, Any] | None:
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM training_jobs WHERE job_id = ?", (job_id,)
            )
            return cursor.fetchone()

    # Columns returned by list queries (excludes large result_json blob).
    _LIST_COLS = (
        "job_id, study_id, model_type, job_type, status, config_json,"
        " decoder_id, error_message, started_at, completed_at, created_at"
    )

    def list_jobs(
        self,
        study_id: int | None = None,
        job_type: str | None = None,
    ) -> list[dict[str, Any]]:
        with self.db.get_connection() as conn:
            clauses = []
            params: list[Any] = []
            if study_id is not None:
                clauses.append("study_id = ?")
                params.append(study_id)
            if job_type is not None:
                clauses.append("job_type = ?")
                params.append(job_type)
            where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
            cursor = conn.execute(
                f"SELECT {self._LIST_COLS} FROM training_jobs{where}"
                " ORDER BY job_id DESC",
                params,
            )
            return cursor.fetchall()

    def update_status(
        self,
        job_id: int,
        status: str,
        error_message: str | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
    ) -> bool:
        updates = ["status = ?"]
        values: list[Any] = [status]
        if error_message is not None:
            updates.append("error_message = ?")
            values.append(error_message)
        if started_at is not None:
            updates.append("started_at = ?")
            values.append(started_at)
        if completed_at is not None:
            updates.append("completed_at = ?")
            values.append(completed_at)
        values.append(job_id)
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                f"UPDATE training_jobs SET {', '.join(updates)} WHERE job_id = ?",
                values,
            )
            conn.commit()
            return cursor.rowcount > 0

    def set_result(self, job_id: int, result_json: str) -> bool:
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "UPDATE training_jobs SET result_json = ? WHERE job_id = ?",
                (result_json, job_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def link_decoder(self, job_id: int, decoder_id: int) -> bool:
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "UPDATE training_jobs SET decoder_id = ? WHERE job_id = ?",
                (decoder_id, job_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_job(self, job_id: int) -> bool:
        with self.db.get_connection() as conn:
            try:
                cursor = conn.execute(
                    "DELETE FROM training_jobs WHERE job_id = ?", (job_id,)
                )
                conn.commit()
                return cursor.rowcount > 0
            except sqlite3.Error as e:
                logger.warning(f"Failed to delete training job {job_id}: {e}")
                return False
