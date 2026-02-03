"""Session I/O Worker for background thread operations."""

import json
import logging
import os
from datetime import datetime

from PyQt6 import QtCore

from dendrite.constants import get_study_paths
from dendrite.data.storage.database import Database, RecordingRepository, StudyRepository
from dendrite.utils import SharedState
from dendrite.utils.logger_central import configure_file_logging, get_logger, set_study_name


class SessionIOWorker(QtCore.QObject):
    """
    Background worker for session I/O operations.

    Performs database queries and file I/O in a background thread to keep
    the GUI responsive during session startup. Only receives primitive data
    (no Qt widget references) to ensure thread safety.
    """

    finished = QtCore.pyqtSignal(dict)  # Returns {run_number, file_identifier, timestamp, log_file}
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        subject_id: str,
        session_id: str,
        recording_name: str,
        study_name: str,
        config_to_save: dict,
    ):
        super().__init__()
        self._subject_id = subject_id
        self._session_id = session_id
        self._recording_name = recording_name
        self._study_name = study_name
        self._config = config_to_save
        self._logger = get_logger("SessionIO")

    @QtCore.pyqtSlot()
    def run(self):
        """Execute all I/O operations in background thread."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # DB query for run number
            db = Database()
            repo = RecordingRepository(db)
            run_number = repo.get_next_run_number(
                self._subject_id, self._session_id, self._recording_name
            )

            # Build BIDS file identifier
            parts = []
            if self._subject_id:
                parts.append(f"sub-{self._subject_id}")
            if self._session_id:
                parts.append(f"ses-{self._session_id}")
            parts.append(f"task-{self._recording_name}")
            parts.append(f"run-{run_number:02d}")
            parts.append(timestamp)
            file_identifier = "_".join(parts)

            # Update config with identifiers for saving
            self._config["run_number"] = run_number
            self._config["file_identifier"] = file_identifier

            # Setup session logging
            set_study_name(self._study_name)
            log_file = configure_file_logging(file_identifier=file_identifier, level=logging.DEBUG)
            self._logger.info(f"Session: {file_identifier} | Study: {self._study_name}")

            # Save configuration to file
            self._save_configuration(file_identifier)

            # Add recording to database (reuse db connection)
            self._add_recording_to_database(db, timestamp, file_identifier, run_number)

            # Create SharedState in background thread (Manager() is blocking)
            shared_state = SharedState()

            self.finished.emit(
                {
                    "run_number": run_number,
                    "file_identifier": file_identifier,
                    "timestamp": timestamp,
                    "log_file": log_file,
                    "shared_state": shared_state,
                }
            )
        except Exception as e:
            self._logger.error(f"Session I/O error: {e}", exc_info=True)
            self.error.emit(str(e))

    def _save_configuration(self, file_identifier: str):
        """Save configuration to JSON file."""
        paths = get_study_paths(self._study_name)
        config_dir = paths["config"]
        os.makedirs(config_dir, exist_ok=True)
        config_path = config_dir / f"{file_identifier}_config.json"

        with open(config_path, "w") as f:
            json.dump(self._config, f, indent=2, default=str)
        self._logger.info(f"Configuration saved: {config_path}")

    def _add_recording_to_database(
        self, db: Database, timestamp: str, file_identifier: str, run_number: int
    ):
        """Add the recording to the database."""
        paths = get_study_paths(self._study_name)
        hdf5_path = str(paths["raw"] / f"{file_identifier}_eeg.h5")
        config_path = str(paths["config"] / f"{file_identifier}_config.json")
        metrics_path = str(paths["metrics"] / f"{file_identifier}_metrics.h5")

        try:
            study_repo = StudyRepository(db)
            recording_repo = RecordingRepository(db)

            study = study_repo.get_or_create(self._study_name)
            study_id = study["study_id"]

            recording_id = recording_repo.add_recording(
                study_id=study_id,
                recording_name=self._recording_name,
                session_timestamp=timestamp,
                hdf5_file_path=hdf5_path,
                subject_id=self._subject_id,
                session_id=self._session_id,
                run_number=run_number,
                config_file_path=config_path,
                metrics_file_path=metrics_path,
            )
            if recording_id:
                self._logger.info(f"Recording added to database with ID: {recording_id}")
            else:
                self._logger.warning("Failed to add recording to database")
        except Exception as e:
            self._logger.error(f"Database error: {e}", exc_info=True)
