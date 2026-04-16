"""
Centralized logging for multi-process Dendrite application.

Propagates log file path to child processes via environment variable so all
processes write to a single rotating log file organized by study name.
"""

import logging
import multiprocessing
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dendrite.constants import DATA_DIR

ENV_LOG_FILE = "DENDRITE_LOG_FILE"

_current_study_name = "default_study"
_current_log_file = None
_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(processName)s - %(message)s"


def set_study_name(study_name: str) -> str:
    """Set the current study name for logging directory organization."""
    global _current_study_name
    _current_study_name = study_name
    return _current_study_name


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)


def _create_handlers(log_file: str | None = None, rotating: bool = True) -> list[logging.Handler]:
    """Create console and optional file handlers.

    Args:
        rotating: If True (main process), use RotatingFileHandler.
            If False (child processes), use plain FileHandler to avoid
            Windows PermissionError when multiple processes share the file.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        if rotating:
            handlers.append(
                RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
            )
        else:
            handlers.append(logging.FileHandler(log_file, mode="a"))
    return handlers


def configure_file_logging(
    file_identifier: str | None = None, log_dir: str | None = None, level: int = logging.DEBUG
) -> str | None:
    """Configure file-based logging for the main process."""
    global _current_log_file

    existing_log = _current_log_file or os.environ.get(ENV_LOG_FILE)
    if existing_log and os.path.exists(existing_log):
        _current_log_file = existing_log
        return existing_log

    if file_identifier is None:
        file_identifier = f"app_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir_path: Path = (
        Path(log_dir)
        if log_dir is not None
        else DATA_DIR / "studies" / _current_study_name / "logs"
    )

    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_file = f"{log_dir_path}/{file_identifier}.log"

    try:
        logging.basicConfig(
            level=level,
            format=_LOG_FORMAT,
            handlers=_create_handlers(log_file),
            force=True,
        )

        _current_log_file = log_file
        os.environ[ENV_LOG_FILE] = log_file

        logging.info(f"File logging configured to {log_file}")
        return log_file

    except (OSError, PermissionError):
        logging.exception("Error setting up file logging")
        return None


def setup_logger(process_name: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure logging for child processes or standalone applications."""
    if process_name:
        multiprocessing.current_process().name = process_name

    parent_log_file = os.environ.get(ENV_LOG_FILE)
    log_file = parent_log_file if (parent_log_file and os.path.exists(parent_log_file)) else None

    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        handlers=_create_handlers(log_file, rotating=False),
        force=True,
    )

    logger = get_logger(process_name)
    logger.info(f"Logger configured for {process_name or 'unnamed process'}")

    return logger
