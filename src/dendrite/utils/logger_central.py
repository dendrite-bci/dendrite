"""
Centralized logging system for multi-process Dendrite application.

Provides unified logging configuration across main and child processes with automatic
log file propagation via environment variables. Supports study-based log organization
and process-specific log identification.

Key Functions:
    - configure_file_logging(): Setup logging for main process with file output
    - setup_logger(): Configure logging for child processes (auto-connects to parent's log)
    - set_study_name(): Organize logs by study name
"""

import logging
import multiprocessing
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dendrite import DATA_DIR

# Environment variable name for sharing log file path
ENV_LOG_FILE = "DENDRITE_LOG_FILE"

# Simple global state
_current_study_name = "default_study"
_current_log_file = None

# Expose standard log levels for convenience
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL


def set_study_name(study_name: str) -> str:
    """Set the current study name for logging directory organization."""
    global _current_study_name
    _current_study_name = study_name
    return _current_study_name


def get_study_name() -> str:
    """Get the current study name."""
    return _current_study_name


def get_active_log_file() -> str | None:
    """Get the path to the active log file (checks both global state and environment)."""
    return _current_log_file or os.environ.get(ENV_LOG_FILE)


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)


def _create_handlers(log_file: str | None = None) -> list[logging.Handler]:
    """Create standard console and optional rotating file handlers."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(
            RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
            )
        )
    return handlers


def set_level(level: int) -> None:
    """Set logging level for the root logger and all handlers."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Update level for all handlers
    for handler in root_logger.handlers:
        handler.setLevel(level)


def configure_file_logging(
    file_identifier: str | None = None, log_dir: str | None = None, level: int = DEBUG
) -> str | None:
    """Configure file-based logging for the main process."""
    global _current_log_file

    # Return existing log file if already configured
    existing_log = _current_log_file or os.environ.get(ENV_LOG_FILE)
    if existing_log and os.path.exists(existing_log):
        _current_log_file = existing_log
        return existing_log

    # Generate file identifier if not provided
    if file_identifier is None:
        file_identifier = f"app_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Use study-specific directory if not provided
    if log_dir is None:
        log_dir = DATA_DIR / "logger" / get_study_name()

    # Create log directory and file path
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = f"{log_dir}/{file_identifier}.log"

    try:
        # Configure root logger with both console and file handlers
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(levelname)s - %(processName)s - %(message)s",
            handlers=_create_handlers(log_file),
            force=True,  # Override any existing configuration
        )

        # Store globally and in environment for child processes
        _current_log_file = log_file
        os.environ[ENV_LOG_FILE] = log_file

        logging.info(f"File logging configured to {log_file}")
        return log_file

    except (OSError, PermissionError):
        logging.exception("Error setting up file logging")
        return None


def setup_logger(process_name: str | None = None, level: int = INFO) -> logging.Logger:
    """Configure logging for child processes or standalone applications."""
    # Set process name for better log identification
    if process_name:
        multiprocessing.current_process().name = process_name

    # Check if parent process set up file logging
    parent_log_file = os.environ.get(ENV_LOG_FILE)

    if parent_log_file and os.path.exists(parent_log_file):
        # Child process - connect to parent's log file
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(levelname)s - %(processName)s - %(message)s",
            handlers=_create_handlers(parent_log_file),
            force=True,
        )
    else:
        # Standalone process - console only
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(levelname)s - %(processName)s - %(message)s",
            handlers=_create_handlers(),
            force=True,
        )

    logger = get_logger(process_name)
    logger.info(f"Logger configured for {process_name or 'unnamed process'}")

    return logger
