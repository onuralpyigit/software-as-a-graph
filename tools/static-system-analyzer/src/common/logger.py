"""Logging module.

This module provides centralized logging functionality for all pipeline stages,
including file and console logging with timestamps.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Module-level logger
_logger: Optional[logging.Logger] = None
_current_stage: Optional[str] = None


def setup_logger(
    log_dir: str, 
    platform_name: str, 
    stage: str = "pipeline",
    log_level: str = "INFO"
) -> logging.Logger:
    """Setup and configure the logger.

    Args:
        log_dir: Directory path for log files.
        platform_name: Platform name for log file naming.
        stage: Pipeline stage name (clone, analyze, aggregate, pipeline).
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured logger instance.
    """
    global _logger, _current_stage

    _current_stage = stage
    
    # Convert string level to logging constant
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Use stage specific logger name to allow separate configuration if needed
    logger = logging.getLogger(f"static_analyzer_{stage}")
    logger.setLevel(level)

    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # Only create log files for actual module stages, not for pipeline orchestrator
    if stage != "pipeline":
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        log_filename = f"{stage}_{platform_name}.log"
        log_file = log_path / log_filename

        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(level)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info("=" * 60)
        logger.info(f"{stage.upper()} session started for platform: {platform_name}")
        logger.info("=" * 60)
    else:
        logger.addHandler(logging.NullHandler())

    _logger = logger

    return logger


def get_logger() -> logging.Logger:
    """Get the configured logger instance.

    Returns:
        Logger instance or a basic logger if not configured.
    """
    if _logger is None:
        # Return basic logger if not configured
        return logging.getLogger("static_analyzer")
    return _logger


def log_info(message: str) -> None:
    """Log an info message.

    Args:
        message: Info message.
    """
    logger = get_logger()
    logger.info(message)


def log_debug(message: str) -> None:
    """Log a debug message.

    Args:
        message: Debug message.
    """
    logger = get_logger()
    logger.debug(message)


def log_warning(message: str) -> None:
    """Log a warning message.

    Args:
        message: Warning message.
    """
    logger = get_logger()
    logger.warning(message)


def log_error(message: str) -> None:
    """Log an error message.

    Args:
        message: Error message.
    """
    logger = get_logger()
    logger.error(message)


def log_debug(message: str) -> None:
    """Log a debug message.

    Args:
        message: Debug message.
    """
    logger = get_logger()
    logger.debug(message)


def log_summary(
    title: str,
    success_count: int,
    fail_count: int,
    extra_info: Optional[dict] = None,
) -> None:
    """Log operation summary.

    Args:
        title: Summary title.
        success_count: Number of successful operations.
        fail_count: Number of failed operations.
        extra_info: Optional dictionary with extra info to log.
    """
    logger = get_logger()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info("=" * 60)
    logger.info(f"{title} SUMMARY")
    logger.info(f"  Timestamp:  {timestamp}")
    logger.info(f"  Successful: {success_count}")
    logger.info(f"  Failed:     {fail_count}")
    logger.info(f"  Total:      {success_count + fail_count}")
    
    if extra_info:
        for key, value in extra_info.items():
            logger.info(f"  {key}: {value}")
    
    logger.info("=" * 60)


# Clone-specific logging helpers
def log_clone_start(tag: str, destination: str) -> None:
    """Log the start of a clone operation.

    Args:
        tag: Git tag being cloned.
        destination: Destination path.
    """
    log_info(f"Cloning tag '{tag}' to '{destination}'...")


def log_clone_success(tag: str, destination: str) -> None:
    """Log successful clone operation.

    Args:
        tag: Git tag that was cloned.
        destination: Destination path.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_info(f"SUCCESS | Tag: {tag} | Path: {destination} | Time: {timestamp}")


def log_clone_error(tag: str, error: str) -> None:
    """Log clone error.

    Args:
        tag: Git tag that failed.
        error: Error message.
    """
    log_error(f"FAILED  | Tag: {tag} | Error: {error}")
