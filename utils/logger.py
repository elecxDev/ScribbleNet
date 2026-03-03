"""
ScribbleNet - Logger Module
Centralized logging configuration for the entire project.
"""

import logging
import os
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "scribblenet",
    log_file: Optional[str] = None,
    level: str = "INFO",
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    """
    Create and configure a logger instance.

    Args:
        name: Logger name identifier.
        log_file: Path to log file. If None, logs only to console.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        fmt: Log message format string.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(fmt)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "scribblenet") -> logging.Logger:
    """
    Retrieve an existing logger by name.

    Args:
        name: Logger name identifier.

    Returns:
        logging.Logger instance.
    """
    return logging.getLogger(name)
