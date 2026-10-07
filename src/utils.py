"""Utility functions for logging, file handling, and paths.

Privacy & Security note:
Loggers configured here never output raw patient text, medical notes,
or API keys. Only operational milestones and technical failures are logged.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    # This file is in src/, so parent is project root
    return Path(__file__).resolve().parent.parent


def setup_logger(name: str = "medical_analyzer", level: int = logging.INFO) -> logging.Logger:
    """Configure a sanitized logger for the application.
    
    Ensures safe logs that do not print sensitive payloads.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def format_file_size(size_bytes: int) -> str:
    """Convert bytes into a human-readable string (KB, MB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def safe_truncate(text: str, max_length: int = 100) -> str:
    """Safely truncate a string for display without breaking."""
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + "..."
