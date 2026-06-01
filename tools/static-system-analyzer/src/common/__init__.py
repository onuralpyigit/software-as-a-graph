"""
Common package for static-system-analyzer.

Common utilities shared across all modules.
"""

from .logger import (
    setup_logger,
    get_logger,
    log_info,
    log_warning,
    log_error,
    log_debug,
    log_summary,
)
from .runtime_config import get_runtime_config, load_runtime_config

__all__ = [
    "setup_logger",
    "get_logger",
    "log_info",
    "log_warning",
    "log_error",
    "log_debug",
    "log_summary",
    "get_runtime_config",
    "load_runtime_config",
]
