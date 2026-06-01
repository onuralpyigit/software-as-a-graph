"""
Statistical analysis module for publish-subscribe architecture.

This module provides statistical analysis for nodes, applications,
libraries, and topics in publish-subscribe distributed systems.
"""

from .service import StatService, StatConfig

__all__ = ["StatService", "StatConfig"]
