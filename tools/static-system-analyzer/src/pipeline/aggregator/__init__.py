"""
Aggregator module for collecting and merging project data into JSON format.

This module handles:
- Reading CSV files from analyzed projects
- Parsing SYSTEM_REPO and TypeSupport data
- Creating nodes, topics, applications, libraries
- Generating relationship JSON output
"""

from .models import Topic
from .parsers import SystemRepoParser, TypeSupportParser
from .converter import QosConverter
from .service import AggregatorService, aggregate

__all__ = [
    "Topic",
    "SystemRepoParser",
    "TypeSupportParser",
    "QosConverter",
    "AggregatorService",
    "aggregate",
]
