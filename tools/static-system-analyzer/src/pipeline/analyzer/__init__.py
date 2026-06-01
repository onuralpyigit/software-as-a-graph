"""
Analyzer module for extracting topic entries from project XML files.

This module handles:
- Project scanning and validation
- XML parsing for topics
- Build execution (gmake regenerate_code)
- CSV output generation
- Code metrics analysis (ck)
"""

from .models import TopicEntry, ProjectInfo, TopicRole, expand_pubsub_entries
from .processor import TopicProcessor, ProcessingResult
from .build_runner import run_regenerate_code, check_gmake_available
from .csv_writer import write_topic_csv
from .file_finder import (
    find_file_recursive,
    find_files_recursive,
    find_project_xml,
    find_makefile,
    find_makefile_with_Makefile,
)
from .project_scanner import scan_projects, ScanStats
from .xml_parser import parse_topic_xml
from .java_parser import parse_java_import_dependencies
from .service import AnalyzerService, MetricsService, MetricsServiceConfig
from .metrics_scanner import MetricsConfig, scan_project, scan_all_projects
from .strategies import AnalysisStrategy, ManualStrategy, CodeQLStrategy

__all__ = [
    "TopicEntry",
    "ProjectInfo",
    "TopicRole",
    "expand_pubsub_entries",
    "TopicProcessor",
    "ProcessingResult",
    "run_regenerate_code",
    "check_gmake_available",
    "write_topic_csv",
    "find_file_recursive",
    "find_files_recursive",
    "find_project_xml",
    "find_makefile",
    "find_makefile_with_Makefile",
    "scan_projects",
    "ScanStats",
    "parse_topic_xml",
    "parse_java_import_dependencies",
    "AnalyzerService",
    "MetricsService",
    "MetricsServiceConfig",
    "MetricsConfig",
    "scan_project",
    "scan_all_projects",
    "AnalysisStrategy",
    "ManualStrategy",
    "CodeQLStrategy",
]
