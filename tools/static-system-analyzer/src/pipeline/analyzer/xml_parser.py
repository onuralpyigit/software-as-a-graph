"""
XML parsing utilities for topic extraction.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

from .models import TopicEntry
from common.logger import log_warning, log_error, log_debug
from common.runtime_config import get_runtime_config


def _is_dummy_topic_name(topic_name: str) -> bool:
    """Return True when the topic name should be ignored during analysis."""
    normalized_name = topic_name.strip().lower()
    dummy_topic_names = {
        name.strip().lower()
        for name in get_runtime_config().analyzer.dummy_topic_names
        if name.strip()
    }
    return normalized_name in dummy_topic_names


def parse_topic_xml(xml_path: Path, folder_name: str) -> List[TopicEntry]:
    """
    Parse topic entries from a project XML file.

    Args:
        xml_path: Path to the XML file
        folder_name: Name of the source folder

    Returns:
        List of TopicEntry objects
    """
    entries = []

    if not xml_path.exists():
        log_warning(f"XML file not found: {xml_path}")
        return entries

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Find all topic elements (mapped to TopicEntry in our model)
        topic_tag = get_runtime_config().analyzer.custom_topic_name
        for topic in root.iter(topic_tag):
            name = topic.get("name")
            role = topic.get("role")

            if name and role:
                if _is_dummy_topic_name(name):
                    log_debug(f"Skipping dummy topic in {xml_path}: {name}")
                    continue

                entries.append(TopicEntry(
                    source_folder=folder_name,
                    name=name,
                    role=role.lower()
                ))
                log_debug(f"Found topic: {name} with role: {role}")

    except ET.ParseError as e:
        log_error(f"Error parsing XML file {xml_path}: {e}")
    except Exception as e:
        log_error(f"Unexpected error parsing {xml_path}: {e}")

    return entries


def _strip_makefile_comments(content: str) -> str:
    """Remove comment portions from Makefile content (lines starting with # or inline # comments)."""
    stripped_lines = []
    for line in content.splitlines():
        comment_pos = line.find('#')
        if comment_pos >= 0:
            line = line[:comment_pos]
        stripped_lines.append(line)
    return '\n'.join(stripped_lines)


def check_makefile_contains_Makefile(makefile_path: Path) -> bool:
    """
    Check if Makefile contains include/Makefile_java.mk in an active (non-commented) line.

    Args:
        makefile_path: Path to Makefile

    Returns:
        True if pattern found, False otherwise
    """
    if not makefile_path.exists():
        return False

    try:
        content = makefile_path.read_text(encoding="utf-8")
        active_content = _strip_makefile_comments(content)
        include_patterns = get_runtime_config().analyzer.makefile_include_patterns
        return any(pattern in active_content for pattern in include_patterns)
    except Exception as e:
        log_error(f"Error reading Makefile {makefile_path}: {e}")
        return False
