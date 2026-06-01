"""
Data models for topic extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List


class TopicRole(Enum):
    """Enum representing topic roles."""
    PUB = "pub"
    SUB = "sub"
    PUBSUB = "pubsub"


@dataclass
class TopicEntry:
    """Represents a single topic entry."""
    source_folder: str
    name: str
    role: str

    def to_csv_row(self) -> str:
        """Convert entry to CSV row format."""
        return f"{self.source_folder},{self.name},{self.role}"


@dataclass
class ProjectInfo:
    """Represents project information."""
    folder_name: str           # Project name (e.g., 'abc')
    versioned_name: str        # Versioned folder name (e.g., 'abc_0.1.1')
    folder_path: str           # Full path to versioned folder
    xml_path: str
    makefile_path: str

    def has_valid_makefile(self) -> bool:
        """Check if project has a valid makefile."""
        return self.makefile_path is not None


def expand_pubsub_entries(entries: List[TopicEntry]) -> List[TopicEntry]:
    """
    Expand pubsub entries into separate pub and sub entries.

    Args:
        entries: List of topic entries

    Returns:
        List with pubsub entries expanded
    """
    expanded = []
    for entry in entries:
        if entry.role == TopicRole.PUBSUB.value:
            expanded.append(TopicEntry(
                source_folder=entry.source_folder,
                name=entry.name,
                role=TopicRole.PUB.value
            ))
            expanded.append(TopicEntry(
                source_folder=entry.source_folder,
                name=entry.name,
                role=TopicRole.SUB.value
            ))
        else:
            expanded.append(entry)
    return expanded
