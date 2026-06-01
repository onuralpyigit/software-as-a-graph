"""Manual analysis strategy — XML parse + Java import parse."""

from __future__ import annotations

from pathlib import Path
from typing import List

from .base import AnalysisStrategy
from ..file_finder import find_project_xml
from ..java_parser import parse_java_import_dependencies
from ..models import TopicEntry
from ..xml_parser import parse_topic_xml
from common.logger import log_warning


class ManualStrategy(AnalysisStrategy):
    """Extract topics from project XML and dependencies from Java imports."""

    def extract(self, folder_path: Path, folder_name: str) -> List[TopicEntry]:
        entries: List[TopicEntry] = []

        # Topics from XML
        xml_path = find_project_xml(folder_path, folder_name)
        if xml_path:
            entries.extend(parse_topic_xml(xml_path, folder_name))
        else:
            log_warning(f"XML file not found for {folder_name}")

        # Uses from Java imports
        dependencies = parse_java_import_dependencies(folder_path)
        normalized = folder_name.strip()
        for dep in dependencies:
            if dep != normalized:
                entries.append(TopicEntry(source_folder=folder_name, name=dep, role="uses"))

        return entries
