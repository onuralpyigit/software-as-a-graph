"""
Project scanner for discovering valid projects in the workspace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re
from typing import Generator, List, Optional

from .file_finder import (
    find_makefile_with_Makefile,
    find_project_xml,
)
from .models import ProjectInfo
from common.logger import log_info, log_warning, log_debug, log_error


@dataclass
class ScanStats:
    """Statistics from scanning projects."""
    total_folders: int = 0
    valid_projects: int = 0
    missing_xml: int = 0
    missing_makefile: int = 0
    skipped_folders: List[str] = field(default_factory=list)
    missing_xml_folders: List[str] = field(default_factory=list)
    missing_makefile_folders: List[str] = field(default_factory=list)

    def log_summary(self) -> None:
        """Log a summary of scan statistics."""
        timestamp = datetime.now().strftime("%d-%m-%Y %H:%M")
        log_info("=" * 50)
        log_info(f"Scan Statistics - {timestamp}")
        log_info("=" * 50)
        log_info(f"Total folders scanned: {self.total_folders}")
        log_info(f"Valid projects found: {self.valid_projects}")
        
        if self.missing_xml_folders:
            log_warning(f"Missing XML ({self.missing_xml}): {', '.join(self.missing_xml_folders)}")
        else:
            log_info(f"Missing XML: None")
            
        if self.missing_makefile_folders:
            log_warning(f"Missing Makefile with Makefile ({self.missing_makefile}): {', '.join(self.missing_makefile_folders)}")
        else:
            log_info(f"Missing Makefile with Makefile: None")
            
        log_info("=" * 50)


def scan_projects(
    projects_root: Path,
    stats: Optional[ScanStats] = None,
    analysis_mode: str = "manual",
) -> Generator[ProjectInfo, None, None]:
    """
    Scan the projects root directory for valid projects.

    Expected structure (direct versioned folders):
        projects_root/
        ├── abc_0.1.1/            # Versioned project folder
        │   ├── src/
        │   │   └── abc.xml
        │   └── Makefile

    A valid project has:
    - <project_name>.xml file (anywhere under src/) — required for manual mode only
    - Makefile containing 'include/Makefile_java.mk' (anywhere under project)

    Args:
        projects_root: Root path containing project folders
        stats: Optional ScanStats object to collect statistics
        analysis_mode: 'manual' or 'codeql'

    Yields:
        ProjectInfo objects for valid projects
    """
    if not projects_root.exists():
        log_error(f"Projects root does not exist: {projects_root}")
        return

    if not projects_root.is_dir():
        log_error(f"Projects root is not a directory: {projects_root}")
        return

    for project_dir in sorted(projects_root.iterdir()):
        if not project_dir.is_dir():
            continue

        # Skip hidden directories
        if project_dir.name.startswith("."):
            continue

        if stats:
            stats.total_folders += 1

        # Folder name is the versioned name (e.g., 'abc_0.1.1')
        versioned_name = project_dir.name
        
        # Extract project name from versioned name (e.g., 'abc' from 'abc_0.1.1')
        # Use regex to split at the last underscore followed by a semver-like version
        version_match = re.search(r'_(?=\d+\.\d+)', versioned_name)
        if version_match:
            project_name = versioned_name[:version_match.start()]
        else:
            project_name = versioned_name

        # The project folder itself is the versioned subfolder
        versioned_subfolder = project_dir

        # Find files recursively in the versioned subfolder
        makefile_path = find_makefile_with_Makefile(versioned_subfolder)

        if analysis_mode == "codeql":
            # CodeQL mode: only Makefile is required, XML is not needed
            xml_path = None
            if makefile_path is None:
                log_debug(f"Skipping {project_name}: No Makefile with include pattern found")
                if stats:
                    stats.missing_makefile += 1
                    stats.missing_makefile_folders.append(project_name)
                continue
        else:
            # Manual mode: both XML and Makefile are required
            xml_path = find_project_xml(versioned_subfolder, project_name)
            if xml_path is None:
                log_debug(f"Skipping {project_name}: {project_name}.xml not found in src/")
                if stats:
                    stats.missing_xml += 1
                    stats.missing_xml_folders.append(project_name)
                continue

            if makefile_path is None:
                log_debug(f"Skipping {project_name}: No Makefile with include/Makefile_java.mk found")
                if stats:
                    stats.missing_makefile += 1
                    stats.missing_makefile_folders.append(project_name)
                continue

        if stats:
            stats.valid_projects += 1

        project_info = ProjectInfo(
            folder_name=project_name,
            versioned_name=versioned_name,
            folder_path=str(versioned_subfolder),
            xml_path=str(xml_path) if xml_path else "",
            makefile_path=str(makefile_path),
        )

        log_info(f"Found valid project: {project_name} ({versioned_name})")
        yield project_info
