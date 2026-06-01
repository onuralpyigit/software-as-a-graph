"""
File finder utilities for locating project files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from common.logger import log_debug
from common.runtime_config import get_runtime_config


def find_files_recursive(
    root_path: Path,
    filename: str,
    max_depth: int = 10
) -> list[Path]:
    """
    Recursively search for all files with given name under root_path.

    Args:
        root_path: Directory to start searching from
        filename: Name of the file to find
        max_depth: Maximum directory depth to search

    Returns:
        List of paths to matching files
    """
    if not root_path.exists() or not root_path.is_dir():
        return []

    # Use glob for recursive search
    pattern = f"**/{filename}"
    matches = list(root_path.glob(pattern))

    if matches:
        log_debug(f"Found {len(matches)} {filename} files under {root_path}")

    return matches


def find_file_recursive(
    root_path: Path,
    filename: str,
    max_depth: int = 10
) -> Optional[Path]:
    """
    Recursively search for a file under root_path.

    Args:
        root_path: Directory to start searching from
        filename: Name of the file to find
        max_depth: Maximum directory depth to search

    Returns:
        Path to the file if found, None otherwise
    """
    matches = find_files_recursive(root_path, filename, max_depth)

    if matches:
        # Return the first match
        found_path = matches[0]
        log_debug(f"Found {filename} at {found_path}")
        return found_path

    log_debug(f"{filename} not found under {root_path}")
    return None


def find_project_xml(project_path: Path, folder_name: str) -> Optional[Path]:
    """
    Find <folderadi>.xml under the src/ directory.

    Args:
        project_path: Path to the project folder
        folder_name: Name of the folder (used for xml filename)

    Returns:
        Path to the XML file if found, None otherwise
    """
    src_path = project_path / "src"
    if not src_path.exists():
        log_debug(f"src/ directory not found in {project_path}")
        return None

    xml_filename = f"{folder_name}.xml"
    return find_file_recursive(src_path, xml_filename)


def find_makefile(project_path: Path) -> Optional[Path]:
    """
    Find Makefile anywhere under the project directory.

    Args:
        project_path: Path to the project folder

    Returns:
        Path to the Makefile if found, None otherwise
    """
    return find_file_recursive(project_path, "Makefile")


def find_makefile_with_Makefile(project_path: Path) -> Optional[Path]:
    """
    Find a Makefile that contains 'include/Makefile_java.mk' under the project directory.

    Searches all Makefiles and returns the first one containing the pattern.

    Args:
        project_path: Path to the project folder

    Returns:
        Path to the Makefile containing Makefile include, None otherwise
    """
    makefiles = find_files_recursive(project_path, "Makefile")
    include_patterns = get_runtime_config().analyzer.makefile_include_patterns

    for makefile_path in makefiles:
        try:
            content = makefile_path.read_text(encoding="utf-8")
            # Strip Makefile comments to avoid false positives
            active_lines = []
            for line in content.splitlines():
                comment_pos = line.find('#')
                if comment_pos >= 0:
                    line = line[:comment_pos]
                active_lines.append(line)
            active_content = '\n'.join(active_lines)
            if any(pattern in active_content for pattern in include_patterns):
                log_debug(f"Found Makefile with Makefile at {makefile_path}")
                return makefile_path
        except Exception as e:
            log_debug(f"Error reading {makefile_path}: {e}")
            continue

    log_debug(f"No Makefile with Makefile_java.mk found in {project_path}")
    return None
