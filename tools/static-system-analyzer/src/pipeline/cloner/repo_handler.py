"""Repository list handler module.

This module provides functionality for reading project names
from a text file (one project per line).
"""

import sys
from pathlib import Path
from typing import List

from common.logger import log_error


def read_repo_list(repo_file: str) -> List[str]:
    """Read project names from a text file.

    Args:
        repo_file: Path to the text file with project names.

    Returns:
        List of project names.

    Raises:
        SystemExit: If file not found or empty.
    """
    file_path = Path(repo_file)

    if not file_path.exists():
        log_error(f"Repository list file not found at {repo_file}")
        sys.exit(1)

    projects = _parse_repo_file(file_path)

    if not projects:
        log_error(f"No project names found in {repo_file}")
        sys.exit(1)

    return projects


def _parse_repo_file(file_path: Path) -> List[str]:
    """Parse repository file and extract project names.

    Args:
        file_path: Path object to the text file.

    Returns:
        List of non-empty, stripped project names.
    """
    projects: List[str] = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            project_name = line.strip()
            # Skip empty lines and comments
            if project_name and not project_name.startswith("#"):
                projects.append(project_name)

    return projects
