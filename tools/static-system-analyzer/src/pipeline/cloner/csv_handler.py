"""CSV handling module.

This module provides functionality for reading and filtering
CSV records based on project name.
"""

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from common.logger import log_error


@dataclass
class ProjectRecord:
    """Data class representing a project record from CSV.

    Attributes:
        project_name: Project/platform name.
        pkg_name: Package/repository name.
        pkg_version: Package version (git tag).
    """

    project_name: str
    pkg_name: str
    pkg_version: str


def read_csv_records(
    csv_path: str,
    project_name: str,
) -> List[ProjectRecord]:
    """Read CSV file and filter records by project_name.

    Args:
        csv_path: Path to the CSV file.
        project_name: Project name (platform) to filter by.

    Returns:
        List of ProjectRecord matching the project_name.

    Raises:
        SystemExit: If CSV file not found or missing required columns.
    """
    csv_file = Path(csv_path)

    if not csv_file.exists():
        log_error(f"CSV file not found at {csv_path}")
        sys.exit(1)

    return _parse_csv(csv_file, project_name)


def filter_by_pkg_name(
    records: List[ProjectRecord],
    pkg_name: str,
) -> List[ProjectRecord]:
    """Filter records by package name and return only the latest version.

    Args:
        records: List of all ProjectRecord.
        pkg_name: Package name to filter by.

    Returns:
        List with single ProjectRecord having the latest version,
        or empty list if no match.
    """
    matching = [r for r in records if r.pkg_name == pkg_name]

    if not matching:
        return []

    # Sort by version and get the latest
    latest = max(matching, key=lambda r: _parse_version(r.pkg_version))

    return [latest]


def _parse_version(version: str) -> Tuple[int, ...]:
    """Parse version string into comparable tuple.

    Handles formats like: v1.0.0, 1.0.0, v2.1.3-beta, etc.

    Args:
        version: Version string.

    Returns:
        Tuple of integers for comparison.
    """
    # Remove 'v' prefix if present
    version = version.lstrip("vV")

    # Extract only numeric parts (handles -beta, -rc, etc.)
    version = version.split("-")[0]

    # Split by dots and convert to integers
    parts = re.split(r"[._]", version)

    result = []
    for part in parts:
        # Extract leading digits from each part
        match = re.match(r"(\d+)", part)
        if match:
            result.append(int(match.group(1)))

    # Ensure at least 3 parts for comparison
    while len(result) < 3:
        result.append(0)

    return tuple(result)


def _parse_csv(
    csv_file: Path,
    project_name: str,
) -> List[ProjectRecord]:
    """Parse CSV file and extract records for project.

    Args:
        csv_file: Path object to the CSV file.
        project_name: Project name to filter by.

    Returns:
        List of ProjectRecord matching the project_name.

    Raises:
        SystemExit: If CSV is empty or missing required columns.
    """
    matching_records: List[ProjectRecord] = []

    with open(csv_file, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        _validate_csv_headers(reader.fieldnames)

        for row in reader:
            if row.get("project_name") == project_name:
                record = ProjectRecord(
                    project_name=row["project_name"],
                    pkg_name=row["pkg_name"],
                    pkg_version=row["pkg_version"].strip(),
                )
                matching_records.append(record)

    return matching_records


def _validate_csv_headers(fieldnames: Optional[List[str]]) -> None:
    """Validate that CSV has required headers.

    Args:
        fieldnames: List of column names from CSV.

    Raises:
        SystemExit: If CSV is empty or missing required columns.
    """
    if fieldnames is None:
        log_error("CSV file is empty or has no headers")
        sys.exit(1)

    required_columns = ["project_name", "pkg_name", "pkg_version"]
    missing_columns = [col for col in required_columns if col not in fieldnames]

    if missing_columns:
        log_error(f"Missing required columns in CSV: {missing_columns}. Available columns: {fieldnames}")
        sys.exit(1)
