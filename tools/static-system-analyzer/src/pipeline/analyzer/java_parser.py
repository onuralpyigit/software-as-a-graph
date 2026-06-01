"""Java source parsing utilities for dependency extraction."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Set

from common.logger import log_debug, log_error
from common.runtime_config import get_runtime_config


IMPORT_PATTERN = re.compile(r"^\s*import\s+(?:static\s+)?([^;]+);", re.MULTILINE)
COMMENT_PATTERN = re.compile(r"//.*?$|/\*.*?\*/", re.MULTILINE | re.DOTALL)
PACKAGE_SEGMENT_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _strip_java_comments(content: str) -> str:
    """Remove line and block comments before scanning import statements."""
    return COMMENT_PATTERN.sub("", content)


def _extract_dependency_name(import_target: str) -> str | None:
    """Extract the first package segment after the configured import domain prefix."""
    import_domain_prefix = get_runtime_config().analyzer.import_domain_prefix
    if not import_domain_prefix:
        return None

    domain_prefix = f"{import_domain_prefix}."
    normalized_target = import_target.strip()

    if not normalized_target.startswith(domain_prefix):
        return None

    remainder = normalized_target[len(domain_prefix):]
    if not remainder:
        return None

    dependency_name = remainder.split(".", 1)[0].strip()
    if not dependency_name:
        return None

    if not PACKAGE_SEGMENT_PATTERN.match(dependency_name):
        return None

    suffixes = get_runtime_config().analyzer.dependency_suffixes
    if not any(dependency_name.endswith(suffix) for suffix in suffixes):
        return None

    return dependency_name


def parse_java_import_dependencies(project_path: Path) -> Set[str]:
    """Scan Java import lines and extract the first segment after the domain prefix.

    The scanner walks the project recursively, reads only `.java` files, and
    treats the first package segment after the configured import domain prefix
    in an `import ...;` line as a direct dependency.
    """
    dependencies: Set[str] = set()

    if not project_path.exists():
        return dependencies

    for java_file in project_path.rglob("*.java"):
        try:
            content = java_file.read_text(encoding="utf-8")
        except Exception as exc:
            log_error(f"Error reading Java file {java_file}: {exc}")
            continue

        uncommented_content = _strip_java_comments(content)

        for import_target in IMPORT_PATTERN.findall(uncommented_content):
            dependency_name = _extract_dependency_name(import_target)
            if dependency_name:
                dependencies.add(dependency_name)
                log_debug(f"Found Java import dependency: {dependency_name} in {java_file}")

    return dependencies