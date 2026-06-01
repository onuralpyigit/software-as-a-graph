"""Git operations module.

This module handles git clone operations including
shallow cloning and cleanup of .git directories.
"""

import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from .logger import log_clone_start, log_clone_success, log_clone_error, log_info, log_error

# Clone timeout in seconds (5 minutes)
CLONE_TIMEOUT = 300


@dataclass
class CloneResult:
    """Result of a clone operation.

    Attributes:
        success: Whether the clone was successful.
        tag: The git tag that was cloned.
        path: Path where the repository was cloned.
        error: Error message if clone failed.
        timestamp: Timestamp of the clone operation.
    """

    success: bool
    tag: str
    path: Optional[Path] = None
    error: Optional[str] = None
    timestamp: Optional[str] = None


def shallow_clone(
    url: str,
    tag: str,
    destination: Path,
    repo_name: str,
) -> CloneResult:
    """Perform a shallow clone with a specific tag.

    Clones the repository with depth=1 for the specified tag,
    then removes the .git directory.

    Args:
        url: Authenticated git clone URL.
        tag: Git tag to clone.
        destination: Parent directory for the clone.
        repo_name: Name of the repository for folder naming.

    Returns:
        CloneResult indicating success or failure.
    """
    clone_path = destination / f"{repo_name}_{tag}"

    _ensure_directory(destination)

    result = _execute_clone(url, tag, clone_path)

    if result.success:
        _remove_git_directory(clone_path)

    return result


def _ensure_directory(path: Path) -> None:
    """Ensure directory exists, create if necessary.

    Args:
        path: Directory path to create.
    """
    path.mkdir(parents=True, exist_ok=True)


def _execute_clone(url: str, tag: str, clone_path: Path) -> CloneResult:
    """Execute git clone command.

    Args:
        url: Git clone URL.
        tag: Git tag/branch to clone.
        clone_path: Destination path for clone.

    Returns:
        CloneResult with operation outcome.
    """
    cmd = [
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        tag,
        url,
        str(clone_path),
    ]

    log_clone_start(tag, str(clone_path))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=CLONE_TIMEOUT,
        )

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if result.returncode != 0:
            log_clone_error(tag, result.stderr.strip())
            return CloneResult(
                success=False,
                tag=tag,
                error=result.stderr,
                timestamp=timestamp,
            )

        log_clone_success(tag, str(clone_path))

        return CloneResult(
            success=True,
            tag=tag,
            path=clone_path,
            timestamp=timestamp,
        )

    except subprocess.TimeoutExpired:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_msg = "Clone operation timed out"
        log_clone_error(tag, error_msg)
        return CloneResult(
            success=False,
            tag=tag,
            error=error_msg,
            timestamp=timestamp,
        )
    except OSError as exc:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_clone_error(tag, str(exc))
        return CloneResult(
            success=False,
            tag=tag,
            error=str(exc),
            timestamp=timestamp,
        )


def _remove_git_directory(clone_path: Path) -> None:
    """Remove .git directory from cloned repository.

    Args:
        clone_path: Path to the cloned repository.
    """
    git_dir = clone_path / ".git"

    if git_dir.exists():
        shutil.rmtree(git_dir)
        log_info(f"Removed .git directory from '{clone_path}'")


def get_remote_tags(url: str) -> List[str]:
    """Fetch remote tags from a git repository.
    
    Args:
        url: Authenticated git clone URL.
    
    Returns:
        List of tag names.
    """
    cmd = ["git", "ls-remote", "--tags", "--refs", url]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False
        )
        
        if result.returncode != 0:
            log_error(f"Failed to fetch tags from {url}. Exit code: {result.returncode}")
            if result.stderr:
                 log_error(f"Stderr: {result.stderr.strip()}")
            return []
        
        tags = []
        for line in result.stdout.splitlines():
            # Output format: <sha> refs/tags/<tag_name>
            parts = line.split()
            if len(parts) > 1:
                ref = parts[1]
                if ref.startswith('refs/tags/'):
                    tag_name = ref.replace('refs/tags/', '')
                    # Skip dereferenced tags
                    if not tag_name.endswith('^{}'):
                        tags.append(tag_name)
        
        if not tags:
             log_info(f"Command successful but parsed 0 tags for {url}")
             # Log first few lines of stdout solely for debugging
             if result.stdout:
                 log_info(f"Stdout head: {result.stdout[:200]}...")

        return tags
        
    except (subprocess.SubprocessError, OSError) as e:
        log_error(f"Exception during tag fetch from {url}: {str(e)}")
        return []
