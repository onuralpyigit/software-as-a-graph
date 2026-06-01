"""Helpers for the deduplicated ``_shared`` content-store layout.

Pipeline stages (clone, analyze, metrics) write artifacts once into a
``<base>/_shared/<name>`` location and expose them per-platform via a
relative symlink at ``<base>/<platform>/<name>``.

This module centralizes the symlink bookkeeping so every stage behaves
identically (idempotent, replaces stale links, refuses to clobber real
files).
"""

from __future__ import annotations

from pathlib import Path

from .logger import log_debug, log_warning


SHARED_DIR_NAME = "_shared"


def shared_path(base_dir: Path, name: str) -> Path:
    """Return the canonical shared location for an artifact."""
    return base_dir / SHARED_DIR_NAME / name


def platform_link_path(base_dir: Path, platform: str, name: str) -> Path:
    """Return the per-platform symlink path for an artifact."""
    return base_dir / platform / name


def ensure_platform_link(base_dir: Path, platform: str, name: str) -> Path:
    """Create or refresh a relative symlink ``<platform>/<name>`` → ``_shared/<name>``.

    Idempotent: a symlink already pointing at the shared target is left
    untouched. Stale or broken symlinks are replaced. A real file or
    directory at the link path is preserved (a warning is emitted) to
    avoid accidental data loss.

    Args:
        base_dir: Directory containing both ``_shared`` and per-platform dirs.
        platform: Platform name (subdirectory).
        name: Artifact name (file or directory) under ``_shared``.

    Returns:
        The link path.
    """
    target_shared = shared_path(base_dir, name)
    link_path = platform_link_path(base_dir, platform, name)
    link_path.parent.mkdir(parents=True, exist_ok=True)

    relative_target = Path("..") / SHARED_DIR_NAME / name

    if link_path.is_symlink():
        try:
            current = (link_path.parent / link_path.readlink()).resolve()
        except OSError:
            current = None
        if current == target_shared.resolve():
            return link_path
        link_path.unlink()
    elif link_path.exists():
        log_warning(
            f"Refusing to replace non-symlink path with shared link: {link_path}"
        )
        return link_path

    link_path.symlink_to(relative_target, target_is_directory=target_shared.is_dir())
    log_debug(f"Created shared link: {link_path} -> {relative_target}")
    return link_path
