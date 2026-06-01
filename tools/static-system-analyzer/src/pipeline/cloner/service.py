"""Cloner service module.

This module provides the main service class for the cloning pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from .config import CloneConfig, get_config
from .csv_handler import ProjectRecord, read_csv_records, filter_by_pkg_name, _parse_version
from common.git_operations import shallow_clone, get_remote_tags
from common.shared_store import (
    SHARED_DIR_NAME,
    ensure_platform_link,
    shared_path,
)
from .repo_handler import read_repo_list
from common.logger import setup_logger, log_info, log_warning, log_debug, log_summary


@dataclass
class ClonerConfig:
    """Configuration for the cloner service.
    
    Attributes:
        repo_list: Path to file containing project names.
        destination: Destination directory for cloned repositories.
        platform_name: Platform name to filter from CSV.
        env_file: Path to .env file.
        csv_file: Path to CSV file.
        log_dir: Directory for log files.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    repo_list: str
    destination: str
    platform_name: str
    env_file: str = ".env"
    csv_file: str = "abc.csv"
    log_dir: str = "./logs"
    log_level: str = "INFO"


class ClonerService:
    """Service class for cloning repositories."""

    def __init__(self, config: ClonerConfig):
        """Initialize the cloner service.
        
        Args:
            config: ClonerConfig instance.
        """
        self.config = config
        self._clone_config: CloneConfig = None

    def run(self) -> int:
        """Execute the cloning pipeline.
        
        Returns:
            Exit code (0 for success, 1 for failure).
        """
        # Setup logger
        setup_logger(self.config.log_dir, self.config.platform_name, stage="clone", log_level=self.config.log_level)

        log_info("=" * 60)
        log_info("STAGE 1: CLONE")
        log_info("=" * 60)

        log_info("Loading configuration from .env file...")
        self._clone_config = get_config(self.config.env_file)
        log_info(f"Clone URL configured for: {self._clone_config.username}@{self._clone_config.base_url}")

        # Read project list from file
        log_info(f"Reading repository list from: {self.config.repo_list}")
        projects = read_repo_list(self.config.repo_list)
        log_info(f"Found {len(projects)} project(s) to process")

        # Parse CSV once for the project/platform
        log_info(f"Reading CSV file: {self.config.csv_file} for project: {self.config.platform_name}")
        all_records = read_csv_records(self.config.csv_file, self.config.platform_name)
        log_info(f"Found {len(all_records)} total record(s) for project '{self.config.platform_name}'")

        # Destination layout:
        #   <base>/_shared/<repo>_<tag>/        ← actual clone (deduplicated)
        #   <base>/<platform>/<repo>_<tag>      ← symlink → ../_shared/<repo>_<tag>
        base_destination = Path(self.config.destination).resolve()

        total_success = 0
        total_fail = 0
        total_warning = 0
        fallback_list: List[str] = []

        # Process each repo
        for repo_name in projects:
            log_info(f"\n{'='*50}")
            log_info(f"Processing repo: {repo_name}")
            log_info(f"{'='*50}")

            records = filter_by_pkg_name(all_records, repo_name)

            if not records:
                log_warning(f"No records found for pkg_name: {repo_name}")
                total_fail += 1
                continue

            log_info(f"Found {len(records)} tag(s) for repo '{repo_name}'")

            success, fail, fallbacks = self._clone_all_tags(
                records=records,
                base_destination=base_destination,
                repo_name=repo_name,
            )

            total_success += success
            total_fail += fail
            fallback_list.extend(fallbacks)

        extra_info: dict = {}
        if fallback_list:
            extra_info["Tag Fallbacks"] = f"{len(fallback_list)} repo(s)"
            for entry in fallback_list:
                extra_info[f"  - {entry}"] = ""

        log_summary("ALL_PROJECTS", total_success, total_fail, extra_info=extra_info)

        return 0

    def _clone_all_tags(
        self,
        records: List[ProjectRecord],
        base_destination: Path,
        repo_name: str,
    ) -> Tuple[int, int, List[str]]:
        """Clone all tags from the records into the shared store.

        Each tag is cloned once into ``<base>/_shared/`` and then exposed to
        the current platform via a relative symlink under ``<base>/<platform>/``.
        If the requested tag fails, attempts to find and clone the latest
        available remote tag as a fallback.
        """
        success = 0
        fail = 0
        fallbacks: List[str] = []
        platform = self.config.platform_name
        shared_dir = base_destination / SHARED_DIR_NAME

        for record in records:
            tag = record.pkg_version
            clone_url = self._clone_config.get_clone_url(repo_name)
            artifact_name = f"{repo_name}_{tag}"

            shared_target = shared_path(base_destination, artifact_name)

            # Reuse existing shared clone if present
            if shared_target.exists():
                log_info(f"Reusing shared clone: {shared_target}")
                ensure_platform_link(base_destination, platform, artifact_name)
                success += 1
                continue

            result = shallow_clone(
                url=clone_url,
                tag=tag,
                destination=shared_dir,
                repo_name=repo_name,
            )

            if result.success:
                ensure_platform_link(base_destination, platform, artifact_name)
                success += 1
            else:
                # Tag not found — try fallback to latest available remote tag
                log_warning(f"Tag '{tag}' not found for '{repo_name}', searching for latest available tag...")
                remote_tags = get_remote_tags(clone_url)

                if not remote_tags:
                    log_warning(f"No remote tags found for '{repo_name}', skipping.")
                    fail += 1
                    continue

                latest_tag = max(remote_tags, key=_parse_version)
                log_info(f"Falling back to latest available tag '{latest_tag}' for '{repo_name}' (requested: '{tag}')")

                fallback_artifact = f"{repo_name}_{latest_tag}"
                fallback_shared = shared_path(base_destination, fallback_artifact)

                if fallback_shared.exists():
                    log_info(f"Reusing shared clone for fallback: {fallback_shared}")
                    ensure_platform_link(base_destination, platform, fallback_artifact)
                    success += 1
                    fallbacks.append(f"{repo_name}: {tag} -> {latest_tag}")
                    continue

                fallback_result = shallow_clone(
                    url=clone_url,
                    tag=latest_tag,
                    destination=shared_dir,
                    repo_name=repo_name,
                )

                if fallback_result.success:
                    ensure_platform_link(base_destination, platform, fallback_artifact)
                    success += 1
                    fallbacks.append(f"{repo_name}: {tag} -> {latest_tag}")
                else:
                    fail += 1

        return success, fail, fallbacks
