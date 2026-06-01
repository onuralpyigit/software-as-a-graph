"""
Main processor for topic extraction workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .build_runner import run_regenerate_code
from .csv_writer import write_topic_csv
from .file_finder import find_makefile_with_Makefile
from .models import TopicEntry, ProjectInfo, expand_pubsub_entries
from .project_scanner import ScanStats, scan_projects
from .strategies.base import AnalysisStrategy
from .strategies.manual import ManualStrategy
from common.logger import log_info, log_warning, log_error, log_debug
from common.shared_store import (
    SHARED_DIR_NAME,
    ensure_platform_link,
    shared_path,
)


@dataclass
class ProcessingResult:
    """Result of processing a single project."""
    folder_name: str
    versioned_name: str
    success: bool
    build_success: bool
    csv_path: Optional[str]
    entry_count: int
    uses_count: int = 0
    error_message: Optional[str] = None
    dependencies: Optional[List[str]] = None


class TopicProcessor:
    """
    Main processor for extracting topics from projects.
    """

    def __init__(
        self,
        projects_root: Path,
        run_build: bool = True,
        skip_on_build_failure: bool = False,
        output_dir: Optional[Path] = None,
        verbose: bool = False,
        strategy: Optional[AnalysisStrategy] = None,
        platform: Optional[str] = None,
    ):
        """
        Initialize the processor.

        Args:
            projects_root: Root path containing all project folders
            run_build: Whether to run gmake regenerate_code
            skip_on_build_failure: Skip project if build fails
            output_dir: Per-platform output directory for CSV symlinks (None
                = write directly into the project folder, no dedup)
            verbose: Enable verbose output for build commands
            strategy: Analysis strategy (default: ManualStrategy)
            platform: Platform name; when set together with output_dir the
                CSV is written once into ``<output_dir>/../_shared/`` and
                exposed via a per-platform symlink.
        """
        self.projects_root = Path(projects_root)
        self.run_build = run_build
        self.skip_on_build_failure = skip_on_build_failure
        self.output_dir = Path(output_dir) if output_dir else None
        self.verbose = verbose
        self.strategy = strategy or ManualStrategy()
        self.platform = platform

    def process_project(self, project_info: ProjectInfo) -> ProcessingResult:
        """
        Process a single project.

        Args:
            project_info: ProjectInfo object containing project details

        Returns:
            ProcessingResult with details of the operation
        """
        folder_name = project_info.folder_name
        versioned_name = project_info.versioned_name
        folder_path = Path(project_info.folder_path)
        build_success = True
        error_message = None

        # Deduplicate by versioned_name: if the shared CSV already exists,
        # only refresh the per-platform symlink and skip strategy + build.
        shared_csv: Optional[Path] = None
        platform_csv: Optional[Path] = None
        if self.output_dir and self.platform:
            base_dir = self.output_dir.parent
            artifact = f"{versioned_name}.csv"
            shared_csv = shared_path(base_dir, artifact)
            platform_csv = self.output_dir / artifact
            if shared_csv.exists():
                log_info(
                    f"Reusing shared analysis for {versioned_name} "
                    f"({shared_csv})"
                )
                ensure_platform_link(base_dir, self.platform, artifact)
                topic_count, uses_count, dependencies = _summarize_csv(shared_csv)
                return ProcessingResult(
                    folder_name=folder_name,
                    versioned_name=versioned_name,
                    success=True,
                    build_success=True,
                    csv_path=str(platform_csv),
                    entry_count=topic_count,
                    uses_count=uses_count,
                    error_message=None,
                    dependencies=dependencies if dependencies else None,
                )

        # Run build if enabled (only for manual mode — CodeQL handles builds internally)
        if self.run_build and isinstance(self.strategy, ManualStrategy):
            # Find the Makefile with Makefile pattern
            makefile_path = find_makefile_with_Makefile(folder_path)
            if makefile_path:
                build_success, build_output = run_regenerate_code(
                    makefile_path, 
                    verbose=self.verbose
                )
                if not build_success:
                    error_message = f"Build failed: {build_output}"
                    if self.skip_on_build_failure:
                        return ProcessingResult(
                            folder_name=folder_name,
                            versioned_name=versioned_name,
                            success=False,
                            build_success=False,
                            csv_path=None,
                            entry_count=0,
                            error_message=error_message
                        )
            else:
                log_warning(f"No Makefile with Makefile found for {folder_name}, skipping build")

        # Collect topics via the configured strategy
        all_entries: List[TopicEntry] = self.strategy.extract(folder_path, folder_name)

        # Expand pubsub entries
        expanded_entries = expand_pubsub_entries(all_entries)

        # Replace all source_folder values with the project name
        for entry in expanded_entries:
            entry.source_folder = folder_name

        # Remove duplicates - keep unique (folder, name, role) tuples
        seen = set()
        unique_entries = []
        for entry in expanded_entries:
            key = (entry.source_folder, entry.name, entry.role)
            if key not in seen:
                seen.add(key)
                unique_entries.append(entry)
        
        log_debug(f"Removed {len(expanded_entries) - len(unique_entries)} duplicate entries")

        # Count topic entries (excluding "uses" entries)
        topic_count = sum(1 for e in unique_entries if e.role != "uses")
        uses_count = sum(1 for e in unique_entries if e.role == "uses")
        dependencies = [e.name for e in unique_entries if e.role == "uses"]

        # Write CSV with versioned_name (dedup via _shared + symlink when
        # both output_dir and platform are configured)
        if self.output_dir and self.platform and shared_csv is not None and platform_csv is not None:
            csv_path = shared_csv
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_success = write_topic_csv(csv_path, unique_entries, expand_pubsub=False)
            if csv_success:
                ensure_platform_link(
                    self.output_dir.parent, self.platform, platform_csv.name
                )
                csv_path = platform_csv
        elif self.output_dir:
            csv_path = self.output_dir / f"{versioned_name}.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_success = write_topic_csv(csv_path, unique_entries, expand_pubsub=False)
        else:
            csv_path = folder_path / f"{versioned_name}.csv"
            csv_success = write_topic_csv(csv_path, unique_entries, expand_pubsub=False)

        if not csv_success:
            error_message = "Failed to write CSV file"

        return ProcessingResult(
            folder_name=folder_name,
            versioned_name=versioned_name,
            success=csv_success,
            build_success=build_success,
            csv_path=str(csv_path) if csv_success else None,
            entry_count=topic_count,
            uses_count=uses_count,
            error_message=error_message,
            dependencies=dependencies if dependencies else None
        )

    def process_all_projects(self) -> List[ProcessingResult]:
        """
        Process all valid projects in the projects root.

        Returns:
            List of ProcessingResult objects
        """
        results = []
        stats = ScanStats()

        mode = "codeql" if not isinstance(self.strategy, ManualStrategy) else "manual"
        for project_info in scan_projects(self.projects_root, stats, analysis_mode=mode):
            log_info(f"Processing project: {project_info.folder_name} ({project_info.versioned_name})")
            result = self.process_project(project_info)
            results.append(result)

            if result.success:
                log_info(
                    f"Successfully processed {project_info.folder_name}: "
                    f"{result.entry_count} entries"
                )
            else:
                log_error(
                    f"Failed to process {project_info.folder_name}: "
                    f"{result.error_message}"
                )

        # Log scan statistics
        stats.log_summary()

        # Log processing statistics
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        total_entries = sum(r.entry_count for r in successful_results)
        total_uses = sum(r.uses_count for r in successful_results)
        
        log_info("=" * 50)
        log_info("Processing Statistics")
        log_info("=" * 50)
        log_info(f"Successfully processed: {len(successful_results)}")
        for r in successful_results:
            deps_info = ""
            if r.dependencies:
                deps_info = f" | uses: {', '.join(r.dependencies)}"
            log_info(f"  [OK] {r.versioned_name}: {r.entry_count} topics, {r.uses_count} dependencies{deps_info}")
        
        log_info(f"Failed to process: {len(failed_results)}")
        for r in failed_results:
            log_error(f"  [FAIL] {r.versioned_name}: {r.error_message}")
        
        log_info(f"Total topic entries: {total_entries}")
        log_info(f"Total uses relations: {total_uses}")
        log_info("=" * 50)

        return results


def _summarize_csv(csv_path: Path) -> tuple:
    """Return (topic_count, uses_count, dependencies) parsed from a CSV.

    Used when reusing an already-produced shared CSV: avoids re-running the
    extraction strategy while still surfacing accurate counts in the
    per-platform summary log.
    """
    import csv as _csv

    topic_count = 0
    uses_count = 0
    dependencies: List[str] = []
    try:
        with csv_path.open("r", encoding="utf-8") as fh:
            for row in _csv.reader(fh):
                if not row or len(row) < 3:
                    continue
                role = row[2].strip()
                if role == "uses":
                    uses_count += 1
                    dependencies.append(row[1].strip())
                else:
                    topic_count += 1
    except OSError as exc:
        log_warning(f"Could not summarize shared CSV {csv_path}: {exc}")
    return topic_count, uses_count, dependencies
