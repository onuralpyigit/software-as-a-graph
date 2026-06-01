"""
Analyzer service module.

This module provides the main service class for the analyzer pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .build_runner import check_gmake_available
from .processor import TopicProcessor, ProcessingResult
from common.logger import setup_logger, log_info, log_warning, log_error


@dataclass
class AnalyzerConfig:
    """Configuration for the analyzer service.
    
    Attributes:
        project_path: Root directory containing platform project folders.
        output_path: Root directory for CSV output files.
        log_path: Directory for log files.
        platform: Platform name (used for subdirectory and log file naming).
        run_build: Run gmake regenerate_code before processing.
        skip_on_build_failure: Skip project if build fails.
        verbose: Enable verbose output.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        analysis_mode: Analysis strategy — 'manual' or 'codeql'.
    """
    project_path: str
    output_path: str
    log_path: str
    platform: str
    run_build: bool = False
    skip_on_build_failure: bool = False
    verbose: bool = False
    log_level: str = "INFO"
    analysis_mode: str = "manual"


class AnalyzerService:
    """Service class for analyzing projects."""

    def __init__(self, config: AnalyzerConfig):
        """Initialize the analyzer service.
        
        Args:
            config: AnalyzerConfig instance.
        """
        self.config = config

    def run(self) -> int:
        """Execute the analyzer pipeline.
        
        Returns:
            Exit code (0 for success, non-zero for failure).
        """
        # Setup paths with platform subdirectory
        project_path = Path(self.config.project_path).resolve()
        output_path = Path(self.config.output_path).resolve()
        log_path = Path(self.config.log_path).resolve()
        platform = self.config.platform

        # Platform-specific paths
        projects_root = project_path / platform
        output_dir = output_path / platform

        # Setup logging
        setup_logger(str(log_path), platform, stage="analyze", log_level=self.config.log_level)

        log_info("=" * 60)
        log_info("STAGE 2: ANALYZE")
        log_info("=" * 60)

        log_info(f"Starting analysis for platform: {platform}")
        log_info(f"Projects path: {projects_root}")
        log_info(f"Output path: {output_dir}")
        log_info(f"Log path: {log_path}")

        # Validate projects path
        if not projects_root.exists():
            log_error(f"Projects path does not exist: {projects_root}")
            return 1

        if not projects_root.is_dir():
            log_error(f"Projects path is not a directory: {projects_root}")
            return 1

        # Check gmake availability if build is enabled
        run_build = self.config.run_build
        if run_build and not check_gmake_available():
            log_error("gmake is not available but --build flag was specified")
            return 1

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        log_info(f"Output directory created: {output_dir}")

        # Select analysis strategy
        analysis_mode = self.config.analysis_mode
        log_info(f"Analysis mode: {analysis_mode}")

        if analysis_mode == "codeql":
            from .strategies.codeql import CodeQLStrategy
            strategy = CodeQLStrategy()
        else:
            from .strategies.manual import ManualStrategy
            strategy = ManualStrategy()

        processor = TopicProcessor(
            projects_root=projects_root,
            run_build=run_build,
            skip_on_build_failure=self.config.skip_on_build_failure,
            output_dir=output_dir,
            verbose=self.config.verbose,
            strategy=strategy,
            platform=platform,
        )

        results = processor.process_all_projects()

        # Log summary
        log_info("=" * 60)
        log_info(f"Processing Summary - Platform: {platform}")
        log_info("=" * 60)

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        log_info(f"Total projects processed: {len(results)}")
        log_info(f"Successful: {len(successful)}")
        log_info(f"Failed: {len(failed)}")

        if successful:
            log_info("Successful projects:")
            for result in successful:
                deps_info = ""
                if result.dependencies:
                    deps_info = f" | Uses: {', '.join(result.dependencies)}"
                log_info(f"  [OK] {result.versioned_name}: {result.entry_count} topics, {result.uses_count} dependencies{deps_info}")

        if failed:
            log_info("Failed projects:")
            for result in failed:
                log_error(f"  [FAIL] {result.versioned_name}: {result.error_message}")

        log_info(f"Analysis complete. {len(successful)} successful, {len(failed)} failed.")
        return 0 if not failed else 1


class MetricsService:
    """Service class for running code metrics analysis (ck tool).

    This is the second step of the analyze stage. It runs inside Docker
    where Java and ck.jar are available, scans each Java project, and writes
    per-application _metrics.json files (size, complexity, cohesion) to the
    analyzed output directory.
    """

    def __init__(self, config: 'MetricsServiceConfig'):
        """Initialize the metrics service.

        Args:
            config: MetricsServiceConfig instance.
        """
        self.config = config

    def run(self) -> int:
        """Execute code metrics scanning for all projects.

        Returns:
            Exit code (0 for success, non-zero for failure).
        """
        from .metrics_scanner import MetricsConfig, scan_all_projects

        platform = self.config.platform
        log_path = Path(self.config.log_path).resolve()

        # Setup logging
        setup_logger(str(log_path), platform, stage="metrics", log_level=self.config.log_level)

        log_info("=" * 60)
        log_info("STAGE 2b: CODE METRICS ANALYSIS (ck)")
        log_info("=" * 60)

        # Output directory: same as analyzer output (output/analyzed/<platform>/)
        output_dir = str(Path(self.config.output_path).resolve() / platform)

        metrics_config = MetricsConfig(
            platform_name=platform,
            source_dir=self.config.project_path,
            output_dir=output_dir,
            logs_dir=str(log_path),
            log_level=self.config.log_level,
        )

        log_info(f"Source dir: {self.config.project_path}")
        log_info(f"Output dir: {output_dir}")

        results = scan_all_projects(metrics_config)

        # Summary
        total = len(results)
        success_count = sum(1 for v in results.values() if v is not None)
        fail_count = total - success_count

        log_info("=" * 60)
        log_info(f"Code Metrics Summary - Platform: {platform}")
        log_info("=" * 60)
        log_info(f"Total projects scanned: {total}")
        log_info(f"Successful: {success_count}")
        log_info(f"Failed: {fail_count}")

        return 0


@dataclass
class MetricsServiceConfig:
    """Configuration for the code metrics scanning service.

    Attributes:
        project_path: Root directory containing platform project folders.
        output_path: Root directory for output files (CSV + metrics JSON).
        log_path: Directory for log files.
        platform: Platform name.
        log_level: Logging level.
    """
    project_path: str
    output_path: str
    log_path: str
    platform: str
    log_level: str = "INFO"
