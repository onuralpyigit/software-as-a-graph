#!/usr/bin/env python3
"""
Static System Analyzer - Main Entry Point

This is the main orchestrator for the static analysis pipeline.
It coordinates the following stages:
1. CLONE: Clone repositories from version control
2. ANALYZE: Extract topic data from projects (HOST)
2b. METRICS: Run code metrics analysis (ck) on Java projects (DOCKER)
3. STRUCTURAL: Structural anomaly analysis
4. AGGREGATE: Merge data into JSON format (+ structural results)
5. STAT: Generate statistical analysis

Usage:
    python main.py --platform <name> [options]
    python main.py --platform avionics --all
    python main.py --platform avionics --clone-only
    python main.py --platform avionics --analyze-only
    python main.py --platform avionics --metrics-only
    python main.py --platform avionics --aggregate-only
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Add parent directory to path for imports
# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.logger import setup_logger, log_info, log_error


@dataclass
class PipelineConfig:
    """Configuration for the entire pipeline.
    
    Attributes:
        platform: Platform name (required).
        workspace: Base workspace directory.
        repo_list: Path to repository list file.
        csv_file: Path to CSV file for cloning.
        env_file: Path to .env file.
        config_dir: Directory containing configuration files.
        run_build: Run gmake regenerate_code during analysis.
        skip_on_build_failure: Skip project if build fails.
        verbose: Enable verbose output.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        stat_format: Stat output format (markdown or pdf).
    """
    platform: str
    workspace: str = "."
    repo_list: str = "config/repo_names.txt"
    csv_file: str = "config/abc.csv"
    env_file: str = "config/.env"
    config_dir: str = "config"
    run_build: bool = False
    skip_on_build_failure: bool = False
    verbose: bool = False
    log_level: str = "INFO"
    stat_format: str = "markdown"
    dds_mask: bool = True
    enable_metrics: bool = False
    analysis_mode: str = "manual"

    @property
    def _base_output_dir(self) -> Path:
        """Base output directory at project root."""
        # Return 'output' directory at the project root (src/cmd/main.py -> src/cmd -> src -> root)
        # Use .resolve() to ensure absolute path
        return Path(__file__).resolve().parent.parent.parent / "output"

    @property
    def cloned_dir(self) -> str:
        """Directory for cloned repositories."""
        return str(self._base_output_dir / "cloned")

    @property
    def analyzed_dir(self) -> str:
        """Directory for analysis CSV output."""
        return str(self._base_output_dir / "analyzed")

    @property
    def aggregated_dir(self) -> str:
        """Directory for aggregated JSON output."""
        return str(self._base_output_dir / "aggregated")

    @property
    def stat_dir(self) -> str:
        """Directory for statistical analysis output."""
        return str(self._base_output_dir / "stat")

    @property
    def structural_dir(self) -> str:
        """Directory for structural analysis output."""
        return str(self._base_output_dir / "structural")

    @property
    def logs_dir(self) -> str:
        """Directory for log files."""
        return str(self._base_output_dir / "logs")


class Pipeline:
    """Main pipeline orchestrator."""

    def __init__(self, config: PipelineConfig):
        """Initialize the pipeline.
        
        Args:
            config: PipelineConfig instance.
        """
        self.config = config

    def run_clone(self) -> int:
        """Run the clone stage.
        
        Returns:
            Exit code (0 for success, non-zero for failure).
        """
        from pipeline.cloner import ClonerService
        from pipeline.cloner.service import ClonerConfig

        cloner_config = ClonerConfig(
            repo_list=self.config.repo_list,
            destination=self.config.cloned_dir,
            platform_name=self.config.platform,
            env_file=self.config.env_file,
            csv_file=self.config.csv_file,
            log_dir=self.config.logs_dir,
            log_level=self.config.log_level,
        )

        service = ClonerService(cloner_config)
        return service.run()

    def run_analyze(self) -> int:
        """Run the analyze stage.
        
        Returns:
            Exit code (0 for success, non-zero for failure).
        """
        from pipeline.analyzer import AnalyzerService
        from pipeline.analyzer.service import AnalyzerConfig

        analyzer_config = AnalyzerConfig(
            project_path=self.config.cloned_dir,
            output_path=self.config.analyzed_dir,
            log_path=self.config.logs_dir,
            platform=self.config.platform,
            run_build=self.config.run_build,
            skip_on_build_failure=self.config.skip_on_build_failure,
            verbose=self.config.verbose,
            log_level=self.config.log_level,
            analysis_mode=self.config.analysis_mode,
        )

        service = AnalyzerService(analyzer_config)
        return service.run()

    def run_metrics(self) -> int:
        """Run the code metrics scanning stage (Stage 2b).

        This runs inside Docker where ck (Java metrics JAR) is available.
        It scans each project directory and writes _metrics.json files
        to the analyzed output directory.

        Returns:
            Exit code (0 for success, non-zero for failure).
        """
        from pipeline.analyzer import MetricsService
        from pipeline.analyzer.service import MetricsServiceConfig

        metrics_config = MetricsServiceConfig(
            project_path=self.config.cloned_dir,
            output_path=self.config.analyzed_dir,
            log_path=self.config.logs_dir,
            platform=self.config.platform,
            log_level=self.config.log_level,
        )

        service = MetricsService(metrics_config)
        return service.run()

    def run_aggregate(self) -> int:
        """Run the aggregate stage.
        
        Returns:
            Exit code (0 for success, non-zero for failure).
        """
        from pipeline.aggregator import AggregatorService
        from pipeline.aggregator.service import AggregatorConfig

        aggregator_config = AggregatorConfig(
            root_dir=str(Path(self.config.cloned_dir) / self.config.platform),
            projects_dir=self.config.analyzed_dir,
            output_dir=self.config.aggregated_dir,
            logs_dir=self.config.logs_dir,
            platform_name=self.config.platform,
            config_dir=self.config.config_dir,
            structural_dir=self.config.structural_dir,
            log_level=self.config.log_level,
            dds_mask=self.config.dds_mask,
        )

        service = AggregatorService(aggregator_config)
        return service.run()

    def run_stat(self) -> int:
        """Run the stat stage (statistical analysis).
        
        Returns:
            Exit code (0 for success, non-zero for failure).
        """
        # Input file is the aggregated JSON from previous stage
        input_file = str(
            Path(self.config.aggregated_dir) / f"{self.config.platform}_relations.json"
        )
        
        from pipeline.stat import StatService
        from pipeline.stat.service import StatConfig

        stat_config = StatConfig(
            input_file=input_file,
            output_dir=self.config.stat_dir,
            logs_dir=self.config.logs_dir,
            platform_name=self.config.platform,
            log_level=self.config.log_level,
            output_format=self.config.stat_format,
            dds_mask=self.config.dds_mask,
        )

        service = StatService(stat_config)
        return service.run()

    def run_structural(self) -> int:
        """Run the structural analysis stage.
        
        Returns:
            Exit code (0 for success, non-zero for failure).
        """
        from pipeline.structural import StructuralService
        from pipeline.structural.service import StructuralConfig

        structural_config = StructuralConfig(
            root_dir=str(Path(self.config.cloned_dir) / self.config.platform),
            projects_dir=self.config.analyzed_dir,
            output_dir=self.config.structural_dir,
            logs_dir=self.config.logs_dir,
            platform_name=self.config.platform,
            config_dir=self.config.config_dir,
            log_level=self.config.log_level,
            dds_mask=self.config.dds_mask,
        )

        service = StructuralService(structural_config)
        return service.run()

    def run_all(self) -> int:
        """Run all pipeline stages.
        
        Returns:
            Exit code (0 for success, non-zero for failure).
        """
        log_info("=" * 60)
        log_info(f"RUNNING FULL PIPELINE FOR PLATFORM: {self.config.platform}")
        log_info("=" * 60)

        # Stage 1: Clone
        result = self.run_clone()
        if result != 0:
            log_error(f"Clone stage failed with exit code {result}")
            return result

        # Stage 2: Analyze
        result = self.run_analyze()
        if result != 0:
            log_error(f"Analyze stage failed with exit code {result}")
            return result

        # Stage 2b: Code Metrics Analysis (optional, enabled via --enable-metrics)
        if self.config.enable_metrics:
            result = self.run_metrics()
            if result != 0:
                log_error(f"Code metrics stage failed with exit code {result}")
                # Non-fatal: continue pipeline even if metrics fails
                log_info("Continuing pipeline despite code metrics failure.")
        else:
            log_info("Code metrics scan skipped (enable with --enable-metrics or METRICS=1).")

        # Stage 3: Structural Analysis
        result = self.run_structural()
        if result != 0:
            log_error(f"Structural analysis stage failed with exit code {result}")
            return result

        # Stage 4: Aggregate (incorporates structural results)
        result = self.run_aggregate()
        if result != 0:
            log_error(f"Aggregate stage failed with exit code {result}")
            return result

        # Stage 5: Stat (Statistical Analysis)
        result = self.run_stat()
        if result != 0:
            log_error(f"Stat stage failed with exit code {result}")
            return result

        log_info("=" * 60)
        log_info("PIPELINE COMPLETED SUCCESSFULLY")
        log_info("=" * 60)
        log_info(f"  Platform: {self.config.platform}")
        log_info(f"  Cloned:    {self.config.cloned_dir}/{self.config.platform}/")
        log_info(f"  Analyzed:  {self.config.analyzed_dir}/{self.config.platform}/")
        log_info(f"  Aggregated: {self.config.aggregated_dir}/{self.config.platform}_relations.json")
        log_info(f"  Statistics: {self.config.stat_dir}/{self.config.platform}_statistics.json")
        log_info(f"  Structural: {self.config.structural_dir}/{self.config.platform}/")
        log_info(f"  Logs:      {self.config.logs_dir}/")
        log_info("=" * 60)

        return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Static System Analyzer - Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python main.py --platform avionics --all

  # Run individual stages
  python main.py --platform avionics --clone-only
  python main.py --platform avionics --analyze-only
  python main.py --platform avionics --metrics-only
  python main.py --platform avionics --aggregate-only
  python main.py --platform avionics --stat-only

  # With custom paths
  python main.py --platform avionics --all --workspace /path/to/workspace

  # With build enabled
  python main.py --platform avionics --all --build --verbose
        """
    )

    # Required arguments
    parser.add_argument(
        "--platform", "-p",
        required=True,
        help="Platform name (required)"
    )

    # Stage selection (mutually exclusive group)
    stage_group = parser.add_mutually_exclusive_group(required=True)
    stage_group.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all pipeline stages (clone → analyze → aggregate → stat)"
    )
    stage_group.add_argument(
        "--clone-only",
        action="store_true",
        help="Run only the clone stage"
    )
    stage_group.add_argument(
        "--analyze-only",
        action="store_true",
        help="Run only the analyze stage"
    )
    stage_group.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Run only the aggregate stage"
    )
    stage_group.add_argument(
        "--stat-only",
        action="store_true",
        help="Run only the stat stage (statistical analysis)"
    )
    stage_group.add_argument(
        "--structural-only",
        action="store_true",
        help="Run only the structural analysis stage"
    )
    stage_group.add_argument(
        "--metrics-only",
        action="store_true",
        help="Run only the code metrics scanning stage (ck, requires Docker)"
    )

    # Optional arguments
    parser.add_argument(
        "--workspace", "-w",
        default=".",
        help="Base workspace directory (default: current directory)"
    )
    parser.add_argument(
        "--repo-list",
        default="config/repo_names.txt",
        help="Path to repository list file (default: config/repo_names.txt)"
    )
    parser.add_argument(
        "--csv-file",
        default="config/abc.csv",
        help="Path to CSV file for cloning (default: config/abc.csv)"
    )
    parser.add_argument(
        "--env-file",
        default="config/.env",
        help="Path to .env file (default: config/.env)"
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Run gmake regenerate_code during analysis"
    )
    parser.add_argument(
        "--skip-on-build-failure",
        action="store_true",
        help="Skip project if build fails"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--stat-format",
        choices=["markdown", "pdf"],
        default="markdown",
        help="Stat output format: markdown or pdf (default: markdown)"
    )
    parser.add_argument(
        "--dds-mask",
        action="store_true",
        help="Enable DDS QoS conversion in aggregator (default: disabled)"
    )
    parser.add_argument(
        "--enable-metrics",
        action="store_true",
        help="Enable code metrics scan (ck) in full pipeline (default: disabled)"
    )
    parser.add_argument(
        "--analysis-mode",
        choices=["manual", "codeql"],
        default="manual",
        help="Analysis strategy: manual (XML+import) or codeql (call-graph) (default: manual)"
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point.
    
    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    args = parse_args()

    config = PipelineConfig(
        platform=args.platform,
        workspace=args.workspace,
        repo_list=args.repo_list,
        csv_file=args.csv_file,
        env_file=args.env_file,
        run_build=args.build,
        skip_on_build_failure=args.skip_on_build_failure,
        verbose=args.verbose,
        log_level=args.log_level,
        stat_format=args.stat_format,
        dds_mask=args.dds_mask,
        enable_metrics=args.enable_metrics,
        analysis_mode=args.analysis_mode,
    )

    # Setup logger for main pipeline
    setup_logger(config.logs_dir, config.platform, stage="pipeline", log_level=config.log_level)

    pipeline = Pipeline(config)

    if args.all:
        return pipeline.run_all()
    elif args.clone_only:
        return pipeline.run_clone()
    elif args.analyze_only:
        return pipeline.run_analyze()
    elif args.metrics_only:
        return pipeline.run_metrics()
    elif args.aggregate_only:
        return pipeline.run_aggregate()
    elif args.stat_only:
        return pipeline.run_stat()
    elif args.structural_only:
        return pipeline.run_structural()
    else:
        log_error("No stage selected")
        return 1


if __name__ == "__main__":
    sys.exit(main())
