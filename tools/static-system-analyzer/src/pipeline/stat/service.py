"""
Statistical analysis service module.

This module provides the main service class for running statistical
analysis on publish-subscribe distributed systems.

The analysis pipeline:
1. Load relations.json from aggregator output
2. Analyze nodes, apps, libs, topics
3. Calculate descriptive statistics (mean, median, std, boxplot)
4. Generate markdown report with charts
"""

import os
import json
from dataclasses import dataclass
from pathlib import Path

from common.logger import setup_logger, log_info, log_error

from .analyzer import analyze_data, AnalysisReport
from .reporter import generate_report, log_report_summary

LOG_LEVEL = "INFO"
OUTPUT_FORMAT = "markdown"  # 'markdown' or 'pdf'


@dataclass
class StatConfig:
    """Configuration for the statistical analysis service.
    
    Attributes:
        input_file: Path to the relations JSON file from aggregator.
        output_dir: Directory where analysis output will be written.
        logs_dir: Directory where log files will be written.
        platform_name: Platform name for output file naming.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        output_format: Output format ('markdown' or 'pdf').
    """
    input_file: str
    output_dir: str
    logs_dir: str
    platform_name: str
    log_level: str = LOG_LEVEL
    output_format: str = OUTPUT_FORMAT
    dds_mask: bool = True


def analyze(
    input_file: str,
    output_dir: str,
    logs_dir: str,
    platform_name: str,
    log_level: str = LOG_LEVEL,
    output_format: str = OUTPUT_FORMAT,
    dds_mask: bool = True,
) -> AnalysisReport:
    """Execute the statistical analysis pipeline.
    
    Args:
        input_file: Path to relations JSON file.
        output_dir: Output directory for results.
        logs_dir: Directory for log files.
        platform_name: Platform name for file naming.
        log_level: Logging level.
        output_format: Output format ('markdown' or 'pdf').
    
    Returns:
        AnalysisReport containing all statistics.
    
    Raises:
        FileNotFoundError: If input file does not exist.
        json.JSONDecodeError: If input file is not valid JSON.
    """
    # Setup logging
    os.makedirs(logs_dir, exist_ok=True)
    setup_logger(logs_dir, platform_name, stage="stat", log_level=log_level)
    
    log_info("=" * 60)
    log_info("STAGE 4: STATISTICAL ANALYSIS")
    log_info("=" * 60)
    
    log_info(f"Input file: {input_file}")
    log_info(f"Output dir: {output_dir}")
    
    # Load JSON data
    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    log_info("Loading relations JSON...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Log scale info
    metadata = data.get("metadata", {})
    scale_str = metadata.get("scale", "{}")
    log_info(f"Data scale: {scale_str}")
    
    # Analyze data
    log_info("Analyzing data...")
    report = analyze_data(data, platform_name, dds_mask=dds_mask)
    
    # Export results
    log_info(f"Generating {output_format} report with charts...")
    generate_report(report, output_dir, output_format, dds_mask=dds_mask)
    
    # Log summary
    log_report_summary(report)
    
    log_info("Statistical analysis completed successfully.")
    
    return report


class StatService:
    """Service class for statistical analysis of publish-subscribe systems."""
    
    def __init__(self, config: StatConfig):
        """Initialize the statistical analysis service.
        
        Args:
            config: StatConfig instance with analysis parameters.
        """
        self.config = config
    
    def run(self) -> int:
        """Execute the statistical analysis pipeline.
        
        Returns:
            Exit code (0 for success, non-zero for failure).
        """
        try:
            analyze(
                input_file=self.config.input_file,
                output_dir=self.config.output_dir,
                logs_dir=self.config.logs_dir,
                platform_name=self.config.platform_name,
                log_level=self.config.log_level,
                output_format=self.config.output_format,
                dds_mask=self.config.dds_mask,
            )
            return 0
        except FileNotFoundError as e:
            log_error(f"Input file not found: {e}")
            return 1
        except Exception as e:
            log_error(f"Statistical analysis failed: {e}")
            return 1
