"""
HTML Dashboard Generator Adapter

Implements IReportGenerator for HTML dashboards.
"""

from typing import Any

from src.application.ports.outbound.report_generator import IReportGenerator

# Re-use legacy implementation during migration
from src.services.visualization.dashboard_generator import DashboardGenerator as LegacyDashboardGenerator


class HtmlDashboardGenerator(IReportGenerator):
    """
    HTML dashboard adapter implementing IReportGenerator.
    
    Wraps the legacy DashboardGenerator for incremental migration.
    """
    
    def __init__(self):
        """Initialize dashboard generator."""
        self._legacy = LegacyDashboardGenerator()
    
    def generate_dashboard(
        self,
        analysis_results: Any,
        output_path: str,
        **options
    ) -> str:
        """Generate a dashboard from analysis results."""
        return self._legacy.generate(analysis_results, output_path)
    
    def generate_report(
        self,
        data: Any,
        output_path: str,
        **options
    ) -> str:
        """Generate a report from data (delegates to dashboard)."""
        return self.generate_dashboard(data, output_path, **options)
