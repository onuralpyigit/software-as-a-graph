"""
Report Generator Port

Interface defining the contract for generating reports/dashboards.
"""

from abc import ABC, abstractmethod
from typing import Any


class IReportGenerator(ABC):
    """
    Outbound port for report/dashboard generation.
    
    Defines the contract for generating visual reports
    regardless of output format (HTML, PDF, etc.).
    """
    
    @abstractmethod
    def generate_dashboard(
        self,
        analysis_results: Any,
        output_path: str,
        **options
    ) -> str:
        """
        Generate a dashboard from analysis results.
        
        Args:
            analysis_results: Analysis data to visualize
            output_path: Path to output file
            options: Additional generation options
            
        Returns:
            Path to generated dashboard
        """
        pass
    
    @abstractmethod
    def generate_report(
        self,
        data: Any,
        output_path: str,
        **options
    ) -> str:
        """
        Generate a report from data.
        
        Args:
            data: Data to include in report
            output_path: Path to output file
            options: Additional generation options
            
        Returns:
            Path to generated report
        """
        pass
