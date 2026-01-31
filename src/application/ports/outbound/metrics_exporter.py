"""
Metrics Exporter Port

Interface defining the contract for exporting metrics data.
"""

from abc import ABC, abstractmethod
from typing import Any, List


class IMetricsExporter(ABC):
    """
    Outbound port for metrics export.
    
    Defines the contract for exporting analysis/simulation metrics
    to various formats (CSV, JSON, etc.).
    """
    
    @abstractmethod
    def export_csv(self, data: Any, output_path: str) -> str:
        """
        Export data to CSV format.
        
        Args:
            data: Data to export
            output_path: Path to output file
            
        Returns:
            Path to exported file
        """
        pass
    
    @abstractmethod
    def export_json(self, data: Any, output_path: str) -> str:
        """
        Export data to JSON format.
        
        Args:
            data: Data to export
            output_path: Path to output file
            
        Returns:
            Path to exported file
        """
        pass
