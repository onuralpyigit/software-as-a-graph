"""
JSON Metrics Exporter Adapter

Implements IMetricsExporter for JSON export.
"""

import json
from typing import Any

from src.application.ports.outbound.metrics_exporter import IMetricsExporter


class JsonMetricsExporter(IMetricsExporter):
    """
    JSON adapter implementing IMetricsExporter.
    """
    
    def export_csv(self, data: Any, output_path: str) -> str:
        """Export data to CSV format (not implemented for JSON exporter)."""
        raise NotImplementedError("Use CsvMetricsExporter for CSV export")
    
    def export_json(self, data: Any, output_path: str) -> str:
        """Export data to JSON format."""
        # Convert to dict if has to_dict method
        if hasattr(data, 'to_dict'):
            data = data.to_dict()
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return output_path
