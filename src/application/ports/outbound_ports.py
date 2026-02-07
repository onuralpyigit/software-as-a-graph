"""
Outbound Ports

Interfaces defining contracts for outbound adapters.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from src.domain.models import GraphData


# =============================================================================
# Graph Repository
# =============================================================================

class IGraphRepository(ABC):
    """
    Outbound port for graph data persistence.
    
    Defines the contract for loading and saving graph data
    regardless of the underlying storage mechanism (Neo4j, in-memory, etc.).
    """
    
    @abstractmethod
    def get_graph_data(
        self,
        component_types: Optional[List[str]] = None,
        dependency_types: Optional[List[str]] = None,
        include_raw: bool = False
    ) -> GraphData:
        """
        Retrieve graph data with optional type filtering.
        
        Args:
            component_types: Types of components to include
            dependency_types: Types of dependencies to include
            include_raw: Include raw structural relationships
            
        Returns:
            GraphData containing components and edges
        """
        pass
    
    @abstractmethod
    def get_layer_data(self, layer: str) -> GraphData:
        """
        Retrieve graph data for a specific layer.
        
        Args:
            layer: Layer name (app, infra, mw, system)
            
        Returns:
            GraphData filtered for the layer
        """
        pass
    
    @abstractmethod
    def save_graph(self, data: Dict[str, Any], clear: bool = False) -> None:
        """
        Save graph data to the repository.
        
        Args:
            data: Graph data dictionary
            clear: Clear existing data before import
        """
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retrieve graph statistics.
        
        Returns:
            Dictionary with component and dependency counts
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close any open connections."""
        pass


# =============================================================================
# File Store (NEW)
# =============================================================================

class IFileStore(ABC):
    """
    Outbound port for file operations.
    
    Defines the contract for reading/writing files
    regardless of storage mechanism (local disk, S3, etc.).
    """
    
    @abstractmethod
    def read_json(self, path: str) -> Dict[str, Any]:
        """Read JSON file."""
        pass
    
    @abstractmethod
    def write_json(self, path: str, data: Dict[str, Any]) -> str:
        """Write JSON file. Returns the written path."""
        pass
    
    @abstractmethod
    def read_text(self, path: str) -> str:
        """Read text file."""
        pass
    
    @abstractmethod
    def write_text(self, path: str, content: str) -> str:
        """Write text file. Returns the written path."""
        pass
    
    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if path exists."""
        pass
    
    @abstractmethod
    def makedirs(self, path: str) -> None:
        """Create directory and parents."""
        pass


# =============================================================================
# Reporter (NEW)
# =============================================================================

class IReporter(ABC):
    """
    Outbound port for reporting/output.
    
    Defines the contract for displaying results to console,
    files, or other output channels.
    """
    
    @abstractmethod
    def info(self, message: str) -> None:
        """Display info message."""
        pass
    
    @abstractmethod
    def success(self, message: str) -> None:
        """Display success message."""
        pass
    
    @abstractmethod
    def warning(self, message: str) -> None:
        """Display warning message."""
        pass
    
    @abstractmethod
    def error(self, message: str) -> None:
        """Display error message."""
        pass
    
    @abstractmethod
    def table(self, headers: List[str], rows: List[List[Any]]) -> None:
        """Display tabular data."""
        pass
    
    @abstractmethod
    def section(self, title: str) -> None:
        """Display section header."""
        pass


# =============================================================================
# Metrics Exporter
# =============================================================================

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


# =============================================================================
# Report Generator
# =============================================================================

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
