"""
Neo4j Validation Client - Version 5.0

Loads graph data from Neo4j for validation purposes.

Features:
- Load full graph or specific layers
- Run validation pipeline directly from Neo4j data
- Get validation statistics from database
- Layer-specific validation

Usage:
    from src.validation import Neo4jValidationClient, validate_from_neo4j
    
    # Using context manager
    with Neo4jValidationClient(uri, user, password) as client:
        result = client.run_validation()
        print(f"Status: {result.validation.status.value}")
    
    # Using factory function
    result = validate_from_neo4j(uri, user, password)

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from contextlib import contextmanager

from .pipeline import (
    ValidationPipeline,
    PipelineResult,
    AnalysisMethod,
)
from .metrics import ValidationTargets, ValidationStatus
from .validator import ValidationResult


# Check for Neo4j driver
try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, ClientError
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False
    GraphDatabase = None
    ServiceUnavailable = Exception
    ClientError = Exception


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Neo4jConfig:
    """Neo4j connection configuration."""
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> Neo4jConfig:
        return cls(
            uri=data.get("uri", "bolt://localhost:7687"),
            user=data.get("user", "neo4j"),
            password=data.get("password", "password"),
            database=data.get("database", "neo4j"),
        )
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "uri": self.uri,
            "user": self.user,
            "database": self.database,
        }


# =============================================================================
# Neo4j Validation Client
# =============================================================================

class Neo4jValidationClient:
    """
    Client for running validation on graphs loaded from Neo4j.
    
    Integrates with the simulation module's Neo4j client to load
    graphs and then runs the validation pipeline.
    
    Example:
        with Neo4jValidationClient(uri, user, password) as client:
            # Run full validation
            result = client.run_validation(compare_methods=True)
            print(f"Spearman: {result.spearman:.4f}")
            
            # Validate specific layer
            layer_result = client.validate_layer("application")
            print(f"App Layer: {layer_result.spearman:.4f}")
            
            # Get statistics
            stats = client.get_statistics()
    """
    
    # Layer definitions (same as simulation)
    LAYER_NAMES = {
        "application": "Application Layer (app_to_app)",
        "infrastructure": "Infrastructure Layer (node_to_node)",
        "app_broker": "Application-Broker Layer (app_to_broker)",
        "node_broker": "Node-Broker Layer (node_to_broker)",
    }
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        targets: Optional[ValidationTargets] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize Neo4j validation client.
        
        Args:
            uri: Neo4j bolt URI
            user: Username
            password: Password
            database: Database name
            targets: Validation target thresholds
            seed: Random seed for reproducibility
        
        Raises:
            ImportError: If neo4j driver not installed
        """
        if not HAS_NEO4J:
            raise ImportError(
                "neo4j driver not installed. Install with: pip install neo4j"
            )
        
        self.config = Neo4jConfig(uri, user, password, database)
        self.targets = targets or ValidationTargets()
        self.seed = seed
        self._driver = None
        self._logger = logging.getLogger(__name__)
    
    def __enter__(self) -> Neo4jValidationClient:
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False
    
    def connect(self) -> None:
        """Establish connection to Neo4j."""
        self._driver = GraphDatabase.driver(
            self.config.uri,
            auth=(self.config.user, self.config.password),
        )
        self._driver.verify_connectivity()
        self._logger.info(f"Connected to Neo4j at {self.config.uri}")
    
    def close(self) -> None:
        """Close connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
            self._logger.info("Disconnected from Neo4j")
    
    @contextmanager
    def session(self):
        """Get a database session."""
        if not self._driver:
            self.connect()
        
        session = self._driver.session(database=self.config.database)
        try:
            yield session
        finally:
            session.close()
    
    def verify_connection(self) -> bool:
        """Verify database connection."""
        try:
            with self.session() as session:
                result = session.run("RETURN 1 AS test")
                return result.single()["test"] == 1
        except Exception as e:
            self._logger.error(f"Connection verification failed: {e}")
            return False
    
    def _load_graph(self, layer: Optional[str] = None):
        """Load graph from Neo4j using simulation module."""
        from src.simulation import Neo4jSimulationClient
        
        with Neo4jSimulationClient(
            uri=self.config.uri,
            user=self.config.user,
            password=self.config.password,
            database=self.config.database,
        ) as client:
            if layer:
                return client.load_layer(layer)
            return client.load_full_graph()
    
    # =========================================================================
    # Validation Methods
    # =========================================================================
    
    def run_validation(
        self,
        method: AnalysisMethod = AnalysisMethod.COMPOSITE,
        compare_methods: bool = False,
        layer: Optional[str] = None,
    ) -> PipelineResult:
        """
        Run validation pipeline on graph from Neo4j.
        
        Args:
            method: Analysis method for prediction
            compare_methods: Compare all analysis methods
            layer: Optional layer to validate
        
        Returns:
            PipelineResult with complete validation
        """
        # Load graph
        graph = self._load_graph(layer)
        
        self._logger.info(
            f"Loaded graph: {len(graph.components)} components, "
            f"{len(graph.edges)} edges"
        )
        
        # Create pipeline
        pipeline = ValidationPipeline(
            targets=self.targets,
            seed=self.seed,
            cascade=True,
        )
        
        # Run validation
        result = pipeline.run(
            graph,
            analysis_method=method,
            compare_methods=compare_methods,
        )
        
        return result
    
    def validate_layer(
        self,
        layer: str,
        method: AnalysisMethod = AnalysisMethod.COMPOSITE,
    ) -> PipelineResult:
        """
        Validate a specific layer.
        
        Args:
            layer: Layer name (application, infrastructure, etc.)
            method: Analysis method
        
        Returns:
            PipelineResult for the layer
        """
        if layer not in self.LAYER_NAMES:
            raise ValueError(f"Unknown layer: {layer}. "
                           f"Valid: {list(self.LAYER_NAMES.keys())}")
        
        return self.run_validation(method=method, layer=layer)
    
    def validate_all_layers(
        self,
        method: AnalysisMethod = AnalysisMethod.COMPOSITE,
    ) -> Dict[str, PipelineResult]:
        """
        Validate each layer separately.
        
        Args:
            method: Analysis method
        
        Returns:
            Dict mapping layer name to PipelineResult
        """
        results = {}
        
        for layer in self.LAYER_NAMES:
            try:
                results[layer] = self.validate_layer(layer, method)
            except Exception as e:
                self._logger.warning(f"Failed to validate layer {layer}: {e}")
        
        return results
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with component and edge counts
        """
        from src.simulation import Neo4jSimulationClient
        
        with Neo4jSimulationClient(
            uri=self.config.uri,
            user=self.config.user,
            password=self.config.password,
            database=self.config.database,
        ) as client:
            return client.get_statistics()
    
    def get_layer_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for each layer.
        
        Returns:
            Dict with per-layer statistics
        """
        from src.simulation import Neo4jSimulationClient
        
        with Neo4jSimulationClient(
            uri=self.config.uri,
            user=self.config.user,
            password=self.config.password,
            database=self.config.database,
        ) as client:
            return client.get_layer_statistics()


# =============================================================================
# Factory Functions
# =============================================================================

def validate_from_neo4j(
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str = "password",
    database: str = "neo4j",
    method: AnalysisMethod = AnalysisMethod.COMPOSITE,
    compare_methods: bool = False,
    layer: Optional[str] = None,
    targets: Optional[ValidationTargets] = None,
    seed: Optional[int] = None,
) -> PipelineResult:
    """
    Factory function to run validation on Neo4j graph.
    
    Args:
        uri: Neo4j bolt URI
        user: Username
        password: Password
        database: Database name
        method: Analysis method
        compare_methods: Compare all methods
        layer: Optional layer to validate
        targets: Validation targets
        seed: Random seed
    
    Returns:
        PipelineResult
    
    Example:
        result = validate_from_neo4j(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="mypassword",
            compare_methods=True
        )
        
        print(f"Status: {result.validation.status.value}")
        print(f"Spearman: {result.spearman:.4f}")
    """
    with Neo4jValidationClient(
        uri=uri,
        user=user,
        password=password,
        database=database,
        targets=targets,
        seed=seed,
    ) as client:
        return client.run_validation(
            method=method,
            compare_methods=compare_methods,
            layer=layer,
        )


def check_neo4j_available() -> bool:
    """Check if Neo4j driver is available."""
    return HAS_NEO4J
