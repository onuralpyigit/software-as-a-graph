"""
Neo4j Validation Client - Version 5.0

Retrieves analysis results from Neo4j and runs validation
comparing graph analysis scores with simulation impact scores.

Features:
- Load analysis results from Neo4j GDS
- Run validation directly against Neo4j data
- Component-type specific validation
- Support for multiple analysis methods

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Set

try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False
    GraphDatabase = None

from .metrics import ValidationTargets
from .validator import Validator, ValidationResult
from .pipeline import (
    ValidationPipeline,
    PipelineResult,
    AnalysisMethod,
    MethodComparison,
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Neo4jValidationConfig:
    """Configuration for Neo4j validation"""
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"
    
    # Analysis configuration
    projection_name: str = "validation_projection"
    weighted: bool = True
    
    # Validation configuration
    critical_threshold: float = 0.3
    bootstrap_samples: int = 1000


# =============================================================================
# Neo4j Validation Client
# =============================================================================

class Neo4jValidationClient:
    """
    Client for running validation using Neo4j GDS analysis results.
    
    Retrieves centrality scores computed by Neo4j GDS algorithms
    and validates them against failure simulation results.
    """
    
    # Node labels
    NODE_LABELS = ["Application", "Broker", "Topic", "Node"]
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        targets: Optional[ValidationTargets] = None,
    ):
        if not HAS_NEO4J:
            raise ImportError(
                "neo4j package is required. Install with: pip install neo4j"
            )
        
        self.config = Neo4jValidationConfig(uri, user, password, database)
        self.targets = targets or ValidationTargets()
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._logger = logging.getLogger(__name__)
        
        self.validator = Validator(targets=self.targets)
    
    def __enter__(self) -> 'Neo4jValidationClient':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    def close(self) -> None:
        """Close database connection"""
        if self._driver:
            self._driver.close()
    
    # =========================================================================
    # Analysis Score Retrieval
    # =========================================================================
    
    def get_analysis_scores(
        self,
        score_property: str = "composite_score",
        component_type: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Get analysis scores from Neo4j.
        
        Args:
            score_property: Property name containing scores
            component_type: Filter by component type
        
        Returns:
            Dict mapping component_id -> score
        """
        if component_type:
            labels = [component_type]
        else:
            labels = self.NODE_LABELS
        
        scores = {}
        
        with self._driver.session(database=self.config.database) as session:
            for label in labels:
                query = f"""
                MATCH (n:{label})
                WHERE n.{score_property} IS NOT NULL
                RETURN COALESCE(n.id, n.name) AS id, n.{score_property} AS score
                """
                
                result = session.run(query)
                for record in result:
                    scores[record["id"]] = record["score"]
        
        return scores
    
    def get_centrality_scores(
        self,
        metric: str = "betweenness",
        component_type: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Get specific centrality metric scores.
        
        Args:
            metric: betweenness, pagerank, degree, closeness
            component_type: Filter by type
        
        Returns:
            Dict mapping component_id -> score
        """
        property_map = {
            "betweenness": "betweenness_centrality",
            "pagerank": "pagerank",
            "degree": "degree_centrality",
            "closeness": "closeness_centrality",
            "eigenvector": "eigenvector_centrality",
        }
        
        prop = property_map.get(metric, f"{metric}_centrality")
        return self.get_analysis_scores(prop, component_type)
    
    def get_all_centrality_scores(
        self,
        component_type: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Get all centrality metrics.
        
        Returns:
            Dict mapping metric_name -> {component_id: score}
        """
        metrics = ["betweenness", "pagerank", "degree", "closeness"]
        results = {}
        
        for metric in metrics:
            scores = self.get_centrality_scores(metric, component_type)
            if scores:
                results[metric] = scores
        
        return results
    
    def get_component_types(self) -> Dict[str, str]:
        """Get component type mapping"""
        types = {}
        
        with self._driver.session(database=self.config.database) as session:
            for label in self.NODE_LABELS:
                query = f"""
                MATCH (n:{label})
                RETURN COALESCE(n.id, n.name) AS id
                """
                result = session.run(query)
                for record in result:
                    types[record["id"]] = label
        
        return types
    
    # =========================================================================
    # Simulation Score Retrieval
    # =========================================================================
    
    def get_simulation_scores(
        self,
        score_property: str = "impact_score",
        component_type: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Get simulation impact scores from Neo4j.
        
        These should be stored after running failure simulation.
        """
        return self.get_analysis_scores(score_property, component_type)
    
    # =========================================================================
    # Run GDS Analysis
    # =========================================================================
    
    def run_gds_analysis(
        self,
        projection_name: str = "validation",
        write_property: str = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run GDS algorithms to compute centrality scores.
        
        Creates projection, runs algorithms, optionally writes results.
        
        Returns:
            Dict mapping metric -> {component_id: score}
        """
        results = {}
        
        with self._driver.session(database=self.config.database) as session:
            # Create projection
            try:
                session.run(f"CALL gds.graph.drop('{projection_name}', false)")
            except:
                pass
            
            # Project all component types and DEPENDS_ON relationships
            create_query = f"""
            CALL gds.graph.project(
                '{projection_name}',
                {self.NODE_LABELS},
                {{
                    DEPENDS_ON: {{
                        properties: ['weight']
                    }}
                }}
            )
            """
            session.run(create_query)
            
            # Run betweenness
            bc_query = f"""
            CALL gds.betweenness.stream('{projection_name}')
            YIELD nodeId, score
            RETURN gds.util.asNode(nodeId).id AS id, score
            """
            result = session.run(bc_query)
            results["betweenness"] = {r["id"]: r["score"] for r in result}
            
            # Run PageRank
            pr_query = f"""
            CALL gds.pageRank.stream('{projection_name}')
            YIELD nodeId, score
            RETURN gds.util.asNode(nodeId).id AS id, score
            """
            result = session.run(pr_query)
            results["pagerank"] = {r["id"]: r["score"] for r in result}
            
            # Run degree
            deg_query = f"""
            CALL gds.degree.stream('{projection_name}')
            YIELD nodeId, score
            RETURN gds.util.asNode(nodeId).id AS id, score
            """
            result = session.run(deg_query)
            results["degree"] = {r["id"]: r["score"] for r in result}
            
            # Cleanup
            session.run(f"CALL gds.graph.drop('{projection_name}')")
        
        return results
    
    # =========================================================================
    # Validation
    # =========================================================================
    
    def validate(
        self,
        predicted_property: str = "composite_score",
        actual_property: str = "impact_score",
        component_type: Optional[str] = None,
    ) -> ValidationResult:
        """
        Run validation using scores stored in Neo4j.
        
        Args:
            predicted_property: Property containing predicted scores
            actual_property: Property containing actual impact scores
            component_type: Filter by component type
        
        Returns:
            ValidationResult
        """
        predicted = self.get_analysis_scores(predicted_property, component_type)
        actual = self.get_analysis_scores(actual_property, component_type)
        component_types = self.get_component_types()
        
        return self.validator.validate(predicted, actual, component_types)
    
    def validate_all_metrics(
        self,
        actual_property: str = "impact_score",
    ) -> Dict[str, ValidationResult]:
        """
        Validate all centrality metrics against actual impact.
        
        Returns:
            Dict mapping metric_name -> ValidationResult
        """
        actual = self.get_analysis_scores(actual_property)
        component_types = self.get_component_types()
        all_scores = self.get_all_centrality_scores()
        
        results = {}
        for metric, predicted in all_scores.items():
            results[metric] = self.validator.validate(
                predicted, actual, component_types
            )
        
        return results
    
    def validate_by_component_type(
        self,
        predicted_property: str = "composite_score",
        actual_property: str = "impact_score",
    ) -> Dict[str, ValidationResult]:
        """
        Run validation for each component type separately.
        
        Returns:
            Dict mapping component_type -> ValidationResult
        """
        results = {}
        
        for comp_type in self.NODE_LABELS:
            predicted = self.get_analysis_scores(predicted_property, comp_type)
            actual = self.get_analysis_scores(actual_property, comp_type)
            
            if len(predicted) < 3 or len(actual) < 3:
                continue
            
            component_types = {cid: comp_type for cid in predicted}
            results[comp_type] = self.validator.validate(
                predicted, actual, component_types
            )
        
        return results
    
    # =========================================================================
    # Full Pipeline with Neo4j
    # =========================================================================
    
    def run_full_validation(
        self,
        run_simulation: bool = True,
        cascade: bool = True,
        seed: Optional[int] = None,
    ) -> PipelineResult:
        """
        Run complete validation pipeline using Neo4j data.
        
        1. Load graph from Neo4j
        2. Run GDS analysis (or use stored scores)
        3. Run failure simulation (or use stored scores)
        4. Validate and compare methods
        
        Args:
            run_simulation: Run new simulation or use stored scores
            cascade: Enable cascade in simulation
            seed: Random seed
        
        Returns:
            PipelineResult with complete validation
        """
        from src.simulation import Neo4jSimulationClient, FailureSimulator
        
        # Load graph from Neo4j
        with Neo4jSimulationClient(
            uri=self.config.uri,
            user=self.config.user,
            password=self.config.password,
            database=self.config.database
        ) as sim_client:
            graph = sim_client.load_full_graph()
        
        # Run pipeline
        pipeline = ValidationPipeline(
            targets=self.targets,
            seed=seed,
            cascade=cascade,
        )
        
        return pipeline.run(
            graph,
            analysis_method=AnalysisMethod.COMPOSITE,
            compare_methods=True,
            validate_by_type=True,
        )
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get statistics about data available for validation"""
        stats = {
            "components": {},
            "has_analysis_scores": {},
            "has_impact_scores": {},
        }
        
        with self._driver.session(database=self.config.database) as session:
            for label in self.NODE_LABELS:
                # Count components
                result = session.run(f"MATCH (n:{label}) RETURN count(n) AS count")
                count = result.single()["count"]
                stats["components"][label] = count
                
                # Check for analysis scores
                result = session.run(
                    f"MATCH (n:{label}) WHERE n.composite_score IS NOT NULL "
                    f"RETURN count(n) AS count"
                )
                stats["has_analysis_scores"][label] = result.single()["count"]
                
                # Check for impact scores
                result = session.run(
                    f"MATCH (n:{label}) WHERE n.impact_score IS NOT NULL "
                    f"RETURN count(n) AS count"
                )
                stats["has_impact_scores"][label] = result.single()["count"]
        
        return stats


# =============================================================================
# Factory Functions
# =============================================================================

def validate_from_neo4j(
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str = "password",
    database: str = "neo4j",
    predicted_property: str = "composite_score",
    actual_property: str = "impact_score",
) -> ValidationResult:
    """
    Quick validation using Neo4j stored scores.
    
    Args:
        uri: Neo4j bolt URI
        user: Username
        password: Password
        database: Database name
        predicted_property: Property with predicted scores
        actual_property: Property with actual scores
    
    Returns:
        ValidationResult
    """
    with Neo4jValidationClient(uri, user, password, database) as client:
        return client.validate(predicted_property, actual_property)


def run_neo4j_validation_pipeline(
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str = "password",
    database: str = "neo4j",
    seed: Optional[int] = None,
) -> PipelineResult:
    """
    Run complete validation pipeline with Neo4j.
    
    Returns:
        PipelineResult
    """
    with Neo4jValidationClient(uri, user, password, database) as client:
        return client.run_full_validation(seed=seed)
