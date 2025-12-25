#!/usr/bin/env python3
"""
Integrated Validator - End-to-End Validation Pipeline
=======================================================

Connects analysis, simulation, and validation into a unified pipeline
that validates graph-based criticality predictions.

Pipeline:
1. Load graph from Neo4j
2. Run GDS-based analysis to get predicted criticality scores
3. Run failure simulation to get actual impact scores
4. Compare predictions vs actuals using statistical validation

Author: Software-as-a-Graph Research Project
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional

from .graph_validator import (
    GraphValidator,
    ValidationResult,
    ValidationTargets,
    ValidationStatus
)


@dataclass
class IntegratedValidationResult:
    """Result from integrated validation pipeline"""
    timestamp: datetime
    
    # Graph info
    graph_nodes: int
    graph_edges: int
    
    # Analysis info
    analysis_method: str
    components_analyzed: int
    
    # Simulation info
    simulation_type: str
    simulations_run: int
    
    # Validation result
    validation: ValidationResult
    
    # Timing
    analysis_time_ms: float
    simulation_time_ms: float
    validation_time_ms: float
    total_time_ms: float
    
    # Raw data for debugging
    predicted_scores: Dict[str, float] = field(default_factory=dict)
    actual_impacts: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'graph': {
                'nodes': self.graph_nodes,
                'edges': self.graph_edges
            },
            'analysis': {
                'method': self.analysis_method,
                'components': self.components_analyzed
            },
            'simulation': {
                'type': self.simulation_type,
                'count': self.simulations_run
            },
            'validation': self.validation.to_dict(),
            'timing': {
                'analysis_ms': round(self.analysis_time_ms, 2),
                'simulation_ms': round(self.simulation_time_ms, 2),
                'validation_ms': round(self.validation_time_ms, 2),
                'total_ms': round(self.total_time_ms, 2)
            }
        }
    
    def print_summary(self):
        """Print human-readable summary"""
        v = self.validation
        
        print(f"\n{'='*60}")
        print(f"VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Status: {v.status.value.upper()}")
        print(f"Components: {v.total_components}")
        print()
        print("Correlation Metrics:")
        print(f"  Spearman:  {v.correlation.spearman_coefficient:.4f} (p={v.correlation.spearman_p_value:.4f})")
        print(f"  Pearson:   {v.correlation.pearson_coefficient:.4f}")
        print(f"  Kendall τ: {v.correlation.kendall_tau:.4f}")
        print()
        print("Classification Metrics:")
        print(f"  Precision: {v.classification.precision:.4f}")
        print(f"  Recall:    {v.classification.recall:.4f}")
        print(f"  F1-Score:  {v.classification.f1_score:.4f}")
        print()
        print("Ranking Metrics:")
        for k, overlap in sorted(v.ranking.top_k_overlap.items()):
            print(f"  Top-{k} Overlap: {overlap:.1%}")
        print(f"  Mean Rank Diff: {v.ranking.mean_rank_difference:.1f}")
        print()
        print("Target Achievement:")
        for metric, (value, status) in v.achieved.items():
            target = getattr(v.targets, metric, 0)
            symbol = "✓" if status.value == "met" else "○" if status.value == "borderline" else "✗"
            print(f"  {symbol} {metric}: {value:.4f} (target: {target})")
        print(f"\nTiming: {self.total_time_ms:.0f}ms total")
        print(f"{'='*60}\n")


class IntegratedValidator:
    """
    End-to-end validation pipeline connecting analysis, simulation, and validation.
    
    Usage:
        from src.validation import IntegratedValidator
        
        validator = IntegratedValidator(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"
        )
        
        result = validator.run_validation()
        result.print_summary()
    """
    
    def __init__(self,
                 uri: str = "bolt://localhost:7687",
                 user: str = "neo4j",
                 password: str = "password",
                 database: str = "neo4j",
                 targets: Optional[ValidationTargets] = None):
        """
        Initialize the integrated validator.
        
        Args:
            uri: Neo4j bolt URI
            user: Database username
            password: Database password
            database: Database name
            targets: Custom validation targets
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.targets = targets or ValidationTargets()
        self.logger = logging.getLogger('IntegratedValidator')
        
        # Lazy imports to avoid circular dependencies
        self._gds_client = None
        self._graph_loader = None
    
    def run_validation(self,
                       analysis_method: str = 'composite',
                       enable_cascade: bool = True,
                       bootstrap: bool = False,
                       seed: Optional[int] = None) -> IntegratedValidationResult:
        """
        Run the complete validation pipeline.
        
        Args:
            analysis_method: 'composite', 'betweenness', 'pagerank', or 'degree'
            enable_cascade: Enable cascade in failure simulation
            bootstrap: Run bootstrap confidence intervals
            seed: Random seed for reproducibility
            
        Returns:
            IntegratedValidationResult with all metrics
        """
        from src.analysis import GDSClient
        from src.simulation import Neo4jGraphLoader, FailureSimulator
        
        start_time = datetime.now()
        
        self.logger.info("Starting integrated validation pipeline...")
        
        # Step 1: Load graph from Neo4j
        self.logger.info("Loading graph from Neo4j...")
        loader = Neo4jGraphLoader(
            uri=self.uri,
            user=self.user,
            password=self.password,
            database=self.database
        )
        sim_graph = loader.load_graph()
        loader.close()
        
        graph_nodes = len(sim_graph.components)
        graph_edges = len(sim_graph.dependencies)
        
        self.logger.info(f"Loaded graph: {graph_nodes} nodes, {graph_edges} edges")
        
        # Step 2: Run analysis to get predicted scores
        self.logger.info(f"Running {analysis_method} analysis...")
        analysis_start = datetime.now()
        
        gds = GDSClient(
            uri=self.uri,
            user=self.user,
            password=self.password,
            database=self.database
        )
        
        predicted_scores = self._run_analysis(gds, analysis_method)
        
        gds.close()
        
        analysis_time = (datetime.now() - analysis_start).total_seconds() * 1000
        
        self.logger.info(f"Analysis complete: {len(predicted_scores)} components scored")
        
        # Step 3: Run failure simulation to get actual impacts
        self.logger.info("Running failure simulation...")
        sim_start = datetime.now()
        
        simulator = FailureSimulator(
            cascade_threshold=0.5,
            cascade_probability=0.7,
            max_cascade_depth=5,
            seed=seed
        )
        
        actual_impacts = {}
        components_to_test = list(predicted_scores.keys())
        
        for comp_id in components_to_test:
            if comp_id in sim_graph.components:
                try:
                    result = simulator.simulate_single_failure(
                        sim_graph,
                        comp_id,
                        enable_cascade=enable_cascade
                    )
                    actual_impacts[comp_id] = result.impact.impact_score
                except Exception as e:
                    self.logger.warning(f"Simulation failed for {comp_id}: {e}")
        
        simulation_time = (datetime.now() - sim_start).total_seconds() * 1000
        
        self.logger.info(f"Simulation complete: {len(actual_impacts)} components tested")
        
        # Step 4: Run validation
        self.logger.info("Running validation...")
        validation_start = datetime.now()
        
        # Get component types
        component_types = {
            c: comp.type.value for c, comp in sim_graph.components.items()
        }
        
        validator = GraphValidator(targets=self.targets, seed=seed)
        
        if bootstrap:
            validation_result = validator.validate_with_bootstrap(
                predicted_scores,
                actual_impacts,
                n_iterations=1000
            )
        else:
            validation_result = validator.validate(
                predicted_scores,
                actual_impacts,
                component_types
            )
        
        validation_time = (datetime.now() - validation_start).total_seconds() * 1000
        total_time = (datetime.now() - start_time).total_seconds() * 1000
        
        self.logger.info(f"Validation complete: {validation_result.status.value}")
        
        return IntegratedValidationResult(
            timestamp=datetime.now(),
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
            analysis_method=analysis_method,
            components_analyzed=len(predicted_scores),
            simulation_type='single_failure_campaign',
            simulations_run=len(actual_impacts),
            validation=validation_result,
            analysis_time_ms=analysis_time,
            simulation_time_ms=simulation_time,
            validation_time_ms=validation_time,
            total_time_ms=total_time,
            predicted_scores=predicted_scores,
            actual_impacts=actual_impacts
        )
    
    def compare_methods(self,
                        methods: List[str] = None,
                        enable_cascade: bool = True,
                        seed: Optional[int] = None) -> Dict[str, IntegratedValidationResult]:
        """
        Compare different analysis methods.
        
        Args:
            methods: List of methods to compare
            enable_cascade: Enable cascade in simulation
            seed: Random seed
            
        Returns:
            Dictionary mapping method name to validation result
        """
        if methods is None:
            methods = ['composite', 'betweenness', 'pagerank', 'degree']
        
        results = {}
        
        for method in methods:
            self.logger.info(f"Validating method: {method}")
            results[method] = self.run_validation(
                analysis_method=method,
                enable_cascade=enable_cascade,
                seed=seed
            )
        
        return results
    
    def _run_analysis(self, gds, method: str) -> Dict[str, float]:
        """Run analysis using specified method"""
        from src.analysis.criticality_classifier import GDSCriticalityClassifier
        
        # Create projection
        projection_name = 'validation_projection'
        gds.create_depends_on_projection(
            projection_name=projection_name,
            include_weights=True
        )
        
        scores = {}
        
        try:
            if method == 'composite':
                # Use composite criticality from classifier
                classifier = GDSCriticalityClassifier(gds)
                result = classifier.classify_by_composite_score(projection_name, weighted=True)
                
                for item in result.items:
                    scores[item.item_id] = item.score
            
            elif method == 'betweenness':
                results = gds.betweenness_centrality(projection_name, weighted=True)
                max_score = max(r.score for r in results) if results else 1
                for r in results:
                    scores[r.node_id] = r.score / max_score if max_score > 0 else 0
            
            elif method == 'pagerank':
                results = gds.pagerank(projection_name, weighted=True)
                max_score = max(r.score for r in results) if results else 1
                for r in results:
                    scores[r.node_id] = r.score / max_score if max_score > 0 else 0
            
            elif method == 'degree':
                results = gds.degree_centrality(projection_name, weighted=True)
                max_score = max(r.score for r in results) if results else 1
                for r in results:
                    scores[r.node_id] = r.score / max_score if max_score > 0 else 0
            
            else:
                raise ValueError(f"Unknown analysis method: {method}")
        
        finally:
            gds.cleanup_projections()
        
        return scores


def run_quick_validation(uri: str = "bolt://localhost:7687",
                         user: str = "neo4j",
                         password: str = "password") -> Dict[str, Any]:
    """
    Run quick validation and return key metrics.
    
    Args:
        uri: Neo4j URI
        user: Username
        password: Password
        
    Returns:
        Dictionary with validation summary
    """
    validator = IntegratedValidator(uri=uri, user=user, password=password)
    result = validator.run_validation(analysis_method='composite')
    
    v = result.validation
    return {
        'status': v.status.value,
        'spearman': round(v.correlation.spearman_coefficient, 4),
        'f1_score': round(v.classification.f1_score, 4),
        'precision': round(v.classification.precision, 4),
        'recall': round(v.classification.recall, 4),
        'top_5_overlap': round(v.ranking.top_k_overlap.get(5, 0), 4),
        'components': v.total_components,
        'time_ms': round(result.total_time_ms, 0)
    }
