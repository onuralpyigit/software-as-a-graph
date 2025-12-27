"""
Validation Pipeline - Version 4.0

Integrates graph analysis, failure simulation, and validation into
a single pipeline for end-to-end methodology assessment.

Pipeline Steps:
1. Load graph from JSON
2. Compute predicted criticality scores using graph metrics
3. Run failure simulation to get actual impact scores
4. Validate predictions against simulation results

Works entirely on graph model without requiring Neo4j.

Author: Software-as-a-Graph Research Project
Version: 4.0
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict

from ..simulation import (
    SimulationGraph,
    FailureSimulator,
    ComponentType,
)

from .validator import Validator, ValidationResult
from .metrics import ValidationTargets


# =============================================================================
# Criticality Analyzers (Pure Python - No Neo4j)
# =============================================================================

class GraphAnalyzer:
    """
    Computes criticality scores using pure graph analysis.
    
    Metrics computed:
    - Degree centrality (connection count)
    - Betweenness centrality (path importance)
    - Message path centrality (pub-sub specific)
    - Composite score combining all metrics
    """

    def __init__(self, graph: SimulationGraph):
        self.graph = graph
        self.logger = logging.getLogger(__name__)

    def degree_centrality(self) -> Dict[str, float]:
        """
        Calculate degree centrality for all components.
        
        Degree = (in_degree + out_degree) / max_possible
        """
        scores = {}
        n = len(self.graph.components)
        max_degree = 2 * (n - 1) if n > 1 else 1
        
        for comp_id in self.graph.components:
            in_deg = len(self.graph.get_incoming(comp_id))
            out_deg = len(self.graph.get_outgoing(comp_id))
            scores[comp_id] = (in_deg + out_deg) / max_degree
        
        return scores

    def betweenness_centrality(self) -> Dict[str, float]:
        """
        Calculate betweenness centrality for all components.
        
        Uses Brandes' algorithm for efficiency.
        """
        scores = {c: 0.0 for c in self.graph.components}
        components = list(self.graph.components.keys())
        n = len(components)
        
        for source in components:
            # Single-source shortest paths (BFS)
            dist = {source: 0}
            paths = defaultdict(int)
            paths[source] = 1
            pred = defaultdict(list)
            queue = [source]
            stack = []
            
            while queue:
                current = queue.pop(0)
                stack.append(current)
                
                for neighbor in self.graph.get_neighbors(current):
                    # First visit
                    if neighbor not in dist:
                        dist[neighbor] = dist[current] + 1
                        queue.append(neighbor)
                    
                    # Shortest path to neighbor
                    if dist[neighbor] == dist[current] + 1:
                        paths[neighbor] += paths[current]
                        pred[neighbor].append(current)
            
            # Back propagation
            delta = defaultdict(float)
            while stack:
                w = stack.pop()
                for v in pred[w]:
                    delta[v] += (paths[v] / paths[w]) * (1 + delta[w])
                if w != source:
                    scores[w] += delta[w]
        
        # Normalize
        if n > 2:
            norm = 2.0 / ((n - 1) * (n - 2))
            scores = {k: v * norm for k, v in scores.items()}
        
        return scores

    def message_path_centrality(self) -> Dict[str, float]:
        """
        Calculate centrality based on message paths.
        
        Components appearing in more message paths are more critical.
        """
        paths = self.graph.get_all_message_paths()
        
        if not paths:
            return {c: 0.0 for c in self.graph.components}
        
        counts = defaultdict(int)
        for _, _, _, path in paths:
            for comp in path:
                counts[comp] += 1
        
        max_count = max(counts.values()) if counts else 1
        
        return {
            c: counts.get(c, 0) / max_count
            for c in self.graph.components
        }

    def composite_score(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Calculate composite criticality score.
        
        C_score = α·BC + β·DC + γ·MPC
        
        Default weights: betweenness=0.4, degree=0.3, message_path=0.3
        """
        if weights is None:
            weights = {
                "betweenness": 0.4,
                "degree": 0.3,
                "message_path": 0.3,
            }
        
        bc = self.betweenness_centrality()
        dc = self.degree_centrality()
        mpc = self.message_path_centrality()
        
        scores = {}
        for comp_id in self.graph.components:
            scores[comp_id] = (
                weights.get("betweenness", 0.4) * bc.get(comp_id, 0) +
                weights.get("degree", 0.3) * dc.get(comp_id, 0) +
                weights.get("message_path", 0.3) * mpc.get(comp_id, 0)
            )
        
        return scores

    def analyze_all(self) -> Dict[str, Dict[str, float]]:
        """Compute all centrality metrics"""
        return {
            "degree": self.degree_centrality(),
            "betweenness": self.betweenness_centrality(),
            "message_path": self.message_path_centrality(),
            "composite": self.composite_score(),
        }


# =============================================================================
# Pipeline Result
# =============================================================================

@dataclass
class PipelineResult:
    """Complete pipeline result"""
    timestamp: datetime
    
    # Graph info
    n_components: int
    n_connections: int
    n_paths: int
    
    # Analysis results
    predicted_scores: Dict[str, float]
    analysis_method: str
    analysis_time_ms: float
    
    # Simulation results
    actual_impacts: Dict[str, float]
    simulation_time_ms: float
    
    # Validation results
    validation: ValidationResult
    validation_time_ms: float
    
    # Total time
    total_time_ms: float

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "graph": {
                "components": self.n_components,
                "connections": self.n_connections,
                "paths": self.n_paths,
            },
            "analysis": {
                "method": self.analysis_method,
                "time_ms": round(self.analysis_time_ms, 2),
            },
            "simulation": {
                "time_ms": round(self.simulation_time_ms, 2),
            },
            "validation": self.validation.to_dict(),
            "timing": {
                "analysis_ms": round(self.analysis_time_ms, 2),
                "simulation_ms": round(self.simulation_time_ms, 2),
                "validation_ms": round(self.validation_time_ms, 2),
                "total_ms": round(self.total_time_ms, 2),
            },
        }


# =============================================================================
# Validation Pipeline
# =============================================================================

class ValidationPipeline:
    """
    End-to-end validation pipeline.
    
    1. Analyzes graph to compute predicted criticality scores
    2. Runs failure simulation to compute actual impact scores
    3. Validates predictions against simulation results
    """

    def __init__(
        self,
        targets: Optional[ValidationTargets] = None,
        cascade_threshold: float = 0.5,
        cascade_probability: float = 0.7,
        seed: Optional[int] = None,
    ):
        """
        Initialize the pipeline.
        
        Args:
            targets: Validation target metrics
            cascade_threshold: Cascade propagation threshold
            cascade_probability: Cascade probability
            seed: Random seed for reproducibility
        """
        self.targets = targets or ValidationTargets()
        self.cascade_threshold = cascade_threshold
        self.cascade_probability = cascade_probability
        self.seed = seed
        self.logger = logging.getLogger(__name__)

    def run(
        self,
        graph: SimulationGraph,
        analysis_method: str = "composite",
        component_types: Optional[List[ComponentType]] = None,
        enable_cascade: bool = True,
    ) -> PipelineResult:
        """
        Run the complete validation pipeline.
        
        Args:
            graph: SimulationGraph to analyze
            analysis_method: Method for computing predicted scores
                - "composite": Combined centrality score (default)
                - "betweenness": Betweenness centrality
                - "degree": Degree centrality
                - "message_path": Message path centrality
            component_types: Component types to include (None = all)
            enable_cascade: Enable cascade propagation in simulation
        
        Returns:
            PipelineResult with all analysis and validation results
        """
        pipeline_start = datetime.now()
        
        self.logger.info(f"Starting validation pipeline: {analysis_method}")
        
        # Graph stats
        n_components = len(graph.components)
        n_connections = len(graph.connections)
        n_paths = len(graph.get_all_message_paths())
        
        # Step 1: Analysis - compute predicted scores
        analysis_start = datetime.now()
        
        analyzer = GraphAnalyzer(graph)
        all_metrics = analyzer.analyze_all()
        
        if analysis_method == "composite":
            predicted = all_metrics["composite"]
        elif analysis_method == "betweenness":
            predicted = all_metrics["betweenness"]
        elif analysis_method == "degree":
            predicted = all_metrics["degree"]
        elif analysis_method == "message_path":
            predicted = all_metrics["message_path"]
        else:
            predicted = all_metrics["composite"]
        
        analysis_end = datetime.now()
        analysis_time = (analysis_end - analysis_start).total_seconds() * 1000
        
        self.logger.info(f"Analysis complete: {len(predicted)} components")
        
        # Step 2: Simulation - compute actual impacts
        simulation_start = datetime.now()
        
        simulator = FailureSimulator(
            cascade_threshold=self.cascade_threshold,
            cascade_probability=self.cascade_probability,
            seed=self.seed,
        )
        
        # Run exhaustive failure campaign
        batch = simulator.simulate_all_failures(
            graph,
            component_types=component_types,
            enable_cascade=enable_cascade,
        )
        
        # Extract impact scores
        actual = {
            result.primary_failures[0]: result.impact.impact_score
            for result in batch.results
        }
        
        simulation_end = datetime.now()
        simulation_time = (simulation_end - simulation_start).total_seconds() * 1000
        
        self.logger.info(f"Simulation complete: {len(batch.results)} failures")
        
        # Step 3: Validation - compare predicted vs actual
        validation_start = datetime.now()
        
        # Get component types
        comp_types = {
            c.id: c.type.value for c in graph.components.values()
        }
        
        validator = Validator(targets=self.targets, seed=self.seed)
        validation = validator.validate(predicted, actual, comp_types)
        
        validation_end = datetime.now()
        validation_time = (validation_end - validation_start).total_seconds() * 1000
        
        pipeline_end = datetime.now()
        total_time = (pipeline_end - pipeline_start).total_seconds() * 1000
        
        self.logger.info(f"Validation complete: {validation.status.value}")
        
        return PipelineResult(
            timestamp=pipeline_start,
            n_components=n_components,
            n_connections=n_connections,
            n_paths=n_paths,
            predicted_scores=predicted,
            analysis_method=analysis_method,
            analysis_time_ms=analysis_time,
            actual_impacts=actual,
            simulation_time_ms=simulation_time,
            validation=validation,
            validation_time_ms=validation_time,
            total_time_ms=total_time,
        )

    def compare_methods(
        self,
        graph: SimulationGraph,
        methods: Optional[List[str]] = None,
        enable_cascade: bool = True,
    ) -> Dict[str, PipelineResult]:
        """
        Compare multiple analysis methods.
        
        Args:
            graph: SimulationGraph to analyze
            methods: Methods to compare (default: all)
            enable_cascade: Enable cascade propagation
        
        Returns:
            Dictionary of method -> PipelineResult
        """
        if methods is None:
            methods = ["composite", "betweenness", "degree", "message_path"]
        
        results = {}
        for method in methods:
            self.logger.info(f"Running method: {method}")
            results[method] = self.run(
                graph, analysis_method=method, enable_cascade=enable_cascade
            )
        
        return results


# =============================================================================
# Convenience Functions
# =============================================================================

def run_validation(
    graph: SimulationGraph,
    method: str = "composite",
    targets: Optional[ValidationTargets] = None,
    seed: Optional[int] = None,
) -> PipelineResult:
    """
    Convenience function to run validation pipeline.
    
    Args:
        graph: SimulationGraph to analyze
        method: Analysis method
        targets: Validation targets
        seed: Random seed
    
    Returns:
        PipelineResult
    """
    pipeline = ValidationPipeline(targets=targets, seed=seed)
    return pipeline.run(graph, analysis_method=method)


def quick_pipeline(graph: SimulationGraph) -> Dict[str, Any]:
    """
    Quick pipeline returning key metrics.
    
    Args:
        graph: SimulationGraph to analyze
    
    Returns:
        Dictionary with key validation metrics
    """
    result = run_validation(graph)
    return {
        "status": result.validation.status.value,
        "spearman": result.validation.correlation.spearman,
        "f1": result.validation.classification.f1,
        "precision": result.validation.classification.precision,
        "recall": result.validation.classification.recall,
        "top_5": result.validation.ranking.top_k_overlap.get(5, 0),
        "n_components": result.n_components,
        "total_time_ms": result.total_time_ms,
    }
