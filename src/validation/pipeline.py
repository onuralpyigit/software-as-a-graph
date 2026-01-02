"""
Validation Pipeline - Version 5.0

Complete pipeline for validating graph analysis against simulation.

Pipeline Flow:
1. Analyze graph → predicted criticality scores
2. Run simulation → actual impact scores  
3. Validate predictions against actuals
4. Report results by layer

Layers:
- application: app_to_app dependencies
- infrastructure: node_to_node dependencies
- app_broker: app_to_broker dependencies
- node_broker: node_to_broker dependencies

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional

from .metrics import (
    ValidationStatus,
    ValidationTargets,
    mean,
)
from .validator import (
    Validator,
    ValidationResult,
    LayerValidationResult,
    TypeValidationResult,
)


# =============================================================================
# Analysis Methods
# =============================================================================

class AnalysisMethod(Enum):
    """Analysis methods for criticality prediction."""
    BETWEENNESS = "betweenness"
    PAGERANK = "pagerank"
    DEGREE = "degree"
    COMPOSITE = "composite"


# =============================================================================
# Graph Analyzer
# =============================================================================

class GraphAnalyzer:
    """
    Analyzes graph to compute predicted criticality scores.
    
    Uses NetworkX algorithms for analysis.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.logger = logging.getLogger(__name__)
    
    def analyze(
        self,
        graph: Any,  # SimulationGraph
        method: AnalysisMethod = AnalysisMethod.COMPOSITE,
    ) -> Dict[str, float]:
        """
        Compute criticality scores.
        
        Args:
            graph: SimulationGraph to analyze
            method: Analysis method
        
        Returns:
            {component_id: criticality_score}
        """
        import networkx as nx
        
        # Build NetworkX graph from simulation graph
        G = nx.DiGraph()
        
        for comp_id, comp in graph.components.items():
            G.add_node(comp_id, type=comp.type.value)
        
        for edge in graph.edges:
            weight = edge.weight + edge.qos.criticality_score()
            G.add_edge(edge.source, edge.target, weight=weight)
        
        if len(G) == 0:
            return {}
        
        # Compute centrality based on method
        if method == AnalysisMethod.BETWEENNESS:
            scores = nx.betweenness_centrality(G, weight="weight")
        elif method == AnalysisMethod.PAGERANK:
            try:
                scores = nx.pagerank(G, weight="weight", max_iter=100)
            except nx.PowerIterationFailedConvergence:
                scores = nx.pagerank(G, weight="weight", max_iter=500, tol=1e-4)
        elif method == AnalysisMethod.DEGREE:
            scores = dict(G.degree())
            # Normalize
            max_deg = max(scores.values()) if scores else 1
            scores = {k: v / max_deg for k, v in scores.items()}
        elif method == AnalysisMethod.COMPOSITE:
            scores = self._compute_composite(G)
        else:
            scores = nx.betweenness_centrality(G)
        
        # Normalize to [0, 1]
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}
        
        return scores
    
    def _compute_composite(self, G) -> Dict[str, float]:
        """Compute composite criticality score."""
        import networkx as nx
        
        # Component weights
        weights = {
            "betweenness": 0.40,
            "pagerank": 0.35,
            "degree": 0.25,
        }
        
        # Calculate each metric
        try:
            bc = nx.betweenness_centrality(G, weight="weight")
        except:
            bc = {n: 0.0 for n in G.nodes()}
        
        try:
            pr = nx.pagerank(G, weight="weight", max_iter=100)
        except:
            pr = {n: 1.0 / len(G) for n in G.nodes()}
        
        deg = dict(G.degree())
        max_deg = max(deg.values()) if deg else 1
        deg = {k: v / max_deg for k, v in deg.items()}
        
        # Normalize each
        for scores in [bc, pr, deg]:
            max_val = max(scores.values()) if scores else 1
            if max_val > 0:
                for k in scores:
                    scores[k] /= max_val
        
        # Combine
        composite = {}
        for node in G.nodes():
            composite[node] = (
                weights["betweenness"] * bc.get(node, 0) +
                weights["pagerank"] * pr.get(node, 0) +
                weights["degree"] * deg.get(node, 0)
            )
        
        return composite
    
    def analyze_all_methods(self, graph: Any) -> Dict[str, Dict[str, float]]:
        """
        Analyze using all methods.
        
        Returns:
            {method_name: {component_id: score}}
        """
        results = {}
        for method in AnalysisMethod:
            results[method.value] = self.analyze(graph, method)
        return results


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MethodComparison:
    """Comparison of an analysis method."""
    method: str
    spearman: float
    f1_score: float
    precision: float
    recall: float
    status: ValidationStatus
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "spearman": round(self.spearman, 4),
            "f1_score": round(self.f1_score, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "status": self.status.value,
        }


@dataclass
class PipelineResult:
    """Complete pipeline result."""
    validation: ValidationResult
    predicted_scores: Dict[str, float]
    actual_scores: Dict[str, float]
    analysis_method: str
    method_comparison: Optional[Dict[str, MethodComparison]] = None
    simulation_stats: Optional[Dict[str, Any]] = None
    analysis_time_ms: float = 0.0
    simulation_time_ms: float = 0.0
    validation_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def passed(self) -> bool:
        return self.validation.passed
    
    @property
    def spearman(self) -> float:
        return self.validation.spearman
    
    @property
    def f1_score(self) -> float:
        return self.validation.f1_score
    
    @property
    def by_layer(self) -> Dict[str, LayerValidationResult]:
        return self.validation.by_layer
    
    @property
    def by_type(self) -> Dict[str, TypeValidationResult]:
        return self.validation.by_type
    
    def get_best_method(self) -> Optional[str]:
        """Get method with highest Spearman."""
        if not self.method_comparison:
            return self.analysis_method
        
        best = max(self.method_comparison.items(), key=lambda x: x[1].spearman)
        return best[0]
    
    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Pipeline Result: {self.validation.status.value.upper()}",
            f"  Method: {self.analysis_method}",
            f"  Components: {len(self.predicted_scores)}",
            "",
            self.validation.summary(),
        ]
        
        if self.method_comparison:
            lines.append("\n  Method Comparison:")
            for method, comp in sorted(
                self.method_comparison.items(),
                key=lambda x: -x[1].spearman
            ):
                lines.append(f"    {method}: ρ={comp.spearman:.4f}, F1={comp.f1_score:.4f}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "validation": self.validation.to_dict(),
            "analysis_method": self.analysis_method,
            "predicted_scores": {k: round(v, 6) for k, v in self.predicted_scores.items()},
            "actual_scores": {k: round(v, 6) for k, v in self.actual_scores.items()},
            "method_comparison": {k: v.to_dict() for k, v in (self.method_comparison or {}).items()},
            "simulation_stats": self.simulation_stats,
            "best_method": self.get_best_method(),
            "timing": {
                "analysis_ms": round(self.analysis_time_ms, 2),
                "simulation_ms": round(self.simulation_time_ms, 2),
                "validation_ms": round(self.validation_time_ms, 2),
            },
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# Layer Mapping
# =============================================================================

def get_component_layer(comp_type: str) -> str:
    """Map component type to primary layer."""
    type_to_layer = {
        "Application": "application",
        "Broker": "app_broker",
        "Topic": "application",
        "Node": "infrastructure",
    }
    return type_to_layer.get(comp_type, "unknown")


def build_component_info(graph: Any) -> Dict[str, Dict[str, str]]:
    """
    Build component info dict from simulation graph.
    
    Returns:
        {component_id: {"type": str, "layer": str}}
    """
    info = {}
    
    for comp_id, comp in graph.components.items():
        comp_type = comp.type.value
        layer = get_component_layer(comp_type)
        
        info[comp_id] = {
            "type": comp_type,
            "layer": layer,
        }
    
    return info


# =============================================================================
# Validation Pipeline
# =============================================================================

class ValidationPipeline:
    """
    Complete validation pipeline.
    
    Integrates:
    1. Graph analysis (predicted scores)
    2. Failure simulation (actual scores)
    3. Validation (comparison)
    4. Layer-specific reporting
    
    Example:
        from src.simulation import SimulationGraph
        from src.validation import ValidationPipeline
        
        graph = SimulationGraph.from_json("system.json")
        
        pipeline = ValidationPipeline(seed=42)
        result = pipeline.run(graph, compare_methods=True)
        
        print(f"Status: {result.validation.status.value}")
        print(f"Spearman: {result.spearman:.4f}")
        
        # Layer results
        for layer, layer_result in result.by_layer.items():
            print(f"{layer}: ρ={layer_result.spearman:.4f}")
    """
    
    def __init__(
        self,
        targets: Optional[ValidationTargets] = None,
        seed: Optional[int] = None,
        cascade: bool = True,
    ):
        """
        Initialize pipeline.
        
        Args:
            targets: Validation target thresholds
            seed: Random seed
            cascade: Enable cascade in simulation
        """
        self.targets = targets or ValidationTargets()
        self.seed = seed
        self.cascade = cascade
        self.analyzer = GraphAnalyzer(seed=seed)
        self.validator = Validator(targets=targets, seed=seed)
        self.logger = logging.getLogger(__name__)
    
    def run(
        self,
        graph: Any,  # SimulationGraph
        analysis_method: AnalysisMethod = AnalysisMethod.COMPOSITE,
        compare_methods: bool = False,
    ) -> PipelineResult:
        """
        Run validation pipeline.
        
        Args:
            graph: SimulationGraph to validate
            analysis_method: Primary analysis method
            compare_methods: Compare all analysis methods
        
        Returns:
            PipelineResult with validation results
        """
        from src.simulation import FailureSimulator
        
        self.logger.info("Starting validation pipeline")
        
        # Step 1: Analysis
        self.logger.info(f"Running analysis using {analysis_method.value}")
        start = time.time()
        predicted_scores = self.analyzer.analyze(graph, analysis_method)
        analysis_time = (time.time() - start) * 1000
        
        # Step 2: Simulation
        self.logger.info("Running failure simulation")
        start = time.time()
        simulator = FailureSimulator(
            seed=self.seed,
            cascade=self.cascade,
        )
        campaign = simulator.simulate_all(graph)
        simulation_time = (time.time() - start) * 1000
        
        # Extract actual scores
        actual_scores = {
            r.component_id: r.impact_score
            for r in campaign.results
        }
        
        # Build component info for layer/type grouping
        component_info = build_component_info(graph)
        
        # Step 3: Validation
        self.logger.info("Validating predictions")
        start = time.time()
        validation = self.validator.validate(
            predicted_scores, actual_scores, component_info
        )
        validation_time = (time.time() - start) * 1000
        
        # Step 4: Compare methods if requested
        method_comparison = None
        if compare_methods:
            method_comparison = self._compare_methods(
                graph, actual_scores, component_info
            )
        
        # Simulation stats
        simulation_stats = {
            "total_simulations": campaign.total_simulations,
            "duration_ms": campaign.duration_ms,
            "critical_count": len(campaign.get_critical()),
        }
        
        return PipelineResult(
            validation=validation,
            predicted_scores=predicted_scores,
            actual_scores=actual_scores,
            analysis_method=analysis_method.value,
            method_comparison=method_comparison,
            simulation_stats=simulation_stats,
            analysis_time_ms=analysis_time,
            simulation_time_ms=simulation_time,
            validation_time_ms=validation_time,
        )
    
    def _compare_methods(
        self,
        graph: Any,
        actual_scores: Dict[str, float],
        component_info: Dict[str, Dict[str, str]],
    ) -> Dict[str, MethodComparison]:
        """Compare all analysis methods."""
        comparisons = {}
        
        for method in AnalysisMethod:
            predicted = self.analyzer.analyze(graph, method)
            result = self.validator.validate(
                predicted, actual_scores, component_info,
                compute_ci=False,
            )
            
            comparisons[method.value] = MethodComparison(
                method=method.value,
                spearman=result.spearman,
                f1_score=result.f1_score,
                precision=result.classification.precision,
                recall=result.classification.recall,
                status=result.status,
            )
        
        return comparisons
    
    def validate_layer(
        self,
        graph: Any,
        layer: str,
        analysis_method: AnalysisMethod = AnalysisMethod.COMPOSITE,
    ) -> Optional[LayerValidationResult]:
        """
        Validate a specific layer.
        
        Args:
            graph: SimulationGraph
            layer: Layer name (application, infrastructure, app_broker, node_broker)
            analysis_method: Analysis method
        
        Returns:
            LayerValidationResult or None
        """
        result = self.run(graph, analysis_method, compare_methods=False)
        return result.by_layer.get(layer)


# =============================================================================
# Factory Functions
# =============================================================================

def run_validation(
    graph: Any,  # SimulationGraph
    method: AnalysisMethod = AnalysisMethod.COMPOSITE,
    compare_methods: bool = False,
    seed: Optional[int] = None,
) -> PipelineResult:
    """
    Quick function to run validation.
    
    Args:
        graph: SimulationGraph
        method: Analysis method
        compare_methods: Compare all methods
        seed: Random seed
    
    Returns:
        PipelineResult
    """
    pipeline = ValidationPipeline(seed=seed)
    return pipeline.run(graph, method, compare_methods)


def quick_pipeline(
    graph: Any,  # SimulationGraph
    seed: Optional[int] = None,
) -> ValidationStatus:
    """
    Quick validation status check.
    
    Returns:
        ValidationStatus
    """
    result = run_validation(graph, seed=seed)
    return result.validation.status
