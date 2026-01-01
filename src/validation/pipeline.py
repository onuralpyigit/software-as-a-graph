"""
Validation Pipeline - Version 5.0

Complete validation pipeline that:
1. Runs graph analysis to get predicted criticality scores
2. Runs failure simulation to get actual impact scores
3. Validates predictions against actuals
4. Supports component-type specific validation
5. Compares multiple analysis methods

Pipeline Flow:
    Graph Model → Analysis (Predicted) → Simulation (Actual) → Validation

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

from .metrics import (
    ValidationStatus,
    ValidationTargets,
    CorrelationMetrics,
    ClassificationMetrics,
    RankingMetrics,
    mean,
    std_dev,
)
from .validator import (
    Validator,
    ValidationResult,
    TypeValidationResult,
    ComponentValidation,
)


# =============================================================================
# Analysis Methods
# =============================================================================

class AnalysisMethod(Enum):
    """Available analysis methods for criticality prediction"""
    BETWEENNESS = "betweenness"
    PAGERANK = "pagerank"
    DEGREE = "degree"
    COMPOSITE = "composite"
    CLOSENESS = "closeness"
    EIGENVECTOR = "eigenvector"


# =============================================================================
# Graph Analyzer (Prediction)
# =============================================================================

class GraphAnalyzer:
    """
    Analyzes graph to compute predicted criticality scores.
    
    Uses NetworkX for graph algorithms when Neo4j is not available.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self._logger = logging.getLogger(__name__)
    
    def analyze(
        self,
        graph: Any,  # SimulationGraph
        method: AnalysisMethod = AnalysisMethod.COMPOSITE,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Compute criticality scores using specified method.
        
        Args:
            graph: SimulationGraph instance
            method: Analysis method to use
            weights: Weights for composite method
        
        Returns:
            Dict mapping component_id -> criticality score
        """
        try:
            import networkx as nx
        except ImportError:
            self._logger.error("NetworkX required for graph analysis")
            return {}
        
        # Build NetworkX graph
        G = self._build_nx_graph(graph)
        
        if method == AnalysisMethod.BETWEENNESS:
            return self._betweenness_centrality(G)
        elif method == AnalysisMethod.PAGERANK:
            return self._pagerank(G)
        elif method == AnalysisMethod.DEGREE:
            return self._degree_centrality(G)
        elif method == AnalysisMethod.CLOSENESS:
            return self._closeness_centrality(G)
        elif method == AnalysisMethod.EIGENVECTOR:
            return self._eigenvector_centrality(G)
        elif method == AnalysisMethod.COMPOSITE:
            return self._composite_score(G, weights)
        else:
            return self._composite_score(G, weights)
    
    def analyze_all_methods(
        self,
        graph: Any,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute scores using all analysis methods.
        
        Returns:
            Dict mapping method_name -> {component_id: score}
        """
        results = {}
        for method in AnalysisMethod:
            try:
                results[method.value] = self.analyze(graph, method)
            except Exception as e:
                self._logger.warning(f"Failed to compute {method.value}: {e}")
        return results
    
    def analyze_by_type(
        self,
        graph: Any,
        method: AnalysisMethod = AnalysisMethod.COMPOSITE,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute scores separately for each component type.
        
        Returns:
            Dict mapping component_type -> {component_id: score}
        """
        from src.simulation import ComponentType
        
        results = {}
        for comp_type in ComponentType:
            # Get subgraph for this type
            type_ids = graph.get_component_ids_by_type(comp_type)
            if len(type_ids) < 2:
                continue
            
            subgraph = graph.get_subgraph_by_type(comp_type)
            scores = self.analyze(subgraph, method)
            
            if scores:
                results[comp_type.value] = scores
        
        return results
    
    def _build_nx_graph(self, graph: Any) -> 'nx.DiGraph':
        """Build NetworkX graph from SimulationGraph"""
        import networkx as nx
        
        G = nx.DiGraph()
        
        for comp_id, comp in graph.components.items():
            G.add_node(comp_id, type=comp.type.value)
        
        for edge in graph.edges.values():
            G.add_edge(
                edge.source, edge.target,
                weight=edge.weight,
                edge_type=edge.edge_type.value
            )
        
        return G
    
    def _betweenness_centrality(self, G: 'nx.DiGraph') -> Dict[str, float]:
        """Calculate betweenness centrality"""
        import networkx as nx
        return nx.betweenness_centrality(G, weight='weight', normalized=True)
    
    def _pagerank(self, G: 'nx.DiGraph') -> Dict[str, float]:
        """Calculate PageRank"""
        import networkx as nx
        try:
            return nx.pagerank(G, weight='weight')
        except nx.PowerIterationFailedConvergence:
            return nx.pagerank(G, weight='weight', max_iter=1000)
    
    def _degree_centrality(self, G: 'nx.DiGraph') -> Dict[str, float]:
        """Calculate degree centrality (in + out)"""
        import networkx as nx
        in_deg = nx.in_degree_centrality(G)
        out_deg = nx.out_degree_centrality(G)
        return {n: (in_deg.get(n, 0) + out_deg.get(n, 0)) / 2 for n in G.nodes()}
    
    def _closeness_centrality(self, G: 'nx.DiGraph') -> Dict[str, float]:
        """Calculate closeness centrality"""
        import networkx as nx
        return nx.closeness_centrality(G)
    
    def _eigenvector_centrality(self, G: 'nx.DiGraph') -> Dict[str, float]:
        """Calculate eigenvector centrality"""
        import networkx as nx
        try:
            return nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
        except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
            # Fall back to degree for problematic graphs
            return self._degree_centrality(G)
    
    def _composite_score(
        self,
        G: 'nx.DiGraph',
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Calculate composite criticality score.
        
        Default weights:
        - Betweenness: 0.40 (bottleneck importance)
        - PageRank: 0.35 (dependency importance)
        - Degree: 0.25 (coupling level)
        """
        if weights is None:
            weights = {
                "betweenness": 0.40,
                "pagerank": 0.35,
                "degree": 0.25,
            }
        
        bc = self._betweenness_centrality(G)
        pr = self._pagerank(G)
        dc = self._degree_centrality(G)
        
        # Normalize each metric to [0, 1]
        bc = self._normalize(bc)
        pr = self._normalize(pr)
        dc = self._normalize(dc)
        
        # Compute weighted composite
        composite = {}
        for node in G.nodes():
            composite[node] = (
                weights.get("betweenness", 0.4) * bc.get(node, 0) +
                weights.get("pagerank", 0.35) * pr.get(node, 0) +
                weights.get("degree", 0.25) * dc.get(node, 0)
            )
        
        return composite
    
    def _normalize(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to [0, 1]"""
        if not scores:
            return scores
        
        min_val = min(scores.values())
        max_val = max(scores.values())
        
        if max_val == min_val:
            return {k: 0.5 for k in scores}
        
        return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}


# =============================================================================
# Pipeline Result
# =============================================================================

@dataclass
class MethodComparison:
    """Comparison of analysis methods"""
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
    """Complete pipeline execution result"""
    validation: ValidationResult
    predicted_scores: Dict[str, float]
    actual_scores: Dict[str, float]
    analysis_method: str
    by_component_type: Dict[str, ValidationResult]
    method_comparison: Optional[Dict[str, MethodComparison]] = None
    simulation_stats: Optional[Dict[str, Any]] = None
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
    
    def get_best_method(self) -> Optional[str]:
        """Get method with highest Spearman correlation"""
        if not self.method_comparison:
            return self.analysis_method
        
        best = max(self.method_comparison.items(), key=lambda x: x[1].spearman)
        return best[0]
    
    def summary(self) -> str:
        """Get human-readable summary"""
        lines = [
            f"Pipeline Result: {self.validation.status.value.upper()}",
            f"  Analysis Method: {self.analysis_method}",
            f"  Components: {len(self.predicted_scores)}",
            self.validation.summary(),
        ]
        
        if self.method_comparison:
            lines.append("\n  Method Comparison:")
            for method, comp in sorted(
                self.method_comparison.items(),
                key=lambda x: -x[1].spearman
            ):
                lines.append(f"    {method}: ρ={comp.spearman:.4f}, F1={comp.f1_score:.4f}")
        
        if self.by_component_type:
            lines.append("\n  By Component Type:")
            for comp_type, result in self.by_component_type.items():
                lines.append(f"    {comp_type}: ρ={result.spearman:.4f}, F1={result.f1_score:.4f}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "validation": self.validation.to_dict(),
            "analysis_method": self.analysis_method,
            "predicted_scores": {k: round(v, 6) for k, v in self.predicted_scores.items()},
            "actual_scores": {k: round(v, 6) for k, v in self.actual_scores.items()},
            "by_component_type": {k: v.to_dict() for k, v in self.by_component_type.items()},
            "method_comparison": {k: v.to_dict() for k, v in (self.method_comparison or {}).items()},
            "simulation_stats": self.simulation_stats,
            "best_method": self.get_best_method(),
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# Validation Pipeline
# =============================================================================

class ValidationPipeline:
    """
    Complete validation pipeline integrating analysis and simulation.
    
    Pipeline:
    1. Analyze graph to get predicted criticality scores
    2. Run failure simulation to get actual impact scores
    3. Validate predictions against actuals
    4. Optionally compare multiple analysis methods
    5. Validate by component type
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
            seed: Random seed for reproducibility
            cascade: Enable cascade in failure simulation
        """
        self.targets = targets or ValidationTargets()
        self.seed = seed
        self.cascade = cascade
        self.analyzer = GraphAnalyzer(seed=seed)
        self.validator = Validator(targets=targets, seed=seed)
        self._logger = logging.getLogger(__name__)
    
    def run(
        self,
        graph: Any,  # SimulationGraph
        analysis_method: AnalysisMethod = AnalysisMethod.COMPOSITE,
        compare_methods: bool = False,
        validate_by_type: bool = True,
    ) -> PipelineResult:
        """
        Run complete validation pipeline.
        
        Args:
            graph: SimulationGraph to validate
            analysis_method: Primary analysis method
            compare_methods: Compare all analysis methods
            validate_by_type: Validate each component type separately
        
        Returns:
            PipelineResult with complete validation
        """
        from src.simulation import FailureSimulator, ComponentType
        
        self._logger.info("Starting validation pipeline")
        
        # Step 1: Get predicted scores from analysis
        self._logger.info(f"Running analysis using {analysis_method.value}")
        predicted_scores = self.analyzer.analyze(graph, analysis_method)
        
        # Step 2: Get actual scores from simulation
        self._logger.info("Running failure simulation")
        simulator = FailureSimulator(
            seed=self.seed,
            cascade=self.cascade,
            critical_threshold=0.3,
        )
        campaign = simulator.simulate_all_failures(graph)
        actual_scores = campaign.component_impacts
        
        # Build component type mapping
        component_types = {
            comp_id: comp.type.value 
            for comp_id, comp in graph.components.items()
        }
        
        # Step 3: Validate
        self._logger.info("Validating predictions")
        validation = self.validator.validate(
            predicted_scores, actual_scores, component_types
        )
        
        # Step 4: Compare methods if requested
        method_comparison = None
        if compare_methods:
            method_comparison = self._compare_methods(
                graph, actual_scores, component_types
            )
        
        # Step 5: Validate by component type
        by_component_type = {}
        if validate_by_type:
            by_component_type = self._validate_by_type(
                graph, predicted_scores, actual_scores, component_types
            )
        
        self._logger.info(f"Validation complete: {validation.status.value}")
        
        return PipelineResult(
            validation=validation,
            predicted_scores=predicted_scores,
            actual_scores=actual_scores,
            analysis_method=analysis_method.value,
            by_component_type=by_component_type,
            method_comparison=method_comparison,
            simulation_stats={
                "total_components": len(campaign.results),
                "critical_count": len(campaign.critical_components),
            },
        )
    
    def _compare_methods(
        self,
        graph: Any,
        actual_scores: Dict[str, float],
        component_types: Dict[str, str],
    ) -> Dict[str, MethodComparison]:
        """Compare all analysis methods"""
        all_methods = self.analyzer.analyze_all_methods(graph)
        comparisons = {}
        
        for method_name, predicted in all_methods.items():
            result = self.validator.validate(
                predicted, actual_scores, component_types,
                compute_ci=False
            )
            comparisons[method_name] = MethodComparison(
                method=method_name,
                spearman=result.spearman,
                f1_score=result.f1_score,
                precision=result.classification.precision,
                recall=result.classification.recall,
                status=result.status,
            )
        
        return comparisons
    
    def _validate_by_type(
        self,
        graph: Any,
        predicted_scores: Dict[str, float],
        actual_scores: Dict[str, float],
        component_types: Dict[str, str],
    ) -> Dict[str, ValidationResult]:
        """Validate each component type separately"""
        from src.simulation import ComponentType
        
        results = {}
        
        for comp_type in ComponentType:
            # Filter to this type
            type_ids = [
                cid for cid, ct in component_types.items()
                if ct == comp_type.value
            ]
            
            if len(type_ids) < 3:
                continue
            
            type_predicted = {cid: predicted_scores[cid] for cid in type_ids 
                            if cid in predicted_scores}
            type_actual = {cid: actual_scores[cid] for cid in type_ids 
                         if cid in actual_scores}
            
            if len(type_predicted) < 3:
                continue
            
            result = self.validator.validate(
                type_predicted, type_actual,
                {cid: comp_type.value for cid in type_ids},
                compute_ci=False
            )
            results[comp_type.value] = result
        
        return results
    
    def run_with_custom_scores(
        self,
        predicted_scores: Dict[str, float],
        actual_scores: Dict[str, float],
        component_types: Optional[Dict[str, str]] = None,
    ) -> PipelineResult:
        """
        Run validation with pre-computed scores.
        
        Useful when scores come from Neo4j or external analysis.
        """
        validation = self.validator.validate(
            predicted_scores, actual_scores, component_types
        )
        
        by_component_type = {}
        if component_types:
            # Group by type
            types = set(component_types.values())
            for comp_type in types:
                type_ids = [cid for cid, ct in component_types.items() if ct == comp_type]
                if len(type_ids) < 3:
                    continue
                
                type_pred = {cid: predicted_scores[cid] for cid in type_ids if cid in predicted_scores}
                type_actual = {cid: actual_scores[cid] for cid in type_ids if cid in actual_scores}
                
                if len(type_pred) >= 3:
                    result = self.validator.validate(type_pred, type_actual, compute_ci=False)
                    by_component_type[comp_type] = result
        
        return PipelineResult(
            validation=validation,
            predicted_scores=predicted_scores,
            actual_scores=actual_scores,
            analysis_method="custom",
            by_component_type=by_component_type,
        )


# =============================================================================
# Factory Functions
# =============================================================================

def run_validation(
    graph: Any,
    method: str = "composite",
    compare_methods: bool = False,
    seed: Optional[int] = None,
) -> PipelineResult:
    """
    Quick validation function.
    
    Args:
        graph: SimulationGraph to validate
        method: Analysis method name
        compare_methods: Compare all methods
        seed: Random seed
    
    Returns:
        PipelineResult
    """
    method_enum = AnalysisMethod(method) if method in [m.value for m in AnalysisMethod] else AnalysisMethod.COMPOSITE
    
    pipeline = ValidationPipeline(seed=seed)
    return pipeline.run(graph, method_enum, compare_methods)


def quick_pipeline(
    graph: Any,
    method: str = "composite",
) -> Dict[str, float]:
    """
    Quick pipeline returning just key metrics.
    
    Returns:
        Dict with spearman, f1_score, passed
    """
    result = run_validation(graph, method)
    return {
        "spearman": result.spearman,
        "f1_score": result.f1_score,
        "passed": result.passed,
        "status": result.validation.status.value,
    }
