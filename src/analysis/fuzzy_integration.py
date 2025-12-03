#!/usr/bin/env python3
"""
Fuzzy Criticality Integration Module

This module provides integration utilities for replacing the traditional
composite criticality scoring with fuzzy logic-based scoring.

Features:
- Drop-in replacement adapters for existing code
- Migration utilities for transitioning from composite to fuzzy
- Validation tools for comparing approaches
- Configuration presets for different analysis scenarios

Author: Software-as-a-Graph Research Project
Version: 1.0
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from fuzzy_criticality_scorer import (
    FuzzyCriticalityScorer,
    FuzzyNodeCriticalityScore,
    FuzzyEdgeCriticalityScore,
    FuzzyCriticalityLevel,
    DefuzzificationMethod,
    compare_with_composite_score
)


# ============================================================================
# Configuration Presets
# ============================================================================

class AnalysisPreset(Enum):
    """Preset configurations for different analysis scenarios"""
    CONSERVATIVE = "conservative"    # Emphasizes certainty, lower false positives
    BALANCED = "balanced"            # Default balanced approach
    AGGRESSIVE = "aggressive"        # Catches more potential issues
    REAL_TIME = "real_time"          # Faster, approximate analysis
    RESEARCH = "research"            # Full analysis with all metrics


@dataclass
class FuzzyConfiguration:
    """Configuration for fuzzy criticality analysis"""
    defuzz_method: DefuzzificationMethod = DefuzzificationMethod.CENTROID
    calculate_impact: bool = True
    critical_threshold: float = 0.7
    high_threshold: float = 0.5
    include_membership_degrees: bool = True
    
    @classmethod
    def from_preset(cls, preset: AnalysisPreset) -> 'FuzzyConfiguration':
        """Create configuration from preset"""
        presets = {
            AnalysisPreset.CONSERVATIVE: cls(
                defuzz_method=DefuzzificationMethod.MOM,
                calculate_impact=True,
                critical_threshold=0.8,
                high_threshold=0.6,
                include_membership_degrees=True
            ),
            AnalysisPreset.BALANCED: cls(
                defuzz_method=DefuzzificationMethod.CENTROID,
                calculate_impact=True,
                critical_threshold=0.7,
                high_threshold=0.5,
                include_membership_degrees=True
            ),
            AnalysisPreset.AGGRESSIVE: cls(
                defuzz_method=DefuzzificationMethod.CENTROID,
                calculate_impact=True,
                critical_threshold=0.6,
                high_threshold=0.4,
                include_membership_degrees=True
            ),
            AnalysisPreset.REAL_TIME: cls(
                defuzz_method=DefuzzificationMethod.WEIGHTED_AVERAGE,
                calculate_impact=False,  # Skip expensive impact calculation
                critical_threshold=0.7,
                high_threshold=0.5,
                include_membership_degrees=False
            ),
            AnalysisPreset.RESEARCH: cls(
                defuzz_method=DefuzzificationMethod.CENTROID,
                calculate_impact=True,
                critical_threshold=0.7,
                high_threshold=0.5,
                include_membership_degrees=True
            )
        }
        return presets.get(preset, presets[AnalysisPreset.BALANCED])


# ============================================================================
# Drop-in Replacement Adapter
# ============================================================================

class FuzzyCriticalityScorerAdapter:
    """
    Adapter that provides the same interface as the original CriticalityScorer
    but uses fuzzy logic internally.
    
    This allows existing code to use fuzzy scoring without modification:
    
    Before:
        scorer = CriticalityScorer(alpha=0.4, beta=0.3, gamma=0.3)
        scores = scorer.calculate_all_scores(graph)
        
    After:
        scorer = FuzzyCriticalityScorerAdapter()
        scores = scorer.calculate_all_scores(graph)
    
    The returned scores are compatible with the original CompositeCriticalityScore
    interface while providing additional fuzzy membership information.
    """
    
    def __init__(self, 
                 preset: AnalysisPreset = AnalysisPreset.BALANCED,
                 config: Optional[FuzzyConfiguration] = None):
        """
        Initialize the adapter.
        
        Args:
            preset: Analysis preset to use
            config: Custom configuration (overrides preset if provided)
        """
        self.config = config or FuzzyConfiguration.from_preset(preset)
        self.scorer = FuzzyCriticalityScorer(
            defuzz_method=self.config.defuzz_method,
            calculate_impact=self.config.calculate_impact
        )
        self.logger = logging.getLogger(__name__)
        
        # Store weights for compatibility (not actually used in fuzzy)
        self.weights = {'alpha': 0.4, 'beta': 0.3, 'gamma': 0.3}
    
    def calculate_all_scores(self, 
                            graph,
                            qos_scores: Optional[Dict[str, float]] = None
                            ) -> Dict[str, FuzzyNodeCriticalityScore]:
        """
        Calculate criticality scores for all nodes.
        
        This method mirrors the original CriticalityScorer.calculate_all_scores()
        interface for backward compatibility.
        
        Args:
            graph: NetworkX directed graph
            qos_scores: Optional QoS scores (used for flow importance in edges)
            
        Returns:
            Dictionary mapping node IDs to FuzzyNodeCriticalityScore objects
        """
        self.logger.info("Calculating fuzzy criticality scores (adapter mode)...")
        node_scores, _ = self.scorer.analyze_graph(graph)
        return node_scores
    
    def get_top_critical(self, 
                        scores: Dict[str, FuzzyNodeCriticalityScore],
                        n: int = 10) -> List[FuzzyNodeCriticalityScore]:
        """Get top N most critical components"""
        return self.scorer.get_top_critical_nodes(scores, n)
    
    def get_critical_components(self,
                               scores: Dict[str, FuzzyNodeCriticalityScore],
                               threshold: Optional[float] = None
                               ) -> List[FuzzyNodeCriticalityScore]:
        """Get all components above criticality threshold"""
        threshold = threshold or self.config.critical_threshold
        return self.scorer.get_critical_components(scores, threshold)
    
    def summarize_criticality(self, 
                             scores: Dict[str, FuzzyNodeCriticalityScore]) -> Dict[str, Any]:
        """Generate summary statistics"""
        return self.scorer.summarize_node_criticality(scores)


class FuzzyEdgeCriticalityAdapter:
    """
    Adapter that provides the same interface as EdgeCriticalityAnalyzer
    but uses fuzzy logic internally.
    
    Before:
        analyzer = EdgeCriticalityAnalyzer(alpha=0.5, beta=0.5)
        scores = analyzer.analyze(graph)
        
    After:
        analyzer = FuzzyEdgeCriticalityAdapter()
        scores = analyzer.analyze(graph)
    """
    
    def __init__(self,
                 preset: AnalysisPreset = AnalysisPreset.BALANCED,
                 config: Optional[FuzzyConfiguration] = None):
        """Initialize the adapter"""
        self.config = config or FuzzyConfiguration.from_preset(preset)
        self.scorer = FuzzyCriticalityScorer(
            defuzz_method=self.config.defuzz_method,
            calculate_impact=self.config.calculate_impact
        )
        self.logger = logging.getLogger(__name__)
        
        # Compatibility attributes
        self.alpha = 0.5
        self.beta = 0.5
    
    def analyze(self, graph) -> Dict[Tuple[str, str], FuzzyEdgeCriticalityScore]:
        """
        Perform comprehensive edge criticality analysis.
        
        Args:
            graph: NetworkX directed graph
            
        Returns:
            Dictionary mapping edge tuples to FuzzyEdgeCriticalityScore
        """
        self.logger.info("Analyzing edge criticality with fuzzy logic (adapter mode)...")
        _, edge_scores = self.scorer.analyze_graph(graph)
        return edge_scores
    
    def get_top_critical_edges(self,
                              scores: Dict[Tuple[str, str], FuzzyEdgeCriticalityScore],
                              n: int = 10,
                              min_score: float = 0.0) -> List[FuzzyEdgeCriticalityScore]:
        """Get top N most critical edges"""
        return self.scorer.get_top_critical_edges(scores, n, min_score)
    
    def get_bridges(self,
                   scores: Dict[Tuple[str, str], FuzzyEdgeCriticalityScore]
                   ) -> List[FuzzyEdgeCriticalityScore]:
        """Get all bridge edges"""
        return self.scorer.get_bridges(scores)
    
    def summarize_edge_criticality(self,
                                   scores: Dict[Tuple[str, str], FuzzyEdgeCriticalityScore]
                                   ) -> Dict[str, Any]:
        """Generate summary statistics"""
        return self.scorer.summarize_edge_criticality(scores)


# ============================================================================
# Unified Analysis Orchestrator Integration
# ============================================================================

def create_fuzzy_analysis_orchestrator_patch():
    """
    Returns a patch function to integrate fuzzy scoring into AnalysisOrchestrator.
    
    Usage:
        from analysis_orchestrator import AnalysisOrchestrator
        from fuzzy_integration import create_fuzzy_analysis_orchestrator_patch
        
        patch = create_fuzzy_analysis_orchestrator_patch()
        orchestrator = AnalysisOrchestrator(...)
        patch(orchestrator)  # Now uses fuzzy scoring
    """
    def patch_orchestrator(orchestrator):
        """Patch orchestrator to use fuzzy scoring"""
        orchestrator.criticality_scorer = FuzzyCriticalityScorerAdapter()
        orchestrator.edge_analyzer = FuzzyEdgeCriticalityAdapter()
        orchestrator._using_fuzzy = True
        return orchestrator
    
    return patch_orchestrator


# ============================================================================
# Validation and Comparison Tools
# ============================================================================

@dataclass
class ValidationResult:
    """Result of validating fuzzy scoring against composite scoring"""
    total_nodes: int
    nodes_with_same_level: int
    level_agreement_rate: float
    pearson_correlation: float
    spearman_correlation: float
    avg_score_difference: float
    max_score_difference: float
    nodes_with_higher_fuzzy: int
    nodes_with_lower_fuzzy: int
    level_changes: List[Dict[str, Any]]


def validate_fuzzy_against_composite(
    graph,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3,
    preset: AnalysisPreset = AnalysisPreset.BALANCED
) -> ValidationResult:
    """
    Validate fuzzy scoring by comparing with traditional composite scoring.
    
    This is useful for:
    1. Research validation - ensuring fuzzy approach maintains correlation
    2. Migration testing - verifying behavior before switching
    3. Understanding differences between approaches
    
    Args:
        graph: NetworkX graph to analyze
        alpha, beta, gamma: Weights for composite score
        preset: Fuzzy analysis preset
        
    Returns:
        ValidationResult with detailed comparison statistics
    """
    if not NETWORKX_AVAILABLE:
        raise RuntimeError("NetworkX required for validation")
    
    # Create fuzzy scorer
    config = FuzzyConfiguration.from_preset(preset)
    fuzzy_scorer = FuzzyCriticalityScorer(
        defuzz_method=config.defuzz_method,
        calculate_impact=config.calculate_impact
    )
    
    # Get comparison
    comparison = compare_with_composite_score(graph, fuzzy_scorer, alpha, beta, gamma)
    
    # Analyze level agreement
    level_changes = []
    same_level_count = 0
    higher_count = 0
    lower_count = 0
    
    for comp in comparison['comparisons']:
        fuzzy_level = comp['fuzzy_level']
        composite_score = comp['composite_score']
        
        # Determine composite level
        if composite_score >= 0.8:
            composite_level = 'critical'
        elif composite_score >= 0.6:
            composite_level = 'high'
        elif composite_score >= 0.4:
            composite_level = 'medium'
        elif composite_score >= 0.2:
            composite_level = 'low'
        else:
            composite_level = 'minimal'
        
        if fuzzy_level == composite_level:
            same_level_count += 1
        else:
            level_changes.append({
                'node': comp['node'],
                'fuzzy_level': fuzzy_level,
                'composite_level': composite_level,
                'fuzzy_score': comp['fuzzy_score'],
                'composite_score': composite_score
            })
        
        if comp['difference'] > 0:
            higher_count += 1
        elif comp['difference'] < 0:
            lower_count += 1
    
    total = len(comparison['comparisons'])
    
    return ValidationResult(
        total_nodes=total,
        nodes_with_same_level=same_level_count,
        level_agreement_rate=same_level_count / total if total > 0 else 0,
        pearson_correlation=comparison['pearson_correlation'] or 0,
        spearman_correlation=comparison['spearman_correlation'] or 0,
        avg_score_difference=comparison['avg_difference'],
        max_score_difference=comparison['max_difference'],
        nodes_with_higher_fuzzy=higher_count,
        nodes_with_lower_fuzzy=lower_count,
        level_changes=level_changes
    )


def generate_validation_report(validation: ValidationResult) -> str:
    """Generate human-readable validation report"""
    report = []
    report.append("=" * 70)
    report.append("FUZZY VS COMPOSITE SCORING VALIDATION REPORT")
    report.append("=" * 70)
    report.append("")
    report.append("SUMMARY")
    report.append("-" * 40)
    report.append(f"Total nodes analyzed: {validation.total_nodes}")
    report.append(f"Level agreement rate: {validation.level_agreement_rate:.1%}")
    report.append(f"Pearson correlation: {validation.pearson_correlation:.4f}")
    report.append(f"Spearman rank correlation: {validation.spearman_correlation:.4f}")
    report.append("")
    report.append("SCORE DIFFERENCES")
    report.append("-" * 40)
    report.append(f"Average difference (fuzzy - composite): {validation.avg_score_difference:.4f}")
    report.append(f"Maximum absolute difference: {validation.max_score_difference:.4f}")
    report.append(f"Nodes with higher fuzzy score: {validation.nodes_with_higher_fuzzy}")
    report.append(f"Nodes with lower fuzzy score: {validation.nodes_with_lower_fuzzy}")
    report.append(f"Nodes with same score: {validation.total_nodes - validation.nodes_with_higher_fuzzy - validation.nodes_with_lower_fuzzy}")
    report.append("")
    
    if validation.level_changes:
        report.append("LEVEL CHANGES")
        report.append("-" * 40)
        for change in validation.level_changes[:10]:  # Show top 10
            report.append(f"  {change['node']}: {change['composite_level']} → {change['fuzzy_level']}")
            report.append(f"    Composite: {change['composite_score']:.4f}, Fuzzy: {change['fuzzy_score']:.4f}")
        if len(validation.level_changes) > 10:
            report.append(f"  ... and {len(validation.level_changes) - 10} more")
    
    report.append("")
    report.append("RESEARCH TARGET ASSESSMENT")
    report.append("-" * 40)
    
    # Check against research targets
    spearman_target = 0.7
    if validation.spearman_correlation >= spearman_target:
        report.append(f"✓ Spearman correlation ≥ {spearman_target}: PASS ({validation.spearman_correlation:.4f})")
    else:
        report.append(f"✗ Spearman correlation ≥ {spearman_target}: FAIL ({validation.spearman_correlation:.4f})")
    
    agreement_target = 0.85
    if validation.level_agreement_rate >= agreement_target:
        report.append(f"✓ Level agreement ≥ {agreement_target:.0%}: PASS ({validation.level_agreement_rate:.1%})")
    else:
        report.append(f"✗ Level agreement ≥ {agreement_target:.0%}: FAIL ({validation.level_agreement_rate:.1%})")
    
    report.append("")
    report.append("=" * 70)
    
    return "\n".join(report)


# ============================================================================
# Example Usage and Testing
# ============================================================================

def main():
    """Demonstrate fuzzy integration capabilities"""
    print("=" * 70)
    print("Fuzzy Criticality Integration Module - Demo")
    print("=" * 70)
    
    if not NETWORKX_AVAILABLE:
        print("❌ NetworkX not available")
        return
    
    # Create test graph
    print("\n📊 Creating test graph...")
    G = nx.DiGraph()
    
    nodes = [
        ('central_broker', {'type': 'Broker'}),
        ('edge_broker_1', {'type': 'Broker'}),
        ('edge_broker_2', {'type': 'Broker'}),
        ('critical_topic', {'type': 'Topic', 'qos_score': 0.95}),
        ('normal_topic', {'type': 'Topic', 'qos_score': 0.3}),
        ('producer_1', {'type': 'Application'}),
        ('producer_2', {'type': 'Application'}),
        ('consumer_1', {'type': 'Application'}),
        ('consumer_2', {'type': 'Application'}),
        ('consumer_3', {'type': 'Application'}),
    ]
    G.add_nodes_from(nodes)
    
    edges = [
        ('producer_1', 'critical_topic', {'type': 'PUBLISHES'}),
        ('producer_2', 'critical_topic', {'type': 'PUBLISHES'}),
        ('critical_topic', 'consumer_1', {'type': 'SUBSCRIBES'}),
        ('critical_topic', 'consumer_2', {'type': 'SUBSCRIBES'}),
        ('critical_topic', 'consumer_3', {'type': 'SUBSCRIBES'}),
        ('producer_1', 'normal_topic', {'type': 'PUBLISHES'}),
        ('normal_topic', 'consumer_1', {'type': 'SUBSCRIBES'}),
        ('critical_topic', 'central_broker', {'type': 'HOSTED_ON'}),
        ('normal_topic', 'edge_broker_1', {'type': 'HOSTED_ON'}),
        ('central_broker', 'edge_broker_1', {'type': 'CONNECTS'}),
        ('central_broker', 'edge_broker_2', {'type': 'CONNECTS'}),
    ]
    G.add_edges_from(edges)
    
    print(f"   Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Test drop-in replacement for node scoring
    print("\n🔧 Testing Node Criticality Adapter...")
    node_adapter = FuzzyCriticalityScorerAdapter(preset=AnalysisPreset.BALANCED)
    node_scores = node_adapter.calculate_all_scores(G)
    
    print("\nTop 5 Critical Nodes (Fuzzy):")
    for i, score in enumerate(node_adapter.get_top_critical(node_scores, n=5), 1):
        print(f"   {i}. {score.component}: {score.fuzzy_score:.4f} ({score.criticality_level.value})")
    
    # Test drop-in replacement for edge scoring
    print("\n🔧 Testing Edge Criticality Adapter...")
    edge_adapter = FuzzyEdgeCriticalityAdapter(preset=AnalysisPreset.BALANCED)
    edge_scores = edge_adapter.analyze(G)
    
    print("\nTop 5 Critical Edges (Fuzzy):")
    for i, score in enumerate(edge_adapter.get_top_critical_edges(edge_scores, n=5), 1):
        print(f"   {i}. {score.source} → {score.target}: {score.fuzzy_score:.4f} ({score.criticality_level.value})")
    
    # Validate against composite scoring
    print("\n📋 Validating against composite scoring...")
    validation = validate_fuzzy_against_composite(G)
    report = generate_validation_report(validation)
    print(report)
    
    print("\n✓ Integration demo complete!")


if __name__ == "__main__":
    main()
