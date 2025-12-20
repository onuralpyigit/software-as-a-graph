"""
Analysis Module for Software-as-a-Graph

This module provides comprehensive analysis capabilities for distributed
publish-subscribe systems modeled as graphs.

Components:
- criticality_scorer: Composite criticality scoring (C_score formula)
- fuzzy_criticality_scorer: Fuzzy logic-based criticality assessment
- boxplot_classifier: Statistical box plot classification
- centrality_analyzer: Multiple centrality metrics (13 types)
- edge_criticality_analyzer: Edge-centric criticality analysis
- structural_analyzer: Structural graph properties
- qos_analyzer: QoS-aware analysis
- reachability_analyzer: Connectivity impact analysis
- path_analyzer: Comprehensive path analysis (dependency chains, message flows, redundancy)
"""

from .criticality_scorer import (
    CompositeCriticalityScorer,
    CompositeCriticalityScore,
    CriticalityLevel
)

from .fuzzy_criticality_scorer import (
    FuzzyCriticalityScorer,
    FuzzyNodeCriticalityScore,
    FuzzyEdgeCriticalityScore,
    FuzzyCriticalityLevel,
    DefuzzificationMethod,
    compare_with_composite_score
)

from .boxplot_classifier import (
    BoxPlotClassifier,
    BoxPlotCriticalityLevel,
    BoxPlotStatistics,
    BoxPlotClassificationResult,
    BoxPlotClassificationSummary,
    classify_criticality_with_boxplot,
    classify_edges_with_boxplot
)

from .edge_criticality_analyzer import (
    EdgeCriticalityAnalyzer,
    EdgeCriticalityScore,
    EdgeCriticalityLevel
)

from .qos_analyzer import (
    QoSAnalyzer,
    QoSAnalysisResult
)

from .path_analyzer import (
    PathAnalyzer,
    PathAnalysisResult,
    PathCriticalityLevel,
    PathInfo,
    MessageFlowPath,
    DependencyChain,
    PathRedundancyInfo,
    FailurePropagationPath
)

from .graph_analyzer import (
    GraphAnalyzer,
    DependsOnEdge,
    CriticalityScore,
    AnalysisResult,
    DependencyType,
    CriticalityLevel,
    analyze_pubsub_system,
    derive_dependencies
)

from .neo4j_loader import (
    NEO4J_AVAILABLE
)

from .relationship_analyzer import (
    # Enums
    RelationshipType,
    MotifType,
    ComponentRole,
    
    # Data Classes
    EdgeCriticalityResult,
    HITSRoleResult,
    MotifInstance,
    DependencyChainResult,
    LayerCorrelationResult,
    EnsembleCriticalityResult,
    RelationshipAnalysisResult,
    
    # Analyzers
    EdgeCriticalityAnalyzer,
    HITSRoleAnalyzer,
    MotifDetector,
    DependencyChainAnalyzer,
    LayerCorrelationAnalyzer,
    EnsembleCriticalityScorer,
    RelationshipAnalyzer,
    
    # Convenience Functions
    analyze_relationships,
    get_algorithm_recommendations,
)

__all__ = [
    # Criticality Scoring
    'CompositeCriticalityScorer',
    'CompositeCriticalityScore',
    'CriticalityLevel',

    # Fuzzy Logic
    'FuzzyCriticalityScorer',
    'FuzzyNodeCriticalityScore',
    'FuzzyEdgeCriticalityScore',
    'FuzzyCriticalityLevel',
    'DefuzzificationMethod',
    'compare_with_composite_score',

    # Box Plot Classification
    'BoxPlotClassifier',
    'BoxPlotCriticalityLevel',
    'BoxPlotStatistics',
    'BoxPlotClassificationResult',
    'BoxPlotClassificationSummary',
    'classify_criticality_with_boxplot',
    'classify_edges_with_boxplot',

    # Edge Criticality
    'EdgeCriticalityAnalyzer',
    'EdgeCriticalityScore',
    'EdgeCriticalityLevel',

    # QoS Analysis
    'QoSAnalyzer',
    'QoSAnalysisResult',

    # Path Analysis
    'PathAnalyzer',
    'PathAnalysisResult',
    'PathCriticalityLevel',
    'PathInfo',
    'MessageFlowPath',
    'DependencyChain',
    'PathRedundancyInfo',
    'FailurePropagationPath',

    # Graph Analysis
    'GraphAnalyzer',
    'DependsOnEdge',
    'CriticalityScore',
    'AnalysisResult',
    'DependencyType',
    'CriticalityLevel',
    'analyze_pubsub_system',
    'derive_dependencies',

    # Neo4j Loader
    'NEO4J_AVAILABLE',

        # Relationship Analysis
    'RelationshipType',
    'MotifType',
    'ComponentRole',
    'EdgeCriticalityResult',
    'HITSRoleResult',
    'MotifInstance',
    'DependencyChainResult',
    'LayerCorrelationResult',
    'EnsembleCriticalityResult',
    'RelationshipAnalysisResult',
    'EdgeCriticalityAnalyzer',
    'HITSRoleAnalyzer',
    'MotifDetector',
    'DependencyChainAnalyzer',
    'LayerCorrelationAnalyzer',
    'EnsembleCriticalityScorer',
    'RelationshipAnalyzer',
    'analyze_relationships',
    'get_algorithm_recommendations'
]
