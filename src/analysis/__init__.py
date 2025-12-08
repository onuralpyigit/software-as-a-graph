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
]
