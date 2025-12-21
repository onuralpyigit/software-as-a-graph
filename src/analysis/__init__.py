"""
Quality Attribute Analysis Module
=================================

Provides comprehensive analysis capabilities for distributed pub-sub systems:

Quality Attribute Analyzers:
- ReliabilityAnalyzer: SPOFs, cascade risks, redundancy gaps
- MaintainabilityAnalyzer: Coupling metrics, anti-patterns, modularity
- AvailabilityAnalyzer: k-connectivity, fault tolerance, recovery paths

Supporting Components:
- AntiPatternDetector: Architectural anti-pattern detection
- BoxPlotCriticalityClassifier: Statistical box-plot classification
- GraphAnalysisUtils: Common graph analysis utilities
- Issue/Component/Edge data classes for results

Usage:
    from src.analysis import (
        ReliabilityAnalyzer,
        MaintainabilityAnalyzer, 
        AvailabilityAnalyzer,
        AntiPatternDetector,
        BoxPlotCriticalityClassifier
    )
    
    # Analyze a graph
    reliability = ReliabilityAnalyzer()
    result = reliability.analyze(graph)
    
    # Classify using box-plot method
    classifier = BoxPlotCriticalityClassifier()
    classification = classifier.classify_graph(graph)
"""

from .quality_attribute_analyzer import (
    # Enums
    QualityAttribute,
    Severity,
    IssueCategory,
    ComponentType,
    DependencyType,
    
    # Data Classes
    QualityIssue,
    CriticalComponent,
    CriticalEdge,
    QualityAttributeResult,
    ComprehensiveAnalysisResult,
    
    # Base Classes
    BaseQualityAnalyzer,
    GraphAnalysisUtils,
    
    # Utilities
    IssueFormatter
)

from .reliability_analyzer import (
    ReliabilityAnalyzer,
    DEFAULT_RELIABILITY_CONFIG
)

from .maintainability_analyzer import (
    MaintainabilityAnalyzer,
    DEFAULT_MAINTAINABILITY_CONFIG,
    CouplingMetrics
)

from .availability_analyzer import (
    AvailabilityAnalyzer,
    DEFAULT_AVAILABILITY_CONFIG,
    AvailabilityMetrics
)

from .antipattern_detector import (
    AntiPatternDetector,
    AntiPatternType,
    AntiPatternSeverity,
    AntiPattern,
    AntiPatternAnalysisResult,
    DEFAULT_ANTIPATTERN_CONFIG
)

from .criticality_classifier import (
    BoxPlotCriticalityClassifier,
    CriticalityLevel,
    BoxPlotStatistics,
    ClassifiedComponent,
    ClassifiedEdge,
    ClassificationResult,
    classify_quality_results
)

__all__ = [
    # Quality Attributes
    'QualityAttribute',
    'Severity',
    'IssueCategory',
    'ComponentType',
    'DependencyType',
    
    # Results
    'QualityIssue',
    'CriticalComponent',
    'CriticalEdge',
    'QualityAttributeResult',
    'ComprehensiveAnalysisResult',
    
    # Analyzers
    'ReliabilityAnalyzer',
    'MaintainabilityAnalyzer',
    'AvailabilityAnalyzer',
    'AntiPatternDetector',
    
    # Box-Plot Classification
    'BoxPlotCriticalityClassifier',
    'CriticalityLevel',
    'BoxPlotStatistics',
    'ClassifiedComponent',
    'ClassifiedEdge',
    'ClassificationResult',
    'classify_quality_results',
    
    # Base/Utilities
    'BaseQualityAnalyzer',
    'GraphAnalysisUtils',
    'IssueFormatter',
    
    # Anti-Pattern Types
    'AntiPatternType',
    'AntiPatternSeverity',
    'AntiPattern',
    'AntiPatternAnalysisResult',
    
    # Metrics
    'CouplingMetrics',
    'AvailabilityMetrics',
    
    # Configs
    'DEFAULT_RELIABILITY_CONFIG',
    'DEFAULT_MAINTAINABILITY_CONFIG',
    'DEFAULT_AVAILABILITY_CONFIG',
    'DEFAULT_ANTIPATTERN_CONFIG'
]

__version__ = '1.1.0'