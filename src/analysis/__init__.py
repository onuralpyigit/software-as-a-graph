"""
Software-as-a-Graph Analysis Module - Version 5.0 (Refactored)

Multi-layer graph analysis for distributed pub-sub systems using Neo4j GDS.

Key Features:
- Component-type-specific analysis (Application, Topic, Node, Broker)
- Centrality algorithms: PageRank, Betweenness, Degree via Neo4j GDS
- Box-plot statistical classification (adaptive thresholds)
- Problem detection: Reliability, Maintainability, Availability issues
- Anti-pattern detection with symptoms and recommendations
- Critical edge identification

Usage:
    from src.analysis import (
        GDSAnalyzer,
        ComponentTypeAnalyzer,
        BoxPlotClassifier,
        ProblemDetector,
        AntiPatternDetector,
    )

    # Analyze by component type
    with GDSAnalyzer(uri, user, password) as analyzer:
        # Analyze all applications
        app_results = analyzer.analyze_component_type("Application")
        
        # Analyze all topics
        topic_results = analyzer.analyze_component_type("Topic")
        
        # Full multi-layer analysis
        full_results = analyzer.analyze_all()
        
        # Detect problems
        problems = analyzer.detect_problems()
        
        # Detect anti-patterns
        antipatterns = analyzer.detect_antipatterns()

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

__version__ = "5.0.0"

# Core GDS Client and Analyzer
from .gds_client import (
    GDSClient,
    CentralityResult,
    CommunityResult,
    ProjectionInfo,
)

# Component-Type Analyzer
from .component_analyzer import (
    ComponentTypeAnalyzer,
    ComponentTypeResult,
    ComponentMetrics,
)

# Main GDS Analyzer (Facade)
from .gds_analyzer import (
    GDSAnalyzer,
    MultiLayerAnalysisResult,
    LayerAnalysisResult,
)

# Box-Plot Classification
from .classifier import (
    CriticalityLevel,
    BoxPlotStats,
    ClassifiedItem,
    ClassificationResult,
    BoxPlotClassifier,
)

# Problem Detection
from .problem_detector import (
    ProblemType,
    ProblemSeverity,
    QualityAttribute,
    Problem,
    Symptom,
    ProblemDetectionResult,
    ProblemDetector,
)

# Anti-Pattern Detection
from .antipatterns import (
    AntiPatternType,
    PatternSeverity,
    AntiPattern,
    AntiPatternResult,
    AntiPatternDetector,
)

# Critical Edge Analysis
from .edge_analyzer import (
    EdgeCriticality,
    EdgeAnalysisResult,
    EdgeAnalyzer,
)

__all__ = [
    # Version
    "__version__",
    # GDS Client
    "GDSClient",
    "CentralityResult",
    "CommunityResult",
    "ProjectionInfo",
    # Component Analyzer
    "ComponentTypeAnalyzer",
    "ComponentTypeResult",
    "ComponentMetrics",
    # Main Analyzer
    "GDSAnalyzer",
    "MultiLayerAnalysisResult",
    "LayerAnalysisResult",
    # Classifier
    "CriticalityLevel",
    "BoxPlotStats",
    "ClassifiedItem",
    "ClassificationResult",
    "BoxPlotClassifier",
    # Problem Detection
    "ProblemType",
    "ProblemSeverity",
    "QualityAttribute",
    "Problem",
    "Symptom",
    "ProblemDetectionResult",
    "ProblemDetector",
    # Anti-Patterns
    "AntiPatternType",
    "PatternSeverity",
    "AntiPattern",
    "AntiPatternResult",
    "AntiPatternDetector",
    # Edge Analysis
    "EdgeCriticality",
    "EdgeAnalysisResult",
    "EdgeAnalyzer",
]
