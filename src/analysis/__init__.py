"""
Software-as-a-Graph Analysis Module - Version 4.0

GDS-based analysis for distributed pub-sub systems providing:
- Centrality analysis (PageRank, Betweenness, Degree)
- Quality attribute assessment (Reliability, Maintainability, Availability)
- Box-plot statistical classification
- Anti-pattern detection

Usage:
    from src.analysis import GDSClient, GDSClassifier
    from src.analysis import ReliabilityAnalyzer, MaintainabilityAnalyzer, AvailabilityAnalyzer
    from src.analysis import AntiPatternDetector

    # Connect to Neo4j
    with GDSClient(uri, user, password) as gds:
        # Create projection
        gds.create_projection("my_graph", ["app_to_app", "node_to_node"])
        
        # Run centrality analysis with box-plot classification
        classifier = GDSClassifier(gds, k_factor=1.5)
        result = classifier.classify_by_composite("my_graph")
        
        # Assess quality attributes
        reliability = ReliabilityAnalyzer(gds).analyze("my_graph")
        maintainability = MaintainabilityAnalyzer(gds).analyze("my_graph")
        availability = AvailabilityAnalyzer(gds).analyze("my_graph")
        
        # Detect anti-patterns
        patterns = AntiPatternDetector(gds).detect_all()
        
        # Cleanup
        gds.drop_projection("my_graph")

Author: Software-as-a-Graph Research Project
Version: 4.0
"""

# GDS Client
from .gds_client import (
    GDSClient,
    CentralityResult,
    CommunityResult,
    ProjectionInfo,
)

# Box-Plot Classification
from .classifier import (
    # Enums
    CriticalityLevel,
    # Data classes
    BoxPlotStats,
    ClassifiedItem,
    ClassificationResult,
    # Classifiers
    BoxPlotClassifier,
    GDSClassifier,
    # Utilities
    merge_classifications,
)

# Quality Attribute Analyzers
from .analyzers import (
    # Enums
    QualityAttribute,
    Severity,
    # Data classes
    Finding,
    CriticalComponent,
    AnalysisResult,
    # Analyzers
    BaseAnalyzer,
    ReliabilityAnalyzer,
    MaintainabilityAnalyzer,
    AvailabilityAnalyzer,
)

# Anti-Pattern Detection
from .antipatterns import (
    # Enums
    AntiPatternType,
    PatternSeverity,
    # Data classes
    AntiPattern,
    AntiPatternResult,
    # Detector
    AntiPatternDetector,
)

__all__ = [
    # GDS Client
    "GDSClient",
    "CentralityResult",
    "CommunityResult",
    "ProjectionInfo",
    # Classification
    "CriticalityLevel",
    "BoxPlotStats",
    "ClassifiedItem",
    "ClassificationResult",
    "BoxPlotClassifier",
    "GDSClassifier",
    "merge_classifications",
    # Quality Attributes
    "QualityAttribute",
    "Severity",
    "Finding",
    "CriticalComponent",
    "AnalysisResult",
    "BaseAnalyzer",
    "ReliabilityAnalyzer",
    "MaintainabilityAnalyzer",
    "AvailabilityAnalyzer",
    # Anti-Patterns
    "AntiPatternType",
    "PatternSeverity",
    "AntiPattern",
    "AntiPatternResult",
    "AntiPatternDetector",
]

__version__ = "4.0.0"