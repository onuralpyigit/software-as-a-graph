"""
Quality Attribute Analysis Module (GDS Edition)
===============================================

Provides GDS-based analysis capabilities for distributed pub-sub systems.

GDS-Based Analyzers (recommended):
- GDSClient: Neo4j Graph Data Science client
- ReliabilityAnalyzer: SPOFs, cascade risks, redundancy (via GDS)
- MaintainabilityAnalyzer: Coupling, cycles, modularity (via GDS)
- AvailabilityAnalyzer: Connectivity, fault tolerance (via GDS)

GDS algorithms used:
- Betweenness centrality
- PageRank
- Degree centrality
- Closeness centrality
- Louvain community detection
- Weakly connected components

Legacy NetworkX-based analyzers are still available for compatibility.

Usage (GDS - Recommended):
    from src.analysis import GDSClient
    from src.analysis.gds_analyzers import (
        ReliabilityAnalyzer,
        MaintainabilityAnalyzer,
        AvailabilityAnalyzer
    )
    
    # Connect to Neo4j with GDS
    gds = GDSClient(uri="bolt://localhost:7687", user="neo4j", password="password")
    
    # Create projection for DEPENDS_ON relationships
    projection = gds.create_depends_on_projection("my_graph")
    
    # Run analysis
    reliability = ReliabilityAnalyzer(gds)
    result = reliability.analyze("my_graph")
    
    # Cleanup
    gds.close()

Usage (Legacy NetworkX):
    from src.analysis import (
        ReliabilityAnalyzer as LegacyReliabilityAnalyzer,
        MaintainabilityAnalyzer as LegacyMaintainabilityAnalyzer,
        AvailabilityAnalyzer as LegacyAvailabilityAnalyzer
    )
"""

# GDS-based components (recommended)
from .gds_client import (
    GDSClient,
    GDSProjection,
    CentralityResult,
    CommunityResult,
    PathResult
)

from .gds_analyzers import (
    # Base
    BaseGDSAnalyzer,
    
    # Analyzers
    ReliabilityAnalyzer,
    MaintainabilityAnalyzer,
    AvailabilityAnalyzer,
    
    # Data classes
    Finding,
    CriticalComponent,
    AnalysisResult,
    
    # Enums
    Severity,
    QualityAttribute
)

# Try to import legacy components for backwards compatibility
try:
    from .quality_attribute_analyzer import (
        QualityAttribute as LegacyQualityAttribute,
        Severity as LegacySeverity,
        IssueCategory,
        ComponentType,
        DependencyType,
        QualityIssue,
        CriticalComponent as LegacyCriticalComponent,
        CriticalEdge,
        QualityAttributeResult,
        ComprehensiveAnalysisResult,
        BaseQualityAnalyzer,
        GraphAnalysisUtils,
        IssueFormatter
    )
    
    from .reliability_analyzer import (
        ReliabilityAnalyzer as LegacyReliabilityAnalyzer,
        DEFAULT_RELIABILITY_CONFIG
    )
    
    from .maintainability_analyzer import (
        MaintainabilityAnalyzer as LegacyMaintainabilityAnalyzer,
        DEFAULT_MAINTAINABILITY_CONFIG,
        CouplingMetrics
    )
    
    from .availability_analyzer import (
        AvailabilityAnalyzer as LegacyAvailabilityAnalyzer,
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
    
    _LEGACY_AVAILABLE = True
except ImportError:
    _LEGACY_AVAILABLE = False

__all__ = [
    # GDS Client
    'GDSClient',
    'GDSProjection',
    'CentralityResult',
    'CommunityResult',
    'PathResult',
    
    # GDS Analyzers
    'BaseGDSAnalyzer',
    'ReliabilityAnalyzer',
    'MaintainabilityAnalyzer',
    'AvailabilityAnalyzer',
    
    # GDS Data Classes
    'Finding',
    'CriticalComponent',
    'AnalysisResult',
    
    # GDS Enums
    'Severity',
    'QualityAttribute',
]

# Add legacy exports if available
if _LEGACY_AVAILABLE:
    __all__.extend([
        # Legacy Quality Attributes
        'LegacyQualityAttribute',
        'LegacySeverity',
        'IssueCategory',
        'ComponentType',
        'DependencyType',
        
        # Legacy Results
        'QualityIssue',
        'LegacyCriticalComponent',
        'CriticalEdge',
        'QualityAttributeResult',
        'ComprehensiveAnalysisResult',
        
        # Legacy Analyzers
        'LegacyReliabilityAnalyzer',
        'LegacyMaintainabilityAnalyzer',
        'LegacyAvailabilityAnalyzer',
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
    ])

__version__ = '2.0.0'