"""
Edge Analyzer - Version 5.0

Analyzes DEPENDS_ON edges to identify critical connections.

Critical edges are identified based on:
- Edge weight (dependency strength from QoS)
- Bridge status (only path between components)
- Endpoint criticality (connects to critical components)

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Set

from .gds_client import GDSClient, DEPENDENCY_TYPES
from .classifier import BoxPlotClassifier, ClassificationResult, CriticalityLevel


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EdgeMetrics:
    """
    Metrics for a single DEPENDS_ON edge.
    """
    source_id: str
    target_id: str
    source_type: str
    target_type: str
    dependency_type: str
    
    # Weight and scores
    weight: float = 1.0
    criticality_score: float = 0.0
    
    # Structural flags
    is_bridge: bool = False
    connects_critical: bool = False
    
    # Classification
    level: CriticalityLevel = CriticalityLevel.MEDIUM
    
    @property
    def edge_key(self) -> str:
        return f"{self.source_id}->{self.target_id}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "source_type": self.source_type,
            "target_type": self.target_type,
            "dependency_type": self.dependency_type,
            "weight": round(self.weight, 4),
            "criticality_score": round(self.criticality_score, 4),
            "is_bridge": self.is_bridge,
            "connects_critical": self.connects_critical,
            "level": self.level.value,
        }


@dataclass
class EdgeAnalysisResult:
    """
    Complete edge analysis result.
    """
    timestamp: str
    edges: List[EdgeMetrics] = field(default_factory=list)
    classification: Optional[ClassificationResult] = None
    by_level: Dict[CriticalityLevel, List[EdgeMetrics]] = field(default_factory=dict)
    by_type: Dict[str, List[EdgeMetrics]] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.by_level:
            self.by_level = {level: [] for level in CriticalityLevel}
            for edge in self.edges:
                self.by_level[edge.level].append(edge)
        
        if not self.by_type:
            self.by_type = {t: [] for t in DEPENDENCY_TYPES}
            for edge in self.edges:
                if edge.dependency_type in self.by_type:
                    self.by_type[edge.dependency_type].append(edge)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "edge_count": len(self.edges),
            "edges": [e.to_dict() for e in self.edges],
            "classification": self.classification.to_dict() if self.classification else None,
            "by_level": {
                level.value: len(edges)
                for level, edges in self.by_level.items()
            },
            "by_type": {
                dep_type: len(edges)
                for dep_type, edges in self.by_type.items()
            },
            "summary": self.summary,
        }
    
    def get_critical(self) -> List[EdgeMetrics]:
        """Get critical edges."""
        return self.by_level.get(CriticalityLevel.CRITICAL, [])
    
    def get_bridges(self) -> List[EdgeMetrics]:
        """Get bridge edges."""
        return [e for e in self.edges if e.is_bridge]
    
    def get_by_dependency_type(self, dep_type: str) -> List[EdgeMetrics]:
        """Get edges by dependency type."""
        return self.by_type.get(dep_type, [])
    
    def top_n(self, n: int = 10) -> List[EdgeMetrics]:
        """Get top N edges by criticality score."""
        return sorted(self.edges, key=lambda e: e.criticality_score, reverse=True)[:n]


# =============================================================================
# Edge Analyzer
# =============================================================================

class EdgeAnalyzer:
    """
    Analyzes DEPENDS_ON edges to identify critical connections.
    
    Uses weighted combination of:
    - Edge weight (from QoS-based calculation)
    - Bridge status (structural importance)
    - Endpoint criticality (connection to critical nodes)
    
    Example:
        with GDSClient(uri, user, password) as gds:
            analyzer = EdgeAnalyzer(gds)
            
            # Analyze all edges
            result = analyzer.analyze()
            
            # Get critical edges
            for edge in result.get_critical():
                print(f"{edge.source_id} -> {edge.target_id}: {edge.criticality_score:.4f}")
            
            # Get bridges
            for edge in result.get_bridges():
                print(f"BRIDGE: {edge.source_id} -> {edge.target_id}")
    """
    
    # Weights for criticality score calculation
    DEFAULT_WEIGHTS = {
        "weight": 0.35,      # Edge weight contribution
        "bridge": 0.40,      # Bridge bonus
        "endpoint": 0.25,    # Critical endpoint bonus
    }
    
    def __init__(
        self,
        gds_client: GDSClient,
        k_factor: float = 1.5,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize analyzer.
        
        Args:
            gds_client: Connected GDS client
            k_factor: Box-plot k factor
            weights: Weights for criticality calculation
        """
        self.gds = gds_client
        self.classifier = BoxPlotClassifier(k_factor=k_factor)
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.logger = logging.getLogger(__name__)
    
    def analyze(
        self,
        dependency_types: Optional[List[str]] = None,
        critical_components: Optional[Set[str]] = None,
    ) -> EdgeAnalysisResult:
        """
        Analyze all DEPENDS_ON edges.
        
        Args:
            dependency_types: Filter to specific types (None = all)
            critical_components: Set of known critical component IDs
        
        Returns:
            EdgeAnalysisResult with all edge analysis
        """
        timestamp = datetime.now().isoformat()
        
        self.logger.info("Starting edge analysis")
        
        # Get all edges
        edges_data = self._get_edges(dependency_types)
        
        if not edges_data:
            return self._empty_result(timestamp)
        
        # Get bridges
        bridge_set = self._get_bridge_set()
        
        # Get critical components if not provided
        if critical_components is None:
            critical_components = self._identify_critical_components()
        
        # Normalize weights
        max_weight = max(e["weight"] for e in edges_data) or 1.0
        
        # Build edge metrics
        edges: List[EdgeMetrics] = []
        
        for edge_data in edges_data:
            source = edge_data["source_id"]
            target = edge_data["target_id"]
            weight = edge_data["weight"]
            
            # Check flags
            is_bridge = (source, target) in bridge_set
            connects_critical = (
                source in critical_components or 
                target in critical_components
            )
            
            # Calculate criticality score
            score = self._calculate_criticality(
                weight_norm=weight / max_weight,
                is_bridge=is_bridge,
                connects_critical=connects_critical,
            )
            
            edges.append(EdgeMetrics(
                source_id=source,
                target_id=target,
                source_type=edge_data["source_type"],
                target_type=edge_data["target_type"],
                dependency_type=edge_data["dependency_type"],
                weight=weight,
                criticality_score=score,
                is_bridge=is_bridge,
                connects_critical=connects_critical,
            ))
        
        # Classify edges
        items = [
            {"id": e.edge_key, "type": "edge", "score": e.criticality_score}
            for e in edges
        ]
        classification = self.classifier.classify(items, metric_name="edge_criticality")
        
        # Assign levels
        level_map = {item.id: item.level for item in classification.items}
        for edge in edges:
            edge.level = level_map.get(edge.edge_key, CriticalityLevel.MEDIUM)
        
        # Sort by score
        edges.sort(key=lambda e: e.criticality_score, reverse=True)
        
        # Generate summary
        summary = self._generate_summary(edges, classification)
        
        return EdgeAnalysisResult(
            timestamp=timestamp,
            edges=edges,
            classification=classification,
            summary=summary,
        )
    
    def analyze_by_layer(
        self,
        critical_components: Optional[Set[str]] = None,
    ) -> Dict[str, EdgeAnalysisResult]:
        """
        Analyze edges grouped by dependency type (layer).
        
        Args:
            critical_components: Known critical component IDs
        
        Returns:
            Dict mapping dependency type to EdgeAnalysisResult
        """
        results = {}
        
        for dep_type in DEPENDENCY_TYPES:
            results[dep_type] = self.analyze(
                dependency_types=[dep_type],
                critical_components=critical_components,
            )
        
        return results
    
    def _get_edges(
        self,
        dependency_types: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        """Get all DEPENDS_ON edges from database."""
        type_filter = ""
        if dependency_types:
            types_str = ", ".join(f"'{t}'" for t in dependency_types)
            type_filter = f"WHERE r.dependency_type IN [{types_str}]"
        
        query = f"""
        MATCH (a)-[r:DEPENDS_ON]->(b)
        {type_filter}
        RETURN a.id AS source_id,
               b.id AS target_id,
               labels(a)[0] AS source_type,
               labels(b)[0] AS target_type,
               r.dependency_type AS dependency_type,
               coalesce(r.weight, 1.0) AS weight
        """
        
        edges = []
        with self.gds.session() as session:
            for record in session.run(query):
                edges.append({
                    "source_id": record["source_id"],
                    "target_id": record["target_id"],
                    "source_type": record["source_type"],
                    "target_type": record["target_type"],
                    "dependency_type": record["dependency_type"],
                    "weight": record["weight"],
                })
        
        return edges
    
    def _get_bridge_set(self) -> Set[tuple]:
        """Get set of bridge edges as (source, target) tuples."""
        bridges = self.gds.find_bridges()
        return {(b["source_id"], b["target_id"]) for b in bridges}
    
    def _identify_critical_components(self) -> Set[str]:
        """Identify critical components based on betweenness centrality."""
        projection_name = "edge_analysis_critical"
        
        try:
            self.gds.create_projection(projection_name)
            bc_results = self.gds.betweenness(projection_name)
            
            if not bc_results:
                return set()
            
            # Classify and get high+ components
            items = [
                {"id": r.node_id, "type": r.node_type, "score": r.score}
                for r in bc_results
            ]
            classification = self.classifier.classify(items, metric_name="betweenness")
            
            return {item.id for item in classification.get_high_and_above()}
        
        finally:
            self.gds.drop_projection(projection_name)
    
    def _calculate_criticality(
        self,
        weight_norm: float,
        is_bridge: bool,
        connects_critical: bool,
    ) -> float:
        """Calculate edge criticality score."""
        score = self.weights["weight"] * weight_norm
        
        if is_bridge:
            score += self.weights["bridge"]
        
        if connects_critical:
            score += self.weights["endpoint"]
        
        return min(1.0, score)
    
    def _generate_summary(
        self,
        edges: List[EdgeMetrics],
        classification: ClassificationResult,
    ) -> Dict[str, Any]:
        """Generate analysis summary."""
        weights = [e.weight for e in edges]
        scores = [e.criticality_score for e in edges]
        
        by_type_counts = {}
        for dep_type in DEPENDENCY_TYPES:
            by_type_counts[dep_type] = len([e for e in edges if e.dependency_type == dep_type])
        
        return {
            "total_edges": len(edges),
            "bridge_count": len([e for e in edges if e.is_bridge]),
            "connects_critical_count": len([e for e in edges if e.connects_critical]),
            "by_level": classification.summary(),
            "by_type": by_type_counts,
            "weight_stats": {
                "min": round(min(weights), 4) if weights else 0,
                "max": round(max(weights), 4) if weights else 0,
                "mean": round(sum(weights) / len(weights), 4) if weights else 0,
            },
            "score_stats": classification.stats.to_dict(),
        }
    
    def _empty_result(self, timestamp: str) -> EdgeAnalysisResult:
        """Create empty result."""
        return EdgeAnalysisResult(
            timestamp=timestamp,
            edges=[],
            summary={"total_edges": 0},
        )
