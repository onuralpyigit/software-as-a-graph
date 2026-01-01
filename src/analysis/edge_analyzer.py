"""
Edge Analyzer - Version 5.0

Analyzes DEPENDS_ON edges to identify critical connections.

Critical edges are those whose failure would have significant impact:
- Bridge edges (only path between components)
- High-weight edges (heavy dependencies)
- Edges connecting critical components

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional

from .gds_client import GDSClient
from .classifier import BoxPlotClassifier, CriticalityLevel


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EdgeCriticality:
    """
    Criticality assessment for a single edge.
    """
    source_id: str
    target_id: str
    source_type: str
    target_type: str
    dependency_type: str
    
    # Criticality scores
    weight: float = 1.0
    criticality_score: float = 0.0
    
    # Flags
    is_bridge: bool = False
    connects_critical: bool = False
    
    # Classification
    level: CriticalityLevel = CriticalityLevel.MEDIUM
    
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
    Complete result from edge analysis.
    """
    timestamp: str
    edge_count: int
    edges: List[EdgeCriticality]
    by_level: Dict[CriticalityLevel, List[EdgeCriticality]]
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "edge_count": self.edge_count,
            "edges": [e.to_dict() for e in self.edges],
            "by_level": {
                level.value: [e.to_dict() for e in edges]
                for level, edges in self.by_level.items()
            },
            "summary": self.summary,
        }
    
    def get_critical_edges(self) -> List[EdgeCriticality]:
        """Get edges classified as CRITICAL"""
        return self.by_level.get(CriticalityLevel.CRITICAL, [])
    
    def get_bridges(self) -> List[EdgeCriticality]:
        """Get bridge edges"""
        return [e for e in self.edges if e.is_bridge]
    
    def top_n(self, n: int = 10) -> List[EdgeCriticality]:
        """Get top N edges by criticality score"""
        return sorted(
            self.edges, 
            key=lambda e: e.criticality_score, 
            reverse=True
        )[:n]


# =============================================================================
# Edge Analyzer
# =============================================================================

class EdgeAnalyzer:
    """
    Analyzes edges in the DEPENDS_ON graph.
    
    Identifies critical edges based on:
    - Weight (dependency strength)
    - Bridge status (only path between components)
    - Endpoint criticality (connects critical components)
    
    Example:
        with GDSClient(uri, user, password) as gds:
            analyzer = EdgeAnalyzer(gds)
            result = analyzer.analyze()
            
            for edge in result.get_critical_edges():
                print(f"{edge.source_id} -> {edge.target_id}: {edge.criticality_score:.4f}")
    """

    # Weights for criticality score
    DEFAULT_WEIGHTS = {
        "weight": 0.30,        # Edge weight
        "bridge": 0.40,        # Bridge bonus
        "endpoint": 0.30,      # Critical endpoint bonus
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
        critical_component_ids: Optional[set] = None,
    ) -> EdgeAnalysisResult:
        """
        Analyze all DEPENDS_ON edges.
        
        Args:
            dependency_types: Types to analyze (None = all)
            critical_component_ids: Set of known critical component IDs
        
        Returns:
            EdgeAnalysisResult with all edge analysis
        """
        timestamp = datetime.now().isoformat()
        
        self.logger.info("Starting edge analysis")
        
        # Get all edges
        edges_data = self._get_all_edges(dependency_types)
        
        if not edges_data:
            return self._empty_result(timestamp)
        
        # Get bridges
        bridges = self._get_bridge_set()
        
        # Get critical components if not provided
        if critical_component_ids is None:
            critical_component_ids = self._get_critical_components()
        
        # Normalize weights
        max_weight = max(e["weight"] for e in edges_data) or 1.0
        
        # Build edge criticality objects
        edges: List[EdgeCriticality] = []
        
        for edge_data in edges_data:
            source = edge_data["source_id"]
            target = edge_data["target_id"]
            weight = edge_data["weight"]
            
            # Check flags
            is_bridge = (source, target) in bridges
            connects_critical = (
                source in critical_component_ids or 
                target in critical_component_ids
            )
            
            # Calculate criticality score
            score = self._calculate_criticality_score(
                weight / max_weight,
                is_bridge,
                connects_critical,
            )
            
            edges.append(EdgeCriticality(
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
            {"id": f"{e.source_id}->{e.target_id}", "type": "edge", "score": e.criticality_score}
            for e in edges
        ]
        classification = self.classifier.classify(items, metric_name="edge_criticality")
        
        # Assign levels
        level_map = {item.id: item.level for item in classification.items}
        for edge in edges:
            edge_key = f"{edge.source_id}->{edge.target_id}"
            edge.level = level_map.get(edge_key, CriticalityLevel.MEDIUM)
        
        # Group by level
        by_level: Dict[CriticalityLevel, List[EdgeCriticality]] = {
            level: [] for level in CriticalityLevel
        }
        for edge in edges:
            by_level[edge.level].append(edge)
        
        # Sort by criticality score
        edges.sort(key=lambda e: e.criticality_score, reverse=True)
        
        # Generate summary
        summary = self._generate_summary(edges, by_level)
        
        return EdgeAnalysisResult(
            timestamp=timestamp,
            edge_count=len(edges),
            edges=edges,
            by_level=by_level,
            summary=summary,
        )

    def _get_all_edges(
        self, 
        dependency_types: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Get all DEPENDS_ON edges"""
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

    def _get_bridge_set(self) -> set:
        """Get set of bridge edges as (source, target) tuples"""
        bridges = self.gds.find_bridges()
        return {
            (b["source_id"], b["target_id"]) 
            for b in bridges
        }

    def _get_critical_components(self) -> set:
        """Get set of critical component IDs based on betweenness"""
        projection_name = "edge_analysis_temp"
        
        try:
            self.gds.create_projection(projection_name)
            bc_results = self.gds.betweenness(projection_name)
            
            if not bc_results:
                return set()
            
            items = [
                {"id": r.node_id, "type": r.node_type, "score": r.score}
                for r in bc_results
            ]
            classification = self.classifier.classify(items, metric_name="betweenness")
            
            return {item.id for item in classification.get_high_and_above()}
        
        finally:
            self.gds.drop_projection(projection_name)

    def _calculate_criticality_score(
        self,
        normalized_weight: float,
        is_bridge: bool,
        connects_critical: bool,
    ) -> float:
        """Calculate edge criticality score"""
        score = 0.0
        
        # Weight contribution
        score += self.weights.get("weight", 0.3) * normalized_weight
        
        # Bridge bonus
        if is_bridge:
            score += self.weights.get("bridge", 0.4)
        
        # Critical endpoint bonus
        if connects_critical:
            score += self.weights.get("endpoint", 0.3)
        
        return min(1.0, score)

    def _generate_summary(
        self,
        edges: List[EdgeCriticality],
        by_level: Dict[CriticalityLevel, List[EdgeCriticality]],
    ) -> Dict[str, Any]:
        """Generate summary statistics"""
        weights = [e.weight for e in edges]
        scores = [e.criticality_score for e in edges]
        
        return {
            "total_edges": len(edges),
            "bridge_count": sum(1 for e in edges if e.is_bridge),
            "connects_critical_count": sum(1 for e in edges if e.connects_critical),
            "by_level": {
                level.value: len(level_edges) 
                for level, level_edges in by_level.items()
            },
            "weight_stats": {
                "min": min(weights) if weights else 0,
                "max": max(weights) if weights else 0,
                "mean": sum(weights) / len(weights) if weights else 0,
            },
            "criticality_stats": {
                "min": min(scores) if scores else 0,
                "max": max(scores) if scores else 0,
                "mean": sum(scores) / len(scores) if scores else 0,
            },
        }

    def _empty_result(self, timestamp: str) -> EdgeAnalysisResult:
        """Create empty result"""
        return EdgeAnalysisResult(
            timestamp=timestamp,
            edge_count=0,
            edges=[],
            by_level={level: [] for level in CriticalityLevel},
            summary={"total_edges": 0},
        )
