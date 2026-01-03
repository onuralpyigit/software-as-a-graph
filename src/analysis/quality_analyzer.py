"""
Quality Analyzer - Version 5.0

Assesses system quality attributes (Reliability, Maintainability, Availability)
using graph topological metrics.

Formulations:
- Reliability: PageRank (Influence) + In-Degree (Dependency)
- Maintainability: Betweenness (Coupling) + Degree (Complexity)
- Availability: Articulation Points (SPOF) + PageRank (Criticality)

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional

from .gds_client import GDSClient, COMPONENT_TYPES
from .classifier import BoxPlotClassifier, CriticalityLevel

@dataclass
class QualityMetrics:
    """Quality scores for a single component."""
    component_id: str
    component_type: str
    
    # Quality Scores (0.0 - 1.0)
    reliability_score: float = 0.0
    maintainability_score: float = 0.0
    availability_score: float = 0.0
    overall_quality_score: float = 0.0
    
    # Classification
    reliability_level: CriticalityLevel = CriticalityLevel.MEDIUM
    maintainability_level: CriticalityLevel = CriticalityLevel.MEDIUM
    availability_level: CriticalityLevel = CriticalityLevel.MEDIUM
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.component_id,
            "type": self.component_type,
            "scores": {
                "reliability": round(self.reliability_score, 4),
                "maintainability": round(self.maintainability_score, 4),
                "availability": round(self.availability_score, 4),
                "overall": round(self.overall_quality_score, 4),
            },
            "levels": {
                "reliability": self.reliability_level.value,
                "maintainability": self.maintainability_level.value,
                "availability": self.availability_level.value,
            }
        }

@dataclass
class QualityAnalysisResult:
    """Result of quality assessment."""
    timestamp: str
    components: List[QualityMetrics]
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "component_count": len(self.components),
            "summary": self.summary,
            "components": [c.to_dict() for c in self.components]
        }

class QualityAnalyzer:
    """Analyzes Reliability, Maintainability, and Availability."""
    
    # Default weights for formulations
    WEIGHTS = {
        "reliability": {"pagerank": 0.6, "in_degree": 0.4},
        "maintainability": {"betweenness": 0.5, "degree": 0.5},
        "availability": {"articulation_point": 0.4, "pagerank": 0.6}
    }
    
    def __init__(self, gds_client: GDSClient, k_factor: float = 1.5):
        self.gds = gds_client
        self.classifier = BoxPlotClassifier(k_factor=k_factor)
        self.logger = logging.getLogger(__name__)

    def analyze_quality(self, weighted: bool = True) -> QualityAnalysisResult:
        """Perform comprehensive quality analysis."""
        timestamp = datetime.now().isoformat()
        proj_name = "quality_analysis_proj"
        
        try:
            self.gds.create_projection(proj_name, include_weights=weighted)
            
            # 1. Fetch Raw Metrics
            pagerank = self._normalize_scores(self.gds.pagerank(proj_name, weighted))
            betweenness = self._normalize_scores(self.gds.betweenness(proj_name, weighted))
            degree = self._normalize_scores(self.gds.degree(proj_name, weighted))
            
            # In-Degree (requires specific query or approximation via Degree for undirected)
            # For simplicity in this example, we reuse Degree or assume GDS calculates total degree
            # Ideally, use distinct in-degree calculation if supported by GDS wrapper
            
            # Articulation Points
            aps = {ap.node_id for ap in self.gds.find_articulation_points()}
            
            # 2. Calculate Scores
            metrics_map = {}
            
            all_node_ids = set(pagerank.keys()) | set(betweenness.keys())
            
            for node_id in all_node_ids:
                pr = pagerank.get(node_id, 0.0)
                bt = betweenness.get(node_id, 0.0)
                deg = degree.get(node_id, 0.0)
                is_ap = 1.0 if node_id in aps else 0.0
                
                # Reliability = w1*PR + w2*Degree (using degree as proxy for dependency complexity)
                rel_score = (self.WEIGHTS["reliability"]["pagerank"] * pr + 
                             self.WEIGHTS["reliability"]["in_degree"] * deg)
                
                # Maintainability = w1*BT + w2*Degree
                maint_score = (self.WEIGHTS["maintainability"]["betweenness"] * bt + 
                               self.WEIGHTS["maintainability"]["degree"] * deg)
                
                # Availability = w1*AP + w2*PR
                avail_score = (self.WEIGHTS["availability"]["articulation_point"] * is_ap + 
                               self.WEIGHTS["availability"]["pagerank"] * pr)
                
                metrics_map[node_id] = QualityMetrics(
                    component_id=node_id,
                    component_type="Unknown", # Would need lookup
                    reliability_score=rel_score,
                    maintainability_score=maint_score,
                    availability_score=avail_score,
                    overall_quality_score=(rel_score + maint_score + avail_score) / 3
                )

            # 3. Classify and Summarize
            components = list(metrics_map.values())
            
            # Simple stats
            summary = {
                "avg_reliability": sum(c.reliability_score for c in components) / len(components) if components else 0,
                "avg_maintainability": sum(c.maintainability_score for c in components) / len(components) if components else 0,
                "avg_availability": sum(c.availability_score for c in components) / len(components) if components else 0,
                "critical_availability_count": len(aps)
            }
            
            return QualityAnalysisResult(timestamp, components, summary)
            
        finally:
            self.gds.drop_projection(proj_name)

    def _normalize_scores(self, results) -> Dict[str, float]:
        """Normalize GDS results to 0.0-1.0 range."""
        if not results: return {}
        scores = {r.node_id: r.score for r in results}
        max_val = max(scores.values()) or 1.0
        return {k: v / max_val for k, v in scores.items()}