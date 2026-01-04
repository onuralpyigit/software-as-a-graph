"""
Quality Analyzer

Implements the RMA (Reliability, Maintainability, Availability) formulations.
Normalizes raw metrics and computes composite scores.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.core.graph_exporter import GraphData
from .structural_analyzer import StructuralAnalysisResult, StructuralMetrics
from .classifier import BoxPlotClassifier, CriticalityLevel, ClassificationResult

@dataclass
class QualityScores:
    reliability: float = 0.0
    maintainability: float = 0.0
    availability: float = 0.0
    overall: float = 0.0

@dataclass
class ComponentQuality:
    id: str
    type: str
    metrics: StructuralMetrics
    scores: QualityScores
    level: CriticalityLevel = CriticalityLevel.MEDIUM
    
    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "scores": self.scores.__dict__,
            "level": self.level.value
        }

@dataclass
class QualityAnalysisResult:
    timestamp: str
    components: List[ComponentQuality]
    classification: ClassificationResult
    summary: Dict[str, Any]

class QualityAnalyzer:
    # Default weights from quality-formulations.md
    DEFAULT_WEIGHTS = {
        "reliability": {"pagerank": 0.45, "failure_propagation": 0.35, "in_degree": 0.20},
        "maintainability": {"betweenness": 0.45, "clustering": 0.25, "degree": 0.30},
        "availability": {"articulation_point": 0.50, "bridge_ratio": 0.25, "criticality": 0.25},
        "overall": {"reliability": 0.35, "maintainability": 0.30, "availability": 0.35}
    }

    def __init__(self, weights: Dict = None, k_factor: float = 1.5):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.classifier = BoxPlotClassifier(k_factor=k_factor)

    def analyze(self, structural_result: StructuralAnalysisResult) -> QualityAnalysisResult:
        """Calculate quality scores based on structural metrics."""
        
        # 1. Normalize Metrics (Min-Max)
        raw_values = structural_result.metrics.values()
        if not raw_values:
            return self._empty_result()
            
        normalized = self._normalize(raw_values)
        
        # 2. Compute Scores
        results = []
        for comp_id, raw in structural_result.metrics.items():
            norm = normalized[comp_id]
            
            # Reliability: PR + FP + InDegree
            w_r = self.weights["reliability"]
            r_score = (w_r["pagerank"] * norm["pagerank"] +
                       w_r["failure_propagation"] * norm["failure_propagation"] +
                       w_r["in_degree"] * norm["in_degree"])
            
            # Maintainability: BT + (1-CC) + Degree
            w_m = self.weights["maintainability"]
            m_score = (w_m["betweenness"] * norm["betweenness"] +
                       w_m["clustering"] * (1.0 - norm["clustering"]) +
                       w_m["degree"] * norm["degree"])
            
            # Availability: AP + BR + Criticality(PR*Deg)
            w_a = self.weights["availability"]
            crit_norm = norm["pagerank"] * norm["degree"] # Simplified Criticality proxy
            a_score = (w_a["articulation_point"] * raw.is_articulation_point +
                       w_a["bridge_ratio"] * raw.bridge_ratio +
                       w_a["criticality"] * crit_norm)
            
            # Overall
            w_o = self.weights["overall"]
            q_score = (w_o["reliability"] * r_score +
                       w_o["maintainability"] * m_score +
                       w_o["availability"] * a_score)
            
            results.append(ComponentQuality(
                id=comp_id,
                type=raw.component_type,
                metrics=raw,
                scores=QualityScores(r_score, m_score, a_score, q_score)
            ))

        # 3. Classify
        items = [{"id": c.id, "type": c.type, "score": c.scores.overall} for c in results]
        classification = self.classifier.classify(items, metric_name="overall_quality")
        
        # Assign levels
        level_map = {item.id: item.level for item in classification.items}
        for comp in results:
            comp.level = level_map.get(comp.id, CriticalityLevel.MEDIUM)
            
        results.sort(key=lambda x: x.scores.overall, reverse=True)
        
        return QualityAnalysisResult(
            timestamp=datetime.now().isoformat(),
            components=results,
            classification=classification,
            summary=classification.summary()
        )

    def _normalize(self, metrics: Any) -> Dict[str, Dict[str, float]]:
        # Helper to normalize all fields in StructuralMetrics
        fields = ["pagerank", "failure_propagation", "in_degree", "betweenness", 
                  "degree", "clustering_coefficient"]
        
        # Extract bounds
        bounds = {}
        for f in fields:
            vals = [getattr(m, f) for m in metrics]
            bounds[f] = (min(vals), max(vals))
            
        # Normalize
        norm_map = {}
        for m in metrics:
            norm_map[m.component_id] = {}
            for f in fields:
                mn, mx = bounds[f]
                val = getattr(m, f)
                # Map 'clustering_coefficient' to 'clustering' key for easier lookup
                key = "clustering" if f == "clustering_coefficient" else f
                norm_map[m.component_id][key] = (val - mn) / (mx - mn) if mx > mn else 0.5
                
        return norm_map

    def _empty_result(self):
        return QualityAnalysisResult("", [], None, {})