"""
Quality Analyzer

Computes Criticality Scores for Reliability (R), Maintainability (M), and Availability (A).
Uses Box-Plot Classification (Dynamic Thresholds) to assign levels.
"""

from dataclasses import dataclass
from typing import Dict, List, Any
from datetime import datetime

from .structural_analyzer import StructuralAnalysisResult
from .classifier import BoxPlotClassifier, CriticalityLevel, ClassificationResult

@dataclass
class QualityScores:
    """
    Scores represent Criticality/Importance.
    Higher Score = Higher Criticality = Higher Risk if component fails.
    """
    reliability: float    # How critical is this for system reliability?
    maintainability: float # How hard is this to maintain/change?
    availability: float    # How critical is this for system availability?
    overall: float

@dataclass
class ComponentQuality:
    id: str
    type: str
    scores: QualityScores
    level: CriticalityLevel

@dataclass
class EdgeQuality:
    source: str
    target: str
    type: str
    scores: QualityScores
    level: CriticalityLevel
    
    @property
    def id(self): return f"{self.source}->{self.target}"

@dataclass
class QualityAnalysisResult:
    timestamp: str
    components: List[ComponentQuality]
    edges: List[EdgeQuality]
    classification_summary: Dict[str, Any]

class QualityAnalyzer:
    def __init__(self, k_factor: float = 1.5):
        self.classifier = BoxPlotClassifier(k_factor=k_factor)
        
        # Coefficients for composite scores
        # Reliability: Importance (PageRank) + Failure Impact (Reverse PR)
        self.W_R = {"pr": 0.45, "fp": 0.35, "id": 0.20} 
        
        # Maintainability: Complexity (Betweenness) + Coupling (Degree)
        # Note: High Score here means "Hard to Maintain / High Ripple Effect"
        self.W_M = {"bt": 0.45, "cc": 0.25, "dc": 0.30}
        
        # Availability: Single Point of Failure potential
        self.W_A = {"ap": 0.50, "br": 0.25, "cr": 0.25} 

    def analyze(self, struct: StructuralAnalysisResult) -> QualityAnalysisResult:
        comp_results = []
        edge_results = []
        
        # 1. Component Analysis
        if struct.components:
            # Normalize metrics relative to the current dataset (Type or Full System)
            normalized = self._normalize_components(struct.components.values())
            
            for c in struct.components.values():
                scores = self._compute_component_scores(c, normalized[c.id])
                comp_results.append(ComponentQuality(c.id, c.type, scores, CriticalityLevel.MINIMAL))
            
            # Classify dynamically based on distribution within this group
            # We classify based on 'overall' score to set the main Level
            self._classify_items(comp_results)

        # 2. Edge Analysis
        if struct.edges:
            c_scores = {c.id: c.scores for c in comp_results}
            normalized_edges = self._normalize_edges(struct.edges.values())
            
            for e_key, e in struct.edges.items():
                if e.source in c_scores and e.target in c_scores:
                    scores = self._compute_edge_scores(e, normalized_edges[e_key], c_scores[e.source], c_scores[e.target])
                    edge_results.append(EdgeQuality(e.source, e.target, e.dependency_type, scores, CriticalityLevel.MINIMAL))
            
            self._classify_items(edge_results)

        return QualityAnalysisResult(
            timestamp=datetime.now().isoformat(),
            components=comp_results,
            edges=edge_results,
            classification_summary=self._summarize(comp_results, edge_results)
        )

    def _compute_component_scores(self, raw, norm) -> QualityScores:
        # Reliability Criticality (R):
        # High PR (Important) + High In-Degree (Dependents) = Critical for Reliability
        r = self.W_R["pr"] * norm["pagerank"] + \
            self.W_R["fp"] * norm["pagerank"] + \
            self.W_R["id"] * norm["in_degree"]
        
        # Maintainability Criticality (M):
        # High Betweenness (Central) + High Degree (Coupled) + Low Clustering = Hard to change safely
        # Note: We use (1 - clustering) because low clustering usually implies higher structural bridge role
        m = self.W_M["bt"] * norm["betweenness"] + \
            self.W_M["cc"] * (1.0 - norm["clustering"]) + \
            self.W_M["dc"] * norm["degree"]
        
        # Availability Criticality (A):
        # Is Articulation Point (SPOF) + Bridge Ratio + General Importance
        crit_factor = norm["pagerank"] * norm["degree"]
        a = self.W_A["ap"] * float(raw.is_articulation_point) + \
            self.W_A["br"] * raw.bridge_ratio + \
            self.W_A["cr"] * crit_factor
        
        # Overall Criticality
        q = 0.35 * r + 0.30 * m + 0.35 * a
        return QualityScores(r, m, a, q)

    def _compute_edge_scores(self, raw, norm, src_s, tgt_s) -> QualityScores:
        # Edge Importance depends on:
        # 1. Structural Betweenness (norm['betweenness'])
        # 2. Importance of connected components (ep_r)
        
        ep_r = (src_s.reliability + tgt_s.reliability) / 2
        r = 0.4 * norm["weight"] + 0.6 * ep_r
        
        # Edge Availability Risk (Bridge status)
        ep_a = (src_s.availability + tgt_s.availability) / 2
        a = 0.6 * float(raw.is_bridge) + 0.4 * ep_a
        
        q = 0.5 * r + 0.5 * a
        return QualityScores(r, 0.0, a, q)

    def _classify_items(self, items: List[Any]):
        if not items: return
        
        # Group by type to ensure fair comparison (Nodes vs Apps)
        groups = {}
        for i in items: groups.setdefault(i.type, []).append(i)

        for g_name, g_items in groups.items():
            if len(g_items) < 2: continue
            
            # Use BoxPlotClassifier to set levels based on Overall Score
            to_classify = [{"id": x.id, "score": x.scores.overall} for x in g_items]
            result = self.classifier.classify(to_classify, metric_name=f"{g_name}_quality")
            
            level_map = {item.id: item.level for item in result.items}
            for x in g_items:
                if x.id in level_map: x.level = level_map[x.id]

    def _normalize_components(self, metrics):
        return self._min_max_normalize(metrics, ["pagerank", "in_degree", "betweenness", "degree", "clustering_coefficient"])

    def _normalize_edges(self, metrics):
        return self._min_max_normalize(metrics, ["weight", "betweenness"])

    def _min_max_normalize(self, items, fields):
        # Safe normalization handling single-item or zero-variance cases
        item_list = list(items)
        if not item_list: return {}

        values = {f: [getattr(i, f) for i in item_list] for f in fields}
        bounds = {f: (min(v), max(v)) for f, v in values.items()}
        
        normalized = {}
        for item in item_list:
            key = getattr(item, 'id', None)
            if not key: key = (item.source, item.target)
            
            normalized[key] = {}
            for f in fields:
                mn, mx = bounds[f]
                val = getattr(item, f)
                diff = mx - mn
                
                # Normalize to 0-1
                if diff == 0:
                    norm_val = 0.0 # If all values are same, assume baseline
                else:
                    norm_val = (val - mn) / diff
                
                normalized[key][f.replace("_coefficient", "")] = norm_val
        return normalized

    def _summarize(self, comps, edges):
        return {
            "components": {l.value: len([c for c in comps if c.level == l]) for l in CriticalityLevel},
            "edges": {l.value: len([e for e in edges if e.level == l]) for l in CriticalityLevel}
        }