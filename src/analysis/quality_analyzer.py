"""
Quality Analyzer

Computes composite Quality Scores (Reliability, Maintainability, Availability)
for Components AND Edges. Uses statistical classification.
"""

from dataclasses import dataclass
from typing import Dict, List, Any
from datetime import datetime

from .structural_analyzer import StructuralAnalysisResult
from .classifier import BoxPlotClassifier, CriticalityLevel, ClassificationResult

@dataclass
class QualityScores:
    reliability: float
    maintainability: float
    availability: float
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
        # Weights from formulations
        self.W_C = {
            "R": {"pr": 0.45, "fp": 0.35, "id": 0.20},
            "M": {"bt": 0.45, "cc": 0.25, "dc": 0.30},
            "A": {"ap": 0.50, "br": 0.25, "cr": 0.25},
            "Q": {"r": 0.35, "m": 0.30, "a": 0.35}
        }
        self.W_E = {
            "R": {"w": 0.40, "ep": 0.60},
            "A": {"br": 0.60, "ep": 0.40}
        }

    def analyze(self, struct: StructuralAnalysisResult) -> QualityAnalysisResult:
        # 1. Component Analysis
        comp_results = []
        if struct.components:
            normalized = self._normalize_components(struct.components.values())
            for c in struct.components.values():
                scores = self._compute_component_scores(c, normalized[c.id])
                comp_results.append(ComponentQuality(c.id, c.type, scores, CriticalityLevel.MINIMAL))
            
            # Classify Components (Grouped by Type for fairness)
            self._classify_items(comp_results, group_by_type=True)

        # 2. Edge Analysis
        edge_results = []
        if struct.edges:
            # Need lookup for endpoint scores
            c_scores = {c.id: c.scores for c in comp_results}
            normalized_edges = self._normalize_edges(struct.edges.values())
            
            for e_key, e in struct.edges.items():
                if e.source in c_scores and e.target in c_scores:
                    scores = self._compute_edge_scores(e, normalized_edges[e_key], c_scores[e.source], c_scores[e.target])
                    edge_results.append(EdgeQuality(e.source, e.target, e.dependency_type, scores, CriticalityLevel.MINIMAL))
            
            # Classify Edges
            self._classify_items(edge_results, group_by_type=True)

        return QualityAnalysisResult(
            timestamp=datetime.now().isoformat(),
            components=comp_results,
            edges=edge_results,
            classification_summary=self._summarize(comp_results, edge_results)
        )

    def _compute_component_scores(self, raw, norm) -> QualityScores:
        w = self.W_C
        # Reliability (R)
        r = w["R"]["pr"] * norm["pagerank"] + w["R"]["fp"] * norm["failure_propagation"] + w["R"]["id"] * norm["in_degree"]
        
        # Maintainability (M) - Invert Clustering
        m = w["M"]["bt"] * norm["betweenness"] + w["M"]["cc"] * (1 - norm["clustering"]) + w["M"]["dc"] * norm["degree"]
        
        # Availability (A) - Criticality proxy = PR * Degree
        crit = norm["pagerank"] * norm["degree"]
        a = w["A"]["ap"] * float(raw.is_articulation_point) + w["A"]["br"] * raw.bridge_ratio + w["A"]["cr"] * crit
        
        # Overall (Q)
        q = w["Q"]["r"] * r + w["Q"]["m"] * m + w["Q"]["a"] * a
        return QualityScores(r, m, a, q)

    def _compute_edge_scores(self, raw, norm, src_s, tgt_s) -> QualityScores:
        w = self.W_E
        # Edge Reliability: Weight + Avg Endpoint Reliability
        ep_r = (src_s.reliability + tgt_s.reliability) / 2
        r = w["R"]["w"] * norm["weight"] + w["R"]["ep"] * ep_r
        
        # Edge Availability: Bridge Status + Avg Endpoint Availability (using Endpoint AP status as proxy)
        # Note: Using Endpoint Availability Score is smoother than binary AP status
        ep_a = (src_s.availability + tgt_s.availability) / 2
        a = w["A"]["br"] * float(raw.is_bridge) + w["A"]["ep"] * ep_a
        
        # Edge Overall
        q = 0.5 * r + 0.5 * a
        return QualityScores(r, 0.0, a, q)

    def _classify_items(self, items: List[Any], group_by_type: bool = False):
        if not items: return
        
        groups = {}
        if group_by_type:
            for i in items: groups.setdefault(i.type, []).append(i)
        else:
            groups["all"] = items

        for g_name, g_items in groups.items():
            if len(g_items) < 2: continue # Cannot classify single items statistically
            
            # Prepare for classifier
            to_classify = [{"id": x.id, "score": x.scores.overall} for x in g_items]
            result = self.classifier.classify(to_classify, metric_name=f"{g_name}_quality")
            
            # Map back
            level_map = {item.id: item.level for item in result.items}
            for x in g_items:
                if x.id in level_map: x.level = level_map[x.id]

    def _normalize_components(self, metrics):
        return self._min_max_normalize(metrics, ["pagerank", "failure_propagation", "in_degree", "betweenness", "degree", "clustering_coefficient"])

    def _normalize_edges(self, metrics):
        return self._min_max_normalize(metrics, ["weight", "betweenness"])

    def _min_max_normalize(self, items, fields):
        # Extract columns
        values = {f: [getattr(i, f) for i in items] for f in fields}
        bounds = {f: (min(v), max(v)) for f, v in values.items()}
        
        normalized = {}
        item_list = list(items) # items is a dict_values view, convert to list for indexing if needed, or iterate
        
        # Assuming items have 'id' or we map by object
        # Since struct.components is a dict, values() are objects. 
        # Need to map back to ID.
        for item in items:
            key = getattr(item, 'id', None)
            if not key: key = (item.source, item.target) # Edge case
            
            normalized[key] = {}
            for f in fields:
                mn, mx = bounds[f]
                val = getattr(item, f)
                normalized[key][f.replace("_coefficient", "")] = (val - mn) / (mx - mn) if mx > mn else 0.5
        return normalized

    def _summarize(self, comps, edges):
        return {
            "components": {l.value: len([c for c in comps if c.level == l]) for l in CriticalityLevel},
            "edges": {l.value: len([e for e in edges if e.level == l]) for l in CriticalityLevel}
        }