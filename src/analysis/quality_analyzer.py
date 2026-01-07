"""
Quality Analyzer

Computes Criticality Scores for Reliability (R), Maintainability (M), and Availability (A).
Uses Box-Plot Classification (Dynamic Thresholds) to assign levels for EACH dimension.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime

from .structural_analyzer import StructuralAnalysisResult, StructuralMetrics
from .classifier import BoxPlotClassifier, CriticalityLevel, ClassificationResult

@dataclass
class QualityScores:
    """Raw scores for quality dimensions."""
    reliability: float
    maintainability: float
    availability: float
    overall: float

@dataclass
class QualityLevels:
    """Classified levels for each quality dimension."""
    reliability: CriticalityLevel = CriticalityLevel.MINIMAL
    maintainability: CriticalityLevel = CriticalityLevel.MINIMAL
    availability: CriticalityLevel = CriticalityLevel.MINIMAL
    overall: CriticalityLevel = CriticalityLevel.MINIMAL

@dataclass
class ComponentQuality:
    id: str
    type: str
    scores: QualityScores
    levels: QualityLevels
    structural: StructuralMetrics # Carried forward for context-aware detection

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
    context: str
    components: List[ComponentQuality]
    edges: List[EdgeQuality]
    classification_summary: Dict[str, Any]

class QualityAnalyzer:
    def __init__(self, k_factor: float = 1.5):
        self.classifier = BoxPlotClassifier(k_factor=k_factor)
        
        # Coefficients for composite scores
        # Reliability: Global Importance (PageRank) + Inward Dependency (In-Degree)
        self.W_R = {"pagerank": 0.45, "reverse_pagerank": 0.35, "in_degree": 0.20} 
        
        # Maintainability: Structural Centrality (Betweenness) + Coupling + (1-Clustering)
        self.W_M = {"betweenness": 0.45, "degree": 0.30, "clustering": 0.25}
        
        # Availability: Bridge Status + Articulation Point + Global Importance
        self.W_A = {"articulation": 0.50, "bridge_ratio": 0.25, "importance": 0.25} 

    def analyze(self, struct: StructuralAnalysisResult, context: str = "System") -> QualityAnalysisResult:
        comp_results = []
        edge_results = []
        
        # 1. Component Analysis
        if struct.components:
            # Normalize metrics relative to the current dataset
            normalized = self._normalize_components(struct.components.values())
            
            # Compute raw scores
            temp_components = []
            for c in struct.components.values():
                scores = self._compute_component_scores(c, normalized.get(c.id, {}))
                temp_components.append({
                    "data": c,
                    "scores": scores
                })
            
            # Classify EACH dimension separately (R, M, A, Overall)
            # This allows us to say "Critical for Availability" specifically.
            levels_map = self._classify_multi_dimensional(temp_components)

            # Assemble final objects
            for item in temp_components:
                c = item["data"]
                scores = item["scores"]
                comp_results.append(ComponentQuality(
                    id=c.id,
                    type=c.type,
                    scores=scores,
                    levels=levels_map[c.id],
                    structural=c
                ))

        # 2. Edge Analysis
        if struct.edges:
            # Quick lookup for endpoint scores
            c_lookup = {c.id: c.scores for c in comp_results}
            normalized_edges = self._normalize_edges(struct.edges.values())
            
            for e_key, e in struct.edges.items():
                if e.source in c_lookup and e.target in c_lookup:
                    scores = self._compute_edge_scores(
                        e, 
                        normalized_edges.get(e_key, {}), 
                        c_lookup[e.source], 
                        c_lookup[e.target]
                    )
                    edge_results.append(EdgeQuality(
                        e.source, e.target, e.dependency_type, 
                        scores, CriticalityLevel.MINIMAL
                    ))
            
            # Classify edges based on Overall score
            self._classify_edges(edge_results)

        return QualityAnalysisResult(
            timestamp=datetime.now().isoformat(),
            context=context,
            components=comp_results,
            edges=edge_results,
            classification_summary=self._summarize(comp_results, edge_results)
        )

    def _compute_component_scores(self, raw: StructuralMetrics, norm: Dict[str, float]) -> QualityScores:
        # Default to 0.0 if metric missing
        get_n = lambda k: norm.get(k, 0.0)

        # Reliability (R): High usage + High importance
        r = self.W_R["pagerank"] * get_n("pagerank") + \
            self.W_R["reverse_pagerank"] * get_n("pagerank") + \
            self.W_R["in_degree"] * get_n("in_degree")
        
        # Maintainability (M): High centrality + High coupling + Low clustering (structural bridge)
        # Note: (1 - clustering) implies the node connects disparate groups
        m = self.W_M["betweenness"] * get_n("betweenness") + \
            self.W_M["degree"] * get_n("degree") + \
            self.W_M["clustering"] * (1.0 - get_n("clustering_coefficient"))
        
        # Availability (A): SPOF characteristics
        # Articulation Point is binary in raw, but we weight it heavily
        is_ap = 1.0 if raw.is_articulation_point else 0.0
        # Importance factor for availability
        imp = (get_n("pagerank") + get_n("degree")) / 2
        
        a = self.W_A["articulation"] * is_ap + \
            self.W_A["bridge_ratio"] * raw.bridge_ratio + \
            self.W_A["importance"] * imp
        
        # Overall Criticality
        overall = 0.35 * r + 0.30 * m + 0.35 * a
        return QualityScores(r, m, a, overall)

    def _compute_edge_scores(self, raw, norm, src_s: QualityScores, tgt_s: QualityScores) -> QualityScores:
        # Edge Importance
        end_point_avg = (src_s.reliability + tgt_s.reliability) / 2
        r = 0.4 * norm.get("weight", 0) + 0.6 * end_point_avg
        
        # Edge Risk (Bridge)
        is_br = 1.0 if raw.is_bridge else 0.0
        a = 0.6 * is_br + 0.4 * ((src_s.availability + tgt_s.availability) / 2)
        
        q = 0.5 * r + 0.5 * a
        return QualityScores(r, 0.0, a, q)

    def _classify_multi_dimensional(self, items: List[Dict]) -> Dict[str, QualityLevels]:
        """Runs the Box-Plot classifier on R, M, A, and Overall scores separately."""
        if not items: return {}
        
        # Helper to extract scores
        extract = lambda dim: [{"id": i["data"].id, "score": getattr(i["scores"], dim)} for i in items]
        
        # Run classification for each dimension
        results = {}
        for dim in ["reliability", "maintainability", "availability", "overall"]:
            # We group by type implicitly because we pass the whole list. 
            # If strictly by type is needed, filter `items` first. 
            # Here we classify against the passed context (Layer or Type).
            dataset = extract(dim)
            classified = self.classifier.classify(dataset, metric_name=dim)
            results[dim] = {item.id: item.level for item in classified.items}
            
        # Map back to QualityLevels objects
        final_map = {}
        for item in items:
            uid = item["data"].id
            final_map[uid] = QualityLevels(
                reliability=results["reliability"].get(uid, CriticalityLevel.MINIMAL),
                maintainability=results["maintainability"].get(uid, CriticalityLevel.MINIMAL),
                availability=results["availability"].get(uid, CriticalityLevel.MINIMAL),
                overall=results["overall"].get(uid, CriticalityLevel.MINIMAL),
            )
        return final_map

    def _classify_edges(self, edges: List[EdgeQuality]):
        if not edges: return
        data = [{"id": e.id, "score": e.scores.overall} for e in edges]
        res = self.classifier.classify(data, metric_name="edge_quality")
        mapping = {item.id: item.level for item in res.items}
        for e in edges:
            e.level = mapping.get(e.id, CriticalityLevel.MINIMAL)

    def _normalize_components(self, metrics):
        fields = ["pagerank", "in_degree", "betweenness", "degree", "clustering_coefficient"]
        return self._min_max_normalize(metrics, fields)

    def _normalize_edges(self, metrics):
        return self._min_max_normalize(metrics, ["weight", "betweenness"])

    def _min_max_normalize(self, items, fields):
        item_list = list(items)
        if not item_list: return {}

        # Extract values
        values = {f: [getattr(i, f, 0.0) for i in item_list] for f in fields}
        bounds = {f: (min(v), max(v)) for f, v in values.items()}
        
        normalized = {}
        for item in item_list:
            key = getattr(item, 'id', None)
            if not key: key = (item.source, item.target) # Edge tuple key
            
            normalized[key] = {}
            for f in fields:
                mn, mx = bounds[f]
                val = getattr(item, f, 0.0)
                diff = mx - mn
                
                if diff == 0:
                    # If no variance, check if value is significant
                    normalized[key][f] = 1.0 if val > 0 else 0.0
                else:
                    normalized[key][f] = (val - mn) / diff
        return normalized

    def _summarize(self, comps, edges):
        return {
            "components": {l.value: len([c for c in comps if c.levels.overall == l]) for l in CriticalityLevel},
            "edges": {l.value: len([e for e in edges if e.level == l]) for l in CriticalityLevel}
        }