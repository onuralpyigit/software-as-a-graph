"""
GDS-based Quality Assessment

Quality assessment using Neo4j Graph Data Science algorithms
on DEPENDS_ON relationships.

Integrates with existing analyzers and provides:
- Composite criticality scoring using GDS metrics
- Quality-specific analysis (R/M/A)
- Problem detection from graph patterns
- Edge criticality via GDS centrality

Author: Ibrahim Onuralp Yigit
"""

from __future__ import annotations
import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING

from .graph_algorithms import GDSClient, CentralityResult
from .analyzers import (
    QualityAttribute,
    Severity,
    Finding,
    CriticalComponent,
    AnalysisResult,
    ReliabilityAnalyzer,
    MaintainabilityAnalyzer,
    AvailabilityAnalyzer,
)
from .classifier import BoxPlotClassifier, CriticalityLevel


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GDSQualityMetrics:
    """Quality metrics computed from GDS algorithms."""
    component_id: str
    component_type: str
    
    # Centrality scores (from GDS)
    betweenness: float = 0.0
    pagerank: float = 0.0
    degree: float = 0.0
    in_degree: float = 0.0
    out_degree: float = 0.0
    
    # Normalized scores
    betweenness_norm: float = 0.0
    pagerank_norm: float = 0.0
    degree_norm: float = 0.0
    
    # Structural properties
    is_articulation_point: bool = False
    is_bridge_endpoint: bool = False
    community_id: int = -1
    
    # Derived quality metrics
    reliability_score: float = 0.0
    maintainability_score: float = 0.0
    availability_score: float = 0.0
    composite_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "component_id": self.component_id,
            "component_type": self.component_type,
            "betweenness": round(self.betweenness, 6),
            "pagerank": round(self.pagerank, 6),
            "degree": round(self.degree, 6),
            "betweenness_norm": round(self.betweenness_norm, 4),
            "pagerank_norm": round(self.pagerank_norm, 4),
            "degree_norm": round(self.degree_norm, 4),
            "is_articulation_point": self.is_articulation_point,
            "is_bridge_endpoint": self.is_bridge_endpoint,
            "community_id": self.community_id,
            "reliability_score": round(self.reliability_score, 4),
            "maintainability_score": round(self.maintainability_score, 4),
            "availability_score": round(self.availability_score, 4),
            "composite_score": round(self.composite_score, 4),
        }


@dataclass
class GDSComponentScore:
    """Quality scores for a component using GDS metrics."""
    component_id: str
    component_type: str
    
    # Quality scores
    reliability_score: float
    maintainability_score: float
    availability_score: float
    composite_score: float
    
    # Criticality levels
    reliability_level: CriticalityLevel
    maintainability_level: CriticalityLevel
    availability_level: CriticalityLevel
    overall_level: CriticalityLevel
    
    # GDS metrics
    metrics: GDSQualityMetrics
    
    # Findings for this component
    findings: List[Finding] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "component_id": self.component_id,
            "component_type": self.component_type,
            "reliability_score": round(self.reliability_score, 4),
            "maintainability_score": round(self.maintainability_score, 4),
            "availability_score": round(self.availability_score, 4),
            "composite_score": round(self.composite_score, 4),
            "reliability_level": self.reliability_level.value,
            "maintainability_level": self.maintainability_level.value,
            "availability_level": self.availability_level.value,
            "overall_level": self.overall_level.value,
            "metrics": self.metrics.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
        }


@dataclass
class GDSEdgeCriticality:
    """Edge criticality from GDS analysis."""
    source_id: str
    target_id: str
    dependency_type: str
    weight: float
    
    # Criticality scores
    reliability_score: float
    maintainability_score: float
    availability_score: float
    composite_score: float
    
    # Properties
    is_bridge: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "dependency_type": self.dependency_type,
            "weight": round(self.weight, 4),
            "reliability_score": round(self.reliability_score, 4),
            "maintainability_score": round(self.maintainability_score, 4),
            "availability_score": round(self.availability_score, 4),
            "composite_score": round(self.composite_score, 4),
            "is_bridge": self.is_bridge,
        }


@dataclass
class GDSQualityResult:
    """Complete GDS-based quality assessment result."""
    timestamp: str
    projection_name: str
    dependency_types: List[str]
    
    # Graph statistics
    node_count: int
    relationship_count: int
    
    # Component scores
    component_scores: List[GDSComponentScore]
    
    # Edge criticality
    edge_criticality: List[GDSEdgeCriticality]
    
    # All findings
    findings: List[Finding]
    
    # Analyzer results
    reliability_result: Optional[AnalysisResult] = None
    maintainability_result: Optional[AnalysisResult] = None
    availability_result: Optional[AnalysisResult] = None
    
    # Summary
    summary: Dict[str, Any] = field(default_factory=dict)
    
    # Weights used
    weights: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "projection_name": self.projection_name,
            "dependency_types": self.dependency_types,
            "node_count": self.node_count,
            "relationship_count": self.relationship_count,
            "component_scores": [c.to_dict() for c in self.component_scores],
            "edge_criticality": [e.to_dict() for e in self.edge_criticality],
            "findings": [f.to_dict() for f in self.findings],
            "reliability_result": self.reliability_result.to_dict() if self.reliability_result else None,
            "maintainability_result": self.maintainability_result.to_dict() if self.maintainability_result else None,
            "availability_result": self.availability_result.to_dict() if self.availability_result else None,
            "summary": self.summary,
            "weights": self.weights,
        }


# =============================================================================
# GDS Criticality Formulas
# =============================================================================

class GDSCriticalityFormulas:
    """
    Composite Criticality Score Formulations using GDS metrics.
    
    Uses centrality algorithms from Neo4j GDS:
    - Betweenness: identifies bottlenecks and SPOFs
    - PageRank: measures importance based on dependencies
    - Degree: measures coupling level
    """
    
    # =========================================================================
    # RELIABILITY CRITICALITY SCORE (GDS-based)
    # =========================================================================
    #
    # C_reliability = α₁·AP + α₂·BC_norm + α₃·PR_norm + α₄·Bridge
    #
    # Where:
    #   AP     = 1 if articulation point, 0 otherwise
    #   BC     = Normalized betweenness centrality (from GDS)
    #   PR     = Normalized PageRank (from GDS)
    #   Bridge = 1 if endpoint of bridge edge, 0 otherwise
    #
    # Default weights: α₁=0.30, α₂=0.35, α₃=0.20, α₄=0.15
    # =========================================================================
    
    RELIABILITY_WEIGHTS = {
        "articulation_point": 0.30,
        "betweenness": 0.35,
        "pagerank": 0.20,
        "bridge_endpoint": 0.15,
    }
    
    # =========================================================================
    # MAINTAINABILITY CRITICALITY SCORE (GDS-based)
    # =========================================================================
    #
    # C_maintainability = β₁·DC_norm + β₂·OD_norm + β₃·ID_norm + β₄·BC_norm
    #
    # Where:
    #   DC = Normalized degree centrality (total coupling)
    #   OD = Normalized out-degree (fan-out coupling)
    #   ID = Normalized in-degree (fan-in coupling)
    #   BC = Normalized betweenness (change propagation)
    #
    # Default weights: β₁=0.35, β₂=0.25, β₃=0.20, β₄=0.20
    # =========================================================================
    
    MAINTAINABILITY_WEIGHTS = {
        "degree": 0.35,
        "out_degree": 0.25,
        "in_degree": 0.20,
        "betweenness": 0.20,
    }
    
    # =========================================================================
    # AVAILABILITY CRITICALITY SCORE (GDS-based)
    # =========================================================================
    #
    # C_availability = γ₁·PR_norm + γ₂·BC_norm + γ₃·DC_norm + γ₄·AP
    #
    # Where:
    #   PR = Normalized PageRank (importance for availability)
    #   BC = Normalized betweenness (bottleneck potential)
    #   DC = Normalized degree (load factor)
    #   AP = Articulation point indicator
    #
    # Default weights: γ₁=0.30, γ₂=0.30, γ₃=0.25, γ₄=0.15
    # =========================================================================
    
    AVAILABILITY_WEIGHTS = {
        "pagerank": 0.30,
        "betweenness": 0.30,
        "degree": 0.25,
        "articulation_point": 0.15,
    }
    
    @staticmethod
    def compute_reliability_score(
        metrics: GDSQualityMetrics,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute reliability criticality score from GDS metrics."""
        w = weights or GDSCriticalityFormulas.RELIABILITY_WEIGHTS
        
        score = 0.0
        score += w.get("articulation_point", 0.30) * (1.0 if metrics.is_articulation_point else 0.0)
        score += w.get("betweenness", 0.35) * metrics.betweenness_norm
        score += w.get("pagerank", 0.20) * metrics.pagerank_norm
        score += w.get("bridge_endpoint", 0.15) * (1.0 if metrics.is_bridge_endpoint else 0.0)
        
        return min(1.0, max(0.0, score))
    
    @staticmethod
    def compute_maintainability_score(
        metrics: GDSQualityMetrics,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute maintainability criticality score from GDS metrics."""
        w = weights or GDSCriticalityFormulas.MAINTAINABILITY_WEIGHTS
        
        score = 0.0
        score += w.get("degree", 0.35) * metrics.degree_norm
        score += w.get("out_degree", 0.25) * (metrics.out_degree / max(metrics.degree, 1)) if metrics.degree > 0 else 0
        score += w.get("in_degree", 0.20) * (metrics.in_degree / max(metrics.degree, 1)) if metrics.degree > 0 else 0
        score += w.get("betweenness", 0.20) * metrics.betweenness_norm
        
        return min(1.0, max(0.0, score))
    
    @staticmethod
    def compute_availability_score(
        metrics: GDSQualityMetrics,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute availability criticality score from GDS metrics."""
        w = weights or GDSCriticalityFormulas.AVAILABILITY_WEIGHTS
        
        score = 0.0
        score += w.get("pagerank", 0.30) * metrics.pagerank_norm
        score += w.get("betweenness", 0.30) * metrics.betweenness_norm
        score += w.get("degree", 0.25) * metrics.degree_norm
        score += w.get("articulation_point", 0.15) * (1.0 if metrics.is_articulation_point else 0.0)
        
        return min(1.0, max(0.0, score))
    
    @staticmethod
    def compute_composite_score(
        reliability: float,
        maintainability: float,
        availability: float,
        weights: Tuple[float, float, float] = (0.40, 0.25, 0.35),
    ) -> float:
        """Compute overall composite criticality score."""
        w_r, w_m, w_a = weights
        return w_r * reliability + w_m * maintainability + w_a * availability


# =============================================================================
# GDS Quality Assessor
# =============================================================================

class GDSQualityAssessor:
    """
    Quality assessment using Neo4j GDS algorithms.
    
    Integrates with:
    - GDSClient for running GDS algorithms
    - Existing quality analyzers (Reliability, Maintainability, Availability)
    - BoxPlotClassifier for criticality classification
    
    Workflow:
    1. Create GDS projection from DEPENDS_ON relationships
    2. Run centrality algorithms (PageRank, Betweenness, Degree)
    3. Detect structural properties (articulation points, bridges)
    4. Compute quality scores using GDS metrics
    5. Run quality analyzers for detailed findings
    6. Classify components using box-plot method
    7. Analyze edge criticality
    """
    
    def __init__(
        self,
        gds_client: GDSClient,
        reliability_weight: float = 0.40,
        maintainability_weight: float = 0.25,
        availability_weight: float = 0.35,
        k_factor: float = 1.5,
    ):
        self.gds = gds_client
        self.weights = (reliability_weight, maintainability_weight, availability_weight)
        self.classifier = BoxPlotClassifier(k_factor=k_factor)
        self.logger = logging.getLogger(__name__)
        
        # Initialize analyzers
        self.reliability_analyzer = ReliabilityAnalyzer(gds_client)
        self.maintainability_analyzer = MaintainabilityAnalyzer(gds_client)
        self.availability_analyzer = AvailabilityAnalyzer(gds_client)
    
    def assess(
        self,
        projection_name: str = "quality_assessment",
        dependency_types: Optional[List[str]] = None,
        run_analyzers: bool = True,
        include_edges: bool = True,
    ) -> GDSQualityResult:
        """
        Run complete GDS-based quality assessment.
        
        Args:
            projection_name: Name for the GDS graph projection
            dependency_types: DEPENDS_ON types to include
            run_analyzers: Whether to run detailed quality analyzers
            include_edges: Whether to analyze edge criticality
        
        Returns:
            Complete quality assessment result
        """
        timestamp = datetime.now().isoformat()
        
        if dependency_types is None:
            dependency_types = ["app_to_app", "node_to_node"]
        
        self.logger.info(f"Starting GDS quality assessment on '{projection_name}'")
        
        # 1. Create projection
        projection_info = self.gds.create_projection(
            projection_name,
            dependency_types=dependency_types,
            include_weights=True,
        )
        
        try:
            # 2. Compute GDS metrics for all components
            component_metrics = self._compute_component_metrics(projection_name)
            
            # 3. Compute quality scores
            component_scores = self._compute_component_scores(component_metrics)
            
            # 4. Run quality analyzers if requested
            reliability_result = None
            maintainability_result = None
            availability_result = None
            all_findings = []
            
            if run_analyzers:
                self.logger.info("Running quality analyzers...")
                
                reliability_result = self.reliability_analyzer.analyze(projection_name)
                all_findings.extend(reliability_result.findings)
                
                maintainability_result = self.maintainability_analyzer.analyze(projection_name)
                all_findings.extend(maintainability_result.findings)
                
                availability_result = self.availability_analyzer.analyze(projection_name)
                all_findings.extend(availability_result.findings)
                
                # Assign findings to components
                for score in component_scores:
                    score.findings = [
                        f for f in all_findings 
                        if f.component_id == score.component_id
                    ]
            
            # 5. Analyze edge criticality
            edge_criticality = []
            if include_edges:
                edge_criticality = self._compute_edge_criticality(projection_name, component_metrics)
            
            # 6. Generate summary
            summary = self._generate_summary(
                component_scores, 
                all_findings,
                reliability_result,
                maintainability_result,
                availability_result,
            )
            
            return GDSQualityResult(
                timestamp=timestamp,
                projection_name=projection_name,
                dependency_types=dependency_types,
                node_count=projection_info.node_count,
                relationship_count=projection_info.relationship_count,
                component_scores=component_scores,
                edge_criticality=edge_criticality,
                findings=all_findings,
                reliability_result=reliability_result,
                maintainability_result=maintainability_result,
                availability_result=availability_result,
                summary=summary,
                weights={
                    "reliability": self.weights[0],
                    "maintainability": self.weights[1],
                    "availability": self.weights[2],
                },
            )
        
        finally:
            # Cleanup projection
            self.gds.drop_projection(projection_name)
    
    def _compute_component_metrics(self, projection_name: str) -> Dict[str, GDSQualityMetrics]:
        """Compute GDS metrics for all components."""
        metrics = {}
        
        # Get centrality scores
        self.logger.info("Computing betweenness centrality...")
        betweenness_results = self.gds.betweenness(projection_name, weighted=True)
        
        self.logger.info("Computing PageRank...")
        pagerank_results = self.gds.pagerank(projection_name, weighted=True)
        
        self.logger.info("Computing degree centrality...")
        degree_results = self.gds.degree(projection_name, weighted=True, orientation="UNDIRECTED")
        in_degree_results = self.gds.degree(projection_name, weighted=True, orientation="REVERSE")
        out_degree_results = self.gds.degree(projection_name, weighted=True, orientation="NATURAL")
        
        # Get structural properties
        self.logger.info("Finding articulation points...")
        articulation_points = {ap["node_id"] for ap in self.gds.find_articulation_points()}
        
        self.logger.info("Finding bridge edges...")
        bridges = self.gds.find_bridges()
        bridge_endpoints = set()
        for bridge in bridges:
            bridge_endpoints.add(bridge.get("source_id", ""))
            bridge_endpoints.add(bridge.get("target_id", ""))
        
        # Get community assignments
        self.logger.info("Detecting communities...")
        community_results, _ = self.gds.weakly_connected_components(projection_name)
        community_map = {c.node_id: c.community_id for c in community_results}
        
        # Normalize scores
        max_bc = max([r.score for r in betweenness_results], default=1.0) or 1.0
        max_pr = max([r.score for r in pagerank_results], default=1.0) or 1.0
        max_degree = max([r.score for r in degree_results], default=1.0) or 1.0
        
        # Build metrics dictionary - start with all nodes from betweenness
        for result in betweenness_results:
            metrics[result.node_id] = GDSQualityMetrics(
                component_id=result.node_id,
                component_type=result.node_type,
                betweenness=result.score,
                betweenness_norm=result.score / max_bc,
                is_articulation_point=result.node_id in articulation_points,
                is_bridge_endpoint=result.node_id in bridge_endpoints,
                community_id=community_map.get(result.node_id, -1),
            )
        
        # Add PageRank scores
        for result in pagerank_results:
            if result.node_id in metrics:
                metrics[result.node_id].pagerank = result.score
                metrics[result.node_id].pagerank_norm = result.score / max_pr
        
        # Add degree scores
        for result in degree_results:
            if result.node_id in metrics:
                metrics[result.node_id].degree = result.score
                metrics[result.node_id].degree_norm = result.score / max_degree
        
        # Add in/out degree
        for result in in_degree_results:
            if result.node_id in metrics:
                metrics[result.node_id].in_degree = result.score
        
        for result in out_degree_results:
            if result.node_id in metrics:
                metrics[result.node_id].out_degree = result.score
        
        return metrics
    
    def _compute_component_scores(
        self, 
        component_metrics: Dict[str, GDSQualityMetrics]
    ) -> List[GDSComponentScore]:
        """Compute quality scores from GDS metrics."""
        scores = []
        
        # First, compute all scores
        reliability_scores = {}
        maintainability_scores = {}
        availability_scores = {}
        composite_scores = {}
        
        for comp_id, metrics in component_metrics.items():
            r_score = GDSCriticalityFormulas.compute_reliability_score(metrics)
            m_score = GDSCriticalityFormulas.compute_maintainability_score(metrics)
            a_score = GDSCriticalityFormulas.compute_availability_score(metrics)
            c_score = GDSCriticalityFormulas.compute_composite_score(
                r_score, m_score, a_score, self.weights
            )
            
            # Store in metrics object
            metrics.reliability_score = r_score
            metrics.maintainability_score = m_score
            metrics.availability_score = a_score
            metrics.composite_score = c_score
            
            reliability_scores[comp_id] = r_score
            maintainability_scores[comp_id] = m_score
            availability_scores[comp_id] = a_score
            composite_scores[comp_id] = c_score
        
        # Classify using box-plot
        r_levels = self._classify_scores(reliability_scores, "reliability")
        m_levels = self._classify_scores(maintainability_scores, "maintainability")
        a_levels = self._classify_scores(availability_scores, "availability")
        c_levels = self._classify_scores(composite_scores, "composite")
        
        # Build component scores
        for comp_id, metrics in component_metrics.items():
            scores.append(GDSComponentScore(
                component_id=comp_id,
                component_type=metrics.component_type,
                reliability_score=reliability_scores[comp_id],
                maintainability_score=maintainability_scores[comp_id],
                availability_score=availability_scores[comp_id],
                composite_score=composite_scores[comp_id],
                reliability_level=r_levels.get(comp_id, CriticalityLevel.MEDIUM),
                maintainability_level=m_levels.get(comp_id, CriticalityLevel.MEDIUM),
                availability_level=a_levels.get(comp_id, CriticalityLevel.MEDIUM),
                overall_level=c_levels.get(comp_id, CriticalityLevel.MEDIUM),
                metrics=metrics,
                findings=[],
            ))
        
        # Sort by composite score
        scores.sort(key=lambda x: x.composite_score, reverse=True)
        
        return scores
    
    def _classify_scores(
        self,
        scores: Dict[str, float],
        metric_name: str,
    ) -> Dict[str, CriticalityLevel]:
        """Classify scores using box-plot method."""
        items = [
            {"id": k, "type": "component", "score": v}
            for k, v in scores.items()
        ]
        
        if not items:
            return {}
        
        result = self.classifier.classify(items, metric_name=metric_name)
        return {item.id: item.level for item in result.items}
    
    def _compute_edge_criticality(
        self,
        projection_name: str,
        component_metrics: Dict[str, GDSQualityMetrics],
    ) -> List[GDSEdgeCriticality]:
        """Compute edge criticality scores."""
        edges = []
        
        # Get bridge edges
        bridges = self.gds.find_bridges()
        bridge_set = {(b.get("source_id"), b.get("target_id")) for b in bridges}
        
        # Query all DEPENDS_ON relationships
        with self.gds.session() as session:
            query = """
            MATCH (a)-[r:DEPENDS_ON]->(b)
            RETURN a.id AS source_id, b.id AS target_id, 
                   r.dependency_type AS dep_type, r.weight AS weight
            """
            
            for record in session.run(query):
                source_id = record["source_id"]
                target_id = record["target_id"]
                dep_type = record["dep_type"]
                weight = record["weight"] or 1.0
                
                # Get endpoint metrics
                source_metrics = component_metrics.get(source_id)
                target_metrics = component_metrics.get(target_id)
                
                if not source_metrics or not target_metrics:
                    continue
                
                # Compute edge scores based on endpoints
                is_bridge = (source_id, target_id) in bridge_set
                
                r_score = (
                    0.35 * (source_metrics.reliability_score + target_metrics.reliability_score) / 2 +
                    0.30 * weight +
                    0.35 * (1.0 if is_bridge else 0.0)
                )
                
                m_score = (
                    0.40 * weight +
                    0.30 * source_metrics.betweenness_norm +
                    0.30 * target_metrics.degree_norm
                )
                
                a_score = (
                    0.35 * max(source_metrics.availability_score, target_metrics.availability_score) +
                    0.35 * (source_metrics.pagerank_norm + target_metrics.pagerank_norm) / 2 +
                    0.30 * (1.0 if is_bridge else 0.0)
                )
                
                c_score = GDSCriticalityFormulas.compute_composite_score(
                    r_score, m_score, a_score, self.weights
                )
                
                edges.append(GDSEdgeCriticality(
                    source_id=source_id,
                    target_id=target_id,
                    dependency_type=dep_type,
                    weight=weight,
                    reliability_score=min(1.0, r_score),
                    maintainability_score=min(1.0, m_score),
                    availability_score=min(1.0, a_score),
                    composite_score=min(1.0, c_score),
                    is_bridge=is_bridge,
                ))
        
        # Sort by composite score
        edges.sort(key=lambda x: x.composite_score, reverse=True)
        
        return edges
    
    def _generate_summary(
        self,
        component_scores: List[GDSComponentScore],
        findings: List[Finding],
        reliability_result: Optional[AnalysisResult],
        maintainability_result: Optional[AnalysisResult],
        availability_result: Optional[AnalysisResult],
    ) -> Dict[str, Any]:
        """Generate summary statistics."""
        from collections import defaultdict
        
        # Count levels
        def count_levels(scores, attr):
            counts = defaultdict(int)
            for s in scores:
                level = getattr(s, f"{attr}_level")
                counts[level.value] += 1
            return dict(counts)
        
        # Count findings
        severity_counts = defaultdict(int)
        category_counts = defaultdict(int)
        
        for f in findings:
            severity_counts[f.severity.value] += 1
            category_counts[f.category] += 1
        
        # Average scores
        if component_scores:
            avg_r = statistics.mean([s.reliability_score for s in component_scores])
            avg_m = statistics.mean([s.maintainability_score for s in component_scores])
            avg_a = statistics.mean([s.availability_score for s in component_scores])
            avg_c = statistics.mean([s.composite_score for s in component_scores])
        else:
            avg_r = avg_m = avg_a = avg_c = 0
        
        # Analyzer scores
        analyzer_scores = {}
        if reliability_result:
            analyzer_scores["reliability"] = reliability_result.score
        if maintainability_result:
            analyzer_scores["maintainability"] = maintainability_result.score
        if availability_result:
            analyzer_scores["availability"] = availability_result.score
        
        return {
            "total_components": len(component_scores),
            "total_findings": len(findings),
            
            "average_scores": {
                "reliability": round(avg_r, 4),
                "maintainability": round(avg_m, 4),
                "availability": round(avg_a, 4),
                "composite": round(avg_c, 4),
            },
            
            "analyzer_scores": analyzer_scores,
            
            "levels": {
                "reliability": count_levels(component_scores, "reliability"),
                "maintainability": count_levels(component_scores, "maintainability"),
                "availability": count_levels(component_scores, "availability"),
                "overall": count_levels(component_scores, "overall"),
            },
            
            "findings_by_severity": dict(severity_counts),
            "findings_by_category": dict(category_counts),
            
            "top_critical_components": [
                {"id": s.component_id, "score": round(s.composite_score, 4)}
                for s in component_scores[:5]
            ],
            
            "health_score": round(1 - avg_c, 4),
        }