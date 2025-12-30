#!/usr/bin/env python3
"""
Software Quality Assessment Module

Assesses distributed pub-sub systems from three perspectives:
1. RELIABILITY - System's ability to perform correctly under stated conditions
2. MAINTAINABILITY - Ease of modifying, updating, or fixing the system  
3. AVAILABILITY - System's operational readiness and uptime

Each perspective has:
- Tailored criticality score formulation
- Specific graph metrics
- Custom simulation scenarios
- Validation approach
- Visualization dashboard

Author: Ibrahim Onuralp Yigit
Research: Graph-Based Modeling and Analysis of Distributed Pub-Sub Systems
"""

import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.simulation import SimulationGraph, FailureSimulator, ComponentType, Connection
from src.validation import GraphAnalyzer
from src.analysis import BoxPlotClassifier, CriticalityLevel

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


# =============================================================================
# Quality Attribute Enums
# =============================================================================

class QualityAttribute(Enum):
    """Software quality attributes for assessment."""
    RELIABILITY = "reliability"
    MAINTAINABILITY = "maintainability"
    AVAILABILITY = "availability"


class ProblemType(Enum):
    """Types of problems detected in the system."""
    # Reliability Problems
    SINGLE_POINT_OF_FAILURE = "single_point_of_failure"
    CASCADE_RISK = "cascade_risk"
    NO_REDUNDANCY = "no_redundancy"
    FRAGILE_PATH = "fragile_path"
    FAILURE_AMPLIFIER = "failure_amplifier"
    
    # Maintainability Problems
    GOD_TOPIC = "god_topic"
    CHATTY_APPLICATION = "chatty_application"
    TIGHT_COUPLING = "tight_coupling"
    HIDDEN_DEPENDENCY = "hidden_dependency"
    CHANGE_AMPLIFIER = "change_amplifier"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    
    # Availability Problems
    BOTTLENECK = "bottleneck"
    CAPACITY_RISK = "capacity_risk"
    NO_FAILOVER = "no_failover"
    SINGLE_ROUTE = "single_route"
    OVERLOADED_BROKER = "overloaded_broker"


class Severity(Enum):
    """Problem severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Problem:
    """Detected problem in the system."""
    problem_type: ProblemType
    quality_attribute: QualityAttribute
    severity: Severity
    component_id: str
    component_type: str
    description: str
    impact: float  # 0-1 scale
    recommendation: str
    related_components: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "problem_type": self.problem_type.value,
            "quality_attribute": self.quality_attribute.value,
            "severity": self.severity.value,
            "component_id": self.component_id,
            "component_type": self.component_type,
            "description": self.description,
            "impact": round(self.impact, 4),
            "recommendation": self.recommendation,
            "related_components": self.related_components,
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
        }


@dataclass
class EdgeCriticality:
    """Criticality assessment for an edge/connection."""
    source_id: str
    target_id: str
    edge_type: str
    reliability_score: float
    maintainability_score: float
    availability_score: float
    composite_score: float
    problems: List[Problem] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
            "reliability_score": round(self.reliability_score, 4),
            "maintainability_score": round(self.maintainability_score, 4),
            "availability_score": round(self.availability_score, 4),
            "composite_score": round(self.composite_score, 4),
            "problems": [p.to_dict() for p in self.problems],
        }


@dataclass
class ComponentQualityScore:
    """Quality scores for a component."""
    component_id: str
    component_type: str
    
    # Individual quality scores
    reliability_score: float
    maintainability_score: float
    availability_score: float
    
    # Composite score
    composite_score: float
    
    # Levels
    reliability_level: CriticalityLevel
    maintainability_level: CriticalityLevel
    availability_level: CriticalityLevel
    overall_level: CriticalityLevel
    
    # Component metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Detected problems
    problems: List[Problem] = field(default_factory=list)
    
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
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "problems": [p.to_dict() for p in self.problems],
        }


@dataclass
class QualityAssessmentResult:
    """Complete quality assessment result."""
    timestamp: str
    graph_stats: Dict[str, int]
    
    # Component scores
    component_scores: List[ComponentQualityScore]
    
    # Edge criticality
    edge_criticality: List[EdgeCriticality]
    
    # All detected problems
    problems: List[Problem]
    
    # Summary statistics
    summary: Dict[str, Any]
    
    # Simulation results
    simulation_results: Optional[Dict] = None
    
    # Validation results
    validation_results: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "graph_stats": self.graph_stats,
            "component_scores": [c.to_dict() for c in self.component_scores],
            "edge_criticality": [e.to_dict() for e in self.edge_criticality],
            "problems": [p.to_dict() for p in self.problems],
            "summary": self.summary,
            "simulation_results": self.simulation_results,
            "validation_results": self.validation_results,
        }


# =============================================================================
# Criticality Score Formulations
# =============================================================================

class CriticalityFormulas:
    """
    Composite Criticality Score Formulations for R/M/A Assessment.
    
    Each formula combines multiple graph metrics weighted by their
    relevance to the specific quality attribute.
    """
    
    # =========================================================================
    # RELIABILITY CRITICALITY SCORE
    # =========================================================================
    # 
    # C_reliability(v) = α₁·SPOF(v) + α₂·BC_norm(v) + α₃·CF(v) + α₄·(1-R(v)) + α₅·FPI(v)
    #
    # Where:
    #   SPOF(v)  = 1 if v is articulation point, 0 otherwise (Single Point of Failure)
    #   BC(v)    = Normalized betweenness centrality (failure propagation paths)
    #   CF(v)    = Cascade factor - expected cascade extent if v fails
    #   R(v)     = Redundancy factor - alternative paths available
    #   FPI(v)   = Failure Propagation Impact - reachability loss on failure
    #
    # Default weights: α₁=0.25, α₂=0.20, α₃=0.25, α₄=0.15, α₅=0.15
    # =========================================================================
    
    RELIABILITY_WEIGHTS = {
        "spof": 0.25,           # Single Point of Failure indicator
        "betweenness": 0.20,    # Failure propagation paths
        "cascade_factor": 0.25, # Cascade potential
        "redundancy": 0.15,     # Lack of redundancy (1 - redundancy)
        "failure_impact": 0.15, # Reachability loss on failure
    }
    
    # =========================================================================
    # MAINTAINABILITY CRITICALITY SCORE
    # =========================================================================
    #
    # C_maintainability(v) = β₁·DC_norm(v) + β₂·CP(v) + β₃·CC(v) + β₄·AP(v) + β₅·HD(v)
    #
    # Where:
    #   DC(v)  = Normalized degree centrality (coupling)
    #   CP(v)  = Change propagation factor - components affected by change
    #   CC(v)  = Cyclomatic complexity proxy (path diversity through v)
    #   AP(v)  = Anti-pattern indicator (god topic, chatty app, etc.)
    #   HD(v)  = Hidden dependency factor - indirect dependencies
    #
    # Default weights: β₁=0.25, β₂=0.25, β₃=0.15, β₄=0.20, β₅=0.15
    # =========================================================================
    
    MAINTAINABILITY_WEIGHTS = {
        "degree": 0.25,              # Coupling (high degree = hard to maintain)
        "change_propagation": 0.25,  # Change impact
        "complexity": 0.15,          # Path complexity
        "antipattern": 0.20,         # Anti-pattern severity
        "hidden_deps": 0.15,         # Hidden/indirect dependencies
    }
    
    # =========================================================================
    # AVAILABILITY CRITICALITY SCORE
    # =========================================================================
    #
    # C_availability(v) = γ₁·BN(v) + γ₂·LF(v) + γ₃·(1-FO(v)) + γ₄·CPL(v) + γ₅·SR(v)
    #
    # Where:
    #   BN(v)   = Bottleneck score (throughput constraint)
    #   LF(v)   = Load factor - traffic through component
    #   FO(v)   = Failover availability (1 - FO = no failover)
    #   CPL(v)  = Critical path length involvement
    #   SR(v)   = Single route factor - paths with no alternatives
    #
    # Default weights: γ₁=0.30, γ₂=0.20, γ₃=0.20, γ₄=0.15, γ₅=0.15
    # =========================================================================
    
    AVAILABILITY_WEIGHTS = {
        "bottleneck": 0.30,      # Throughput constraint
        "load_factor": 0.20,     # Traffic/load
        "failover": 0.20,        # Lack of failover (1 - failover)
        "critical_path": 0.15,   # Critical path involvement
        "single_route": 0.15,    # No alternative routes
    }
    
    @staticmethod
    def compute_reliability_score(metrics: Dict[str, float]) -> float:
        """
        Compute reliability criticality score.
        
        Formula:
        C_reliability = Σ αᵢ × metricᵢ
        """
        score = 0.0
        weights = CriticalityFormulas.RELIABILITY_WEIGHTS
        
        score += weights["spof"] * metrics.get("spof", 0)
        score += weights["betweenness"] * metrics.get("betweenness_norm", 0)
        score += weights["cascade_factor"] * metrics.get("cascade_factor", 0)
        score += weights["redundancy"] * (1 - metrics.get("redundancy", 0))
        score += weights["failure_impact"] * metrics.get("failure_impact", 0)
        
        return min(1.0, max(0.0, score))
    
    @staticmethod
    def compute_maintainability_score(metrics: Dict[str, float]) -> float:
        """
        Compute maintainability criticality score.
        
        Higher score = harder to maintain = more critical from maintenance view.
        """
        score = 0.0
        weights = CriticalityFormulas.MAINTAINABILITY_WEIGHTS
        
        score += weights["degree"] * metrics.get("degree_norm", 0)
        score += weights["change_propagation"] * metrics.get("change_propagation", 0)
        score += weights["complexity"] * metrics.get("complexity", 0)
        score += weights["antipattern"] * metrics.get("antipattern_score", 0)
        score += weights["hidden_deps"] * metrics.get("hidden_deps", 0)
        
        return min(1.0, max(0.0, score))
    
    @staticmethod
    def compute_availability_score(metrics: Dict[str, float]) -> float:
        """
        Compute availability criticality score.
        
        Higher score = greater availability risk = more critical.
        """
        score = 0.0
        weights = CriticalityFormulas.AVAILABILITY_WEIGHTS
        
        score += weights["bottleneck"] * metrics.get("bottleneck_score", 0)
        score += weights["load_factor"] * metrics.get("load_factor", 0)
        score += weights["failover"] * (1 - metrics.get("failover", 0))
        score += weights["critical_path"] * metrics.get("critical_path", 0)
        score += weights["single_route"] * metrics.get("single_route", 0)
        
        return min(1.0, max(0.0, score))
    
    @staticmethod
    def compute_composite_score(
        reliability: float,
        maintainability: float,
        availability: float,
        weights: Tuple[float, float, float] = (0.40, 0.25, 0.35),
    ) -> float:
        """
        Compute overall composite criticality score.
        
        Default weights prioritize reliability and availability
        over maintainability for operational systems.
        """
        w_r, w_m, w_a = weights
        return w_r * reliability + w_m * maintainability + w_a * availability


# =============================================================================
# Quality Metrics Calculator
# =============================================================================

class QualityMetricsCalculator:
    """
    Calculates quality-specific metrics for graph components and edges.
    """
    
    def __init__(self, graph: SimulationGraph):
        self.graph = graph
        self._build_networkx_graph()
        self._cache = {}
    
    def _build_networkx_graph(self):
        """Build NetworkX graph for advanced analysis."""
        if not HAS_NETWORKX:
            self.nx_graph = None
            return
        
        self.nx_graph = nx.DiGraph()
        
        for comp_id, comp in self.graph.components.items():
            self.nx_graph.add_node(comp_id, type=comp.type.value)
        
        for conn in self.graph.connections:
            self.nx_graph.add_edge(
                conn.source, 
                conn.target,
                weight=conn.weight,
                edge_type=conn.type.value
            )
    
    # =========================================================================
    # Reliability Metrics
    # =========================================================================
    
    def calculate_spof(self, component_id: str) -> float:
        """Check if component is a Single Point of Failure (articulation point)."""
        if not HAS_NETWORKX or self.nx_graph is None:
            return 0.0
        
        # Convert to undirected for articulation point detection
        undirected = self.nx_graph.to_undirected()
        try:
            aps = set(nx.articulation_points(undirected))
            return 1.0 if component_id in aps else 0.0
        except Exception:
            return 0.0
    
    def calculate_cascade_factor(self, component_id: str) -> float:
        """
        Calculate cascade factor - expected cascade extent if component fails.
        
        CF(v) = |reachable_from(v)| / |V|
        """
        reachable = self.graph.calculate_reachability(component_id)
        total = len(self.graph.components)
        return len(reachable) / total if total > 0 else 0.0
    
    def calculate_redundancy(self, component_id: str) -> float:
        """
        Calculate redundancy factor - availability of alternative paths.
        
        R(v) = average number of alternative paths / max possible alternatives
        """
        if not HAS_NETWORKX or self.nx_graph is None:
            return 0.0
        
        component = self.graph.get_component(component_id)
        if not component:
            return 0.0
        
        # For topics: count alternative routes through other brokers
        if component.type == ComponentType.TOPIC:
            brokers = self.graph.get_components_by_type(ComponentType.BROKER)
            if len(brokers) > 1:
                return min(1.0, (len(brokers) - 1) / len(brokers))
            return 0.0
        
        # For other components: count alternative paths to neighbors
        neighbors = self.graph.get_neighbors(component_id)
        if not neighbors:
            return 0.0
        
        alt_paths = 0
        for neighbor in neighbors:
            # Check for alternative paths (excluding direct connection)
            try:
                paths = list(nx.all_simple_paths(
                    self.nx_graph.to_undirected(), 
                    component_id, 
                    neighbor, 
                    cutoff=3
                ))
                if len(paths) > 1:
                    alt_paths += 1
            except Exception:
                pass
        
        return alt_paths / len(neighbors) if neighbors else 0.0
    
    def calculate_failure_impact(self, component_id: str) -> float:
        """
        Calculate failure propagation impact - reachability loss on failure.
        
        FPI(v) = |paths_lost| / |total_paths|
        """
        # Get current paths
        original_paths = self.graph.count_active_paths()
        if original_paths == 0:
            return 0.0
        
        # Simulate failure
        graph_copy = self.graph.copy()
        graph_copy.get_component(component_id).is_active = False
        
        remaining_paths = graph_copy.count_active_paths()
        paths_lost = original_paths - remaining_paths
        
        return paths_lost / original_paths
    
    # =========================================================================
    # Maintainability Metrics
    # =========================================================================
    
    def calculate_change_propagation(self, component_id: str) -> float:
        """
        Calculate change propagation factor.
        
        CP(v) = |components affected by change to v| / |V|
        """
        # Components affected = downstream dependencies
        affected = set()
        
        # BFS to find all downstream components
        queue = [component_id]
        visited = set()
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            for conn in self.graph.get_outgoing(current):
                if conn.target not in visited:
                    affected.add(conn.target)
                    queue.append(conn.target)
        
        total = len(self.graph.components)
        return len(affected) / total if total > 0 else 0.0
    
    def calculate_complexity(self, component_id: str) -> float:
        """
        Calculate complexity proxy based on path diversity through component.
        
        CC(v) = |distinct paths through v| / |total paths|
        """
        all_paths = self.graph.get_all_message_paths()
        if not all_paths:
            return 0.0
        
        paths_through = sum(1 for p in all_paths if component_id in p[3])
        return paths_through / len(all_paths)
    
    def calculate_antipattern_score(self, component_id: str) -> float:
        """
        Calculate anti-pattern score based on detected patterns.
        
        AP(v) = weighted sum of anti-pattern indicators
        """
        component = self.graph.get_component(component_id)
        if not component:
            return 0.0
        
        score = 0.0
        
        # God Topic: Topic with too many connections
        if component.type == ComponentType.TOPIC:
            publishers = len(self.graph.get_publishers(component_id))
            subscribers = len(self.graph.get_subscribers(component_id))
            total_apps = len(self.graph.get_components_by_type(ComponentType.APPLICATION))
            
            if total_apps > 0:
                connection_ratio = (publishers + subscribers) / total_apps
                if connection_ratio > 0.5:  # >50% of apps use this topic
                    score += 0.5
                if connection_ratio > 0.7:  # >70% = severe god topic
                    score += 0.3
        
        # Chatty Application: App with too many topic connections
        if component.type == ComponentType.APPLICATION:
            pub_topics = len(self.graph.get_published_topics(component_id))
            sub_topics = len(self.graph.get_subscribed_topics(component_id))
            total_topics = len(self.graph.get_components_by_type(ComponentType.TOPIC))
            
            if total_topics > 0:
                topic_ratio = (pub_topics + sub_topics) / total_topics
                if topic_ratio > 0.3:  # >30% of topics
                    score += 0.4
                if topic_ratio > 0.5:  # >50% = severe chatty
                    score += 0.3
        
        # Bottleneck Broker: Broker routing too many topics
        if component.type == ComponentType.BROKER:
            routed = len([c for c in self.graph.get_outgoing(component_id)])
            total_topics = len(self.graph.get_components_by_type(ComponentType.TOPIC))
            total_brokers = len(self.graph.get_components_by_type(ComponentType.BROKER))
            
            if total_topics > 0 and total_brokers > 0:
                expected_per_broker = total_topics / total_brokers
                if routed > expected_per_broker * 1.5:
                    score += 0.5
        
        return min(1.0, score)
    
    def calculate_hidden_deps(self, component_id: str) -> float:
        """
        Calculate hidden dependency factor.
        
        HD(v) = |indirect dependencies| / |total dependencies|
        """
        direct = self.graph.get_neighbors(component_id)
        
        # Find indirect dependencies (2+ hops away)
        indirect = set()
        for neighbor in direct:
            second_level = self.graph.get_neighbors(neighbor)
            indirect.update(second_level - direct - {component_id})
        
        total_deps = len(direct) + len(indirect)
        return len(indirect) / total_deps if total_deps > 0 else 0.0
    
    # =========================================================================
    # Availability Metrics
    # =========================================================================
    
    def calculate_bottleneck_score(self, component_id: str) -> float:
        """
        Calculate bottleneck score based on traffic flow concentration.
        
        BN(v) = normalized(|paths through v| × betweenness)
        """
        if "betweenness" not in self._cache:
            self._cache["betweenness"] = self._calculate_betweenness()
        
        bc = self._cache["betweenness"].get(component_id, 0)
        path_involvement = self.calculate_complexity(component_id)
        
        return min(1.0, bc * 0.6 + path_involvement * 0.4)
    
    def calculate_load_factor(self, component_id: str) -> float:
        """
        Calculate load factor based on connection count and message flow.
        
        LF(v) = normalized degree centrality
        """
        if "degree" not in self._cache:
            self._cache["degree"] = self._calculate_degree()
        
        return self._cache["degree"].get(component_id, 0)
    
    def calculate_failover(self, component_id: str) -> float:
        """
        Calculate failover availability.
        
        FO(v) = 1 if alternative paths exist, scaled by alternatives count
        """
        return self.calculate_redundancy(component_id)
    
    def calculate_critical_path(self, component_id: str) -> float:
        """
        Calculate critical path involvement.
        
        CPL(v) = fraction of shortest paths containing v
        """
        if not HAS_NETWORKX or self.nx_graph is None:
            return 0.0
        
        # Use betweenness as proxy for critical path involvement
        if "betweenness" not in self._cache:
            self._cache["betweenness"] = self._calculate_betweenness()
        
        return self._cache["betweenness"].get(component_id, 0)
    
    def calculate_single_route(self, component_id: str) -> float:
        """
        Calculate single route factor - paths with no alternatives.
        
        SR(v) = |paths with v as only route| / |paths through v|
        """
        all_paths = self.graph.get_all_message_paths()
        if not all_paths:
            return 0.0
        
        paths_through = [p for p in all_paths if component_id in p[3]]
        if not paths_through:
            return 0.0
        
        single_route_count = 0
        for pub, topic, sub, path in paths_through:
            # Check if there's an alternative path
            has_alternative = False
            if HAS_NETWORKX and self.nx_graph is not None:
                try:
                    # Remove component and check if path still exists
                    temp_graph = self.nx_graph.copy()
                    temp_graph.remove_node(component_id)
                    if nx.has_path(temp_graph, pub, sub):
                        has_alternative = True
                except Exception:
                    pass
            
            if not has_alternative:
                single_route_count += 1
        
        return single_route_count / len(paths_through)
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _calculate_betweenness(self) -> Dict[str, float]:
        """Calculate normalized betweenness centrality."""
        if not HAS_NETWORKX or self.nx_graph is None:
            return {}
        
        try:
            bc = nx.betweenness_centrality(self.nx_graph, normalized=True)
            return bc
        except Exception:
            return {}
    
    def _calculate_degree(self) -> Dict[str, float]:
        """Calculate normalized degree centrality."""
        if not HAS_NETWORKX or self.nx_graph is None:
            return {}
        
        try:
            dc = nx.degree_centrality(self.nx_graph)
            return dc
        except Exception:
            return {}
    
    def get_all_metrics(self, component_id: str) -> Dict[str, float]:
        """Get all quality metrics for a component."""
        # Precompute cached metrics
        if "betweenness" not in self._cache:
            self._cache["betweenness"] = self._calculate_betweenness()
        if "degree" not in self._cache:
            self._cache["degree"] = self._calculate_degree()
        
        return {
            # Reliability metrics
            "spof": self.calculate_spof(component_id),
            "betweenness_norm": self._cache["betweenness"].get(component_id, 0),
            "cascade_factor": self.calculate_cascade_factor(component_id),
            "redundancy": self.calculate_redundancy(component_id),
            "failure_impact": self.calculate_failure_impact(component_id),
            
            # Maintainability metrics
            "degree_norm": self._cache["degree"].get(component_id, 0),
            "change_propagation": self.calculate_change_propagation(component_id),
            "complexity": self.calculate_complexity(component_id),
            "antipattern_score": self.calculate_antipattern_score(component_id),
            "hidden_deps": self.calculate_hidden_deps(component_id),
            
            # Availability metrics
            "bottleneck_score": self.calculate_bottleneck_score(component_id),
            "load_factor": self.calculate_load_factor(component_id),
            "failover": self.calculate_failover(component_id),
            "critical_path": self.calculate_critical_path(component_id),
            "single_route": self.calculate_single_route(component_id),
        }


# =============================================================================
# Problem Detector
# =============================================================================

class ProblemDetector:
    """Detects problems affecting reliability, maintainability, and availability."""
    
    def __init__(
        self,
        graph: SimulationGraph,
        metrics_calculator: QualityMetricsCalculator,
    ):
        self.graph = graph
        self.calculator = metrics_calculator
    
    def detect_all_problems(self) -> List[Problem]:
        """Detect all problems across all quality attributes."""
        problems = []
        
        for comp_id in self.graph.components:
            problems.extend(self.detect_reliability_problems(comp_id))
            problems.extend(self.detect_maintainability_problems(comp_id))
            problems.extend(self.detect_availability_problems(comp_id))
        
        return problems
    
    def detect_reliability_problems(self, component_id: str) -> List[Problem]:
        """Detect reliability-related problems for a component."""
        problems = []
        component = self.graph.get_component(component_id)
        if not component:
            return problems
        
        comp_type = component.type.value
        
        # Check for Single Point of Failure
        if self.calculator.calculate_spof(component_id) > 0:
            cascade = self.calculator.calculate_cascade_factor(component_id)
            severity = Severity.CRITICAL if cascade > 0.3 else Severity.HIGH
            
            problems.append(Problem(
                problem_type=ProblemType.SINGLE_POINT_OF_FAILURE,
                quality_attribute=QualityAttribute.RELIABILITY,
                severity=severity,
                component_id=component_id,
                component_type=comp_type,
                description=f"Component is an articulation point - removal disconnects the graph",
                impact=cascade,
                recommendation="Add redundant component or alternative routing paths",
                metrics={"cascade_factor": cascade},
            ))
        
        # Check for Cascade Risk
        cascade_factor = self.calculator.calculate_cascade_factor(component_id)
        if cascade_factor > 0.4:
            severity = Severity.CRITICAL if cascade_factor > 0.6 else Severity.HIGH
            
            problems.append(Problem(
                problem_type=ProblemType.CASCADE_RISK,
                quality_attribute=QualityAttribute.RELIABILITY,
                severity=severity,
                component_id=component_id,
                component_type=comp_type,
                description=f"Failure could cascade to {cascade_factor:.0%} of system components",
                impact=cascade_factor,
                recommendation="Implement circuit breakers and failure isolation",
                metrics={"cascade_factor": cascade_factor},
            ))
        
        # Check for No Redundancy
        redundancy = self.calculator.calculate_redundancy(component_id)
        if redundancy < 0.2:
            failure_impact = self.calculator.calculate_failure_impact(component_id)
            if failure_impact > 0.2:
                problems.append(Problem(
                    problem_type=ProblemType.NO_REDUNDANCY,
                    quality_attribute=QualityAttribute.RELIABILITY,
                    severity=Severity.HIGH if failure_impact > 0.4 else Severity.MEDIUM,
                    component_id=component_id,
                    component_type=comp_type,
                    description="No alternative paths available - single point of routing",
                    impact=failure_impact,
                    recommendation="Add redundant routing through additional brokers/paths",
                    metrics={"redundancy": redundancy, "failure_impact": failure_impact},
                ))
        
        return problems
    
    def detect_maintainability_problems(self, component_id: str) -> List[Problem]:
        """Detect maintainability-related problems for a component."""
        problems = []
        component = self.graph.get_component(component_id)
        if not component:
            return problems
        
        comp_type = component.type.value
        
        # Check for God Topic
        if component.type == ComponentType.TOPIC:
            publishers = len(self.graph.get_publishers(component_id))
            subscribers = len(self.graph.get_subscribers(component_id))
            total_apps = len(self.graph.get_components_by_type(ComponentType.APPLICATION))
            
            if total_apps > 0:
                connection_ratio = (publishers + subscribers) / total_apps
                if connection_ratio > 0.5:
                    problems.append(Problem(
                        problem_type=ProblemType.GOD_TOPIC,
                        quality_attribute=QualityAttribute.MAINTAINABILITY,
                        severity=Severity.HIGH if connection_ratio > 0.7 else Severity.MEDIUM,
                        component_id=component_id,
                        component_type=comp_type,
                        description=f"Topic used by {connection_ratio:.0%} of applications",
                        impact=connection_ratio,
                        recommendation="Split into domain-specific topics or use topic hierarchies",
                        related_components=self.graph.get_publishers(component_id) + 
                                          self.graph.get_subscribers(component_id),
                        metrics={"publishers": publishers, "subscribers": subscribers},
                    ))
        
        # Check for Chatty Application
        if component.type == ComponentType.APPLICATION:
            pub_topics = len(self.graph.get_published_topics(component_id))
            sub_topics = len(self.graph.get_subscribed_topics(component_id))
            total_topics = len(self.graph.get_components_by_type(ComponentType.TOPIC))
            
            if total_topics > 0:
                topic_ratio = (pub_topics + sub_topics) / total_topics
                if topic_ratio > 0.3:
                    problems.append(Problem(
                        problem_type=ProblemType.CHATTY_APPLICATION,
                        quality_attribute=QualityAttribute.MAINTAINABILITY,
                        severity=Severity.HIGH if topic_ratio > 0.5 else Severity.MEDIUM,
                        component_id=component_id,
                        component_type=comp_type,
                        description=f"Application connects to {topic_ratio:.0%} of topics",
                        impact=topic_ratio,
                        recommendation="Decompose into smaller, focused microservices",
                        metrics={"pub_topics": pub_topics, "sub_topics": sub_topics},
                    ))
        
        # Check for Tight Coupling (high change propagation)
        change_prop = self.calculator.calculate_change_propagation(component_id)
        if change_prop > 0.3:
            problems.append(Problem(
                problem_type=ProblemType.TIGHT_COUPLING,
                quality_attribute=QualityAttribute.MAINTAINABILITY,
                severity=Severity.HIGH if change_prop > 0.5 else Severity.MEDIUM,
                component_id=component_id,
                component_type=comp_type,
                description=f"Changes affect {change_prop:.0%} of downstream components",
                impact=change_prop,
                recommendation="Introduce abstraction layers or event versioning",
                metrics={"change_propagation": change_prop},
            ))
        
        # Check for Hidden Dependencies
        hidden_deps = self.calculator.calculate_hidden_deps(component_id)
        if hidden_deps > 0.4:
            problems.append(Problem(
                problem_type=ProblemType.HIDDEN_DEPENDENCY,
                quality_attribute=QualityAttribute.MAINTAINABILITY,
                severity=Severity.MEDIUM,
                component_id=component_id,
                component_type=comp_type,
                description=f"{hidden_deps:.0%} of dependencies are indirect/hidden",
                impact=hidden_deps,
                recommendation="Document dependency chains and consider direct connections",
                metrics={"hidden_deps": hidden_deps},
            ))
        
        return problems
    
    def detect_availability_problems(self, component_id: str) -> List[Problem]:
        """Detect availability-related problems for a component."""
        problems = []
        component = self.graph.get_component(component_id)
        if not component:
            return problems
        
        comp_type = component.type.value
        
        # Check for Bottleneck
        bottleneck = self.calculator.calculate_bottleneck_score(component_id)
        if bottleneck > 0.5:
            problems.append(Problem(
                problem_type=ProblemType.BOTTLENECK,
                quality_attribute=QualityAttribute.AVAILABILITY,
                severity=Severity.HIGH if bottleneck > 0.7 else Severity.MEDIUM,
                component_id=component_id,
                component_type=comp_type,
                description=f"Component is a throughput bottleneck (score: {bottleneck:.2f})",
                impact=bottleneck,
                recommendation="Scale horizontally or add load balancing",
                metrics={"bottleneck_score": bottleneck},
            ))
        
        # Check for No Failover
        failover = self.calculator.calculate_failover(component_id)
        load = self.calculator.calculate_load_factor(component_id)
        if failover < 0.2 and load > 0.3:
            problems.append(Problem(
                problem_type=ProblemType.NO_FAILOVER,
                quality_attribute=QualityAttribute.AVAILABILITY,
                severity=Severity.HIGH,
                component_id=component_id,
                component_type=comp_type,
                description="High-load component with no failover mechanism",
                impact=load * (1 - failover),
                recommendation="Implement active-passive or active-active redundancy",
                metrics={"failover": failover, "load_factor": load},
            ))
        
        # Check for Single Route
        single_route = self.calculator.calculate_single_route(component_id)
        if single_route > 0.5:
            problems.append(Problem(
                problem_type=ProblemType.SINGLE_ROUTE,
                quality_attribute=QualityAttribute.AVAILABILITY,
                severity=Severity.MEDIUM if single_route < 0.8 else Severity.HIGH,
                component_id=component_id,
                component_type=comp_type,
                description=f"{single_route:.0%} of paths have no alternative route",
                impact=single_route,
                recommendation="Add redundant message paths through multiple brokers",
                metrics={"single_route": single_route},
            ))
        
        # Check for Overloaded Broker
        if component.type == ComponentType.BROKER:
            routed = len([c for c in self.graph.get_outgoing(component_id)])
            total_topics = len(self.graph.get_components_by_type(ComponentType.TOPIC))
            total_brokers = len(self.graph.get_components_by_type(ComponentType.BROKER))
            
            if total_brokers > 0 and total_topics > 0:
                expected = total_topics / total_brokers
                if routed > expected * 1.5:
                    overload_ratio = routed / expected
                    problems.append(Problem(
                        problem_type=ProblemType.OVERLOADED_BROKER,
                        quality_attribute=QualityAttribute.AVAILABILITY,
                        severity=Severity.HIGH if overload_ratio > 2 else Severity.MEDIUM,
                        component_id=component_id,
                        component_type=comp_type,
                        description=f"Broker handles {overload_ratio:.1f}x expected load",
                        impact=min(1.0, overload_ratio / 3),
                        recommendation="Redistribute topics across brokers or add capacity",
                        metrics={"topics_routed": routed, "expected": expected},
                    ))
        
        return problems


# =============================================================================
# Quality Assessor (Main Class)
# =============================================================================

class QualityAssessor:
    """
    Main class for assessing system quality from R/M/A perspectives.
    
    Integrates:
    - Criticality score calculation
    - Problem detection
    - Edge criticality analysis
    - Quality-specific simulations
    - Validation
    - Visualization
    """
    
    def __init__(
        self,
        graph: SimulationGraph,
        reliability_weight: float = 0.40,
        maintainability_weight: float = 0.25,
        availability_weight: float = 0.35,
    ):
        self.graph = graph
        self.weights = (reliability_weight, maintainability_weight, availability_weight)
        
        self.calculator = QualityMetricsCalculator(graph)
        self.detector = ProblemDetector(graph, self.calculator)
        self.classifier = BoxPlotClassifier(k_factor=1.5)
    
    def assess(self, run_simulation: bool = True) -> QualityAssessmentResult:
        """
        Run complete quality assessment.
        
        Returns comprehensive assessment covering:
        - Component criticality scores for R/M/A
        - Edge criticality
        - Problem detection
        - Simulation results (optional)
        - Validation metrics
        """
        timestamp = datetime.now().isoformat()
        
        # Graph statistics
        graph_stats = {
            "total_components": len(self.graph.components),
            "applications": len(self.graph.get_components_by_type(ComponentType.APPLICATION)),
            "topics": len(self.graph.get_components_by_type(ComponentType.TOPIC)),
            "brokers": len(self.graph.get_components_by_type(ComponentType.BROKER)),
            "nodes": len(self.graph.get_components_by_type(ComponentType.NODE)),
            "connections": len(self.graph.connections),
            "message_paths": len(self.graph.get_all_message_paths()),
        }
        
        # Calculate component scores
        component_scores = self._calculate_component_scores()
        
        # Calculate edge criticality
        edge_criticality = self._calculate_edge_criticality()
        
        # Detect problems
        problems = self.detector.detect_all_problems()
        
        # Assign problems to components
        for score in component_scores:
            score.problems = [p for p in problems if p.component_id == score.component_id]
        
        # Run simulations if requested
        simulation_results = None
        if run_simulation:
            simulation_results = self._run_quality_simulations()
        
        # Generate summary
        summary = self._generate_summary(component_scores, problems)
        
        return QualityAssessmentResult(
            timestamp=timestamp,
            graph_stats=graph_stats,
            component_scores=component_scores,
            edge_criticality=edge_criticality,
            problems=problems,
            summary=summary,
            simulation_results=simulation_results,
        )
    
    def _calculate_component_scores(self) -> List[ComponentQualityScore]:
        """Calculate quality scores for all components."""
        scores = []
        
        # Collect all scores for classification
        reliability_scores = {}
        maintainability_scores = {}
        availability_scores = {}
        composite_scores = {}
        
        for comp_id, component in self.graph.components.items():
            metrics = self.calculator.get_all_metrics(comp_id)
            
            r_score = CriticalityFormulas.compute_reliability_score(metrics)
            m_score = CriticalityFormulas.compute_maintainability_score(metrics)
            a_score = CriticalityFormulas.compute_availability_score(metrics)
            c_score = CriticalityFormulas.compute_composite_score(
                r_score, m_score, a_score, self.weights
            )
            
            reliability_scores[comp_id] = r_score
            maintainability_scores[comp_id] = m_score
            availability_scores[comp_id] = a_score
            composite_scores[comp_id] = c_score
        
        # Classify each dimension
        r_classified = self._classify_scores(reliability_scores, "reliability")
        m_classified = self._classify_scores(maintainability_scores, "maintainability")
        a_classified = self._classify_scores(availability_scores, "availability")
        c_classified = self._classify_scores(composite_scores, "composite")
        
        # Build results
        for comp_id, component in self.graph.components.items():
            metrics = self.calculator.get_all_metrics(comp_id)
            
            scores.append(ComponentQualityScore(
                component_id=comp_id,
                component_type=component.type.value,
                reliability_score=reliability_scores[comp_id],
                maintainability_score=maintainability_scores[comp_id],
                availability_score=availability_scores[comp_id],
                composite_score=composite_scores[comp_id],
                reliability_level=r_classified.get(comp_id, CriticalityLevel.MEDIUM),
                maintainability_level=m_classified.get(comp_id, CriticalityLevel.MEDIUM),
                availability_level=a_classified.get(comp_id, CriticalityLevel.MEDIUM),
                overall_level=c_classified.get(comp_id, CriticalityLevel.MEDIUM),
                metrics=metrics,
                problems=[],
            ))
        
        # Sort by composite score (descending)
        scores.sort(key=lambda x: x.composite_score, reverse=True)
        
        return scores
    
    def _classify_scores(
        self, 
        scores: Dict[str, float], 
        metric_name: str
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
    
    def _calculate_edge_criticality(self) -> List[EdgeCriticality]:
        """Calculate criticality for edges/connections."""
        edge_scores = []
        
        for conn in self.graph.connections:
            # Get source and target metrics
            source_metrics = self.calculator.get_all_metrics(conn.source)
            target_metrics = self.calculator.get_all_metrics(conn.target)
            
            # Edge reliability: based on endpoints and redundancy
            r_score = (
                0.4 * (source_metrics.get("failure_impact", 0) + 
                       target_metrics.get("failure_impact", 0)) / 2 +
                0.3 * conn.weight +
                0.3 * (1 - min(source_metrics.get("redundancy", 0),
                              target_metrics.get("redundancy", 0)))
            )
            
            # Edge maintainability: coupling strength
            m_score = (
                0.5 * conn.weight +
                0.25 * source_metrics.get("change_propagation", 0) +
                0.25 * target_metrics.get("hidden_deps", 0)
            )
            
            # Edge availability: bottleneck and load
            a_score = (
                0.4 * max(source_metrics.get("bottleneck_score", 0),
                         target_metrics.get("bottleneck_score", 0)) +
                0.3 * (source_metrics.get("load_factor", 0) + 
                       target_metrics.get("load_factor", 0)) / 2 +
                0.3 * max(source_metrics.get("single_route", 0),
                         target_metrics.get("single_route", 0))
            )
            
            c_score = CriticalityFormulas.compute_composite_score(
                r_score, m_score, a_score, self.weights
            )
            
            edge_scores.append(EdgeCriticality(
                source_id=conn.source,
                target_id=conn.target,
                edge_type=conn.type.value,
                reliability_score=min(1.0, r_score),
                maintainability_score=min(1.0, m_score),
                availability_score=min(1.0, a_score),
                composite_score=min(1.0, c_score),
            ))
        
        # Sort by composite score
        edge_scores.sort(key=lambda x: x.composite_score, reverse=True)
        
        return edge_scores
    
    def _run_quality_simulations(self) -> Dict[str, Any]:
        """Run quality-specific simulations."""
        simulator = FailureSimulator(
            cascade_threshold=0.5,
            cascade_probability=0.7,
            max_cascade_depth=5,
            seed=42,
        )
        
        results = {
            "reliability_simulation": {},
            "availability_simulation": {},
        }
        
        # Reliability simulation: failure impact analysis
        batch = simulator.simulate_all_failures(self.graph, enable_cascade=True)
        
        results["reliability_simulation"] = {
            "total_simulations": len(batch.results),
            "critical_components": batch.critical_components[:10],
            "average_impact": statistics.mean([r.impact.impact_score for r in batch.results]),
            "max_impact": max([r.impact.impact_score for r in batch.results]),
            "cascade_prone": [
                r.primary_failures[0] for r in batch.results 
                if len(r.cascade_failures) > 2
            ][:5],
        }
        
        # Availability simulation: bottleneck stress test
        # Simulate removing high-load components
        high_load = sorted(
            self.graph.components.keys(),
            key=lambda x: self.calculator.calculate_load_factor(x),
            reverse=True
        )[:5]
        
        availability_impacts = []
        for comp_id in high_load:
            result = simulator.simulate_failure(self.graph, comp_id, enable_cascade=False)
            availability_impacts.append({
                "component": comp_id,
                "paths_lost": result.impact.paths_lost,
                "reachability_loss": result.impact.reachability_loss,
            })
        
        results["availability_simulation"] = {
            "high_load_components": high_load,
            "availability_impacts": availability_impacts,
        }
        
        return results
    
    def _generate_summary(
        self,
        component_scores: List[ComponentQualityScore],
        problems: List[Problem],
    ) -> Dict[str, Any]:
        """Generate summary statistics."""
        # Count by level for each attribute
        def count_levels(scores, attr):
            counts = defaultdict(int)
            for s in scores:
                level = getattr(s, f"{attr}_level")
                counts[level.value] += 1
            return dict(counts)
        
        # Count problems by type and severity
        problem_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        attribute_counts = defaultdict(int)
        
        for p in problems:
            problem_counts[p.problem_type.value] += 1
            severity_counts[p.severity.value] += 1
            attribute_counts[p.quality_attribute.value] += 1
        
        # Calculate average scores
        avg_reliability = statistics.mean([s.reliability_score for s in component_scores])
        avg_maintainability = statistics.mean([s.maintainability_score for s in component_scores])
        avg_availability = statistics.mean([s.availability_score for s in component_scores])
        avg_composite = statistics.mean([s.composite_score for s in component_scores])
        
        return {
            "total_components": len(component_scores),
            "total_problems": len(problems),
            
            "average_scores": {
                "reliability": round(avg_reliability, 4),
                "maintainability": round(avg_maintainability, 4),
                "availability": round(avg_availability, 4),
                "composite": round(avg_composite, 4),
            },
            
            "levels": {
                "reliability": count_levels(component_scores, "reliability"),
                "maintainability": count_levels(component_scores, "maintainability"),
                "availability": count_levels(component_scores, "availability"),
                "overall": count_levels(component_scores, "overall"),
            },
            
            "problems_by_type": dict(problem_counts),
            "problems_by_severity": dict(severity_counts),
            "problems_by_attribute": dict(attribute_counts),
            
            "top_critical_components": [
                {"id": s.component_id, "score": round(s.composite_score, 4)}
                for s in component_scores[:5]
            ],
            
            "health_score": round(1 - avg_composite, 4),  # Inverse of criticality
        }


# =============================================================================
# Quality Dashboard Generator
# =============================================================================

class QualityDashboardGenerator:
    """Generate interactive HTML dashboard for quality assessment."""
    
    def generate(self, result: QualityAssessmentResult) -> str:
        """Generate complete HTML dashboard."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>System Quality Assessment Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f7fa; color: #333; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 12px; margin-bottom: 20px; }}
        .header h1 {{ font-size: 2em; margin-bottom: 10px; }}
        .header .subtitle {{ opacity: 0.9; }}
        
        .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }}
        .card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
        .card.reliability {{ border-left: 4px solid #3498db; }}
        .card.maintainability {{ border-left: 4px solid #9b59b6; }}
        .card.availability {{ border-left: 4px solid #2ecc71; }}
        .card.health {{ border-left: 4px solid #e74c3c; }}
        .card h3 {{ color: #666; font-size: 0.9em; text-transform: uppercase; margin-bottom: 8px; }}
        .card .value {{ font-size: 2em; font-weight: bold; }}
        .card .subvalue {{ color: #888; font-size: 0.85em; }}
        
        .section {{ background: white; padding: 25px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); margin-bottom: 20px; }}
        .section h2 {{ margin-bottom: 20px; color: #444; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        
        .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
        .chart-container {{ position: relative; height: 300px; }}
        
        .problems-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 15px; }}
        .problem {{ background: #fafafa; padding: 15px; border-radius: 8px; border-left: 4px solid #e74c3c; }}
        .problem.high {{ border-color: #e67e22; }}
        .problem.medium {{ border-color: #f1c40f; }}
        .problem.low {{ border-color: #95a5a6; }}
        .problem h4 {{ margin-bottom: 8px; }}
        .problem .type {{ color: #666; font-size: 0.85em; }}
        .problem .recommendation {{ background: #e8f4f8; padding: 8px; border-radius: 4px; margin-top: 10px; font-size: 0.9em; }}
        
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; color: #555; }}
        tr:hover {{ background: #f8f9fa; }}
        
        .badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.8em; font-weight: 500; }}
        .badge.critical {{ background: #e74c3c; color: white; }}
        .badge.high {{ background: #e67e22; color: white; }}
        .badge.medium {{ background: #f1c40f; color: #333; }}
        .badge.low {{ background: #3498db; color: white; }}
        .badge.minimal {{ background: #95a5a6; color: white; }}
        
        .score-bar {{ height: 8px; background: #eee; border-radius: 4px; overflow: hidden; }}
        .score-fill {{ height: 100%; border-radius: 4px; }}
        .score-fill.r {{ background: linear-gradient(90deg, #3498db, #2980b9); }}
        .score-fill.m {{ background: linear-gradient(90deg, #9b59b6, #8e44ad); }}
        .score-fill.a {{ background: linear-gradient(90deg, #2ecc71, #27ae60); }}
        
        .tabs {{ display: flex; gap: 10px; margin-bottom: 20px; }}
        .tab {{ padding: 10px 20px; background: #eee; border-radius: 8px; cursor: pointer; }}
        .tab.active {{ background: #667eea; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 System Quality Assessment</h1>
            <div class="subtitle">Reliability • Maintainability • Availability Analysis</div>
            <div class="subtitle" style="margin-top: 10px; opacity: 0.7;">Generated: {result.timestamp}</div>
        </div>
        
        <!-- Summary Cards -->
        <div class="cards">
            <div class="card reliability">
                <h3>Reliability Score</h3>
                <div class="value">{result.summary['average_scores']['reliability']:.2f}</div>
                <div class="subvalue">Avg criticality (lower is better)</div>
            </div>
            <div class="card maintainability">
                <h3>Maintainability Score</h3>
                <div class="value">{result.summary['average_scores']['maintainability']:.2f}</div>
                <div class="subvalue">Avg criticality (lower is better)</div>
            </div>
            <div class="card availability">
                <h3>Availability Score</h3>
                <div class="value">{result.summary['average_scores']['availability']:.2f}</div>
                <div class="subvalue">Avg criticality (lower is better)</div>
            </div>
            <div class="card health">
                <h3>System Health</h3>
                <div class="value">{result.summary['health_score']:.0%}</div>
                <div class="subvalue">{result.summary['total_problems']} problems detected</div>
            </div>
        </div>
        
        <!-- Charts -->
        <div class="section">
            <h2>📊 Quality Distribution</h2>
            <div class="charts">
                <div class="chart-container">
                    <canvas id="radarChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="problemsChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Problems -->
        <div class="section">
            <h2>⚠️ Detected Problems ({result.summary['total_problems']})</h2>
            <div class="problems-grid">
                {self._generate_problems_html(result.problems[:12])}
            </div>
        </div>
        
        <!-- Component Table -->
        <div class="section">
            <h2>📋 Component Criticality Scores</h2>
            <table>
                <thead>
                    <tr>
                        <th>Component</th>
                        <th>Type</th>
                        <th>Reliability</th>
                        <th>Maintainability</th>
                        <th>Availability</th>
                        <th>Composite</th>
                        <th>Level</th>
                        <th>Problems</th>
                    </tr>
                </thead>
                <tbody>
                    {self._generate_component_rows(result.component_scores[:20])}
                </tbody>
            </table>
        </div>
        
        <!-- Edge Criticality -->
        <div class="section">
            <h2>🔗 Critical Edges</h2>
            <table>
                <thead>
                    <tr>
                        <th>Source</th>
                        <th>Target</th>
                        <th>Type</th>
                        <th>Reliability</th>
                        <th>Maintainability</th>
                        <th>Availability</th>
                        <th>Composite</th>
                    </tr>
                </thead>
                <tbody>
                    {self._generate_edge_rows(result.edge_criticality[:15])}
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        // Radar Chart
        new Chart(document.getElementById('radarChart'), {{
            type: 'radar',
            data: {{
                labels: ['Reliability', 'Maintainability', 'Availability'],
                datasets: [{{
                    label: 'Criticality Scores',
                    data: [
                        {result.summary['average_scores']['reliability']},
                        {result.summary['average_scores']['maintainability']},
                        {result.summary['average_scores']['availability']}
                    ],
                    backgroundColor: 'rgba(102, 126, 234, 0.2)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 2
                }}]
            }},
            options: {{
                scales: {{
                    r: {{
                        beginAtZero: true,
                        max: 1
                    }}
                }}
            }}
        }});
        
        // Problems by Attribute Chart
        new Chart(document.getElementById('problemsChart'), {{
            type: 'doughnut',
            data: {{
                labels: ['Reliability', 'Maintainability', 'Availability'],
                datasets: [{{
                    data: [
                        {result.summary['problems_by_attribute'].get('reliability', 0)},
                        {result.summary['problems_by_attribute'].get('maintainability', 0)},
                        {result.summary['problems_by_attribute'].get('availability', 0)}
                    ],
                    backgroundColor: ['#3498db', '#9b59b6', '#2ecc71']
                }}]
            }},
            options: {{
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Problems by Quality Attribute'
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
    
    def _generate_problems_html(self, problems: List[Problem]) -> str:
        html = ""
        for p in problems:
            severity_class = p.severity.value
            html += f"""
            <div class="problem {severity_class}">
                <h4>{p.component_id}</h4>
                <div class="type">{p.problem_type.value.replace('_', ' ').title()} • {p.quality_attribute.value.title()}</div>
                <p style="margin-top: 8px;">{p.description}</p>
                <div class="recommendation">💡 {p.recommendation}</div>
            </div>
            """
        return html
    
    def _generate_component_rows(self, scores: List[ComponentQualityScore]) -> str:
        html = ""
        for s in scores:
            html += f"""
            <tr>
                <td><strong>{s.component_id}</strong></td>
                <td>{s.component_type}</td>
                <td>
                    <div class="score-bar"><div class="score-fill r" style="width: {s.reliability_score*100}%"></div></div>
                    <small>{s.reliability_score:.3f}</small>
                </td>
                <td>
                    <div class="score-bar"><div class="score-fill m" style="width: {s.maintainability_score*100}%"></div></div>
                    <small>{s.maintainability_score:.3f}</small>
                </td>
                <td>
                    <div class="score-bar"><div class="score-fill a" style="width: {s.availability_score*100}%"></div></div>
                    <small>{s.availability_score:.3f}</small>
                </td>
                <td><strong>{s.composite_score:.3f}</strong></td>
                <td><span class="badge {s.overall_level.value.lower()}">{s.overall_level.value}</span></td>
                <td>{len(s.problems)}</td>
            </tr>
            """
        return html
    
    def _generate_edge_rows(self, edges: List[EdgeCriticality]) -> str:
        html = ""
        for e in edges:
            html += f"""
            <tr>
                <td>{e.source_id}</td>
                <td>{e.target_id}</td>
                <td>{e.edge_type}</td>
                <td>{e.reliability_score:.3f}</td>
                <td>{e.maintainability_score:.3f}</td>
                <td>{e.availability_score:.3f}</td>
                <td><strong>{e.composite_score:.3f}</strong></td>
            </tr>
            """
        return html


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """CLI interface for quality assessment."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Assess system quality from R/M/A perspectives"
    )
    parser.add_argument("--input", "-i", required=True, help="Input graph JSON")
    parser.add_argument("--output", "-o", default="output/quality", help="Output directory")
    parser.add_argument("--no-simulation", action="store_true", help="Skip simulation")
    parser.add_argument("--reliability-weight", type=float, default=0.40)
    parser.add_argument("--maintainability-weight", type=float, default=0.25)
    parser.add_argument("--availability-weight", type=float, default=0.35)
    
    args = parser.parse_args()
    
    # Load graph
    from src.core import generate_graph
    
    input_path = Path(args.input)
    if input_path.exists():
        graph = SimulationGraph.from_json(input_path)
    else:
        print(f"Generating sample graph.")
        data = generate_graph(scale="medium", scenario="iot", seed=42)
        graph = SimulationGraph.from_dict(data)
    
    # Run assessment
    assessor = QualityAssessor(
        graph,
        reliability_weight=args.reliability_weight,
        maintainability_weight=args.maintainability_weight,
        availability_weight=args.availability_weight,
    )
    
    print("Running quality assessment.")
    result = assessor.assess(run_simulation=not args.no_simulation)
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON results
    json_path = output_dir / "quality_assessment.json"
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"Saved JSON: {json_path}")
    
    # Dashboard
    dashboard = QualityDashboardGenerator()
    html = dashboard.generate(result)
    html_path = output_dir / "quality_dashboard.html"
    html_path.write_text(html)
    print(f"Saved dashboard: {html_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("QUALITY ASSESSMENT SUMMARY")
    print("="*60)
    print(f"Components analyzed: {result.summary['total_components']}")
    print(f"Problems detected: {result.summary['total_problems']}")
    print(f"\nAverage Scores (lower = less critical = better):")
    print(f"  Reliability:     {result.summary['average_scores']['reliability']:.4f}")
    print(f"  Maintainability: {result.summary['average_scores']['maintainability']:.4f}")
    print(f"  Availability:    {result.summary['average_scores']['availability']:.4f}")
    print(f"  Composite:       {result.summary['average_scores']['composite']:.4f}")
    print(f"\nSystem Health Score: {result.summary['health_score']:.0%}")
    print("="*60)


if __name__ == "__main__":
    main()
