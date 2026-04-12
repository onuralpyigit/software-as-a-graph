"""
Anti-Pattern Detector for Pub-Sub Architectural Systems

[DEPRECATED] This module is preserved for backward compatibility. 
New implementations should use src.analysis.antipattern_detector.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Set, Optional

import networkx as nx
from src.core.metrics import ComponentQuality, EdgeQuality
from .models import QualityAnalysisResult, DetectedProblem
from src.core.criticality import CriticalityLevel

class AntiPatternTier(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"

class AntiPatternDetector:
    """
    Specialized detector for pub-sub architectural anti-patterns.
    """

    def __init__(self, quality_result: QualityAnalysisResult):
        self.quality = quality_result
        self.components = {c.id: c for c in quality_result.components}
        self.edges = quality_result.edges
        self.graph = self._build_graph()
        self.stats = self._compute_stats()

    def _compute_stats(self) -> Dict[str, Any]:
        """Compute relative thresholds (aligned with canonical engine)."""
        import statistics
        pts = [c.structural.pagerank for c in self.components.values()]
        bts = [c.structural.betweenness for c in self.components.values()]
        ids = [c.structural.in_degree_raw for c in self.components.values()]
        
        def _fence(v):
            if not v: return 0.0
            sv = sorted(v)
            n = len(sv)
            q1, q3 = sv[n // 4], sv[min(3 * n // 4, n - 1)]
            return q3 + 1.5 * (q3 - q1)

        return {
            "pagerank_fence": _fence(pts),
            "betweenness_fence": _fence(bts),
            "in_degree_fence": _fence(ids),
            "avg_pr": statistics.mean(pts) if pts else 0.0
        }

    def _build_graph(self) -> nx.DiGraph:
        """Build a NetworkX directed graph from analysis edges."""
        G = nx.DiGraph()
        for e in self.edges:
            G.add_edge(e.source, e.target, **e.to_dict())
        return G

    def detect_all(self) -> List[DetectedProblem]:
        """Run all 12 anti-pattern detection rules."""
        findings = []
        
        # CRITICAL
        findings.extend(self._detect_spof())
        findings.extend(self._detect_systemic_risk())
        findings.extend(self._detect_cyclic_dependency())
        
        # HIGH
        findings.extend(self._detect_god_component())
        findings.extend(self._detect_bottleneck_edge())
        findings.extend(self._detect_broker_overload())
        findings.extend(self._detect_deep_pipeline())
        
        # MEDIUM
        findings.extend(self._detect_topic_fanout())
        findings.extend(self._detect_chatty_pair())
        findings.extend(self._detect_qos_mismatch())
        findings.extend(self._detect_orphaned_topic())
        findings.extend(self._detect_unstable_interface())
        
        # Sort by severity priority or as needed
        return findings

    # --- CRITICAL TIER ---

    def _detect_spof(self) -> List[DetectedProblem]:
        """SPOF: Single Point of Failure — structural graph cut vertex."""
        problems = []
        for c in self.components.values():
            if getattr(c.structural, 'is_directed_ap', False) or c.structural.is_articulation_point:
                problems.append(DetectedProblem(
                    entity_id=c.id, entity_type="Component",
                    category="Availability", severity="CRITICAL",
                    name="SPOF",
                    description=f"'{c.id}' is a single point of failure (cut vertex).",
                    recommendation="Introduce redundancy: backup instances, alternative paths, or event-driven decoupling.",
                    evidence={"is_directed": getattr(c.structural, 'is_directed_ap', False), "availability_score": c.scores.availability}
                ))
        return problems

    def _detect_systemic_risk(self) -> List[DetectedProblem]:
        """SYSTEMIC_RISK: Correlated failure cluster — CRITICAL clique."""
        problems = []
        total = len(self.components)
        if total == 0: return []
        
        critical_nodes = [c for c in self.components.values() if c.levels.overall == CriticalityLevel.CRITICAL]
        crit_ratio = len(critical_nodes) / total
        
        if crit_ratio > 0.2:
            problems.append(DetectedProblem(
                entity_id="SYSTEM", entity_type="System",
                category="Reliability", severity="CRITICAL",
                name="SYSTEMIC_RISK",
                description=f"Over 20% of system components ({crit_ratio:.0%}) are classified as CRITICAL.",
                recommendation="Comprehensive architectural review required. Decouple core hubs and distribute responsibilities.",
                evidence={"critical_ratio": crit_ratio, "critical_count": len(critical_nodes)}
            ))
        return problems

    def _detect_cyclic_dependency(self) -> List[DetectedProblem]:
        """CYCLIC_DEPENDENCY: Circular pub-sub feedback loop (SCC > 1)."""
        problems = []
        sccs = list(nx.strongly_connected_components(self.graph))
        for scc in sccs:
            if len(scc) > 1:
                path = " -> ".join(list(scc) + [list(scc)[0]])
                problems.append(DetectedProblem(
                    entity_id=path, entity_type="Architecture",
                    category="Maintainability", severity="CRITICAL",
                    name="CYCLIC_DEPENDENCY",
                    description=f"Circular dependency detected among: {', '.join(scc)}.",
                    recommendation="Break the cycle by introducing an abstraction layer or using asynchronous eventing.",
                    evidence={"scc_size": len(scc), "nodes": list(scc)}
                ))
        return problems

    # --- HIGH TIER ---

    def _detect_god_component(self) -> List[DetectedProblem]:
        """GOD_COMPONENT: Dependency magnet — absorbs too many responsibilities."""
        problems = []
        # Issue #8: Use relative fences instead of absolute 0.3/10
        bt_fence = self.stats["betweenness_fence"]
        id_fence = self.stats["in_degree_fence"]
        
        for c in self.components.values():
            if c.structural.betweenness > bt_fence and c.structural.in_degree_raw > id_fence:
                problems.append(DetectedProblem(
                    entity_id=c.id, entity_type="Component",
                    category="Maintainability", severity="HIGH",
                    name="GOD_COMPONENT",
                    description=f"'{c.id}' acts as a dependency magnet (centrality > fence).",
                    recommendation="Apply the Single Responsibility Principle: decompose into smaller, focused services.",
                    evidence={"betweenness": c.structural.betweenness, "in_degree": c.structural.in_degree_raw}
                ))
        return problems

    def _detect_bottleneck_edge(self) -> List[DetectedProblem]:
        """BOTTLENECK_EDGE: High-traffic bridge with no redundant path."""
        problems = []
        for e in self.edges:
            if e.structural and e.structural.is_bridge and e.structural.betweenness > 0.2:
                problems.append(DetectedProblem(
                    entity_id=e.id, entity_type="Edge",
                    category="Maintainability", severity="HIGH",
                    name="BOTTLENECK_EDGE",
                    description=f"Edge '{e.id}' is a bridge carrying significant traffic with no alternatives.",
                    recommendation="Add redundant connections or introduce a load balancer/broker to distribute the load.",
                    evidence={"is_bridge": True, "edge_betweenness": e.structural.betweenness}
                ))
        return problems

    def _detect_broker_overload(self) -> List[DetectedProblem]:
        """BROKER_OVERLOAD: Broker saturation — disproportionate routing share."""
        problems = []
        brokers = [c for c in self.components.values() if c.type.lower() == "broker"]
        if not brokers: return []
        
        # Issue #9: Aligned gate using PageRank fence
        pr_fence = self.stats["pagerank_fence"]
        avg_pr = self.stats["avg_pr"]
        
        for b in brokers:
            if b.structural.pagerank > pr_fence:
                problems.append(DetectedProblem(
                    entity_id=b.id, entity_type="Component",
                    category="Availability", severity="HIGH",
                    name="BROKER_OVERLOAD",
                    description=f"Broker '{b.id}' handles a disproportionate share of system routing.",
                    recommendation="Shard topics across additional brokers or implement broker clustering.",
                    evidence={"pagerank": b.structural.pagerank, "avg_broker_pagerank": avg_pr}
                ))
        return problems

    def _detect_deep_pipeline(self) -> List[DetectedProblem]:
        """DEEP_PIPELINE: Excessive processing chain depth — latency amplifier."""
        problems = []
        # Simple longest path in DAG (or approximate for general graphs)
        # Here we just look for nodes at the end of long paths
        try:
            dag = self.graph.copy()
            # Remove cycles to make it a DAG for path analysis
            cycles = list(nx.simple_cycles(dag))
            for cycle in cycles:
                if len(cycle) > 1:
                    dag.remove_edge(cycle[0], cycle[1])
            
            for node in dag.nodes():
                # This is computationally expensive for large graphs, but works for typical system layers
                longest_path_len = nx.dag_longest_path_length(dag.subgraph(nx.ancestors(dag, node) | {node}))
                if longest_path_len > 5:
                    problems.append(DetectedProblem(
                        entity_id=node, entity_type="Component",
                        category="Reliability", severity="HIGH",
                        name="DEEP_PIPELINE",
                        description=f"'{node}' is at the end of an excessive dependency chain (depth > 5).",
                        recommendation="Flatten the architecture using direct events or consolidate intermediate processing steps.",
                        evidence={"pipeline_depth": longest_path_len}
                    ))
        except:
            pass
        return problems

    # --- MEDIUM TIER ---

    def _detect_topic_fanout(self) -> List[DetectedProblem]:
        """TOPIC_FANOUT: Topic fan-out explosion — broadcast blast radius."""
        problems = []
        topics = [c for c in self.components.values() if c.type.lower() == "topic"]
        if not topics: return []
        
        avg_od = sum(t.structural.out_degree_raw for t in topics) / len(topics)
        for t in topics:
            if t.structural.out_degree_raw > max(avg_od * 3, 10):
                problems.append(DetectedProblem(
                    entity_id=t.id, entity_type="Component",
                    category="Reliability", severity="MEDIUM",
                    name="TOPIC_FANOUT",
                    description=f"Topic '{t.id}' has an excessive number of subscribers ({t.structural.out_degree_raw}).",
                    recommendation="Partition the topic into more granular sub-topics or use message filtering.",
                    evidence={"out_degree": t.structural.out_degree_raw, "avg_topic_out_degree": avg_od}
                ))
        return problems

    def _detect_chatty_pair(self) -> List[DetectedProblem]:
        """CHATTY_PAIR: Bidirectional tight coupling through topics."""
        problems = []
        checked_pairs = set()
        for u, v in self.graph.edges():
            pair = tuple(sorted((u, v)))
            if pair in checked_pairs: continue
            checked_pairs.add(pair)
            
            if self.graph.has_edge(v, u):
                # Check for "chatty" behavior - high mutual traffic/dependency
                # For now, any bidirectional link between apps is a smell
                u_comp = self.components.get(u)
                v_comp = self.components.get(v)
                if u_comp and v_comp and u_comp.type.lower() == "application" and v_comp.type.lower() == "application":
                    problems.append(DetectedProblem(
                        entity_id=f"{u} <-> {v}", entity_type="Architecture",
                        category="Maintainability", severity="MEDIUM",
                        name="CHATTY_PAIR",
                        description=f"Mutual dependency detected between '{u}' and '{v}'.",
                        recommendation="Unify the components or move shared logic to a common library/service.",
                        evidence={"nodes": [u, v]}
                    ))
        return problems

    def _detect_qos_mismatch(self) -> List[DetectedProblem]:
        """QOS_MISMATCH: Publisher/subscriber QoS incompatibility."""
        problems = []
        # This requires looking at the raw graph structure or metadata
        # Assuming for now we can infer from component weights or edge types
        # Placeholder for real QoS logic
        return problems

    def _detect_orphaned_topic(self) -> List[DetectedProblem]:
        """ORPHANED_TOPIC: Topic with no publishers OR no subscribers."""
        problems = []
        for c in self.components.values():
            if c.type.lower() == "topic":
                if c.structural.in_degree_raw == 0:
                    problems.append(DetectedProblem(
                        entity_id=c.id, entity_type="Component",
                        category="Availability", severity="MEDIUM",
                        name="ORPHANED_TOPIC",
                        description=f"Topic '{c.id}' has no publishers.",
                        recommendation="Remove the unused topic or verify publisher configuration.",
                        evidence={"in_degree": 0}
                    ))
                elif c.structural.out_degree_raw == 0:
                    problems.append(DetectedProblem(
                        entity_id=c.id, entity_type="Component",
                        category="Availability", severity="MEDIUM",
                        name="ORPHANED_TOPIC",
                        description=f"Topic '{c.id}' has no subscribers.",
                        recommendation="Remove the unused topic or verify subscriber configuration.",
                        evidence={"out_degree": 0}
                    ))
        return problems

    def _detect_unstable_interface(self) -> List[DetectedProblem]:
        """UNSTABLE_INTERFACE: High churn potential — extreme coupling imbalance."""
        problems = []
        for c in self.components.values():
            # Using coupling risk from QualityAnalyzer logic (instability imbalance)
            id_n = c.structural.in_degree_raw
            od_n = c.structural.out_degree_raw
            _eps = 1e-9
            instability = od_n / (id_n + od_n + _eps)
            coupling_risk = 1.0 - abs(2.0 * instability - 1.0)
            
            if coupling_risk > 0.8 and (id_n + od_n) > 5:
                problems.append(DetectedProblem(
                    entity_id=c.id, entity_type="Component",
                    category="Maintainability", severity="MEDIUM",
                    name="UNSTABLE_INTERFACE",
                    description=f"'{c.id}' sits at an unstable coupling boundary (risk: {coupling_risk:.2f}).",
                    recommendation="Stable Dependency Principle: ensure components depend in the direction of stability.",
                    evidence={"coupling_risk": coupling_risk, "total_degree": id_n + od_n}
                ))
        return problems
