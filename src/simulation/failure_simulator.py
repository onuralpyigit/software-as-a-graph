"""
Failure Simulator - Version 5.0

Simulates component failures and measures impact on the system.
Uses ORIGINAL edge types (NOT derived DEPENDS_ON).

Features:
- Single component failure simulation
- Cascade failure propagation
- Component-type specific simulation
- Failure campaign (test all components)
- Impact scoring based on reachability loss

Impact Calculation:
- Uses message flow paths (PUBLISHES_TO -> Topic -> SUBSCRIBES_TO)
- Considers infrastructure dependencies (RUNS_ON, CONNECTS_TO)
- Measures affected components / total components

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Set, Any, Optional, Tuple, Iterator
from collections import defaultdict

from .simulation_graph import (
    SimulationGraph,
    Component,
    ComponentType,
    EdgeType,
    ComponentStatus,
)


# =============================================================================
# Enums
# =============================================================================

class FailureMode(Enum):
    """Types of component failures"""
    CRASH = "crash"                    # Complete failure
    DEGRADED = "degraded"              # Partial functionality
    NETWORK_PARTITION = "partition"    # Network isolation
    OVERLOAD = "overload"             # Performance degradation
    
    @property
    def impact_factor(self) -> float:
        """Impact multiplier for this failure mode"""
        return {
            FailureMode.CRASH: 1.0,
            FailureMode.DEGRADED: 0.5,
            FailureMode.NETWORK_PARTITION: 0.8,
            FailureMode.OVERLOAD: 0.3,
        }[self]


class PropagationRule(Enum):
    """Rules for cascade propagation"""
    NONE = "none"                      # No cascade
    IMMEDIATE = "immediate"            # Direct connections only
    TRANSITIVE = "transitive"          # Full cascade propagation
    THRESHOLD = "threshold"            # Cascade if threshold exceeded


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FailureResult:
    """Result of a single failure simulation"""
    failed_component: str
    component_type: ComponentType
    failure_mode: FailureMode
    directly_affected: Set[str]
    cascade_affected: Set[str]
    total_affected: int
    total_components: int
    impact_score: float
    affected_by_type: Dict[str, int]
    disconnected_components: Set[str]
    message_paths_broken: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def all_affected(self) -> Set[str]:
        """All affected components including the failed one"""
        return {self.failed_component} | self.directly_affected | self.cascade_affected
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "failed_component": self.failed_component,
            "component_type": self.component_type.value,
            "failure_mode": self.failure_mode.value,
            "directly_affected": list(self.directly_affected),
            "cascade_affected": list(self.cascade_affected),
            "total_affected": self.total_affected,
            "total_components": self.total_components,
            "impact_score": round(self.impact_score, 6),
            "affected_by_type": self.affected_by_type,
            "disconnected_components": list(self.disconnected_components),
            "message_paths_broken": self.message_paths_broken,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CampaignResult:
    """Result of a failure simulation campaign"""
    results: List[FailureResult]
    total_components: int
    critical_threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def component_impacts(self) -> Dict[str, float]:
        """Map of component ID to impact score"""
        return {r.failed_component: r.impact_score for r in self.results}
    
    @property
    def critical_components(self) -> List[str]:
        """Components with impact above threshold"""
        return [r.failed_component for r in self.results 
                if r.impact_score >= self.critical_threshold]
    
    @property
    def ranked_by_impact(self) -> List[Tuple[str, float]]:
        """Components ranked by impact (descending)"""
        return sorted(
            [(r.failed_component, r.impact_score) for r in self.results],
            key=lambda x: -x[1]
        )
    
    def get_by_type(self, comp_type: ComponentType) -> List[FailureResult]:
        """Get results for a specific component type"""
        return [r for r in self.results if r.component_type == comp_type]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "total_components": self.total_components,
            "critical_threshold": self.critical_threshold,
            "critical_count": len(self.critical_components),
            "component_impacts": self.component_impacts,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# Failure Simulator
# =============================================================================

class FailureSimulator:
    """
    Simulates component failures and measures system impact.
    
    Uses ORIGINAL edge types (PUBLISHES_TO, SUBSCRIBES_TO, etc.)
    NOT derived DEPENDS_ON relationships.
    
    This allows accurate simulation of message flow disruption
    when components fail.
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        cascade: bool = True,
        cascade_threshold: float = 0.5,
        critical_threshold: float = 0.3,
    ):
        """
        Initialize the failure simulator.
        
        Args:
            seed: Random seed for reproducibility
            cascade: Enable cascade failure propagation
            cascade_threshold: Fraction of dependencies that must fail 
                             to trigger cascade (0.0-1.0)
            critical_threshold: Impact threshold for critical classification
        """
        self.rng = random.Random(seed)
        self.cascade = cascade
        self.cascade_threshold = cascade_threshold
        self.critical_threshold = critical_threshold
        self._logger = logging.getLogger(__name__)
    
    # =========================================================================
    # Single Failure Simulation
    # =========================================================================
    
    def simulate_failure(
        self,
        graph: SimulationGraph,
        component_id: str,
        failure_mode: FailureMode = FailureMode.CRASH,
    ) -> FailureResult:
        """
        Simulate failure of a single component.
        
        Args:
            graph: The simulation graph
            component_id: ID of component to fail
            failure_mode: Type of failure
        
        Returns:
            FailureResult with impact analysis
        """
        component = graph.get_component(component_id)
        if not component:
            raise ValueError(f"Component not found: {component_id}")
        
        # Track affected components
        directly_affected: Set[str] = set()
        cascade_affected: Set[str] = set()
        disconnected: Set[str] = set()
        
        # Calculate direct impact based on component type
        directly_affected = self._calculate_direct_impact(
            graph, component_id, component.type
        )
        
        # Calculate cascade if enabled
        if self.cascade and failure_mode == FailureMode.CRASH:
            cascade_affected = self._propagate_cascade(
                graph, 
                {component_id} | directly_affected,
                visited={component_id}
            )
        
        # Calculate disconnected components
        disconnected = self._find_disconnected(
            graph, 
            {component_id} | directly_affected | cascade_affected
        )
        
        # Count broken message paths
        broken_paths = self._count_broken_paths(graph, component_id, component.type)
        
        # Calculate impact score
        all_affected = {component_id} | directly_affected | cascade_affected
        total = len(graph.components)
        
        # Impact = affected / total, weighted by failure mode
        raw_impact = len(all_affected) / total if total > 0 else 0
        impact = raw_impact * failure_mode.impact_factor
        
        # Count affected by type
        affected_by_type: Dict[str, int] = defaultdict(int)
        for comp_id in all_affected:
            comp = graph.get_component(comp_id)
            if comp:
                affected_by_type[comp.type.value] += 1
        
        return FailureResult(
            failed_component=component_id,
            component_type=component.type,
            failure_mode=failure_mode,
            directly_affected=directly_affected,
            cascade_affected=cascade_affected,
            total_affected=len(all_affected),
            total_components=total,
            impact_score=impact,
            affected_by_type=dict(affected_by_type),
            disconnected_components=disconnected,
            message_paths_broken=broken_paths,
        )
    
    def _calculate_direct_impact(
        self,
        graph: SimulationGraph,
        component_id: str,
        comp_type: ComponentType,
    ) -> Set[str]:
        """
        Calculate components directly affected by a failure.
        Uses original edge types to trace impact.
        """
        affected = set()
        
        if comp_type == ComponentType.APPLICATION:
            # App failure affects topics it publishes to (no more messages)
            for topic_id in graph.get_topics_published_by(component_id):
                affected.add(topic_id)
                # Which affects all subscribers to those topics
                for subscriber in graph.get_subscribers(topic_id):
                    if subscriber != component_id:
                        affected.add(subscriber)
        
        elif comp_type == ComponentType.TOPIC:
            # Topic failure affects all subscribers
            for subscriber in graph.get_subscribers(component_id):
                affected.add(subscriber)
            # And affects publishers (they can't publish)
            for publisher in graph.get_publishers(component_id):
                affected.add(publisher)
        
        elif comp_type == ComponentType.BROKER:
            # Broker failure affects all topics it routes
            for edge in graph.get_edges_by_type(EdgeType.ROUTES):
                if edge.source == component_id:
                    affected.add(edge.target)
                    # And all pub/sub relationships for those topics
                    for sub in graph.get_subscribers(edge.target):
                        affected.add(sub)
                    for pub in graph.get_publishers(edge.target):
                        affected.add(pub)
        
        elif comp_type == ComponentType.NODE:
            # Node failure affects all components running on it
            for comp_id in graph.get_components_on_node(component_id):
                affected.add(comp_id)
                # Recursively calculate their impact
                comp = graph.get_component(comp_id)
                if comp:
                    affected.update(
                        self._calculate_direct_impact(graph, comp_id, comp.type)
                    )
        
        return affected
    
    def _propagate_cascade(
        self,
        graph: SimulationGraph,
        failed: Set[str],
        visited: Set[str],
    ) -> Set[str]:
        """
        Propagate cascade failures based on threshold rule.
        
        A component fails in cascade if more than cascade_threshold
        of its dependencies have failed.
        """
        cascade = set()
        
        for comp_id, component in graph.components.items():
            if comp_id in visited:
                continue
            
            # Count failed dependencies
            dependencies = self._get_dependencies(graph, comp_id, component.type)
            if not dependencies:
                continue
            
            failed_deps = len(dependencies & failed)
            total_deps = len(dependencies)
            
            if total_deps > 0 and (failed_deps / total_deps) >= self.cascade_threshold:
                cascade.add(comp_id)
                visited.add(comp_id)
        
        # Recursive propagation
        if cascade:
            cascade.update(
                self._propagate_cascade(graph, failed | cascade, visited)
            )
        
        return cascade
    
    def _get_dependencies(
        self,
        graph: SimulationGraph,
        component_id: str,
        comp_type: ComponentType,
    ) -> Set[str]:
        """Get components this component depends on"""
        deps = set()
        
        if comp_type == ComponentType.APPLICATION:
            # Apps depend on topics they subscribe to
            deps.update(graph.get_topics_subscribed_by(component_id))
            # And the node they run on
            node = graph.get_node_for_component(component_id)
            if node:
                deps.add(node)
        
        elif comp_type == ComponentType.TOPIC:
            # Topics depend on their broker
            broker = graph.get_broker_for_topic(component_id)
            if broker:
                deps.add(broker)
            # And on publishers (need at least one)
            deps.update(graph.get_publishers(component_id))
        
        elif comp_type == ComponentType.BROKER:
            # Brokers depend on their node
            node = graph.get_node_for_component(component_id)
            if node:
                deps.add(node)
        
        return deps
    
    def _find_disconnected(
        self,
        graph: SimulationGraph,
        failed: Set[str],
    ) -> Set[str]:
        """Find components that become disconnected due to failures"""
        disconnected = set()
        
        # Simple check: components with all neighbors failed
        for comp_id in graph.components:
            if comp_id in failed:
                continue
            
            neighbors = graph.get_neighbors(comp_id, "both")
            if neighbors and neighbors.issubset(failed):
                disconnected.add(comp_id)
        
        return disconnected
    
    def _count_broken_paths(
        self,
        graph: SimulationGraph,
        failed_id: str,
        comp_type: ComponentType,
    ) -> int:
        """Count message paths broken by this failure"""
        broken = 0
        
        if comp_type == ComponentType.TOPIC:
            # All pub-sub pairs through this topic are broken
            publishers = graph.get_publishers(failed_id)
            subscribers = graph.get_subscribers(failed_id)
            broken = len(publishers) * len(subscribers)
        
        elif comp_type == ComponentType.BROKER:
            # All topics routed by this broker
            for edge in graph.get_edges_by_type(EdgeType.ROUTES):
                if edge.source == failed_id:
                    pubs = graph.get_publishers(edge.target)
                    subs = graph.get_subscribers(edge.target)
                    broken += len(pubs) * len(subs)
        
        elif comp_type == ComponentType.APPLICATION:
            # Paths where this app is publisher
            for topic in graph.get_topics_published_by(failed_id):
                subs = graph.get_subscribers(topic)
                broken += len(subs)
        
        return broken
    
    # =========================================================================
    # Campaign Simulation
    # =========================================================================
    
    def simulate_all_failures(
        self,
        graph: SimulationGraph,
        failure_mode: FailureMode = FailureMode.CRASH,
        component_types: Optional[List[ComponentType]] = None,
    ) -> CampaignResult:
        """
        Simulate failure of every component and measure impact.
        
        Args:
            graph: The simulation graph
            failure_mode: Type of failure to simulate
            component_types: If specified, only test these types
        
        Returns:
            CampaignResult with all failure results
        """
        results = []
        
        # Determine which components to test
        if component_types:
            component_ids = set()
            for ct in component_types:
                component_ids.update(graph.get_component_ids_by_type(ct))
        else:
            component_ids = set(graph.components.keys())
        
        total = len(component_ids)
        self._logger.info(f"Starting failure campaign for {total} components")
        
        for i, comp_id in enumerate(sorted(component_ids)):
            result = self.simulate_failure(graph, comp_id, failure_mode)
            results.append(result)
            
            if (i + 1) % 100 == 0:
                self._logger.info(f"Progress: {i + 1}/{total} components tested")
        
        return CampaignResult(
            results=results,
            total_components=len(graph.components),
            critical_threshold=self.critical_threshold,
        )
    
    def simulate_by_component_type(
        self,
        graph: SimulationGraph,
        comp_type: ComponentType,
        failure_mode: FailureMode = FailureMode.CRASH,
    ) -> CampaignResult:
        """
        Simulate failures for a specific component type.
        
        Args:
            graph: The simulation graph
            comp_type: Component type to test
            failure_mode: Type of failure
        
        Returns:
            CampaignResult for this component type
        """
        return self.simulate_all_failures(
            graph, failure_mode, [comp_type]
        )
    
    def simulate_random_failures(
        self,
        graph: SimulationGraph,
        count: int,
        failure_mode: FailureMode = FailureMode.CRASH,
    ) -> CampaignResult:
        """
        Simulate random component failures.
        
        Args:
            graph: The simulation graph
            count: Number of random failures to simulate
            failure_mode: Type of failure
        
        Returns:
            CampaignResult for random failures
        """
        component_ids = list(graph.components.keys())
        selected = self.rng.sample(component_ids, min(count, len(component_ids)))
        
        results = []
        for comp_id in selected:
            result = self.simulate_failure(graph, comp_id, failure_mode)
            results.append(result)
        
        return CampaignResult(
            results=results,
            total_components=len(graph.components),
            critical_threshold=self.critical_threshold,
        )
    
    def simulate_simultaneous_failures(
        self,
        graph: SimulationGraph,
        component_ids: List[str],
        failure_mode: FailureMode = FailureMode.CRASH,
    ) -> FailureResult:
        """
        Simulate multiple simultaneous failures.
        
        Args:
            graph: The simulation graph
            component_ids: Components to fail simultaneously
            failure_mode: Type of failure
        
        Returns:
            Combined FailureResult
        """
        all_directly_affected: Set[str] = set()
        all_cascade_affected: Set[str] = set()
        
        # Calculate direct impact of all failures
        for comp_id in component_ids:
            comp = graph.get_component(comp_id)
            if comp:
                affected = self._calculate_direct_impact(graph, comp_id, comp.type)
                all_directly_affected.update(affected)
        
        # Remove the failed components from affected
        all_directly_affected -= set(component_ids)
        
        # Calculate cascade from combined failures
        if self.cascade:
            failed_set = set(component_ids) | all_directly_affected
            all_cascade_affected = self._propagate_cascade(
                graph, failed_set, set(component_ids)
            )
        
        # Calculate disconnected
        all_failed = set(component_ids) | all_directly_affected | all_cascade_affected
        disconnected = self._find_disconnected(graph, all_failed)
        
        # Impact calculation
        total = len(graph.components)
        impact = len(all_failed) / total if total > 0 else 0
        impact *= failure_mode.impact_factor
        
        # Count by type
        affected_by_type: Dict[str, int] = defaultdict(int)
        for comp_id in all_failed:
            comp = graph.get_component(comp_id)
            if comp:
                affected_by_type[comp.type.value] += 1
        
        return FailureResult(
            failed_component=",".join(sorted(component_ids)[:5]) + 
                            ("..." if len(component_ids) > 5 else ""),
            component_type=ComponentType.APPLICATION,  # Mixed
            failure_mode=failure_mode,
            directly_affected=all_directly_affected,
            cascade_affected=all_cascade_affected,
            total_affected=len(all_failed),
            total_components=total,
            impact_score=impact,
            affected_by_type=dict(affected_by_type),
            disconnected_components=disconnected,
            message_paths_broken=0,  # Not calculated for simultaneous
        )


# =============================================================================
# Factory Functions
# =============================================================================

def simulate_single_failure(
    graph: SimulationGraph,
    component_id: str,
    cascade: bool = True,
    failure_mode: FailureMode = FailureMode.CRASH,
) -> FailureResult:
    """
    Quick function to simulate a single failure.
    """
    simulator = FailureSimulator(cascade=cascade)
    return simulator.simulate_failure(graph, component_id, failure_mode)


def simulate_all_components(
    graph: SimulationGraph,
    cascade: bool = True,
    critical_threshold: float = 0.3,
) -> CampaignResult:
    """
    Quick function to run full failure campaign.
    """
    simulator = FailureSimulator(
        cascade=cascade,
        critical_threshold=critical_threshold,
    )
    return simulator.simulate_all_failures(graph)
