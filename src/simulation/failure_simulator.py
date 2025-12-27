"""
Failure Simulator - Version 4.0

Simulates component failures and their impact on the pub-sub system.
Works directly on the graph model without DEPENDS_ON relationships.

Features:
- Single and multiple component failures
- Cascading failure propagation
- Targeted attack simulations
- Exhaustive failure campaigns
- Impact measurement via reachability loss

Author: Software-as-a-Graph Research Project
Version: 4.0
"""

from __future__ import annotations
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from .graph_model import SimulationGraph, Component, ComponentType, ConnectionType


# =============================================================================
# Enums
# =============================================================================

class FailureType(Enum):
    """Types of failures"""
    COMPLETE = "complete"       # Total failure
    PARTIAL = "partial"         # Degraded performance
    INTERMITTENT = "intermittent"  # Occasional failures


class FailureMode(Enum):
    """How failures manifest"""
    CRASH = "crash"             # Immediate stop
    HANG = "hang"               # Unresponsive
    BYZANTINE = "byzantine"     # Incorrect behavior


class AttackStrategy(Enum):
    """Targeted attack strategies"""
    RANDOM = "random"
    HIGHEST_DEGREE = "highest_degree"
    HIGHEST_BETWEENNESS = "highest_betweenness"
    BROKERS_FIRST = "brokers_first"
    NODES_FIRST = "nodes_first"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FailureEvent:
    """Records a single failure event"""
    component: str
    component_type: str
    failure_type: FailureType
    timestamp: datetime
    is_cascade: bool = False
    cascade_depth: int = 0
    cause: str = "primary"

    def to_dict(self) -> Dict:
        return {
            "component": self.component,
            "component_type": self.component_type,
            "failure_type": self.failure_type.value,
            "timestamp": self.timestamp.isoformat(),
            "is_cascade": self.is_cascade,
            "cascade_depth": self.cascade_depth,
            "cause": self.cause,
        }


@dataclass
class ImpactMetrics:
    """Metrics measuring failure impact"""
    # Reachability
    original_reachability: int
    final_reachability: int
    reachability_loss: int
    reachability_loss_pct: float
    
    # Paths
    original_paths: int
    final_paths: int
    paths_lost: int
    paths_lost_pct: float
    
    # Connectivity
    original_components: int
    final_components: int
    fragmentation: int
    
    # Counts
    total_nodes: int
    failed_count: int
    cascade_count: int
    
    # Combined score
    impact_score: float  # 0.0 to 1.0

    def to_dict(self) -> Dict:
        return {
            "reachability": {
                "original": self.original_reachability,
                "final": self.final_reachability,
                "loss": self.reachability_loss,
                "loss_pct": round(self.reachability_loss_pct, 2),
            },
            "paths": {
                "original": self.original_paths,
                "final": self.final_paths,
                "lost": self.paths_lost,
                "lost_pct": round(self.paths_lost_pct, 2),
            },
            "connectivity": {
                "original_components": self.original_components,
                "final_components": self.final_components,
                "fragmentation": self.fragmentation,
            },
            "nodes": {
                "total": self.total_nodes,
                "failed": self.failed_count,
                "cascade": self.cascade_count,
            },
            "impact_score": round(self.impact_score, 4),
        }


@dataclass
class SimulationResult:
    """Result of a failure simulation"""
    simulation_id: str
    simulation_type: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    
    # Failures
    primary_failures: List[str]
    cascade_failures: List[str]
    failure_events: List[FailureEvent]
    
    # Impact
    impact: ImpactMetrics
    
    # Isolated components
    isolated_components: List[str]

    def to_dict(self) -> Dict:
        return {
            "simulation_id": self.simulation_id,
            "simulation_type": self.simulation_type,
            "timing": {
                "start": self.start_time.isoformat(),
                "end": self.end_time.isoformat(),
                "duration_ms": round(self.duration_ms, 2),
            },
            "failures": {
                "primary": self.primary_failures,
                "cascade": self.cascade_failures,
                "total": len(self.primary_failures) + len(self.cascade_failures),
                "events": [e.to_dict() for e in self.failure_events],
            },
            "impact": self.impact.to_dict(),
            "isolated_components": self.isolated_components,
        }


@dataclass
class BatchResult:
    """Result of batch simulations"""
    simulation_count: int
    total_duration_ms: float
    results: List[SimulationResult]
    
    # Aggregated metrics
    avg_impact_score: float
    max_impact_score: float
    critical_components: List[Tuple[str, float]]  # (component, impact_score)

    def to_dict(self) -> Dict:
        return {
            "summary": {
                "simulation_count": self.simulation_count,
                "duration_ms": round(self.total_duration_ms, 2),
                "avg_impact": round(self.avg_impact_score, 4),
                "max_impact": round(self.max_impact_score, 4),
            },
            "critical_components": [
                {"component": c, "impact": round(i, 4)}
                for c, i in self.critical_components[:20]
            ],
            "results": [r.to_dict() for r in self.results],
        }


# =============================================================================
# Failure Simulator
# =============================================================================

class FailureSimulator:
    """
    Simulates failures and measures impact on the pub-sub system.
    
    Works directly on the graph model using native pub-sub connections
    (PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, RUNS_ON) rather than
    derived DEPENDS_ON relationships.
    
    Supports:
    - Single/multiple component failures
    - Cascading failure propagation
    - Targeted attack simulations
    - Exhaustive failure campaigns
    """

    def __init__(
        self,
        cascade_threshold: float = 0.5,
        cascade_probability: float = 0.7,
        max_cascade_depth: int = 5,
        seed: Optional[int] = None,
    ):
        """
        Initialize the failure simulator.
        
        Args:
            cascade_threshold: Fraction of lost dependencies triggering cascade
            cascade_probability: Probability of cascade occurring
            max_cascade_depth: Maximum cascade propagation depth
            seed: Random seed for reproducibility
        """
        self.cascade_threshold = cascade_threshold
        self.cascade_probability = cascade_probability
        self.max_cascade_depth = max_cascade_depth
        
        self._rng = random.Random(seed)
        self._simulation_counter = 0
        self.logger = logging.getLogger(__name__)

    # =========================================================================
    # Single Failure
    # =========================================================================

    def simulate_failure(
        self,
        graph: SimulationGraph,
        component: str,
        failure_type: FailureType = FailureType.COMPLETE,
        enable_cascade: bool = True,
    ) -> SimulationResult:
        """
        Simulate failure of a single component.
        
        Args:
            graph: SimulationGraph to simulate on
            component: Component ID to fail
            failure_type: Type of failure
            enable_cascade: Whether to propagate cascades
        
        Returns:
            SimulationResult with impact analysis
        """
        if component not in graph.components:
            raise ValueError(f"Component '{component}' not found")
        
        self._simulation_counter += 1
        sim_id = f"fail_{self._simulation_counter:05d}"
        start_time = datetime.now()
        
        self.logger.info(f"[{sim_id}] Simulating {failure_type.value} failure of '{component}'")
        
        # Store original metrics
        original_reach = graph.calculate_total_reachability()
        original_paths = graph.count_active_paths()
        original_cc = graph.count_connected_components()
        
        # Create simulation copy
        sim_graph = graph.copy()
        failure_events = []
        
        # Apply primary failure
        comp_obj = sim_graph.components[component]
        event = FailureEvent(
            component=component,
            component_type=comp_obj.type.value,
            failure_type=failure_type,
            timestamp=datetime.now(),
            is_cascade=False,
            cascade_depth=0,
            cause="Primary failure",
        )
        failure_events.append(event)
        
        self._apply_failure(sim_graph, component, failure_type)
        
        # Cascade propagation
        cascade_failures = []
        if enable_cascade and failure_type == FailureType.COMPLETE:
            cascade_failures, cascade_events = self._propagate_cascade(
                sim_graph, [component], 1
            )
            failure_events.extend(cascade_events)
        
        end_time = datetime.now()
        
        return self._build_result(
            sim_id=sim_id,
            sim_type="single_failure",
            original_graph=graph,
            sim_graph=sim_graph,
            original_reach=original_reach,
            original_paths=original_paths,
            original_cc=original_cc,
            primary_failures=[component],
            cascade_failures=cascade_failures,
            failure_events=failure_events,
            start_time=start_time,
            end_time=end_time,
        )

    # =========================================================================
    # Multiple Failures
    # =========================================================================

    def simulate_multiple_failures(
        self,
        graph: SimulationGraph,
        components: List[str],
        failure_type: FailureType = FailureType.COMPLETE,
        enable_cascade: bool = True,
    ) -> SimulationResult:
        """
        Simulate failure of multiple components.
        
        Args:
            graph: SimulationGraph to simulate on
            components: List of component IDs to fail
            failure_type: Type of failure
            enable_cascade: Whether to propagate cascades
        
        Returns:
            SimulationResult with impact analysis
        """
        # Validate components
        for comp in components:
            if comp not in graph.components:
                raise ValueError(f"Component '{comp}' not found")
        
        self._simulation_counter += 1
        sim_id = f"multi_{self._simulation_counter:05d}"
        start_time = datetime.now()
        
        self.logger.info(f"[{sim_id}] Simulating failure of {len(components)} components")
        
        # Store original metrics
        original_reach = graph.calculate_total_reachability()
        original_paths = graph.count_active_paths()
        original_cc = graph.count_connected_components()
        
        # Create simulation copy
        sim_graph = graph.copy()
        failure_events = []
        
        # Apply all failures
        for comp in components:
            comp_obj = sim_graph.components[comp]
            event = FailureEvent(
                component=comp,
                component_type=comp_obj.type.value,
                failure_type=failure_type,
                timestamp=datetime.now(),
                is_cascade=False,
                cascade_depth=0,
                cause="Primary failure (multiple)",
            )
            failure_events.append(event)
            self._apply_failure(sim_graph, comp, failure_type)
        
        # Cascade propagation
        cascade_failures = []
        if enable_cascade and failure_type == FailureType.COMPLETE:
            cascade_failures, cascade_events = self._propagate_cascade(
                sim_graph, components, 1
            )
            failure_events.extend(cascade_events)
        
        end_time = datetime.now()
        
        return self._build_result(
            sim_id=sim_id,
            sim_type="multiple_failure",
            original_graph=graph,
            sim_graph=sim_graph,
            original_reach=original_reach,
            original_paths=original_paths,
            original_cc=original_cc,
            primary_failures=components,
            cascade_failures=cascade_failures,
            failure_events=failure_events,
            start_time=start_time,
            end_time=end_time,
        )

    # =========================================================================
    # Targeted Attack
    # =========================================================================

    def simulate_attack(
        self,
        graph: SimulationGraph,
        strategy: AttackStrategy,
        count: int = 1,
        enable_cascade: bool = True,
    ) -> SimulationResult:
        """
        Simulate targeted attack on components.
        
        Args:
            graph: SimulationGraph to simulate on
            strategy: Attack strategy
            count: Number of components to attack
            enable_cascade: Whether to propagate cascades
        
        Returns:
            SimulationResult with impact analysis
        """
        targets = self._select_targets(graph, strategy, count)
        self.logger.info(f"Attack ({strategy.value}): {targets}")
        
        return self.simulate_multiple_failures(
            graph, targets, enable_cascade=enable_cascade
        )

    # =========================================================================
    # Exhaustive Campaign
    # =========================================================================

    def simulate_all_failures(
        self,
        graph: SimulationGraph,
        component_types: Optional[List[ComponentType]] = None,
        enable_cascade: bool = True,
    ) -> BatchResult:
        """
        Simulate failure of each component individually.
        
        Args:
            graph: SimulationGraph to simulate on
            component_types: Types to include (None = all)
            enable_cascade: Whether to propagate cascades
        
        Returns:
            BatchResult with all simulation results
        """
        self._simulation_counter += 1
        batch_start = datetime.now()
        
        # Select components to test
        if component_types:
            targets = [
                c.id for c in graph.components.values()
                if c.type in component_types
            ]
        else:
            targets = list(graph.components.keys())
        
        self.logger.info(f"Running exhaustive campaign on {len(targets)} components")
        
        results = []
        for comp_id in targets:
            result = self.simulate_failure(
                graph, comp_id, enable_cascade=enable_cascade
            )
            results.append(result)
        
        batch_end = datetime.now()
        total_duration = (batch_end - batch_start).total_seconds() * 1000
        
        # Compute aggregates
        impacts = [(r.primary_failures[0], r.impact.impact_score) for r in results]
        impacts.sort(key=lambda x: -x[1])
        
        avg_impact = sum(r.impact.impact_score for r in results) / len(results) if results else 0
        max_impact = max(r.impact.impact_score for r in results) if results else 0
        
        return BatchResult(
            simulation_count=len(results),
            total_duration_ms=total_duration,
            results=results,
            avg_impact_score=avg_impact,
            max_impact_score=max_impact,
            critical_components=impacts,
        )

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _apply_failure(
        self,
        graph: SimulationGraph,
        component: str,
        failure_type: FailureType,
    ) -> None:
        """Apply failure to a component"""
        comp = graph.components.get(component)
        if not comp:
            return
        
        if failure_type == FailureType.COMPLETE:
            comp.is_active = False
            comp.capacity = 0.0
            
            # Deactivate connections
            for conn in graph.get_outgoing(component):
                conn.is_active = False
            for conn in graph.get_incoming(component):
                conn.is_active = False
        
        elif failure_type == FailureType.PARTIAL:
            comp.is_degraded = True
            comp.capacity = 0.5
        
        elif failure_type == FailureType.INTERMITTENT:
            comp.is_degraded = True
            comp.capacity = 0.7

    def _propagate_cascade(
        self,
        graph: SimulationGraph,
        failed: List[str],
        depth: int,
    ) -> Tuple[List[str], List[FailureEvent]]:
        """Propagate cascading failures based on dependency loss"""
        if depth > self.max_cascade_depth:
            return [], []
        
        cascade_failures = []
        cascade_events = []
        
        # Find components affected by failures
        at_risk = set()
        
        for failed_comp in failed:
            # Find components that depend on the failed component
            # In pub-sub: publishers depend on brokers/topics, subscribers depend on topics
            
            # If broker failed, apps using topics routed by this broker are at risk
            if graph.components.get(failed_comp, Component("", ComponentType.APPLICATION)).type == ComponentType.BROKER:
                for conn in graph.get_outgoing(failed_comp):
                    if conn.type == ConnectionType.ROUTES:
                        topic = conn.target
                        # All publishers/subscribers of this topic are at risk
                        at_risk.update(graph.get_publishers(topic))
                        at_risk.update(graph.get_subscribers(topic))
            
            # If node failed, all apps on it are at risk
            if graph.components.get(failed_comp, Component("", ComponentType.APPLICATION)).type == ComponentType.NODE:
                at_risk.update(graph.get_apps_on_node(failed_comp))
            
            # If topic failed, publishers and subscribers are at risk
            if graph.components.get(failed_comp, Component("", ComponentType.APPLICATION)).type == ComponentType.TOPIC:
                at_risk.update(graph.get_publishers(failed_comp))
                at_risk.update(graph.get_subscribers(failed_comp))
        
        # Remove already failed components
        at_risk -= set(failed)
        at_risk = {c for c in at_risk if graph.components.get(c, Component("", ComponentType.APPLICATION)).is_active}
        
        # Evaluate cascade for each at-risk component
        for comp_id in at_risk:
            comp = graph.components.get(comp_id)
            if not comp or not comp.is_active:
                continue
            
            # Calculate dependency loss ratio
            outgoing = graph.get_outgoing(comp_id)
            if not outgoing:
                continue
            
            active = sum(1 for c in outgoing if c.is_active)
            total = len(outgoing)
            loss_ratio = 1.0 - (active / total) if total > 0 else 0
            
            # Check if cascade occurs
            if loss_ratio >= self.cascade_threshold:
                if self._rng.random() < self.cascade_probability:
                    cascade_failures.append(comp_id)
                    
                    event = FailureEvent(
                        component=comp_id,
                        component_type=comp.type.value,
                        failure_type=FailureType.COMPLETE,
                        timestamp=datetime.now(),
                        is_cascade=True,
                        cascade_depth=depth,
                        cause=f"Cascade depth {depth} (loss: {loss_ratio:.0%})",
                    )
                    cascade_events.append(event)
                    
                    self._apply_failure(graph, comp_id, FailureType.COMPLETE)
        
        # Recursive cascade
        if cascade_failures:
            more_failures, more_events = self._propagate_cascade(
                graph, cascade_failures, depth + 1
            )
            cascade_failures.extend(more_failures)
            cascade_events.extend(more_events)
        
        return cascade_failures, cascade_events

    def _select_targets(
        self,
        graph: SimulationGraph,
        strategy: AttackStrategy,
        count: int,
    ) -> List[str]:
        """Select attack targets based on strategy"""
        candidates = [c for c in graph.components.values() if c.is_active]
        
        if strategy == AttackStrategy.RANDOM:
            targets = self._rng.sample(candidates, min(count, len(candidates)))
            return [t.id for t in targets]
        
        elif strategy == AttackStrategy.HIGHEST_DEGREE:
            # Sort by total connections
            by_degree = sorted(
                candidates,
                key=lambda c: len(graph.get_outgoing(c.id)) + len(graph.get_incoming(c.id)),
                reverse=True,
            )
            return [c.id for c in by_degree[:count]]
        
        elif strategy == AttackStrategy.BROKERS_FIRST:
            brokers = [c for c in candidates if c.type == ComponentType.BROKER]
            others = [c for c in candidates if c.type != ComponentType.BROKER]
            targets = brokers + others
            return [c.id for c in targets[:count]]
        
        elif strategy == AttackStrategy.NODES_FIRST:
            nodes = [c for c in candidates if c.type == ComponentType.NODE]
            others = [c for c in candidates if c.type != ComponentType.NODE]
            targets = nodes + others
            return [c.id for c in targets[:count]]
        
        else:
            return [candidates[0].id] if candidates else []

    def _build_result(
        self,
        sim_id: str,
        sim_type: str,
        original_graph: SimulationGraph,
        sim_graph: SimulationGraph,
        original_reach: int,
        original_paths: int,
        original_cc: int,
        primary_failures: List[str],
        cascade_failures: List[str],
        failure_events: List[FailureEvent],
        start_time: datetime,
        end_time: datetime,
    ) -> SimulationResult:
        """Build simulation result with impact metrics"""
        # Calculate final metrics
        final_reach = sim_graph.calculate_total_reachability()
        final_paths = sim_graph.count_active_paths()
        final_cc = sim_graph.count_connected_components()
        
        # Reachability
        reach_loss = original_reach - final_reach
        reach_loss_pct = (reach_loss / original_reach * 100) if original_reach > 0 else 0
        
        # Paths
        paths_lost = original_paths - final_paths
        paths_lost_pct = (paths_lost / original_paths * 100) if original_paths > 0 else 0
        
        # Impact score (weighted average of losses)
        impact_score = 0.0
        if original_reach > 0:
            impact_score = 0.5 * (reach_loss / original_reach)
        if original_paths > 0:
            impact_score += 0.5 * (paths_lost / original_paths)
        
        # Find isolated components
        isolated = []
        for comp_id, comp in sim_graph.components.items():
            if comp.is_active:
                reachable = sim_graph.calculate_reachability(comp_id)
                if len(reachable) == 1:  # Only reaches itself
                    isolated.append(comp_id)
        
        impact = ImpactMetrics(
            original_reachability=original_reach,
            final_reachability=final_reach,
            reachability_loss=reach_loss,
            reachability_loss_pct=reach_loss_pct,
            original_paths=original_paths,
            final_paths=final_paths,
            paths_lost=paths_lost,
            paths_lost_pct=paths_lost_pct,
            original_components=original_cc,
            final_components=final_cc,
            fragmentation=final_cc - original_cc,
            total_nodes=len(original_graph.components),
            failed_count=len(primary_failures),
            cascade_count=len(cascade_failures),
            impact_score=impact_score,
        )
        
        duration = (end_time - start_time).total_seconds() * 1000
        
        return SimulationResult(
            simulation_id=sim_id,
            simulation_type=sim_type,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration,
            primary_failures=primary_failures,
            cascade_failures=cascade_failures,
            failure_events=failure_events,
            impact=impact,
            isolated_components=isolated,
        )