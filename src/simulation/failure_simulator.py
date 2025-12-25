#!/usr/bin/env python3
"""
Failure Simulator
==================

Simulates component failures and cascading effects in distributed systems.
Uses graph structure to analyze failure impact.

Features:
- Single and multiple component failures
- Cascading failure propagation
- Impact assessment (reachability loss, connectivity)
- Targeted attack strategies
- Batch simulation campaigns

Author: Software-as-a-Graph Research Project
"""

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Tuple

from .neo4j_loader import SimulationGraph, Component, Dependency, ComponentType


class FailureType(Enum):
    """Type of failure"""
    COMPLETE = "complete"      # Component fully unavailable
    PARTIAL = "partial"        # Component degraded
    INTERMITTENT = "intermittent"  # Sporadic availability


class FailureMode(Enum):
    """How the failure manifests"""
    CRASH = "crash"            # Immediate stop
    HANG = "hang"              # Component unresponsive
    BYZANTINE = "byzantine"    # Erratic behavior
    OVERLOAD = "overload"      # Capacity exceeded


class AttackStrategy(Enum):
    """Targeted attack strategies"""
    RANDOM = "random"                  # Random selection
    HIGHEST_DEGREE = "highest_degree"  # Most connected
    HIGHEST_BETWEENNESS = "highest_betweenness"  # Most central
    HIGHEST_WEIGHT = "highest_weight"  # Highest total weight
    BROKERS_FIRST = "brokers_first"    # Target brokers


@dataclass
class FailureEvent:
    """Record of a failure event"""
    component: str
    component_type: str
    failure_type: FailureType
    failure_mode: FailureMode
    timestamp: datetime
    severity: float
    is_cascade: bool
    cascade_depth: int
    cause: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'component': self.component,
            'component_type': self.component_type,
            'failure_type': self.failure_type.value,
            'failure_mode': self.failure_mode.value,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity,
            'is_cascade': self.is_cascade,
            'cascade_depth': self.cascade_depth,
            'cause': self.cause
        }


@dataclass
class ImpactMetrics:
    """Metrics measuring failure impact"""
    # Reachability
    original_reachability: int
    final_reachability: int
    reachability_loss: float
    reachability_loss_pct: float
    
    # Connectivity
    original_components: int
    final_components: int
    components_added: int
    
    # Component counts
    original_node_count: int
    final_active_count: int
    failed_count: int
    cascade_count: int
    
    # Severity
    impact_score: float  # Combined impact metric
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'reachability': {
                'original': self.original_reachability,
                'final': self.final_reachability,
                'loss': self.reachability_loss,
                'loss_pct': round(self.reachability_loss_pct, 2)
            },
            'connectivity': {
                'original_components': self.original_components,
                'final_components': self.final_components,
                'fragmentation': self.components_added
            },
            'nodes': {
                'original': self.original_node_count,
                'active': self.final_active_count,
                'failed': self.failed_count,
                'cascade': self.cascade_count
            },
            'impact_score': round(self.impact_score, 4)
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
    
    # Graph state
    affected_dependencies: int
    isolated_components: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'simulation_id': self.simulation_id,
            'simulation_type': self.simulation_type,
            'timing': {
                'start': self.start_time.isoformat(),
                'end': self.end_time.isoformat(),
                'duration_ms': round(self.duration_ms, 2)
            },
            'failures': {
                'primary': self.primary_failures,
                'cascade': self.cascade_failures,
                'total': len(self.primary_failures) + len(self.cascade_failures),
                'events': [e.to_dict() for e in self.failure_events]
            },
            'impact': self.impact.to_dict(),
            'affected_dependencies': self.affected_dependencies,
            'isolated_components': self.isolated_components
        }


@dataclass
class BatchSimulationResult:
    """Result of batch simulations"""
    simulation_count: int
    total_duration_ms: float
    results: List[SimulationResult]
    
    # Aggregated metrics
    avg_impact_score: float
    max_impact_score: float
    critical_components: List[Tuple[str, float]]  # (component, impact_score)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'summary': {
                'simulation_count': self.simulation_count,
                'duration_ms': round(self.total_duration_ms, 2),
                'avg_impact': round(self.avg_impact_score, 4),
                'max_impact': round(self.max_impact_score, 4)
            },
            'critical_components': [
                {'component': c, 'impact': round(i, 4)} 
                for c, i in self.critical_components[:20]
            ],
            'results': [r.to_dict() for r in self.results]
        }


class FailureSimulator:
    """
    Simulates failures and measures their impact on the system.
    
    Supports:
    - Single component failures
    - Multiple simultaneous failures
    - Cascading failure propagation
    - Targeted attack simulations
    - Exhaustive failure campaigns
    """
    
    def __init__(self,
                 cascade_threshold: float = 0.5,
                 cascade_probability: float = 0.7,
                 max_cascade_depth: int = 5,
                 seed: Optional[int] = None):
        """
        Initialize the failure simulator.
        
        Args:
            cascade_threshold: Dependency loss ratio to trigger cascade (0-1)
            cascade_probability: Probability of cascade propagation (0-1)
            max_cascade_depth: Maximum depth of cascade chain
            seed: Random seed for reproducibility
        """
        self.cascade_threshold = cascade_threshold
        self.cascade_probability = cascade_probability
        self.max_cascade_depth = max_cascade_depth
        
        self._rng = random.Random(seed)
        self._simulation_counter = 0
        self.logger = logging.getLogger('FailureSimulator')
    
    def simulate_single_failure(self,
                                 graph: SimulationGraph,
                                 component: str,
                                 failure_type: FailureType = FailureType.COMPLETE,
                                 failure_mode: FailureMode = FailureMode.CRASH,
                                 severity: float = 1.0,
                                 enable_cascade: bool = True) -> SimulationResult:
        """
        Simulate failure of a single component.
        
        Args:
            graph: SimulationGraph to simulate on
            component: Component ID to fail
            failure_type: Type of failure
            failure_mode: How failure manifests
            severity: Failure severity (0-1)
            enable_cascade: Whether to propagate cascading failures
            
        Returns:
            SimulationResult with impact analysis
        """
        if component not in graph.components:
            raise ValueError(f"Component '{component}' not found in graph")
        
        self._simulation_counter += 1
        sim_id = f"fail_{self._simulation_counter:05d}"
        start_time = datetime.now()
        
        self.logger.info(f"[{sim_id}] Simulating {failure_type.value} failure of '{component}'")
        
        # Store original metrics
        original_reach = graph.calculate_total_reachability()
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
            failure_mode=failure_mode,
            timestamp=datetime.now(),
            severity=severity,
            is_cascade=False,
            cascade_depth=0,
            cause="Primary failure (simulated)"
        )
        failure_events.append(event)
        
        self._apply_failure(sim_graph, component, failure_type, severity)
        
        # Cascade propagation
        cascade_failures = []
        if enable_cascade and failure_type == FailureType.COMPLETE:
            cascade_failures, cascade_events = self._propagate_cascade(
                sim_graph, [component], failure_mode, 1
            )
            failure_events.extend(cascade_events)
        
        end_time = datetime.now()
        
        # Build result
        return self._build_result(
            sim_id=sim_id,
            sim_type="single_failure",
            original_graph=graph,
            sim_graph=sim_graph,
            original_reach=original_reach,
            original_cc=original_cc,
            primary_failures=[component],
            cascade_failures=cascade_failures,
            failure_events=failure_events,
            start_time=start_time,
            end_time=end_time
        )
    
    def simulate_multiple_failures(self,
                                    graph: SimulationGraph,
                                    components: List[str],
                                    failure_type: FailureType = FailureType.COMPLETE,
                                    simultaneous: bool = True,
                                    enable_cascade: bool = True) -> SimulationResult:
        """
        Simulate failure of multiple components.
        
        Args:
            graph: SimulationGraph to simulate on
            components: List of component IDs to fail
            failure_type: Type of failure
            simultaneous: Whether failures occur at once
            enable_cascade: Whether to propagate cascading failures
            
        Returns:
            SimulationResult with impact analysis
        """
        # Validate components
        for comp in components:
            if comp not in graph.components:
                raise ValueError(f"Component '{comp}' not found in graph")
        
        self._simulation_counter += 1
        sim_id = f"multi_{self._simulation_counter:05d}"
        start_time = datetime.now()
        
        self.logger.info(f"[{sim_id}] Simulating failure of {len(components)} components")
        
        # Store original metrics
        original_reach = graph.calculate_total_reachability()
        original_cc = graph.count_connected_components()
        
        # Create simulation copy
        sim_graph = graph.copy()
        failure_events = []
        
        # Apply failures
        for comp in components:
            comp_obj = sim_graph.components[comp]
            event = FailureEvent(
                component=comp,
                component_type=comp_obj.type.value,
                failure_type=failure_type,
                failure_mode=FailureMode.CRASH,
                timestamp=datetime.now(),
                severity=1.0,
                is_cascade=False,
                cascade_depth=0,
                cause="Primary failure (multiple)"
            )
            failure_events.append(event)
            self._apply_failure(sim_graph, comp, failure_type, 1.0)
        
        # Cascade propagation
        cascade_failures = []
        if enable_cascade and failure_type == FailureType.COMPLETE:
            cascade_failures, cascade_events = self._propagate_cascade(
                sim_graph, components, FailureMode.CRASH, 1
            )
            failure_events.extend(cascade_events)
        
        end_time = datetime.now()
        
        return self._build_result(
            sim_id=sim_id,
            sim_type="multiple_failure",
            original_graph=graph,
            sim_graph=sim_graph,
            original_reach=original_reach,
            original_cc=original_cc,
            primary_failures=components,
            cascade_failures=cascade_failures,
            failure_events=failure_events,
            start_time=start_time,
            end_time=end_time
        )
    
    def simulate_all_single_failures(self,
                                      graph: SimulationGraph,
                                      component_types: Optional[List[ComponentType]] = None,
                                      enable_cascade: bool = True) -> BatchSimulationResult:
        """
        Simulate failure of each component individually.
        
        Args:
            graph: SimulationGraph to simulate on
            component_types: Filter by component types (default: all)
            enable_cascade: Whether to propagate cascading failures
            
        Returns:
            BatchSimulationResult with all individual results
        """
        start_time = datetime.now()
        
        # Get components to test
        if component_types:
            components = []
            for ct in component_types:
                components.extend(graph.get_by_type(ct))
        else:
            components = list(graph.components.keys())
        
        self.logger.info(f"Running exhaustive campaign on {len(components)} components")
        
        results = []
        for comp in components:
            result = self.simulate_single_failure(
                graph, comp, enable_cascade=enable_cascade
            )
            results.append(result)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() * 1000
        
        # Aggregate
        impacts = [(r.primary_failures[0], r.impact.impact_score) for r in results]
        impacts.sort(key=lambda x: -x[1])
        
        avg_impact = sum(r.impact.impact_score for r in results) / len(results) if results else 0
        max_impact = max(r.impact.impact_score for r in results) if results else 0
        
        return BatchSimulationResult(
            simulation_count=len(results),
            total_duration_ms=duration,
            results=results,
            avg_impact_score=avg_impact,
            max_impact_score=max_impact,
            critical_components=impacts[:20]
        )
    
    def simulate_targeted_attack(self,
                                  graph: SimulationGraph,
                                  strategy: AttackStrategy,
                                  count: int = 5,
                                  enable_cascade: bool = True) -> SimulationResult:
        """
        Simulate a targeted attack using specified strategy.
        
        Args:
            graph: SimulationGraph to simulate on
            strategy: Attack strategy to use
            count: Number of components to attack
            enable_cascade: Whether to propagate cascading failures
            
        Returns:
            SimulationResult with impact analysis
        """
        # Select targets based on strategy
        targets = self._select_attack_targets(graph, strategy, count)
        
        self.logger.info(f"Targeted attack ({strategy.value}): {targets}")
        
        return self.simulate_multiple_failures(
            graph, targets, enable_cascade=enable_cascade
        )
    
    def _apply_failure(self, graph: SimulationGraph, component: str,
                       failure_type: FailureType, severity: float):
        """Apply failure to a component"""
        comp = graph.components.get(component)
        if not comp:
            return
        
        if failure_type == FailureType.COMPLETE:
            comp.is_active = False
            comp.capacity = 0.0
            # Deactivate all dependencies involving this component
            for dep in graph.get_outgoing(component):
                dep.is_active = False
            for dep in graph.get_incoming(component):
                dep.is_active = False
        
        elif failure_type == FailureType.PARTIAL:
            comp.is_degraded = True
            comp.capacity = max(0, 1.0 - severity)
        
        elif failure_type == FailureType.INTERMITTENT:
            comp.is_degraded = True
            comp.capacity = 0.5  # 50% availability
    
    def _propagate_cascade(self,
                            graph: SimulationGraph,
                            failed: List[str],
                            failure_mode: FailureMode,
                            depth: int) -> Tuple[List[str], List[FailureEvent]]:
        """Propagate cascading failures"""
        if depth > self.max_cascade_depth:
            return [], []
        
        cascade_failures = []
        cascade_events = []
        
        # Find components that depend on failed components
        at_risk = set()
        for failed_comp in failed:
            for dep in graph.get_incoming(failed_comp):
                if dep.is_active:
                    source = dep.source
                    source_comp = graph.components.get(source)
                    if source_comp and source_comp.is_active:
                        at_risk.add(source)
        
        # Evaluate cascade for each at-risk component
        for comp_id in at_risk:
            # Calculate dependency loss
            outgoing = graph.get_outgoing(comp_id)
            if not outgoing:
                continue
            
            active_deps = sum(1 for d in outgoing if d.is_active)
            total_deps = len(outgoing)
            loss_ratio = 1.0 - (active_deps / total_deps)
            
            # Check if cascade occurs
            if loss_ratio >= self.cascade_threshold:
                if self._rng.random() < self.cascade_probability:
                    # Cascade failure occurs
                    cascade_failures.append(comp_id)
                    
                    comp_obj = graph.components[comp_id]
                    event = FailureEvent(
                        component=comp_id,
                        component_type=comp_obj.type.value,
                        failure_type=FailureType.COMPLETE,
                        failure_mode=failure_mode,
                        timestamp=datetime.now(),
                        severity=1.0,
                        is_cascade=True,
                        cascade_depth=depth,
                        cause=f"Cascade from depth {depth-1} (loss: {loss_ratio:.0%})"
                    )
                    cascade_events.append(event)
                    
                    self._apply_failure(graph, comp_id, FailureType.COMPLETE, 1.0)
        
        # Recursive cascade
        if cascade_failures:
            more_failures, more_events = self._propagate_cascade(
                graph, cascade_failures, failure_mode, depth + 1
            )
            cascade_failures.extend(more_failures)
            cascade_events.extend(more_events)
        
        return cascade_failures, cascade_events
    
    def _select_attack_targets(self, graph: SimulationGraph,
                                strategy: AttackStrategy, count: int) -> List[str]:
        """Select attack targets based on strategy"""
        components = list(graph.components.keys())
        
        if strategy == AttackStrategy.RANDOM:
            return self._rng.sample(components, min(count, len(components)))
        
        elif strategy == AttackStrategy.HIGHEST_DEGREE:
            # Sort by total connections
            by_degree = [
                (c, len(graph.get_outgoing(c)) + len(graph.get_incoming(c)))
                for c in components
            ]
            by_degree.sort(key=lambda x: -x[1])
            return [c for c, _ in by_degree[:count]]
        
        elif strategy == AttackStrategy.HIGHEST_WEIGHT:
            # Sort by total weight of incoming dependencies
            by_weight = []
            for c in components:
                total_weight = sum(d.weight for d in graph.get_incoming(c))
                by_weight.append((c, total_weight))
            by_weight.sort(key=lambda x: -x[1])
            return [c for c, _ in by_weight[:count]]
        
        elif strategy == AttackStrategy.BROKERS_FIRST:
            brokers = graph.get_by_type(ComponentType.BROKER)
            if len(brokers) >= count:
                return brokers[:count]
            # Add other components if not enough brokers
            others = [c for c in components if c not in brokers]
            return brokers + others[:count - len(brokers)]
        
        elif strategy == AttackStrategy.HIGHEST_BETWEENNESS:
            # Approximate betweenness using path sampling
            betweenness = {c: 0 for c in components}
            sample_size = min(100, len(components) * 2)
            
            for _ in range(sample_size):
                if len(components) < 2:
                    break
                src, dst = self._rng.sample(components, 2)
                path = self._find_path(graph, src, dst)
                for node in path[1:-1]:  # Exclude endpoints
                    betweenness[node] += 1
            
            by_bc = sorted(betweenness.items(), key=lambda x: -x[1])
            return [c for c, _ in by_bc[:count]]
        
        return components[:count]
    
    def _find_path(self, graph: SimulationGraph, src: str, dst: str) -> List[str]:
        """Find a path between two components (BFS)"""
        if src == dst:
            return [src]
        
        visited = {src}
        queue = [(src, [src])]
        
        while queue:
            current, path = queue.pop(0)
            
            for dep in graph.get_outgoing(current):
                if not dep.is_active:
                    continue
                target = dep.target
                if target == dst:
                    return path + [target]
                if target not in visited:
                    visited.add(target)
                    queue.append((target, path + [target]))
        
        return []
    
    def _build_result(self,
                      sim_id: str,
                      sim_type: str,
                      original_graph: SimulationGraph,
                      sim_graph: SimulationGraph,
                      original_reach: int,
                      original_cc: int,
                      primary_failures: List[str],
                      cascade_failures: List[str],
                      failure_events: List[FailureEvent],
                      start_time: datetime,
                      end_time: datetime) -> SimulationResult:
        """Build simulation result with metrics"""
        # Calculate final metrics
        final_reach = sim_graph.calculate_total_reachability()
        final_cc = sim_graph.count_connected_components()
        
        reach_loss = original_reach - final_reach
        reach_loss_pct = (reach_loss / original_reach * 100) if original_reach > 0 else 0
        
        # Count affected dependencies
        affected_deps = sum(1 for d in sim_graph.dependencies if not d.is_active)
        
        # Find isolated components
        active_comps = sim_graph.get_active_components()
        isolated = []
        for comp_id in active_comps:
            if not sim_graph.get_outgoing(comp_id) and not sim_graph.get_incoming(comp_id):
                # Check if it was connected before
                if original_graph.get_outgoing(comp_id) or original_graph.get_incoming(comp_id):
                    isolated.append(comp_id)
        
        # Calculate impact score
        # Combines reachability loss, fragmentation, and cascade extent
        impact_score = (
            0.5 * (reach_loss_pct / 100) +
            0.3 * (final_cc - original_cc) / max(original_cc, 1) +
            0.2 * len(cascade_failures) / max(len(original_graph.components), 1)
        )
        impact_score = min(1.0, max(0.0, impact_score))
        
        impact = ImpactMetrics(
            original_reachability=original_reach,
            final_reachability=final_reach,
            reachability_loss=reach_loss,
            reachability_loss_pct=reach_loss_pct,
            original_components=original_cc,
            final_components=final_cc,
            components_added=final_cc - original_cc,
            original_node_count=len(original_graph.components),
            final_active_count=len(active_comps),
            failed_count=len(primary_failures) + len(cascade_failures),
            cascade_count=len(cascade_failures),
            impact_score=impact_score
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
            affected_dependencies=affected_deps,
            isolated_components=isolated
        )