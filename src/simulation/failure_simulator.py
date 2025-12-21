#!/usr/bin/env python3
"""
Failure Simulator for Pub-Sub Systems
======================================

Comprehensive failure simulation including:
- Single and multiple component failures
- Cascading failure propagation
- Network/connection failures
- Partial degradation simulation
- Recovery scenario testing
- Impact quantification and analysis

Author: Software-as-a-Graph Research Project
"""

import networkx as nx
import random
import math
from typing import Dict, List, Set, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict
import logging


# ============================================================================
# Enums
# ============================================================================

class FailureType(Enum):
    """Types of failures that can be simulated"""
    COMPLETE = "complete"           # Total component failure
    PARTIAL = "partial"             # Degraded performance
    NETWORK = "network"             # Connection failure
    OVERLOAD = "overload"           # Resource exhaustion
    CASCADE = "cascade"             # Triggered by another failure
    TRANSIENT = "transient"         # Temporary failure


class FailureMode(Enum):
    """How the failure manifests"""
    CRASH = "crash"                 # Immediate stop
    HANG = "hang"                   # Unresponsive
    BYZANTINE = "byzantine"         # Incorrect behavior
    SLOWDOWN = "slowdown"           # Performance degradation
    INTERMITTENT = "intermittent"   # On-off behavior


class RecoveryStrategy(Enum):
    """Recovery strategies for components"""
    NONE = "none"                   # No recovery
    RESTART = "restart"             # Component restart
    FAILOVER = "failover"           # Switch to backup
    DEGRADED = "degraded"           # Continue with reduced capacity
    CIRCUIT_BREAKER = "circuit_breaker"  # Stop and wait


class AttackStrategy(Enum):
    """Targeted attack strategies"""
    RANDOM = "random"               # Random selection
    CRITICALITY = "criticality"     # Target high-criticality nodes
    BETWEENNESS = "betweenness"     # Target high-betweenness nodes
    DEGREE = "degree"               # Target high-degree nodes
    ARTICULATION = "articulation"   # Target articulation points
    BRIDGES = "bridges"             # Target bridge edges


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class FailureEvent:
    """Represents a single failure event"""
    component: str
    failure_type: FailureType
    failure_mode: FailureMode
    timestamp: datetime
    severity: float  # 0.0 to 1.0
    cause: str = ""
    is_cascade: bool = False
    cascade_depth: int = 0
    recovery_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'component': self.component,
            'failure_type': self.failure_type.value,
            'failure_mode': self.failure_mode.value,
            'timestamp': self.timestamp.isoformat(),
            'severity': round(self.severity, 4),
            'cause': self.cause,
            'is_cascade': self.is_cascade,
            'cascade_depth': self.cascade_depth,
            'recovery_time': self.recovery_time
        }


@dataclass
class ImpactMetrics:
    """Quantified impact of a failure"""
    # Reachability
    original_reachability: int
    remaining_reachability: int
    reachability_loss: float
    
    # Connectivity
    original_components: int
    remaining_components: int
    fragmentation: int  # New disconnected components
    
    # Component-level
    affected_nodes: List[str]
    isolated_nodes: List[str]
    degraded_nodes: List[str]
    
    # Service-level
    affected_topics: List[str]
    affected_applications: List[str]
    affected_brokers: List[str]
    
    # Cascade metrics
    cascade_depth: int
    cascade_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'reachability': {
                'original': self.original_reachability,
                'remaining': self.remaining_reachability,
                'loss': round(self.reachability_loss, 4),
                'loss_percentage': round(self.reachability_loss * 100, 2)
            },
            'connectivity': {
                'original_components': self.original_components,
                'remaining_components': self.remaining_components,
                'fragmentation': self.fragmentation
            },
            'affected': {
                'total_nodes': len(self.affected_nodes),
                'isolated_nodes': len(self.isolated_nodes),
                'degraded_nodes': len(self.degraded_nodes),
                'topics': self.affected_topics,
                'applications': self.affected_applications,
                'brokers': self.affected_brokers
            },
            'cascade': {
                'depth': self.cascade_depth,
                'count': self.cascade_count
            }
        }


@dataclass
class SimulationResult:
    """Complete results from a failure simulation"""
    simulation_id: str
    simulation_type: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    
    # Failed components
    primary_failures: List[str]
    cascade_failures: List[str]
    all_failures: List[str]
    failure_events: List[FailureEvent]
    
    # Impact
    impact: ImpactMetrics
    impact_score: float  # 0.0 to 1.0
    resilience_score: float  # 1.0 - impact_score
    
    # Graph state
    original_nodes: int
    original_edges: int
    remaining_nodes: int
    remaining_edges: int
    
    # Analysis
    critical_path_affected: bool = False
    spof_triggered: bool = False
    
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
                'total': len(self.all_failures),
                'events': [e.to_dict() for e in self.failure_events]
            },
            'impact': self.impact.to_dict(),
            'scores': {
                'impact': round(self.impact_score, 4),
                'resilience': round(self.resilience_score, 4)
            },
            'graph': {
                'original_nodes': self.original_nodes,
                'original_edges': self.original_edges,
                'remaining_nodes': self.remaining_nodes,
                'remaining_edges': self.remaining_edges,
                'nodes_lost': self.original_nodes - self.remaining_nodes,
                'edges_lost': self.original_edges - self.remaining_edges
            },
            'analysis': {
                'critical_path_affected': self.critical_path_affected,
                'spof_triggered': self.spof_triggered
            }
        }
    
    def summary(self) -> str:
        """Generate human-readable summary"""
        lines = [
            f"Simulation: {self.simulation_id}",
            f"Type: {self.simulation_type}",
            f"Duration: {self.duration_ms:.2f}ms",
            f"",
            f"Failures:",
            f"  Primary: {len(self.primary_failures)}",
            f"  Cascade: {len(self.cascade_failures)}",
            f"  Total: {len(self.all_failures)}",
            f"",
            f"Impact:",
            f"  Reachability Loss: {self.impact.reachability_loss*100:.1f}%",
            f"  Nodes Affected: {len(self.impact.affected_nodes)}",
            f"  Fragmentation: {self.impact.fragmentation} new components",
            f"",
            f"Scores:",
            f"  Impact: {self.impact_score:.4f}",
            f"  Resilience: {self.resilience_score:.4f}"
        ]
        return "\n".join(lines)


@dataclass
class BatchSimulationResult:
    """Results from batch/exhaustive simulation"""
    total_simulations: int
    completed_simulations: int
    failed_simulations: int
    
    results: List[SimulationResult]
    
    # Aggregated metrics
    avg_impact_score: float
    max_impact_score: float
    min_impact_score: float
    
    # Rankings
    most_critical: List[Tuple[str, float]]  # (component, impact_score)
    least_critical: List[Tuple[str, float]]
    
    # Timing
    total_duration_ms: float
    avg_duration_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'summary': {
                'total': self.total_simulations,
                'completed': self.completed_simulations,
                'failed': self.failed_simulations
            },
            'metrics': {
                'avg_impact': round(self.avg_impact_score, 4),
                'max_impact': round(self.max_impact_score, 4),
                'min_impact': round(self.min_impact_score, 4)
            },
            'rankings': {
                'most_critical': [(c, round(s, 4)) for c, s in self.most_critical[:10]],
                'least_critical': [(c, round(s, 4)) for c, s in self.least_critical[:10]]
            },
            'timing': {
                'total_ms': round(self.total_duration_ms, 2),
                'avg_ms': round(self.avg_duration_ms, 2)
            },
            'results': [r.to_dict() for r in self.results]
        }


# ============================================================================
# Failure Simulator
# ============================================================================

class FailureSimulator:
    """
    Comprehensive failure simulation for pub-sub systems.
    
    Simulates various failure scenarios and measures impact using
    graph-based analysis of DEPENDS_ON relationships.
    """
    
    def __init__(self,
                 cascade_threshold: float = 0.7,
                 cascade_probability: float = 0.5,
                 max_cascade_depth: int = 5,
                 seed: Optional[int] = None):
        """
        Initialize failure simulator.
        
        Args:
            cascade_threshold: Dependency loss ratio to trigger cascade (0-1)
            cascade_probability: Base probability of cascade failure (0-1)
            max_cascade_depth: Maximum cascade propagation depth
            seed: Random seed for reproducibility
        """
        self.cascade_threshold = cascade_threshold
        self.cascade_probability = cascade_probability
        self.max_cascade_depth = max_cascade_depth
        
        if seed is not None:
            random.seed(seed)
        
        self._simulation_counter = 0
        self.logger = logging.getLogger('FailureSimulator')
    
    # =========================================================================
    # Core Simulation Methods
    # =========================================================================
    
    def simulate_single_failure(self,
                               graph: nx.DiGraph,
                               component: str,
                               failure_type: FailureType = FailureType.COMPLETE,
                               failure_mode: FailureMode = FailureMode.CRASH,
                               severity: float = 1.0,
                               enable_cascade: bool = True) -> SimulationResult:
        """
        Simulate failure of a single component.
        
        Args:
            graph: NetworkX directed graph
            component: Component to fail
            failure_type: Type of failure
            failure_mode: How the failure manifests
            severity: Failure severity (0-1)
            enable_cascade: Whether to simulate cascading failures
            
        Returns:
            SimulationResult with comprehensive impact analysis
        """
        if component not in graph.nodes():
            raise ValueError(f"Component '{component}' not found in graph")
        
        self._simulation_counter += 1
        sim_id = f"sim_{self._simulation_counter:05d}"
        start_time = datetime.now()
        
        self.logger.info(f"[{sim_id}] Simulating {failure_type.value} failure of '{component}'")
        
        # Store original metrics
        original_reach = self._calculate_reachability(graph)
        original_cc = nx.number_weakly_connected_components(graph)
        
        # Create working copy
        sim_graph = graph.copy()
        failure_events = []
        primary_failures = [component]
        cascade_failures = []
        
        # Create initial failure event
        event = FailureEvent(
            component=component,
            failure_type=failure_type,
            failure_mode=failure_mode,
            timestamp=start_time,
            severity=severity,
            cause="Primary failure (simulated)",
            is_cascade=False,
            cascade_depth=0
        )
        failure_events.append(event)
        
        # Apply failure
        if failure_type == FailureType.COMPLETE:
            sim_graph.remove_node(component)
        elif failure_type == FailureType.PARTIAL:
            sim_graph.nodes[component]['degraded'] = True
            sim_graph.nodes[component]['capacity'] = 1.0 - severity
        
        # Simulate cascading failures
        if enable_cascade and failure_type == FailureType.COMPLETE:
            cascade_failures, cascade_events = self._propagate_cascade(
                graph, sim_graph, primary_failures, failure_mode, 1
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
            primary_failures=primary_failures,
            cascade_failures=cascade_failures,
            failure_events=failure_events,
            start_time=start_time,
            end_time=end_time
        )
    
    def simulate_multiple_failures(self,
                                  graph: nx.DiGraph,
                                  components: List[str],
                                  failure_type: FailureType = FailureType.COMPLETE,
                                  simultaneous: bool = True,
                                  enable_cascade: bool = True) -> SimulationResult:
        """
        Simulate failure of multiple components.
        
        Args:
            graph: NetworkX directed graph
            components: List of components to fail
            failure_type: Type of failure
            simultaneous: Whether failures occur at same time
            enable_cascade: Whether to simulate cascading failures
            
        Returns:
            SimulationResult with comprehensive impact analysis
        """
        # Validate components
        for comp in components:
            if comp not in graph.nodes():
                raise ValueError(f"Component '{comp}' not found in graph")
        
        self._simulation_counter += 1
        sim_id = f"sim_{self._simulation_counter:05d}"
        start_time = datetime.now()
        
        self.logger.info(f"[{sim_id}] Simulating failure of {len(components)} components")
        
        # Store original metrics
        original_reach = self._calculate_reachability(graph)
        original_cc = nx.number_weakly_connected_components(graph)
        
        # Create working copy
        sim_graph = graph.copy()
        failure_events = []
        primary_failures = list(components)
        cascade_failures = []
        
        # Create failure events
        for i, component in enumerate(components):
            timestamp = start_time if simultaneous else datetime.now()
            event = FailureEvent(
                component=component,
                failure_type=failure_type,
                failure_mode=FailureMode.CRASH,
                timestamp=timestamp,
                severity=1.0,
                cause=f"Multiple failure ({i+1}/{len(components)})",
                is_cascade=False,
                cascade_depth=0
            )
            failure_events.append(event)
        
        # Apply failures
        for comp in components:
            if comp in sim_graph.nodes():
                if failure_type == FailureType.COMPLETE:
                    sim_graph.remove_node(comp)
                elif failure_type == FailureType.PARTIAL:
                    sim_graph.nodes[comp]['degraded'] = True
        
        # Simulate cascading failures
        if enable_cascade:
            cascade_failures, cascade_events = self._propagate_cascade(
                graph, sim_graph, primary_failures, FailureMode.CRASH, 1
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
            primary_failures=primary_failures,
            cascade_failures=cascade_failures,
            failure_events=failure_events,
            start_time=start_time,
            end_time=end_time
        )
    
    def simulate_network_failure(self,
                                graph: nx.DiGraph,
                                source: str,
                                target: str) -> SimulationResult:
        """
        Simulate failure of a network connection (edge).
        
        Args:
            graph: NetworkX directed graph
            source: Source node of edge
            target: Target node of edge
            
        Returns:
            SimulationResult with impact analysis
        """
        if not graph.has_edge(source, target):
            raise ValueError(f"Edge '{source}' -> '{target}' not found")
        
        self._simulation_counter += 1
        sim_id = f"sim_{self._simulation_counter:05d}"
        start_time = datetime.now()
        
        self.logger.info(f"[{sim_id}] Simulating network failure: {source} -> {target}")
        
        # Store original metrics
        original_reach = self._calculate_reachability(graph)
        original_cc = nx.number_weakly_connected_components(graph)
        
        # Create working copy and remove edge
        sim_graph = graph.copy()
        sim_graph.remove_edge(source, target)
        
        # Create failure event
        failure_events = [FailureEvent(
            component=f"{source}->{target}",
            failure_type=FailureType.NETWORK,
            failure_mode=FailureMode.CRASH,
            timestamp=start_time,
            severity=1.0,
            cause="Network connection failure"
        )]
        
        end_time = datetime.now()
        
        return self._build_result(
            sim_id=sim_id,
            sim_type="network_failure",
            original_graph=graph,
            sim_graph=sim_graph,
            original_reach=original_reach,
            original_cc=original_cc,
            primary_failures=[],
            cascade_failures=[],
            failure_events=failure_events,
            start_time=start_time,
            end_time=end_time
        )
    
    def simulate_random_failures(self,
                                graph: nx.DiGraph,
                                failure_probability: float = 0.1,
                                component_types: Optional[List[str]] = None,
                                enable_cascade: bool = True) -> SimulationResult:
        """
        Simulate random failures based on probability.
        
        Args:
            graph: NetworkX directed graph
            failure_probability: Probability of each component failing
            component_types: Limit to specific types
            enable_cascade: Whether to simulate cascading failures
            
        Returns:
            SimulationResult with impact analysis
        """
        # Select candidates
        candidates = []
        for node in graph.nodes():
            node_type = graph.nodes[node].get('type', 'Unknown')
            if component_types is None or node_type in component_types:
                candidates.append(node)
        
        # Apply probability
        to_fail = [c for c in candidates if random.random() < failure_probability]
        
        if not to_fail:
            self.logger.info("No components selected for failure")
            return self._create_empty_result(graph, "random_failure")
        
        return self.simulate_multiple_failures(
            graph, to_fail,
            enable_cascade=enable_cascade
        )
    
    def simulate_targeted_attack(self,
                                graph: nx.DiGraph,
                                strategy: AttackStrategy = AttackStrategy.CRITICALITY,
                                target_count: int = 5,
                                enable_cascade: bool = True) -> SimulationResult:
        """
        Simulate targeted attack using specified strategy.
        
        Args:
            graph: NetworkX directed graph
            strategy: Attack targeting strategy
            target_count: Number of components to target
            enable_cascade: Whether to simulate cascading failures
            
        Returns:
            SimulationResult with impact analysis
        """
        self.logger.info(f"Simulating targeted attack: strategy={strategy.value}, count={target_count}")
        
        # Select targets based on strategy
        targets = self._select_attack_targets(graph, strategy, target_count)
        
        if not targets:
            return self._create_empty_result(graph, "targeted_attack")
        
        return self.simulate_multiple_failures(
            graph, targets,
            enable_cascade=enable_cascade
        )
    
    def simulate_exhaustive(self,
                           graph: nx.DiGraph,
                           component_types: Optional[List[str]] = None,
                           enable_cascade: bool = False) -> BatchSimulationResult:
        """
        Simulate failure of each component individually.
        
        Args:
            graph: NetworkX directed graph
            component_types: Limit to specific component types
            enable_cascade: Whether to simulate cascading failures
            
        Returns:
            BatchSimulationResult with all individual results
        """
        # Select components to test
        components = []
        for node in graph.nodes():
            node_type = graph.nodes[node].get('type', 'Unknown')
            if component_types is None or node_type in component_types:
                components.append(node)
        
        self.logger.info(f"Running exhaustive simulation for {len(components)} components")
        
        results = []
        start_time = datetime.now()
        
        for comp in components:
            try:
                result = self.simulate_single_failure(
                    graph, comp,
                    enable_cascade=enable_cascade
                )
                results.append(result)
            except Exception as e:
                self.logger.warning(f"Failed to simulate {comp}: {e}")
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds() * 1000
        
        return self._build_batch_result(results, len(components), total_duration)
    
    # =========================================================================
    # Cascade Propagation
    # =========================================================================
    
    def _propagate_cascade(self,
                          original_graph: nx.DiGraph,
                          sim_graph: nx.DiGraph,
                          failed: List[str],
                          failure_mode: FailureMode,
                          depth: int) -> Tuple[List[str], List[FailureEvent]]:
        """Propagate cascading failures"""
        if depth > self.max_cascade_depth:
            return [], []
        
        cascade_failures = []
        cascade_events = []
        
        # Find components that depend on failed components
        for failed_comp in failed:
            dependents = self._find_dependents(original_graph, failed_comp)
            
            for dep in dependents:
                if dep not in sim_graph.nodes():
                    continue  # Already failed
                
                if self._should_cascade(original_graph, sim_graph, dep, failed):
                    # Create cascade event
                    event = FailureEvent(
                        component=dep,
                        failure_type=FailureType.CASCADE,
                        failure_mode=failure_mode,
                        timestamp=datetime.now(),
                        severity=1.0,
                        cause=f"Cascade from {failed_comp}",
                        is_cascade=True,
                        cascade_depth=depth
                    )
                    cascade_events.append(event)
                    cascade_failures.append(dep)
                    
                    # Remove from graph
                    sim_graph.remove_node(dep)
        
        # Recurse for new failures
        if cascade_failures:
            new_failures, new_events = self._propagate_cascade(
                original_graph, sim_graph, cascade_failures, failure_mode, depth + 1
            )
            cascade_failures.extend(new_failures)
            cascade_events.extend(new_events)
        
        return cascade_failures, cascade_events
    
    def _find_dependents(self, graph: nx.DiGraph, component: str) -> List[str]:
        """Find components that depend on the given component"""
        dependents = []
        
        # Direct predecessors (components that have edges TO this component)
        for pred in graph.predecessors(component):
            dependents.append(pred)
        
        # Components connected via topics/brokers
        for succ in graph.successors(component):
            for pred in graph.predecessors(succ):
                if pred != component and pred not in dependents:
                    dependents.append(pred)
        
        return dependents
    
    def _should_cascade(self, original: nx.DiGraph, current: nx.DiGraph,
                       component: str, failed: List[str]) -> bool:
        """Determine if a component should fail due to cascade"""
        # Count original dependencies
        original_deps = set(original.predecessors(component)) | set(original.successors(component))
        
        # Count remaining dependencies
        if component in current:
            current_deps = set(current.predecessors(component)) | set(current.successors(component))
        else:
            return False
        
        # Calculate dependency loss ratio
        if len(original_deps) == 0:
            return False
        
        lost_deps = len(original_deps) - len(current_deps)
        loss_ratio = lost_deps / len(original_deps)
        
        # Check threshold and apply probability
        if loss_ratio >= self.cascade_threshold:
            return random.random() < self.cascade_probability
        
        return False
    
    def _select_attack_targets(self, graph: nx.DiGraph,
                              strategy: AttackStrategy,
                              count: int) -> List[str]:
        """Select attack targets based on strategy"""
        targets = []
        
        if strategy == AttackStrategy.RANDOM:
            candidates = list(graph.nodes())
            targets = random.sample(candidates, min(count, len(candidates)))
        
        elif strategy == AttackStrategy.BETWEENNESS:
            bc = nx.betweenness_centrality(graph)
            sorted_nodes = sorted(bc.items(), key=lambda x: -x[1])
            targets = [n for n, _ in sorted_nodes[:count]]
        
        elif strategy == AttackStrategy.DEGREE:
            degrees = dict(graph.degree())
            sorted_nodes = sorted(degrees.items(), key=lambda x: -x[1])
            targets = [n for n, _ in sorted_nodes[:count]]
        
        elif strategy == AttackStrategy.ARTICULATION:
            try:
                aps = list(nx.articulation_points(graph.to_undirected()))
                targets = aps[:count]
            except:
                targets = []
        
        elif strategy == AttackStrategy.CRITICALITY:
            # Use combination of metrics
            bc = nx.betweenness_centrality(graph)
            try:
                aps = set(nx.articulation_points(graph.to_undirected()))
            except:
                aps = set()
            
            scores = {}
            for node in graph.nodes():
                score = bc.get(node, 0) * 0.6
                if node in aps:
                    score += 0.4
                scores[node] = score
            
            sorted_nodes = sorted(scores.items(), key=lambda x: -x[1])
            targets = [n for n, _ in sorted_nodes[:count]]
        
        return targets
    
    # =========================================================================
    # Impact Analysis
    # =========================================================================
    
    def _calculate_reachability(self, graph: nx.DiGraph) -> int:
        """Calculate total reachable pairs in graph"""
        total = 0
        for node in graph.nodes():
            try:
                descendants = nx.descendants(graph, node)
                total += len(descendants)
            except:
                pass
        return total
    
    def _analyze_impact(self, original: nx.DiGraph, current: nx.DiGraph,
                       failed: List[str]) -> ImpactMetrics:
        """Analyze the impact of failures"""
        # Reachability
        orig_reach = self._calculate_reachability(original)
        curr_reach = self._calculate_reachability(current)
        reach_loss = 1.0 - (curr_reach / orig_reach) if orig_reach > 0 else 0.0
        
        # Connectivity
        orig_cc = nx.number_weakly_connected_components(original)
        curr_cc = nx.number_weakly_connected_components(current)
        
        # Find affected components by type
        affected_nodes = list(failed)
        isolated_nodes = []
        degraded_nodes = []
        
        affected_topics = []
        affected_apps = []
        affected_brokers = []
        
        for node in failed:
            node_type = original.nodes[node].get('type', 'Unknown')
            if node_type == 'Topic':
                affected_topics.append(node)
            elif node_type == 'Application':
                affected_apps.append(node)
            elif node_type == 'Broker':
                affected_brokers.append(node)
        
        # Find isolated nodes (nodes with no remaining connections)
        for node in current.nodes():
            if current.degree(node) == 0:
                isolated_nodes.append(node)
            if current.nodes[node].get('degraded', False):
                degraded_nodes.append(node)
        
        # Calculate cascade metrics
        cascade_count = sum(1 for f in failed if f not in failed[:1])
        cascade_depth = 0
        
        return ImpactMetrics(
            original_reachability=orig_reach,
            remaining_reachability=curr_reach,
            reachability_loss=reach_loss,
            original_components=orig_cc,
            remaining_components=curr_cc,
            fragmentation=curr_cc - orig_cc,
            affected_nodes=affected_nodes,
            isolated_nodes=isolated_nodes,
            degraded_nodes=degraded_nodes,
            affected_topics=affected_topics,
            affected_applications=affected_apps,
            affected_brokers=affected_brokers,
            cascade_depth=cascade_depth,
            cascade_count=cascade_count
        )
    
    def _build_result(self,
                     sim_id: str,
                     sim_type: str,
                     original_graph: nx.DiGraph,
                     sim_graph: nx.DiGraph,
                     original_reach: int,
                     original_cc: int,
                     primary_failures: List[str],
                     cascade_failures: List[str],
                     failure_events: List[FailureEvent],
                     start_time: datetime,
                     end_time: datetime) -> SimulationResult:
        """Build simulation result"""
        all_failures = primary_failures + cascade_failures
        
        # Analyze impact
        impact = self._analyze_impact(original_graph, sim_graph, all_failures)
        
        # Calculate scores
        nodes_affected = len(all_failures) / original_graph.number_of_nodes() if original_graph.number_of_nodes() > 0 else 0
        impact_score = (impact.reachability_loss * 0.5 + nodes_affected * 0.3 + 
                       (impact.fragmentation / max(original_cc, 1)) * 0.2)
        impact_score = min(1.0, impact_score)
        
        # Check for SPOF
        try:
            aps = set(nx.articulation_points(original_graph.to_undirected()))
            spof_triggered = any(f in aps for f in primary_failures)
        except:
            spof_triggered = False
        
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        return SimulationResult(
            simulation_id=sim_id,
            simulation_type=sim_type,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            primary_failures=primary_failures,
            cascade_failures=cascade_failures,
            all_failures=all_failures,
            failure_events=failure_events,
            impact=impact,
            impact_score=impact_score,
            resilience_score=1.0 - impact_score,
            original_nodes=original_graph.number_of_nodes(),
            original_edges=original_graph.number_of_edges(),
            remaining_nodes=sim_graph.number_of_nodes(),
            remaining_edges=sim_graph.number_of_edges(),
            critical_path_affected=False,
            spof_triggered=spof_triggered
        )
    
    def _build_batch_result(self, results: List[SimulationResult],
                           total: int, duration_ms: float) -> BatchSimulationResult:
        """Build batch simulation result"""
        if not results:
            return BatchSimulationResult(
                total_simulations=total,
                completed_simulations=0,
                failed_simulations=total,
                results=[],
                avg_impact_score=0.0,
                max_impact_score=0.0,
                min_impact_score=0.0,
                most_critical=[],
                least_critical=[],
                total_duration_ms=duration_ms,
                avg_duration_ms=0.0
            )
        
        impacts = [r.impact_score for r in results]
        
        # Create rankings
        component_impacts = []
        for r in results:
            if r.primary_failures:
                component_impacts.append((r.primary_failures[0], r.impact_score))
        
        sorted_impacts = sorted(component_impacts, key=lambda x: -x[1])
        
        return BatchSimulationResult(
            total_simulations=total,
            completed_simulations=len(results),
            failed_simulations=total - len(results),
            results=results,
            avg_impact_score=sum(impacts) / len(impacts),
            max_impact_score=max(impacts),
            min_impact_score=min(impacts),
            most_critical=sorted_impacts[:10],
            least_critical=sorted_impacts[-10:][::-1],
            total_duration_ms=duration_ms,
            avg_duration_ms=duration_ms / len(results)
        )
    
    def _create_empty_result(self, graph: nx.DiGraph, sim_type: str) -> SimulationResult:
        """Create empty result when no failures occur"""
        now = datetime.now()
        self._simulation_counter += 1
        
        return SimulationResult(
            simulation_id=f"sim_{self._simulation_counter:05d}",
            simulation_type=sim_type,
            start_time=now,
            end_time=now,
            duration_ms=0.0,
            primary_failures=[],
            cascade_failures=[],
            all_failures=[],
            failure_events=[],
            impact=ImpactMetrics(
                original_reachability=self._calculate_reachability(graph),
                remaining_reachability=self._calculate_reachability(graph),
                reachability_loss=0.0,
                original_components=nx.number_weakly_connected_components(graph),
                remaining_components=nx.number_weakly_connected_components(graph),
                fragmentation=0,
                affected_nodes=[],
                isolated_nodes=[],
                degraded_nodes=[],
                affected_topics=[],
                affected_applications=[],
                affected_brokers=[],
                cascade_depth=0,
                cascade_count=0
            ),
            impact_score=0.0,
            resilience_score=1.0,
            original_nodes=graph.number_of_nodes(),
            original_edges=graph.number_of_edges(),
            remaining_nodes=graph.number_of_nodes(),
            remaining_edges=graph.number_of_edges()
        )