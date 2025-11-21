"""
Enhanced Failure Simulator for Lightweight DDS Simulation

Integrates failure simulation capabilities with the lightweight event-driven simulator.
Supports:
- Single component failures (node, application, topic, broker)
- Multiple simultaneous failures
- Cascading failure propagation with load modeling
- Network partition simulation
- Degraded performance scenarios
- Recovery and healing scenarios
- Real-time failure impact metrics
- Failure injection during active simulation

Features:
- Load-based cascade propagation
- Dependency-aware failure impact
- QoS-aware degradation
- Time-series impact tracking
- Automated resilience recommendations
"""

import asyncio
import time
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime
from enum import Enum
import json
import logging
import random
import copy


class FailureType(Enum):
    """Types of failures"""
    COMPLETE = "complete"           # Total failure
    PARTIAL = "partial"             # Degraded performance
    INTERMITTENT = "intermittent"   # On/off failure
    CASCADE = "cascade"             # Propagated failure
    NETWORK = "network"             # Network partition
    OVERLOAD = "overload"           # Resource exhaustion


class ComponentType(Enum):
    """Component types that can fail"""
    NODE = "node"
    APPLICATION = "application"
    TOPIC = "topic"
    BROKER = "broker"
    NETWORK_LINK = "network_link"


@dataclass
class FailureEvent:
    """Represents a failure event"""
    component_id: str
    component_type: ComponentType
    failure_type: FailureType
    timestamp: float
    severity: float  # 0.0 - 1.0
    cause: Optional[str] = None
    propagated_from: Optional[str] = None
    recovered: bool = False
    recovery_time: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'component_id': self.component_id,
            'component_type': self.component_type.value,
            'failure_type': self.failure_type.value,
            'timestamp': self.timestamp,
            'severity': self.severity,
            'cause': self.cause,
            'propagated_from': self.propagated_from,
            'recovered': self.recovered,
            'recovery_time': self.recovery_time
        }


@dataclass
class FailureImpact:
    """Tracks impact of failures"""
    failed_components: Set[str] = field(default_factory=set)
    affected_components: Set[str] = field(default_factory=set)
    isolated_applications: Set[str] = field(default_factory=set)
    unavailable_topics: Set[str] = field(default_factory=set)
    
    messages_lost: int = 0
    messages_delayed: int = 0
    deadline_violations: int = 0
    
    avg_latency_before: float = 0.0
    avg_latency_after: float = 0.0
    
    throughput_before: float = 0.0
    throughput_after: float = 0.0
    
    delivery_rate_before: float = 0.0
    delivery_rate_after: float = 0.0
    
    cascade_depth: int = 0
    cascade_width: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'failed_components': list(self.failed_components),
            'affected_components': list(self.affected_components),
            'isolated_applications': list(self.isolated_applications),
            'unavailable_topics': list(self.unavailable_topics),
            'messages_lost': self.messages_lost,
            'messages_delayed': self.messages_delayed,
            'deadline_violations': self.deadline_violations,
            'latency_increase': self.avg_latency_after - self.avg_latency_before,
            'latency_increase_pct': 
                ((self.avg_latency_after - self.avg_latency_before) / self.avg_latency_before * 100) 
                if self.avg_latency_before > 0 else 0,
            'throughput_decrease': self.throughput_before - self.throughput_after,
            'throughput_decrease_pct':
                ((self.throughput_before - self.throughput_after) / self.throughput_before * 100)
                if self.throughput_before > 0 else 0,
            'delivery_rate_decrease': self.delivery_rate_before - self.delivery_rate_after,
            'cascade_depth': self.cascade_depth,
            'cascade_width': self.cascade_width
        }


class FailureSimulator:
    """
    Enhanced failure simulator for lightweight DDS simulation
    
    Integrates with LightweightDDSSimulator to inject failures and track impact
    """
    
    def __init__(self, 
                 cascade_threshold: float = 0.8,
                 cascade_probability: float = 0.6,
                 recovery_enabled: bool = False):
        """
        Initialize failure simulator
        
        Args:
            cascade_threshold: Load threshold for cascading (0-1)
            cascade_probability: Probability of cascade propagation (0-1)
            recovery_enabled: Enable automatic recovery
        """
        self.cascade_threshold = cascade_threshold
        self.cascade_probability = cascade_probability
        self.recovery_enabled = recovery_enabled
        
        self.failure_events: List[FailureEvent] = []
        self.active_failures: Dict[str, FailureEvent] = {}
        self.component_loads: Dict[str, float] = defaultdict(float)
        
        self.logger = logging.getLogger(__name__)
    
    def inject_failure(self,
                      simulator: 'LightweightDDSSimulator',
                      component_id: str,
                      component_type: ComponentType,
                      failure_type: FailureType = FailureType.COMPLETE,
                      severity: float = 1.0,
                      enable_cascade: bool = True) -> FailureEvent:
        """
        Inject failure into running simulation
        
        Args:
            simulator: Active LightweightDDSSimulator instance
            component_id: ID of component to fail
            component_type: Type of component
            failure_type: Type of failure
            severity: Failure severity (0-1)
            enable_cascade: Enable cascading failures
            
        Returns:
            FailureEvent describing the failure
        """
        event = FailureEvent(
            component_id=component_id,
            component_type=component_type,
            failure_type=failure_type,
            timestamp=time.time(),
            severity=severity,
            cause="injected"
        )
        
        self.failure_events.append(event)
        self.active_failures[component_id] = event
        
        self.logger.info(f"Injecting {failure_type} failure: {component_type} {component_id}")
        
        # Apply failure to simulator
        self._apply_failure(simulator, event)
        
        # Check for cascading failures
        if enable_cascade and failure_type != FailureType.CASCADE:
            self._propagate_cascade(simulator, component_id, component_type, severity)
        
        return event
    
    def _apply_failure(self, simulator: 'LightweightDDSSimulator', event: FailureEvent):
        """Apply failure to simulator components"""
        
        if event.component_type == ComponentType.APPLICATION:
            app = simulator.applications.get(event.component_id)
            if app:
                if event.failure_type == FailureType.COMPLETE:
                    # Stop application completely
                    app.running = False
                    self.logger.info(f"  Stopped application {event.component_id}")
                    
                elif event.failure_type == FailureType.PARTIAL:
                    # Reduce publishing rate
                    for i, (topic, period, size) in enumerate(app.publish_topics):
                        app.publish_topics[i] = (
                            topic,
                            int(period / (1 - event.severity)),  # Slower publishing
                            size
                        )
                    self.logger.info(f"  Degraded application {event.component_id} performance")
        
        elif event.component_type == ComponentType.BROKER:
            broker = simulator.brokers.get(event.component_id)
            if broker:
                if event.failure_type == FailureType.COMPLETE:
                    # Mark all broker topics as unavailable
                    for topic_id in list(broker.topics.keys()):
                        self.logger.info(f"  Topic {topic_id} unavailable due to broker failure")
                        # Topics remain in broker but we'll track unavailability
                        
                elif event.failure_type == FailureType.PARTIAL:
                    # Increase routing delay by dropping some messages
                    broker.stats.messages_dropped += int(broker.stats.messages_delivered * event.severity)
        
        elif event.component_type == ComponentType.TOPIC:
            topic = simulator.topics.get(event.component_id)
            if topic:
                if event.failure_type == FailureType.COMPLETE:
                    # Clear all subscribers
                    topic.subscribers.clear()
                    self.logger.info(f"  Topic {event.component_id} unavailable")
        
        elif event.component_type == ComponentType.NODE:
            # Fail all applications on this node
            for app_id, app in simulator.applications.items():
                # Check if app runs on this node (simplified - would need runs_on mapping)
                if app_id.startswith(event.component_id):  # Simplified check
                    app.running = False
                    self.logger.info(f"  Application {app_id} failed due to node failure")
    
    def _propagate_cascade(self,
                          simulator: 'LightweightDDSSimulator',
                          source_component: str,
                          source_type: ComponentType,
                          severity: float,
                          depth: int = 0,
                          max_depth: int = 5):
        """
        Propagate cascading failures based on dependencies and load
        
        Args:
            simulator: Simulator instance
            source_component: Failed component
            source_type: Type of failed component
            severity: Failure severity
            depth: Current cascade depth
            max_depth: Maximum cascade depth
        """
        if depth >= max_depth:
            return
        
        self.logger.info(f"  Checking cascade propagation (depth {depth})")
        
        # Find dependent components
        dependent_components = self._find_dependents(simulator, source_component, source_type)
        
        for dep_id, dep_type in dependent_components:
            # Skip if already failed
            if dep_id in self.active_failures:
                continue
            
            # Calculate load on dependent component
            load_increase = self._calculate_load_increase(
                simulator, dep_id, dep_type, source_component
            )
            
            self.component_loads[dep_id] += load_increase
            
            # Check if load exceeds threshold
            if self.component_loads[dep_id] > self.cascade_threshold:
                # Probabilistic cascade
                if random.random() < self.cascade_probability:
                    self.logger.info(f"  Cascading failure to {dep_type.value} {dep_id} "
                                   f"(load: {self.component_loads[dep_id]:.2f})")
                    
                    # Create cascade failure event
                    cascade_event = FailureEvent(
                        component_id=dep_id,
                        component_type=dep_type,
                        failure_type=FailureType.CASCADE,
                        timestamp=time.time(),
                        severity=severity * 0.8,  # Reduced severity
                        propagated_from=source_component
                    )
                    
                    self.failure_events.append(cascade_event)
                    self.active_failures[dep_id] = cascade_event
                    
                    self._apply_failure(simulator, cascade_event)
                    
                    # Continue cascade
                    self._propagate_cascade(
                        simulator, dep_id, dep_type, 
                        cascade_event.severity, depth + 1, max_depth
                    )
    
    def _find_dependents(self,
                        simulator: 'LightweightDDSSimulator',
                        component_id: str,
                        component_type: ComponentType) -> List[Tuple[str, ComponentType]]:
        """Find components that depend on the given component"""
        
        dependents = []
        
        if component_type == ComponentType.BROKER:
            # Applications using this broker's topics depend on it
            broker = simulator.brokers.get(component_id)
            if broker:
                for topic_id in broker.topics.keys():
                    topic = simulator.topics.get(topic_id)
                    if topic:
                        for sub_id in topic.subscribers:
                            dependents.append((sub_id, ComponentType.APPLICATION))
        
        elif component_type == ComponentType.TOPIC:
            # Subscribers depend on this topic
            topic = simulator.topics.get(component_id)
            if topic:
                for sub_id in topic.subscribers:
                    dependents.append((sub_id, ComponentType.APPLICATION))
        
        elif component_type == ComponentType.APPLICATION:
            # Other apps subscribing to topics published by this app depend on it
            app = simulator.applications.get(component_id)
            if app:
                for topic_id, _, _ in app.publish_topics:
                    topic = simulator.topics.get(topic_id)
                    if topic:
                        for sub_id in topic.subscribers:
                            if sub_id != component_id:
                                dependents.append((sub_id, ComponentType.APPLICATION))
        
        return dependents
    
    def _calculate_load_increase(self,
                                simulator: 'LightweightDDSSimulator',
                                component_id: str,
                                component_type: ComponentType,
                                failed_component: str) -> float:
        """Calculate load increase on component due to failure"""
        
        # Simplified load calculation
        # In reality, would calculate based on message rerouting
        
        if component_type == ComponentType.APPLICATION:
            app = simulator.applications.get(component_id)
            if app:
                # Load increases if we're receiving more due to failover
                return 0.3
        
        elif component_type == ComponentType.BROKER:
            # Broker load increases significantly if another broker fails
            return 0.5
        
        return 0.2
    
    def recover_component(self,
                         simulator: 'LightweightDDSSimulator',
                         component_id: str) -> bool:
        """
        Recover a failed component
        
        Args:
            simulator: Simulator instance
            component_id: Component to recover
            
        Returns:
            True if recovered successfully
        """
        if component_id not in self.active_failures:
            return False
        
        event = self.active_failures[component_id]
        event.recovered = True
        event.recovery_time = time.time()
        
        self.logger.info(f"Recovering {event.component_type.value} {component_id}")
        
        # Restore component
        if event.component_type == ComponentType.APPLICATION:
            app = simulator.applications.get(component_id)
            if app:
                app.running = True
        
        elif event.component_type == ComponentType.BROKER:
            broker = simulator.brokers.get(component_id)
            if broker:
                # Topics become available again
                pass
        
        elif event.component_type == ComponentType.TOPIC:
            # Topic becomes available again
            pass
        
        # Remove from active failures
        del self.active_failures[component_id]
        
        # Reduce load on dependent components
        self.component_loads[component_id] = 0.0
        
        return True
    
    def analyze_impact(self,
                      simulator: 'LightweightDDSSimulator',
                      baseline_stats: Optional[Dict] = None) -> FailureImpact:
        """
        Analyze impact of active failures
        
        Args:
            simulator: Simulator instance
            baseline_stats: Statistics before failures (for comparison)
            
        Returns:
            FailureImpact analysis
        """
        impact = FailureImpact()
        
        # Collect failed components
        for component_id, event in self.active_failures.items():
            impact.failed_components.add(component_id)
            
            if event.component_type == ComponentType.APPLICATION:
                impact.affected_components.add(component_id)
                
                # Check if application is isolated
                app = simulator.applications.get(component_id)
                if app and not app.running:
                    impact.isolated_applications.add(component_id)
            
            elif event.component_type == ComponentType.TOPIC:
                impact.unavailable_topics.add(component_id)
                
                # All subscribers are affected
                topic = simulator.topics.get(component_id)
                if topic:
                    for sub_id in topic.subscribers:
                        impact.affected_components.add(sub_id)
            
            elif event.component_type == ComponentType.BROKER:
                # All topics routed by this broker are affected
                broker = simulator.brokers.get(component_id)
                if broker:
                    for topic_id in broker.topics.keys():
                        impact.unavailable_topics.add(topic_id)
        
        # Calculate cascade metrics
        cascade_events = [e for e in self.failure_events 
                         if e.failure_type == FailureType.CASCADE]
        
        if cascade_events:
            impact.cascade_depth = max(
                len([e for e in cascade_events if e.propagated_from == event.component_id])
                for event in self.failure_events if event.failure_type != FailureType.CASCADE
            )
            impact.cascade_width = len(cascade_events)
        
        # Compare with baseline if provided
        if baseline_stats:
            impact.avg_latency_before = baseline_stats.get('avg_latency_ms', 0)
            impact.throughput_before = baseline_stats.get('throughput_msg_s', 0)
            impact.delivery_rate_before = baseline_stats.get('delivery_rate', 0)
        
        # Get current stats
        current_delivered = simulator.global_stats.messages_delivered
        current_sent = simulator.global_stats.messages_sent
        
        if simulator.global_stats.messages_delivered > 0:
            impact.avg_latency_after = (
                simulator.global_stats.total_latency_ms / 
                simulator.global_stats.messages_delivered
            )
        
        if current_sent > 0:
            impact.delivery_rate_after = current_delivered / current_sent
        
        impact.messages_lost = simulator.global_stats.messages_dropped
        impact.deadline_violations = simulator.global_stats.deadline_misses
        
        return impact
    
    async def inject_scheduled_failures(self,
                                       simulator: 'LightweightDDSSimulator',
                                       failure_schedule: List[Dict]):
        """
        Inject failures according to a schedule during simulation
        
        Args:
            simulator: Running simulator
            failure_schedule: List of {time: float, component: str, type: str, ...}
        """
        for failure_spec in sorted(failure_schedule, key=lambda x: x['time']):
            # Wait until scheduled time
            await asyncio.sleep(failure_spec['time'])
            
            # Inject failure
            self.inject_failure(
                simulator,
                failure_spec['component'],
                ComponentType(failure_spec['component_type']),
                FailureType(failure_spec.get('failure_type', 'complete')),
                failure_spec.get('severity', 1.0),
                failure_spec.get('enable_cascade', True)
            )
    
    def generate_recommendations(self,
                               simulator: 'LightweightDDSSimulator',
                               impact: FailureImpact) -> Dict[str, List[str]]:
        """
        Generate resilience recommendations based on failure impact
        
        Args:
            simulator: Simulator instance
            impact: Failure impact analysis
            
        Returns:
            Dictionary of recommendation categories
        """
        recommendations = {
            'replication': [],
            'load_balancing': [],
            'redundancy': [],
            'recovery': []
        }
        
        # Check for single points of failure
        for component_id in impact.failed_components:
            affected = len([c for c in impact.affected_components 
                          if c in simulator.applications])
            
            if affected > 10:
                recommendations['replication'].append(
                    f"Replicate {component_id} - failure affects {affected} applications"
                )
        
        # Check for broker overload
        for broker_id, broker in simulator.brokers.items():
            if len(broker.topics) > 50:
                recommendations['load_balancing'].append(
                    f"Distribute topics from {broker_id} - routing {len(broker.topics)} topics"
                )
        
        # Check for cascade susceptibility
        if impact.cascade_depth > 2:
            recommendations['redundancy'].append(
                f"Add redundancy - cascade depth of {impact.cascade_depth} detected"
            )
        
        # Check for recovery needs
        if len(impact.isolated_applications) > 5:
            recommendations['recovery'].append(
                f"Implement automatic recovery - {len(impact.isolated_applications)} "
                f"applications isolated"
            )
        
        return recommendations


async def run_failure_simulation(json_path: str,
                                 duration_seconds: int,
                                 failure_schedule: List[Dict],
                                 output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run simulation with scheduled failures
    
    Args:
        json_path: Path to graph JSON
        duration_seconds: Simulation duration
        failure_schedule: List of failures to inject
        output_path: Optional path to save results
        
    Returns:
        Complete simulation results with failure analysis
    """
    from .lightweight_dds_simulator import LightweightDDSSimulator
    
    # Create simulator and load topology
    simulator = LightweightDDSSimulator()
    simulator.load_from_json(json_path)
    
    # Create failure simulator
    failure_sim = FailureSimulator(
        cascade_threshold=0.7,
        cascade_probability=0.6
    )
    
    # Get baseline stats
    baseline_time = 10  # Run 10s for baseline
    print(f"Collecting baseline for {baseline_time}s...")
    
    baseline_results = await simulator.run_simulation(baseline_time)
    baseline_stats = {
        'avg_latency_ms': baseline_results['global_stats']['avg_latency_ms'],
        'throughput_msg_s': (
            baseline_results['global_stats']['messages_delivered'] / baseline_time
        ),
        'delivery_rate': baseline_results['global_stats']['delivery_rate']
    }
    
    print(f"Baseline: {baseline_stats['avg_latency_ms']:.2f}ms latency, "
          f"{baseline_stats['throughput_msg_s']:.1f} msg/s")
    
    # Reset simulator for main run
    simulator = LightweightDDSSimulator()
    simulator.load_from_json(json_path)
    
    # Start failure injection task
    failure_task = asyncio.create_task(
        failure_sim.inject_scheduled_failures(simulator, failure_schedule)
    )
    
    # Run simulation
    print(f"\nRunning simulation with failures for {duration_seconds}s...")
    results = await simulator.run_simulation(duration_seconds)
    
    # Cancel failure task if still running
    failure_task.cancel()
    
    # Analyze impact
    impact = failure_sim.analyze_impact(simulator, baseline_stats)
    
    # Generate recommendations
    recommendations = failure_sim.generate_recommendations(simulator, impact)
    
    # Compile complete results
    complete_results = {
        'simulation': results,
        'baseline': baseline_stats,
        'failures': {
            'events': [e.to_dict() for e in failure_sim.failure_events],
            'active_failures': len(failure_sim.active_failures),
            'total_failures': len(failure_sim.failure_events)
        },
        'impact': impact.to_dict(),
        'recommendations': recommendations
    }
    
    # Save if requested
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(complete_results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("FAILURE SIMULATION SUMMARY")
    print("=" * 70)
    
    print(f"\nFailures Injected: {len(failure_sim.failure_events)}")
    print(f"Cascade Events: {impact.cascade_width}")
    print(f"Cascade Depth: {impact.cascade_depth}")
    
    print(f"\nImpact:")
    print(f"  Failed Components: {len(impact.failed_components)}")
    print(f"  Affected Components: {len(impact.affected_components)}")
    print(f"  Isolated Applications: {len(impact.isolated_applications)}")
    print(f"  Unavailable Topics: {len(impact.unavailable_topics)}")
    
    impact_dict = impact.to_dict()
    print(f"\nPerformance Degradation:")
    print(f"  Latency Increase: {impact_dict['latency_increase_pct']:.1f}%")
    print(f"  Throughput Decrease: {impact_dict['throughput_decrease_pct']:.1f}%")
    print(f"  Delivery Rate Decrease: {impact_dict['delivery_rate_decrease']:.2%}")
    
    print(f"\nRecommendations:")
    for category, recs in recommendations.items():
        if recs:
            print(f"  {category.title()}:")
            for rec in recs[:3]:
                print(f"    - {rec}")
    
    return complete_results
