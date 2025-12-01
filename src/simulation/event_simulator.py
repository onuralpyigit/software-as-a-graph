"""
Event-Driven Simulator

Lightweight event-driven simulation engine for pub-sub systems.
Simulates message flow, latency, throughput, and system behavior
under various conditions.

Features:
- Message flow simulation
- Baseline traffic generation
- Failure injection during simulation
- Performance metrics collection
- QoS policy enforcement simulation
- Achieves 100-1000x real-time speedup
"""

import heapq
import random
import time
from typing import Dict, List, Set, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import logging

try:
    import networkx as nx
except ImportError:
    nx = None


class EventType(Enum):
    """Types of simulation events"""
    MESSAGE_PUBLISH = "message_publish"
    MESSAGE_DELIVER = "message_deliver"
    MESSAGE_DROP = "message_drop"
    COMPONENT_FAIL = "component_fail"
    COMPONENT_RECOVER = "component_recover"
    LOAD_SPIKE = "load_spike"
    TIMEOUT = "timeout"
    HEARTBEAT = "heartbeat"


@dataclass(order=True)
class SimEvent:
    """A single simulation event"""
    timestamp: float  # Simulation time in ms
    event_type: EventType = field(compare=False)
    source: str = field(compare=False)
    target: str = field(compare=False)
    payload: Dict[str, Any] = field(default_factory=dict, compare=False)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp_ms': self.timestamp,
            'event_type': self.event_type.value,
            'source': self.source,
            'target': self.target,
            'payload': self.payload
        }


@dataclass
class Message:
    """A message in the pub-sub system"""
    message_id: str
    topic: str
    publisher: str
    publish_time: float
    size_bytes: int
    qos_level: str = "best_effort"
    deadline_ms: Optional[float] = None
    delivered_to: List[str] = field(default_factory=list)
    delivery_times: Dict[str, float] = field(default_factory=dict)
    dropped: bool = False
    drop_reason: str = ""


@dataclass
class SimulationMetrics:
    """Collected simulation metrics"""
    # Message metrics
    messages_published: int = 0
    messages_delivered: int = 0
    messages_dropped: int = 0
    
    # Latency metrics (in ms)
    total_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    latencies: List[float] = field(default_factory=list)
    
    # Throughput
    bytes_sent: int = 0
    bytes_delivered: int = 0
    
    # QoS
    deadline_violations: int = 0
    qos_failures: int = 0
    
    # Component metrics
    component_failures: int = 0
    cascade_failures: int = 0
    
    # Time
    simulation_duration_ms: float = 0.0
    wall_clock_duration_s: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        avg_latency = self.total_latency_ms / max(1, self.messages_delivered)
        delivery_rate = self.messages_delivered / max(1, self.messages_published)
        
        return {
            'messages': {
                'published': self.messages_published,
                'delivered': self.messages_delivered,
                'dropped': self.messages_dropped,
                'delivery_rate': round(delivery_rate, 4)
            },
            'latency_ms': {
                'avg': round(avg_latency, 2),
                'min': round(self.min_latency_ms, 2) if self.min_latency_ms != float('inf') else 0,
                'max': round(self.max_latency_ms, 2),
                'p50': round(self._percentile(50), 2),
                'p95': round(self._percentile(95), 2),
                'p99': round(self._percentile(99), 2)
            },
            'throughput': {
                'bytes_sent': self.bytes_sent,
                'bytes_delivered': self.bytes_delivered,
                'messages_per_sec': round(
                    self.messages_published / max(0.001, self.simulation_duration_ms / 1000), 2
                )
            },
            'qos': {
                'deadline_violations': self.deadline_violations,
                'qos_failures': self.qos_failures
            },
            'failures': {
                'component_failures': self.component_failures,
                'cascade_failures': self.cascade_failures
            },
            'performance': {
                'simulation_duration_ms': round(self.simulation_duration_ms, 2),
                'wall_clock_duration_s': round(self.wall_clock_duration_s, 4),
                'speedup': round(
                    self.simulation_duration_ms / max(0.001, self.wall_clock_duration_s * 1000), 1
                )
            }
        }
    
    def _percentile(self, p: int) -> float:
        """Calculate percentile of latencies"""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * p / 100)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]


class EventDrivenSimulator:
    """
    Event-driven simulation engine for pub-sub systems.
    
    Simulates message flow, component behavior, and system performance
    under various conditions including failures.
    """
    
    def __init__(self,
                 graph: 'nx.DiGraph',
                 graph_data: Optional[Dict] = None,
                 seed: Optional[int] = None):
        """
        Initialize the simulator.
        
        Args:
            graph: NetworkX directed graph of the system
            graph_data: Original JSON data with additional metadata
            seed: Random seed for reproducibility
        """
        if nx is None:
            raise ImportError("networkx is required for simulation")
        
        self.graph = graph
        self.graph_data = graph_data or {}
        
        if seed is not None:
            random.seed(seed)
        
        # Event queue (min-heap by timestamp)
        self.event_queue: List[SimEvent] = []
        
        # Simulation state
        self.current_time: float = 0.0
        self.failed_components: Set[str] = set()
        self.degraded_components: Dict[str, float] = {}  # component -> capacity
        
        # Message tracking
        self.messages: Dict[str, Message] = {}
        self.message_counter: int = 0
        
        # Metrics
        self.metrics = SimulationMetrics()
        
        # Build topic mappings
        self._build_topic_mappings()
        
        self.logger = logging.getLogger(__name__)
    
    def _build_topic_mappings(self):
        """Build publisher/subscriber mappings for topics"""
        self.topic_publishers: Dict[str, List[str]] = defaultdict(list)
        self.topic_subscribers: Dict[str, List[str]] = defaultdict(list)
        self.topic_qos: Dict[str, Dict] = {}
        
        for source, target, data in self.graph.edges(data=True):
            edge_type = data.get('type', '')
            
            if edge_type == 'PUBLISHES_TO':
                self.topic_publishers[target].append(source)
            elif edge_type == 'SUBSCRIBES_TO':
                self.topic_subscribers[target].append(source)
        
        # Extract QoS from topic nodes
        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'Topic':
                qos = {}
                for key, value in data.items():
                    if key.startswith('qos_'):
                        qos[key[4:]] = value
                if qos:
                    self.topic_qos[node] = qos
    
    def reset(self):
        """Reset simulation state"""
        self.event_queue.clear()
        self.current_time = 0.0
        self.failed_components.clear()
        self.degraded_components.clear()
        self.messages.clear()
        self.message_counter = 0
        self.metrics = SimulationMetrics()
    
    def schedule_event(self, event: SimEvent):
        """Schedule an event for future execution"""
        heapq.heappush(self.event_queue, event)
    
    def run_baseline_simulation(self,
                               duration_ms: float = 10000,
                               message_rate_per_sec: float = 100) -> SimulationMetrics:
        """
        Run baseline simulation without failures.
        
        Args:
            duration_ms: Simulation duration in milliseconds
            message_rate_per_sec: Average messages per second
        
        Returns:
            SimulationMetrics from the run
        """
        self.logger.info(f"Running baseline simulation ({duration_ms}ms, {message_rate_per_sec} msg/s)")
        self.reset()
        
        # Schedule initial messages
        self._schedule_traffic(duration_ms, message_rate_per_sec)
        
        # Run simulation
        return self._run_simulation(duration_ms)
    
    def run_failure_simulation(self,
                              duration_ms: float = 60000,
                              failure_time_ms: float = 30000,
                              failure_components: Optional[List[str]] = None,
                              message_rate_per_sec: float = 100,
                              enable_cascade: bool = True) -> Dict[str, Any]:
        """
        Run simulation with failure injection.
        
        Args:
            duration_ms: Total simulation duration
            failure_time_ms: When to inject failures
            failure_components: Components to fail (or auto-select critical)
            message_rate_per_sec: Message rate
            enable_cascade: Enable cascading failures
        
        Returns:
            Results including pre/post failure metrics
        """
        self.logger.info(f"Running failure simulation (failure at {failure_time_ms}ms)")
        self.reset()
        
        # Auto-select failure components if not specified
        if failure_components is None:
            failure_components = self._select_critical_components(1)
        
        # Schedule traffic
        self._schedule_traffic(duration_ms, message_rate_per_sec)
        
        # Schedule failures
        for component in failure_components:
            self.schedule_event(SimEvent(
                timestamp=failure_time_ms,
                event_type=EventType.COMPONENT_FAIL,
                source="simulator",
                target=component,
                payload={'cascade': enable_cascade}
            ))
        
        # Run pre-failure phase
        pre_metrics = self._run_simulation(failure_time_ms)
        
        # Run post-failure phase
        post_metrics = self._run_simulation(duration_ms)
        
        return {
            'pre_failure_metrics': pre_metrics.to_dict(),
            'post_failure_metrics': post_metrics.to_dict(),
            'failed_components': list(self.failed_components),
            'impact_analysis': self._analyze_impact(pre_metrics, post_metrics)
        }
    
    def run_load_test(self,
                     duration_ms: float = 30000,
                     initial_rate: float = 10,
                     peak_rate: float = 1000,
                     ramp_time_ms: float = 10000) -> Dict[str, Any]:
        """
        Run load test with increasing message rate.
        
        Args:
            duration_ms: Total duration
            initial_rate: Starting message rate
            peak_rate: Maximum message rate
            ramp_time_ms: Time to reach peak rate
        
        Returns:
            Load test results with throughput analysis
        """
        self.logger.info(f"Running load test ({initial_rate} -> {peak_rate} msg/s)")
        self.reset()
        
        # Schedule ramping traffic
        current_time = 0.0
        rate = initial_rate
        rate_increment = (peak_rate - initial_rate) / (ramp_time_ms / 1000)
        
        while current_time < duration_ms:
            # Schedule messages at current rate
            interval_ms = 1000.0 / max(1, rate)
            self._schedule_message_at(current_time)
            current_time += interval_ms
            
            # Update rate during ramp
            if current_time < ramp_time_ms:
                elapsed_sec = current_time / 1000
                rate = initial_rate + rate_increment * elapsed_sec
            else:
                rate = peak_rate
        
        metrics = self._run_simulation(duration_ms)
        
        return {
            'metrics': metrics.to_dict(),
            'load_profile': {
                'initial_rate': initial_rate,
                'peak_rate': peak_rate,
                'ramp_time_ms': ramp_time_ms
            },
            'saturation_analysis': self._analyze_saturation(metrics)
        }
    
    def run_chaos_simulation(self,
                            duration_ms: float = 60000,
                            failure_probability: float = 0.05,
                            recovery_probability: float = 0.3,
                            message_rate_per_sec: float = 100) -> Dict[str, Any]:
        """
        Run chaos engineering simulation with random failures and recoveries.
        
        Args:
            duration_ms: Simulation duration
            failure_probability: Probability of random failure per interval
            recovery_probability: Probability of recovery per interval
            message_rate_per_sec: Message rate
        
        Returns:
            Chaos simulation results
        """
        self.logger.info(f"Running chaos simulation (p_fail={failure_probability})")
        self.reset()
        
        # Schedule traffic
        self._schedule_traffic(duration_ms, message_rate_per_sec)
        
        # Schedule chaos events
        interval_ms = 1000  # Check every second
        current = 0.0
        
        while current < duration_ms:
            # Random failures
            for node in self.graph.nodes():
                if node not in self.failed_components:
                    if random.random() < failure_probability:
                        self.schedule_event(SimEvent(
                            timestamp=current,
                            event_type=EventType.COMPONENT_FAIL,
                            source="chaos",
                            target=node
                        ))
            
            # Random recoveries
            for node in list(self.failed_components):
                if random.random() < recovery_probability:
                    self.schedule_event(SimEvent(
                        timestamp=current,
                        event_type=EventType.COMPONENT_RECOVER,
                        source="chaos",
                        target=node
                    ))
            
            current += interval_ms
        
        metrics = self._run_simulation(duration_ms)
        
        return {
            'metrics': metrics.to_dict(),
            'chaos_profile': {
                'failure_probability': failure_probability,
                'recovery_probability': recovery_probability
            },
            'resilience_score': self._calculate_resilience(metrics)
        }
    
    # =========================================================================
    # Internal Simulation Methods
    # =========================================================================
    
    def _schedule_traffic(self, duration_ms: float, rate_per_sec: float):
        """Schedule message traffic for the duration"""
        interval_ms = 1000.0 / max(1, rate_per_sec)
        current = 0.0
        
        while current < duration_ms:
            self._schedule_message_at(current)
            current += interval_ms + random.uniform(-interval_ms * 0.2, interval_ms * 0.2)
    
    def _schedule_message_at(self, timestamp: float):
        """Schedule a message publication at given time"""
        # Select random publisher and topic
        publishers = [n for n, d in self.graph.nodes(data=True) 
                     if d.get('type') == 'Application']
        
        if not publishers:
            return
        
        publisher = random.choice(publishers)
        
        # Find topics this publisher publishes to
        topics = [t for s, t, d in self.graph.out_edges(publisher, data=True)
                 if d.get('type') == 'PUBLISHES_TO']
        
        if not topics:
            return
        
        topic = random.choice(topics)
        
        self.schedule_event(SimEvent(
            timestamp=timestamp,
            event_type=EventType.MESSAGE_PUBLISH,
            source=publisher,
            target=topic,
            payload={
                'size_bytes': random.randint(100, 10000),
                'qos': self.topic_qos.get(topic, {})
            }
        ))
    
    def _run_simulation(self, end_time: float) -> SimulationMetrics:
        """Run simulation until end time"""
        wall_start = time.time()
        
        while self.event_queue and self.current_time < end_time:
            event = heapq.heappop(self.event_queue)
            
            if event.timestamp > end_time:
                break
            
            self.current_time = event.timestamp
            self._process_event(event)
        
        wall_end = time.time()
        
        self.metrics.simulation_duration_ms = end_time
        self.metrics.wall_clock_duration_s = wall_end - wall_start
        
        return self.metrics
    
    def _process_event(self, event: SimEvent):
        """Process a single simulation event"""
        if event.event_type == EventType.MESSAGE_PUBLISH:
            self._handle_publish(event)
        elif event.event_type == EventType.MESSAGE_DELIVER:
            self._handle_deliver(event)
        elif event.event_type == EventType.MESSAGE_DROP:
            self._handle_drop(event)
        elif event.event_type == EventType.COMPONENT_FAIL:
            self._handle_failure(event)
        elif event.event_type == EventType.COMPONENT_RECOVER:
            self._handle_recovery(event)
    
    def _handle_publish(self, event: SimEvent):
        """Handle message publication"""
        publisher = event.source
        topic = event.target
        
        # Check if publisher is failed
        if publisher in self.failed_components:
            return
        
        # Create message
        self.message_counter += 1
        msg_id = f"msg_{self.message_counter}"
        
        message = Message(
            message_id=msg_id,
            topic=topic,
            publisher=publisher,
            publish_time=self.current_time,
            size_bytes=event.payload.get('size_bytes', 1000),
            qos_level=event.payload.get('qos', {}).get('reliability', 'best_effort'),
            deadline_ms=event.payload.get('qos', {}).get('deadline_ms')
        )
        self.messages[msg_id] = message
        
        self.metrics.messages_published += 1
        self.metrics.bytes_sent += message.size_bytes
        
        # Schedule deliveries to subscribers
        subscribers = self.topic_subscribers.get(topic, [])
        
        for subscriber in subscribers:
            if subscriber in self.failed_components:
                # Schedule drop
                self.schedule_event(SimEvent(
                    timestamp=self.current_time,
                    event_type=EventType.MESSAGE_DROP,
                    source=topic,
                    target=subscriber,
                    payload={'message_id': msg_id, 'reason': 'subscriber_failed'}
                ))
            else:
                # Calculate delivery latency
                latency = self._calculate_latency(publisher, subscriber)
                
                self.schedule_event(SimEvent(
                    timestamp=self.current_time + latency,
                    event_type=EventType.MESSAGE_DELIVER,
                    source=topic,
                    target=subscriber,
                    payload={'message_id': msg_id}
                ))
    
    def _handle_deliver(self, event: SimEvent):
        """Handle message delivery"""
        msg_id = event.payload.get('message_id')
        subscriber = event.target
        
        if msg_id not in self.messages:
            return
        
        message = self.messages[msg_id]
        
        # Check if subscriber is still available
        if subscriber in self.failed_components:
            self._handle_drop(SimEvent(
                timestamp=self.current_time,
                event_type=EventType.MESSAGE_DROP,
                source=event.source,
                target=subscriber,
                payload={'message_id': msg_id, 'reason': 'subscriber_failed'}
            ))
            return
        
        # Calculate latency
        latency = self.current_time - message.publish_time
        
        # Check deadline
        if message.deadline_ms and latency > message.deadline_ms:
            self.metrics.deadline_violations += 1
        
        # Record delivery
        message.delivered_to.append(subscriber)
        message.delivery_times[subscriber] = self.current_time
        
        self.metrics.messages_delivered += 1
        self.metrics.bytes_delivered += message.size_bytes
        self.metrics.total_latency_ms += latency
        self.metrics.latencies.append(latency)
        self.metrics.min_latency_ms = min(self.metrics.min_latency_ms, latency)
        self.metrics.max_latency_ms = max(self.metrics.max_latency_ms, latency)
    
    def _handle_drop(self, event: SimEvent):
        """Handle message drop"""
        msg_id = event.payload.get('message_id')
        reason = event.payload.get('reason', 'unknown')
        
        if msg_id in self.messages:
            self.messages[msg_id].dropped = True
            self.messages[msg_id].drop_reason = reason
        
        self.metrics.messages_dropped += 1
    
    def _handle_failure(self, event: SimEvent):
        """Handle component failure"""
        component = event.target
        
        if component in self.failed_components:
            return
        
        self.failed_components.add(component)
        self.metrics.component_failures += 1
        
        self.logger.debug(f"Component failed: {component} at {self.current_time}ms")
        
        # Simulate cascade if enabled
        if event.payload.get('cascade', False):
            self._trigger_cascade(component)
    
    def _handle_recovery(self, event: SimEvent):
        """Handle component recovery"""
        component = event.target
        
        if component in self.failed_components:
            self.failed_components.remove(component)
            self.logger.debug(f"Component recovered: {component} at {self.current_time}ms")
    
    def _calculate_latency(self, source: str, target: str) -> float:
        """Calculate message latency between components"""
        # Base latency
        base_latency = 5.0  # 5ms base
        
        # Add network latency based on path length
        try:
            path_length = nx.shortest_path_length(self.graph.to_undirected(), source, target)
            network_latency = path_length * 2.0  # 2ms per hop
        except:
            network_latency = 10.0
        
        # Add processing latency
        processing_latency = random.uniform(1.0, 5.0)
        
        # Add congestion factor if component is degraded
        congestion_factor = 1.0
        if target in self.degraded_components:
            congestion_factor = 1.0 / max(0.1, self.degraded_components[target])
        
        total_latency = (base_latency + network_latency + processing_latency) * congestion_factor
        
        return total_latency
    
    def _trigger_cascade(self, failed_component: str):
        """Trigger cascading failures"""
        # Find dependents
        dependents = []
        for source, target, data in self.graph.edges(data=True):
            if target == failed_component and data.get('type') == 'DEPENDS_ON':
                dependents.append(source)
        
        for dependent in dependents:
            if dependent not in self.failed_components:
                # Small delay before cascade
                cascade_delay = random.uniform(100, 500)
                self.schedule_event(SimEvent(
                    timestamp=self.current_time + cascade_delay,
                    event_type=EventType.COMPONENT_FAIL,
                    source=failed_component,
                    target=dependent,
                    payload={'cascade': True}
                ))
                self.metrics.cascade_failures += 1
    
    def _select_critical_components(self, count: int) -> List[str]:
        """Select critical components for failure injection"""
        # Use betweenness centrality
        bc = nx.betweenness_centrality(self.graph)
        sorted_nodes = sorted(bc.items(), key=lambda x: x[1], reverse=True)
        
        # Filter to applications
        apps = [n for n, c in sorted_nodes 
                if self.graph.nodes[n].get('type') == 'Application']
        
        return apps[:count]
    
    def _analyze_impact(self,
                       pre_metrics: SimulationMetrics,
                       post_metrics: SimulationMetrics) -> Dict[str, Any]:
        """Analyze impact of failures by comparing metrics"""
        pre = pre_metrics.to_dict()
        post = post_metrics.to_dict()
        
        # Calculate deltas
        delivery_rate_delta = (
            post['messages']['delivery_rate'] - pre['messages']['delivery_rate']
        )
        latency_delta = (
            post['latency_ms']['avg'] - pre['latency_ms']['avg']
        )
        
        return {
            'delivery_rate_impact': round(delivery_rate_delta, 4),
            'latency_impact_ms': round(latency_delta, 2),
            'additional_drops': post['messages']['dropped'] - pre['messages']['dropped'],
            'severity': self._classify_impact(delivery_rate_delta, latency_delta)
        }
    
    def _classify_impact(self, delivery_delta: float, latency_delta: float) -> str:
        """Classify impact severity"""
        if delivery_delta < -0.3 or latency_delta > 100:
            return "CRITICAL"
        elif delivery_delta < -0.1 or latency_delta > 50:
            return "HIGH"
        elif delivery_delta < -0.05 or latency_delta > 20:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _analyze_saturation(self, metrics: SimulationMetrics) -> Dict[str, Any]:
        """Analyze load test for saturation point"""
        m = metrics.to_dict()
        
        # Check for saturation indicators
        is_saturated = (
            m['messages']['delivery_rate'] < 0.9 or
            m['latency_ms']['p99'] > 100 or
            m['messages']['dropped'] > 0
        )
        
        return {
            'saturated': is_saturated,
            'max_sustainable_rate': m['throughput']['messages_per_sec'] if not is_saturated else None,
            'bottleneck_indicators': {
                'high_latency': m['latency_ms']['p99'] > 100,
                'message_drops': m['messages']['dropped'] > 0,
                'low_delivery_rate': m['messages']['delivery_rate'] < 0.9
            }
        }
    
    def _calculate_resilience(self, metrics: SimulationMetrics) -> float:
        """Calculate resilience score from chaos simulation"""
        m = metrics.to_dict()
        
        # Factors contributing to resilience
        delivery_factor = m['messages']['delivery_rate']
        latency_factor = 1.0 / (1.0 + m['latency_ms']['avg'] / 100)
        failure_factor = 1.0 - (m['failures']['component_failures'] / max(1, len(self.graph.nodes())))
        
        return round(
            delivery_factor * 0.4 + latency_factor * 0.3 + failure_factor * 0.3,
            4
        )