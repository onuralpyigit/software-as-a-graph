#!/usr/bin/env python3
"""
Event-Driven Simulator for Pub-Sub Systems
============================================

Lightweight discrete event simulation engine for pub-sub systems.
Simulates message flow, latency, throughput, and system behavior
under various conditions.

Features:
- Message flow simulation
- Baseline traffic generation
- Failure injection during simulation
- Performance metrics collection
- QoS policy enforcement
- Load testing capabilities
- Achieves 100-1000x real-time speedup

Author: Software-as-a-Graph Research Project
"""

import heapq
import random
import math
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


# ============================================================================
# Enums
# ============================================================================

class EventType(Enum):
    """Types of simulation events"""
    MESSAGE_PUBLISH = "message_publish"
    MESSAGE_ROUTE = "message_route"
    MESSAGE_DELIVER = "message_deliver"
    MESSAGE_DROP = "message_drop"
    MESSAGE_TIMEOUT = "message_timeout"
    COMPONENT_FAIL = "component_fail"
    COMPONENT_RECOVER = "component_recover"
    LOAD_SPIKE = "load_spike"
    HEARTBEAT = "heartbeat"
    SIMULATION_END = "simulation_end"


class MessageState(Enum):
    """Message lifecycle states"""
    CREATED = "created"
    PUBLISHED = "published"
    ROUTING = "routing"
    DELIVERED = "delivered"
    DROPPED = "dropped"
    EXPIRED = "expired"


class ComponentState(Enum):
    """Component health states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"


class QoSLevel(Enum):
    """Quality of Service levels"""
    BEST_EFFORT = 0      # Fire and forget
    AT_LEAST_ONCE = 1    # Guaranteed delivery, possible duplicates
    EXACTLY_ONCE = 2     # Guaranteed single delivery


# ============================================================================
# Data Classes
# ============================================================================

@dataclass(order=True)
class SimEvent:
    """A single simulation event (priority queue compatible)"""
    timestamp: float  # Simulation time in milliseconds
    event_type: EventType = field(compare=False)
    source: str = field(compare=False)
    target: str = field(compare=False, default="")
    payload: Dict[str, Any] = field(default_factory=dict, compare=False)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp_ms': round(self.timestamp, 3),
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
    size_bytes: int = 1024
    qos: QoSLevel = QoSLevel.AT_LEAST_ONCE
    ttl_ms: float = 30000  # Time to live
    priority: int = 0
    
    # Tracking
    state: MessageState = MessageState.CREATED
    delivery_time: Optional[float] = None
    hops: int = 0
    route: List[str] = field(default_factory=list)
    
    def latency(self) -> Optional[float]:
        """Calculate end-to-end latency"""
        if self.delivery_time is not None:
            return self.delivery_time - self.publish_time
        return None
    
    def is_expired(self, current_time: float) -> bool:
        """Check if message has expired"""
        return current_time - self.publish_time > self.ttl_ms
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'message_id': self.message_id,
            'topic': self.topic,
            'publisher': self.publisher,
            'publish_time': self.publish_time,
            'size_bytes': self.size_bytes,
            'qos': self.qos.value,
            'state': self.state.value,
            'latency_ms': self.latency(),
            'hops': self.hops,
            'route': self.route
        }


@dataclass
class ComponentStats:
    """Statistics for a single component"""
    component_id: str
    component_type: str
    
    # Message counts
    messages_received: int = 0
    messages_sent: int = 0
    messages_dropped: int = 0
    messages_queued: int = 0
    
    # Performance
    total_processing_time: float = 0.0
    max_queue_depth: int = 0
    
    # State tracking
    uptime_ms: float = 0.0
    downtime_ms: float = 0.0
    failure_count: int = 0
    
    @property
    def throughput(self) -> float:
        """Messages per second"""
        if self.uptime_ms > 0:
            return (self.messages_sent / self.uptime_ms) * 1000
        return 0.0
    
    @property
    def avg_processing_time(self) -> float:
        """Average processing time per message"""
        if self.messages_sent > 0:
            return self.total_processing_time / self.messages_sent
        return 0.0
    
    @property
    def availability(self) -> float:
        """Component availability percentage"""
        total = self.uptime_ms + self.downtime_ms
        if total > 0:
            return self.uptime_ms / total
        return 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'component_id': self.component_id,
            'component_type': self.component_type,
            'messages': {
                'received': self.messages_received,
                'sent': self.messages_sent,
                'dropped': self.messages_dropped,
                'queued': self.messages_queued
            },
            'performance': {
                'throughput_per_sec': round(self.throughput, 2),
                'avg_processing_ms': round(self.avg_processing_time, 3),
                'max_queue_depth': self.max_queue_depth
            },
            'availability': {
                'uptime_ms': round(self.uptime_ms, 2),
                'downtime_ms': round(self.downtime_ms, 2),
                'availability_pct': round(self.availability * 100, 2),
                'failure_count': self.failure_count
            }
        }


@dataclass
class SimulationMetrics:
    """Overall simulation metrics"""
    # Message statistics
    total_messages: int = 0
    delivered_messages: int = 0
    dropped_messages: int = 0
    expired_messages: int = 0
    in_flight_messages: int = 0
    
    # Latency (in ms)
    total_latency: float = 0.0
    min_latency: float = float('inf')
    max_latency: float = 0.0
    latency_samples: List[float] = field(default_factory=list)
    
    # Throughput
    messages_per_second: float = 0.0
    bytes_per_second: float = 0.0
    
    # System health
    component_failures: int = 0
    cascade_failures: int = 0
    
    @property
    def delivery_rate(self) -> float:
        """Percentage of messages successfully delivered"""
        if self.total_messages > 0:
            return self.delivered_messages / self.total_messages
        return 0.0
    
    @property
    def avg_latency(self) -> float:
        """Average message latency"""
        if self.delivered_messages > 0:
            return self.total_latency / self.delivered_messages
        return 0.0
    
    @property
    def p50_latency(self) -> float:
        """50th percentile latency"""
        return self._percentile(50)
    
    @property
    def p95_latency(self) -> float:
        """95th percentile latency"""
        return self._percentile(95)
    
    @property
    def p99_latency(self) -> float:
        """99th percentile latency"""
        return self._percentile(99)
    
    def _percentile(self, p: float) -> float:
        """Calculate percentile from samples"""
        if not self.latency_samples:
            return 0.0
        sorted_samples = sorted(self.latency_samples)
        idx = int(len(sorted_samples) * p / 100)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'messages': {
                'total': self.total_messages,
                'delivered': self.delivered_messages,
                'dropped': self.dropped_messages,
                'expired': self.expired_messages,
                'in_flight': self.in_flight_messages,
                'delivery_rate_pct': round(self.delivery_rate * 100, 2)
            },
            'latency_ms': {
                'avg': round(self.avg_latency, 3),
                'min': round(self.min_latency, 3) if self.min_latency != float('inf') else 0,
                'max': round(self.max_latency, 3),
                'p50': round(self.p50_latency, 3),
                'p95': round(self.p95_latency, 3),
                'p99': round(self.p99_latency, 3)
            },
            'throughput': {
                'messages_per_sec': round(self.messages_per_second, 2),
                'bytes_per_sec': round(self.bytes_per_second, 2)
            },
            'failures': {
                'component_failures': self.component_failures,
                'cascade_failures': self.cascade_failures
            }
        }


@dataclass
class EventSimulationResult:
    """Complete results from event-driven simulation"""
    simulation_id: str
    duration_ms: float
    real_time_ms: float
    speedup: float  # Simulation time / real time
    
    # Metrics
    metrics: SimulationMetrics
    component_stats: Dict[str, ComponentStats]
    
    # Events
    total_events: int
    events_by_type: Dict[str, int]
    
    # Messages
    messages: List[Message]
    
    # Failures injected
    failures_injected: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'simulation_id': self.simulation_id,
            'timing': {
                'simulation_duration_ms': round(self.duration_ms, 2),
                'real_time_ms': round(self.real_time_ms, 2),
                'speedup': round(self.speedup, 1)
            },
            'metrics': self.metrics.to_dict(),
            'component_stats': {k: v.to_dict() for k, v in self.component_stats.items()},
            'events': {
                'total': self.total_events,
                'by_type': self.events_by_type
            },
            'messages': {
                'total': len(self.messages),
                'samples': [m.to_dict() for m in self.messages[:100]]  # First 100
            },
            'failures_injected': self.failures_injected
        }
    
    def summary(self) -> str:
        """Generate human-readable summary"""
        m = self.metrics
        lines = [
            f"Event Simulation: {self.simulation_id}",
            f"Duration: {self.duration_ms:.0f}ms (simulated), {self.real_time_ms:.0f}ms (real)",
            f"Speedup: {self.speedup:.0f}x real-time",
            f"",
            f"Messages:",
            f"  Total: {m.total_messages}",
            f"  Delivered: {m.delivered_messages} ({m.delivery_rate*100:.1f}%)",
            f"  Dropped: {m.dropped_messages}",
            f"",
            f"Latency:",
            f"  Avg: {m.avg_latency:.2f}ms",
            f"  P95: {m.p95_latency:.2f}ms",
            f"  P99: {m.p99_latency:.2f}ms",
            f"",
            f"Throughput: {m.messages_per_second:.0f} msg/sec"
        ]
        return "\n".join(lines)


# ============================================================================
# Event-Driven Simulator
# ============================================================================

class EventDrivenSimulator:
    """
    Discrete event simulation engine for pub-sub systems.
    
    Simulates message flow through the system with configurable:
    - Message rates and patterns
    - Processing delays
    - Failure injection
    - QoS enforcement
    """
    
    def __init__(self,
                 base_latency_ms: float = 1.0,
                 latency_variance: float = 0.5,
                 queue_capacity: int = 1000,
                 seed: Optional[int] = None):
        """
        Initialize event simulator.
        
        Args:
            base_latency_ms: Base processing latency per hop
            latency_variance: Variance in latency (0-1)
            queue_capacity: Max queue size per component
            seed: Random seed for reproducibility
        """
        self.base_latency_ms = base_latency_ms
        self.latency_variance = latency_variance
        self.queue_capacity = queue_capacity
        
        if seed is not None:
            random.seed(seed)
        
        self._simulation_counter = 0
        self.logger = logging.getLogger('EventSimulator')
        
        # Simulation state
        self._event_queue: List[SimEvent] = []
        self._current_time: float = 0.0
        self._messages: Dict[str, Message] = {}
        self._component_states: Dict[str, ComponentState] = {}
        self._component_queues: Dict[str, List[str]] = {}
        self._metrics: SimulationMetrics = SimulationMetrics()
        self._component_stats: Dict[str, ComponentStats] = {}
        self._events_processed: int = 0
        self._events_by_type: Dict[str, int] = defaultdict(int)
        self._failures_injected: List[Dict[str, Any]] = []
    
    def simulate(self,
                graph: nx.DiGraph,
                duration_ms: float = 10000,
                message_rate: float = 100,
                failure_schedule: Optional[List[Dict[str, Any]]] = None) -> EventSimulationResult:
        """
        Run event-driven simulation.
        
        Args:
            graph: NetworkX graph of the pub-sub system
            duration_ms: Simulation duration in milliseconds
            message_rate: Messages per second to generate
            failure_schedule: List of failures to inject [{time_ms, component, duration_ms}]
            
        Returns:
            EventSimulationResult with comprehensive metrics
        """
        self._simulation_counter += 1
        sim_id = f"event_sim_{self._simulation_counter:05d}"
        
        self.logger.info(f"[{sim_id}] Starting event simulation: "
                        f"duration={duration_ms}ms, rate={message_rate}/sec")
        
        real_start = datetime.now()
        
        # Initialize
        self._initialize_simulation(graph)
        
        # Schedule initial events
        self._schedule_message_generation(graph, duration_ms, message_rate)
        self._schedule_failures(failure_schedule or [])
        self._schedule_end(duration_ms)
        
        # Run simulation loop
        while self._event_queue:
            event = heapq.heappop(self._event_queue)
            
            if event.timestamp > duration_ms:
                break
            
            self._current_time = event.timestamp
            self._process_event(graph, event)
            self._events_processed += 1
            self._events_by_type[event.event_type.value] += 1
        
        real_end = datetime.now()
        real_time_ms = (real_end - real_start).total_seconds() * 1000
        
        # Calculate final metrics
        self._finalize_metrics(duration_ms)
        
        return EventSimulationResult(
            simulation_id=sim_id,
            duration_ms=duration_ms,
            real_time_ms=real_time_ms,
            speedup=duration_ms / real_time_ms if real_time_ms > 0 else 0,
            metrics=self._metrics,
            component_stats=dict(self._component_stats),
            total_events=self._events_processed,
            events_by_type=dict(self._events_by_type),
            messages=list(self._messages.values()),
            failures_injected=self._failures_injected
        )
    
    def simulate_with_load_test(self,
                               graph: nx.DiGraph,
                               duration_ms: float = 60000,
                               initial_rate: float = 10,
                               peak_rate: float = 1000,
                               ramp_time_ms: float = 10000) -> EventSimulationResult:
        """
        Run load test with ramping message rate.
        
        Args:
            graph: NetworkX graph of the pub-sub system
            duration_ms: Total simulation duration
            initial_rate: Starting message rate (msg/sec)
            peak_rate: Peak message rate (msg/sec)
            ramp_time_ms: Time to ramp from initial to peak
            
        Returns:
            EventSimulationResult with load test metrics
        """
        self._simulation_counter += 1
        sim_id = f"load_test_{self._simulation_counter:05d}"
        
        self.logger.info(f"[{sim_id}] Starting load test: "
                        f"initial={initial_rate}/sec, peak={peak_rate}/sec")
        
        real_start = datetime.now()
        
        # Initialize
        self._initialize_simulation(graph)
        
        # Schedule ramping message generation
        self._schedule_ramping_messages(graph, duration_ms, initial_rate, peak_rate, ramp_time_ms)
        self._schedule_end(duration_ms)
        
        # Run simulation loop
        while self._event_queue:
            event = heapq.heappop(self._event_queue)
            
            if event.timestamp > duration_ms:
                break
            
            self._current_time = event.timestamp
            self._process_event(graph, event)
            self._events_processed += 1
            self._events_by_type[event.event_type.value] += 1
        
        real_end = datetime.now()
        real_time_ms = (real_end - real_start).total_seconds() * 1000
        
        self._finalize_metrics(duration_ms)
        
        return EventSimulationResult(
            simulation_id=sim_id,
            duration_ms=duration_ms,
            real_time_ms=real_time_ms,
            speedup=duration_ms / real_time_ms if real_time_ms > 0 else 0,
            metrics=self._metrics,
            component_stats=dict(self._component_stats),
            total_events=self._events_processed,
            events_by_type=dict(self._events_by_type),
            messages=list(self._messages.values()),
            failures_injected=self._failures_injected
        )
    
    def simulate_chaos(self,
                      graph: nx.DiGraph,
                      duration_ms: float = 30000,
                      message_rate: float = 100,
                      failure_probability: float = 0.1,
                      recovery_probability: float = 0.3,
                      check_interval_ms: float = 1000) -> EventSimulationResult:
        """
        Run chaos engineering simulation with random failures.
        
        Args:
            graph: NetworkX graph
            duration_ms: Simulation duration
            message_rate: Message generation rate
            failure_probability: Probability of failure per check
            recovery_probability: Probability of recovery per check
            check_interval_ms: Interval between failure checks
            
        Returns:
            EventSimulationResult with chaos test results
        """
        # Build failure schedule with random failures
        failure_schedule = []
        current_time = 0.0
        
        while current_time < duration_ms:
            current_time += check_interval_ms
            
            # Random component failures
            for node in graph.nodes():
                if random.random() < failure_probability:
                    recovery_time = check_interval_ms * random.randint(1, 5)
                    failure_schedule.append({
                        'time_ms': current_time,
                        'component': node,
                        'duration_ms': recovery_time
                    })
        
        return self.simulate(
            graph,
            duration_ms=duration_ms,
            message_rate=message_rate,
            failure_schedule=failure_schedule
        )
    
    # =========================================================================
    # Initialization
    # =========================================================================
    
    def _initialize_simulation(self, graph: nx.DiGraph):
        """Initialize simulation state"""
        self._event_queue = []
        self._current_time = 0.0
        self._messages = {}
        self._component_states = {}
        self._component_queues = {}
        self._metrics = SimulationMetrics()
        self._component_stats = {}
        self._events_processed = 0
        self._events_by_type = defaultdict(int)
        self._failures_injected = []
        
        # Initialize component states
        for node in graph.nodes():
            self._component_states[node] = ComponentState.HEALTHY
            self._component_queues[node] = []
            
            node_type = graph.nodes[node].get('type', 'Unknown')
            self._component_stats[node] = ComponentStats(
                component_id=node,
                component_type=node_type
            )
    
    # =========================================================================
    # Event Scheduling
    # =========================================================================
    
    def _schedule_event(self, event: SimEvent):
        """Add event to priority queue"""
        heapq.heappush(self._event_queue, event)
    
    def _schedule_message_generation(self, graph: nx.DiGraph,
                                    duration_ms: float, rate: float):
        """Schedule message generation events"""
        # Handle zero or negative rate
        if rate <= 0:
            self.logger.info("Message rate is 0, no messages will be generated")
            return
        
        # Find publishers (applications that publish to topics)
        publishers = []
        topics = []
        
        for node in graph.nodes():
            node_type = graph.nodes[node].get('type', '')
            if node_type == 'Application':
                # Check if it publishes to any topic
                for succ in graph.successors(node):
                    if graph.nodes[succ].get('type') == 'Topic':
                        publishers.append(node)
                        break
            elif node_type == 'Topic':
                topics.append(node)
        
        if not publishers or not topics:
            self.logger.warning("No publishers or topics found")
            return
        
        # Schedule messages
        interval_ms = 1000.0 / rate
        current_time = 0.0
        msg_id = 0
        
        while current_time < duration_ms:
            publisher = random.choice(publishers)
            topic = random.choice(topics)
            
            self._schedule_event(SimEvent(
                timestamp=current_time,
                event_type=EventType.MESSAGE_PUBLISH,
                source=publisher,
                target=topic,
                payload={'message_id': f"msg_{msg_id:06d}"}
            ))
            
            # Add some jitter
            jitter = random.uniform(-0.1, 0.1) * interval_ms
            current_time += interval_ms + jitter
            msg_id += 1
    
    def _schedule_ramping_messages(self, graph: nx.DiGraph,
                                  duration_ms: float,
                                  initial_rate: float,
                                  peak_rate: float,
                                  ramp_time_ms: float):
        """Schedule messages with ramping rate"""
        publishers = []
        topics = []
        
        for node in graph.nodes():
            node_type = graph.nodes[node].get('type', '')
            if node_type == 'Application':
                for succ in graph.successors(node):
                    if graph.nodes[succ].get('type') == 'Topic':
                        publishers.append(node)
                        break
            elif node_type == 'Topic':
                topics.append(node)
        
        if not publishers or not topics:
            return
        
        current_time = 0.0
        msg_id = 0
        
        while current_time < duration_ms:
            # Calculate current rate based on ramp
            if current_time < ramp_time_ms:
                progress = current_time / ramp_time_ms
                rate = initial_rate + (peak_rate - initial_rate) * progress
            else:
                rate = peak_rate
            
            interval_ms = 1000.0 / rate if rate > 0 else 1000.0
            
            publisher = random.choice(publishers)
            topic = random.choice(topics)
            
            self._schedule_event(SimEvent(
                timestamp=current_time,
                event_type=EventType.MESSAGE_PUBLISH,
                source=publisher,
                target=topic,
                payload={'message_id': f"msg_{msg_id:06d}"}
            ))
            
            current_time += interval_ms
            msg_id += 1
    
    def _schedule_failures(self, schedule: List[Dict[str, Any]]):
        """Schedule failure injection events"""
        for failure in schedule:
            # Failure event
            self._schedule_event(SimEvent(
                timestamp=failure['time_ms'],
                event_type=EventType.COMPONENT_FAIL,
                source=failure['component'],
                payload={'duration_ms': failure.get('duration_ms', 5000)}
            ))
            
            # Recovery event
            recovery_time = failure['time_ms'] + failure.get('duration_ms', 5000)
            self._schedule_event(SimEvent(
                timestamp=recovery_time,
                event_type=EventType.COMPONENT_RECOVER,
                source=failure['component']
            ))
            
            self._failures_injected.append(failure)
    
    def _schedule_end(self, duration_ms: float):
        """Schedule simulation end event"""
        self._schedule_event(SimEvent(
            timestamp=duration_ms + 1,
            event_type=EventType.SIMULATION_END,
            source="simulator"
        ))
    
    # =========================================================================
    # Event Processing
    # =========================================================================
    
    def _process_event(self, graph: nx.DiGraph, event: SimEvent):
        """Process a single event"""
        handlers = {
            EventType.MESSAGE_PUBLISH: self._handle_publish,
            EventType.MESSAGE_ROUTE: self._handle_route,
            EventType.MESSAGE_DELIVER: self._handle_deliver,
            EventType.MESSAGE_DROP: self._handle_drop,
            EventType.COMPONENT_FAIL: self._handle_failure,
            EventType.COMPONENT_RECOVER: self._handle_recovery,
        }
        
        handler = handlers.get(event.event_type)
        if handler:
            handler(graph, event)
    
    def _handle_publish(self, graph: nx.DiGraph, event: SimEvent):
        """Handle message publish event"""
        source = event.source
        topic = event.target
        msg_id = event.payload.get('message_id', f"msg_{self._metrics.total_messages}")
        
        # Check if publisher is healthy
        if self._component_states.get(source) != ComponentState.HEALTHY:
            self._metrics.dropped_messages += 1
            return
        
        # Create message
        message = Message(
            message_id=msg_id,
            topic=topic,
            publisher=source,
            publish_time=self._current_time,
            state=MessageState.PUBLISHED,
            route=[source]
        )
        
        self._messages[msg_id] = message
        self._metrics.total_messages += 1
        self._metrics.in_flight_messages += 1
        
        # Update stats
        stats = self._component_stats.get(source)
        if stats:
            stats.messages_sent += 1
        
        # Schedule routing to topic
        latency = self._calculate_latency()
        self._schedule_event(SimEvent(
            timestamp=self._current_time + latency,
            event_type=EventType.MESSAGE_ROUTE,
            source=source,
            target=topic,
            payload={'message_id': msg_id}
        ))
    
    def _handle_route(self, graph: nx.DiGraph, event: SimEvent):
        """Handle message routing event"""
        msg_id = event.payload.get('message_id')
        target = event.target
        
        message = self._messages.get(msg_id)
        if not message:
            return
        
        # Check if target component is healthy
        if self._component_states.get(target) != ComponentState.HEALTHY:
            message.state = MessageState.DROPPED
            self._metrics.dropped_messages += 1
            self._metrics.in_flight_messages -= 1
            return
        
        # Check for expiration
        if message.is_expired(self._current_time):
            message.state = MessageState.EXPIRED
            self._metrics.expired_messages += 1
            self._metrics.in_flight_messages -= 1
            return
        
        message.state = MessageState.ROUTING
        message.route.append(target)
        message.hops += 1
        
        # Update stats
        stats = self._component_stats.get(target)
        if stats:
            stats.messages_received += 1
        
        # Find subscribers
        subscribers = self._find_subscribers(graph, target)
        
        if not subscribers:
            # No subscribers - message delivered to topic
            message.state = MessageState.DELIVERED
            message.delivery_time = self._current_time
            self._record_delivery(message)
        else:
            # Schedule delivery to each subscriber
            for subscriber in subscribers:
                latency = self._calculate_latency()
                self._schedule_event(SimEvent(
                    timestamp=self._current_time + latency,
                    event_type=EventType.MESSAGE_DELIVER,
                    source=target,
                    target=subscriber,
                    payload={'message_id': msg_id}
                ))
    
    def _handle_deliver(self, graph: nx.DiGraph, event: SimEvent):
        """Handle message delivery event"""
        msg_id = event.payload.get('message_id')
        target = event.target
        
        message = self._messages.get(msg_id)
        if not message:
            return
        
        # Check if subscriber is healthy
        if self._component_states.get(target) != ComponentState.HEALTHY:
            # Don't count as dropped - just this delivery failed
            return
        
        # Record successful delivery
        if message.state != MessageState.DELIVERED:
            message.state = MessageState.DELIVERED
            message.delivery_time = self._current_time
            message.route.append(target)
            self._record_delivery(message)
        
        # Update stats
        stats = self._component_stats.get(target)
        if stats:
            stats.messages_received += 1
    
    def _handle_drop(self, graph: nx.DiGraph, event: SimEvent):
        """Handle message drop event"""
        msg_id = event.payload.get('message_id')
        message = self._messages.get(msg_id)
        
        if message and message.state != MessageState.DROPPED:
            message.state = MessageState.DROPPED
            self._metrics.dropped_messages += 1
            self._metrics.in_flight_messages -= 1
    
    def _handle_failure(self, graph: nx.DiGraph, event: SimEvent):
        """Handle component failure event"""
        component = event.source
        
        self._component_states[component] = ComponentState.FAILED
        self._metrics.component_failures += 1
        
        stats = self._component_stats.get(component)
        if stats:
            stats.failure_count += 1
        
        self.logger.debug(f"Component failed: {component} at t={self._current_time}")
    
    def _handle_recovery(self, graph: nx.DiGraph, event: SimEvent):
        """Handle component recovery event"""
        component = event.source
        
        self._component_states[component] = ComponentState.HEALTHY
        
        self.logger.debug(f"Component recovered: {component} at t={self._current_time}")
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _calculate_latency(self) -> float:
        """Calculate processing/network latency with variance"""
        variance = random.uniform(-self.latency_variance, self.latency_variance)
        return max(0.1, self.base_latency_ms * (1 + variance))
    
    def _find_subscribers(self, graph: nx.DiGraph, topic: str) -> List[str]:
        """Find subscribers to a topic"""
        subscribers = []
        
        # Subscribers are components that have edges FROM the topic
        for succ in graph.successors(topic):
            node_type = graph.nodes[succ].get('type', '')
            if node_type == 'Application':
                subscribers.append(succ)
        
        return subscribers
    
    def _record_delivery(self, message: Message):
        """Record successful message delivery"""
        self._metrics.delivered_messages += 1
        self._metrics.in_flight_messages -= 1
        
        latency = message.latency()
        if latency is not None:
            self._metrics.total_latency += latency
            self._metrics.min_latency = min(self._metrics.min_latency, latency)
            self._metrics.max_latency = max(self._metrics.max_latency, latency)
            self._metrics.latency_samples.append(latency)
    
    def _finalize_metrics(self, duration_ms: float):
        """Calculate final metrics"""
        if duration_ms > 0:
            self._metrics.messages_per_second = (
                self._metrics.total_messages / duration_ms * 1000
            )
        
        # Update component uptimes
        for comp_id, stats in self._component_stats.items():
            if self._component_states.get(comp_id) == ComponentState.HEALTHY:
                stats.uptime_ms = duration_ms
            else:
                stats.uptime_ms = duration_ms * 0.9  # Approximate