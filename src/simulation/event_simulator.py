"""
Event-Driven Simulator

Simulates pub-sub message flow using discrete event simulation.
Tracks runtime metrics: throughput, latency, message drops, queue depths.

Works directly on RAW structural relationships without DEPENDS_ON.

Author: Software-as-a-Graph Research Project
"""

from __future__ import annotations
import asyncio
import heapq
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Any, Optional, Callable
from enum import Enum
from collections import defaultdict

from .simulation_graph import SimulationGraph, ComponentState, TopicInfo


class EventType(Enum):
    """Types of events in the simulation."""
    PUBLISH = "publish"
    ROUTE = "route"
    DELIVER = "deliver"
    ACK = "ack"
    TIMEOUT = "timeout"
    FAILURE = "failure"
    RECOVERY = "recovery"


@dataclass(order=True)
class Event:
    """A discrete event in the simulation."""
    time: float
    event_type: EventType = field(compare=False)
    source: str = field(compare=False)
    target: str = field(compare=False)
    message_id: str = field(compare=False)
    data: Dict[str, Any] = field(default_factory=dict, compare=False)
    
    def __post_init__(self):
        if isinstance(self.event_type, str):
            self.event_type = EventType(self.event_type)


@dataclass
class Message:
    """A message being transmitted through the system."""
    id: str
    source_app: str
    topic_id: str
    size: int
    priority: int
    requires_ack: bool
    timestamp: float
    hops: int = 0
    delivered_to: Set[str] = field(default_factory=set)
    dropped_at: Optional[str] = None
    
    @property
    def latency(self) -> float:
        """Latency from creation (not final until delivered)."""
        return 0.0  # Will be computed at delivery time


@dataclass
class EventScenario:
    """Configuration for an event simulation run."""
    source_app: str
    description: str = ""
    
    # Message generation settings
    num_messages: int = 100
    message_interval: float = 0.01  # seconds between messages
    message_size: int = 1024  # bytes
    
    # Simulation settings
    duration: float = 10.0  # simulation duration in seconds
    seed: Optional[int] = None  # random seed for reproducibility
    
    # Component settings
    broker_latency: float = 0.001  # seconds
    network_latency: float = 0.005  # seconds
    processing_latency: float = 0.002  # seconds
    
    # Failure injection
    drop_probability: float = 0.0  # probability of message drop
    failure_probability: float = 0.0  # probability of component failure


@dataclass
class RuntimeMetrics:
    """Runtime metrics collected during simulation."""
    # Throughput metrics
    messages_published: int = 0
    messages_delivered: int = 0
    messages_dropped: int = 0
    messages_in_flight: int = 0
    
    # Latency metrics
    total_latency: float = 0.0
    min_latency: float = float('inf')
    max_latency: float = 0.0
    latency_samples: List[float] = field(default_factory=list)
    
    # Per-component metrics
    component_messages: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    component_drops: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    component_latency: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    
    # Topic metrics
    topic_throughput: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    topic_drops: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Queue depths (max observed)
    max_queue_depth: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    @property
    def delivery_rate(self) -> float:
        """Delivery rate as percentage."""
        if self.messages_published == 0:
            return 0.0
        return self.messages_delivered / self.messages_published * 100
    
    @property
    def drop_rate(self) -> float:
        """Drop rate as percentage."""
        if self.messages_published == 0:
            return 0.0
        return self.messages_dropped / self.messages_published * 100
    
    @property
    def avg_latency(self) -> float:
        """Average end-to-end latency."""
        if self.messages_delivered == 0:
            return 0.0
        return self.total_latency / self.messages_delivered
    
    @property
    def p50_latency(self) -> float:
        """50th percentile latency."""
        if not self.latency_samples:
            return 0.0
        sorted_samples = sorted(self.latency_samples)
        idx = len(sorted_samples) // 2
        return sorted_samples[idx]
    
    @property
    def p99_latency(self) -> float:
        """99th percentile latency."""
        if not self.latency_samples:
            return 0.0
        sorted_samples = sorted(self.latency_samples)
        idx = int(len(sorted_samples) * 0.99)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "throughput": {
                "published": self.messages_published,
                "delivered": self.messages_delivered,
                "dropped": self.messages_dropped,
                "delivery_rate": round(self.delivery_rate, 2),
                "drop_rate": round(self.drop_rate, 2),
            },
            "latency": {
                "avg_ms": round(self.avg_latency * 1000, 3),
                "min_ms": round(self.min_latency * 1000, 3) if self.min_latency != float('inf') else 0,
                "max_ms": round(self.max_latency * 1000, 3),
                "p50_ms": round(self.p50_latency * 1000, 3),
                "p99_ms": round(self.p99_latency * 1000, 3),
            },
            "per_component": {
                "messages": dict(self.component_messages),
                "drops": dict(self.component_drops),
            },
            "per_topic": {
                "throughput": dict(self.topic_throughput),
                "drops": dict(self.topic_drops),
            },
        }


@dataclass
class EventResult:
    """Result of an event simulation run."""
    scenario: str
    source_app: str
    duration: float
    metrics: RuntimeMetrics
    
    # Path analysis
    affected_topics: List[str] = field(default_factory=list)
    reached_subscribers: List[str] = field(default_factory=list)
    failed_paths: List[Tuple[str, str, str]] = field(default_factory=list)
    
    # Component impacts
    component_impacts: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "source_app": self.source_app,
            "duration_seconds": round(self.duration, 3),
            "metrics": self.metrics.to_dict(),
            "path_analysis": {
                "affected_topics": self.affected_topics,
                "reached_subscribers": self.reached_subscribers,
                "failed_paths_count": len(self.failed_paths),
            },
            "component_impacts": {
                k: round(v, 4) for k, v in self.component_impacts.items()
            },
        }


class EventSimulator:
    """
    Discrete event simulator for pub-sub message flow.
    
    Simulates message propagation through the system:
    Publisher -> Topic -> Broker(s) -> Subscribers
    
    Tracks runtime metrics including throughput, latency, and message drops.
    """
    
    def __init__(self, graph: SimulationGraph):
        """
        Initialize the event simulator.
        
        Args:
            graph: SimulationGraph instance with raw structural relationships
        """
        self.graph = graph
        self.logger = logging.getLogger(__name__)
        
        # Event queue (min-heap by time)
        self._event_queue: List[Event] = []
        
        # Active messages
        self._messages: Dict[str, Message] = {}
        
        # Metrics
        self._metrics = RuntimeMetrics()
        
        # Component queues (for queue depth tracking)
        self._queues: Dict[str, List[str]] = defaultdict(list)
        
        # Random generator
        self._rng = random.Random()
        
        # Current simulation time
        self._sim_time = 0.0
    
    def simulate(self, scenario: EventScenario) -> EventResult:
        """
        Run an event simulation.
        
        Args:
            scenario: Configuration for the simulation
            
        Returns:
            EventResult with metrics and analysis
        """
        # Reset state
        self.graph.reset()
        self._reset_state(scenario.seed)
        
        start_time = time.time()
        
        # Get source app info
        if scenario.source_app not in self.graph.components:
            return self._empty_result(scenario, "Source not found")
        
        # Get topics the source publishes to
        publishes_to, _ = self.graph.get_app_topics(scenario.source_app)
        if not publishes_to:
            return self._empty_result(scenario, "Source has no publications")
        
        # Schedule initial message publications
        for i in range(scenario.num_messages):
            msg_time = i * scenario.message_interval
            if msg_time > scenario.duration:
                break
            
            # Round-robin through topics
            topic_id = publishes_to[i % len(publishes_to)]
            topic_info = self.graph.topics.get(topic_id)
            
            msg_id = f"msg_{i:05d}"
            self._schedule_event(Event(
                time=msg_time,
                event_type=EventType.PUBLISH,
                source=scenario.source_app,
                target=topic_id,
                message_id=msg_id,
                data={
                    "size": scenario.message_size,
                    "priority": topic_info.priority_value if topic_info else 1,
                    "requires_ack": topic_info.requires_ack if topic_info else False,
                }
            ))
        
        # Run simulation loop
        while self._event_queue and self._sim_time < scenario.duration:
            event = heapq.heappop(self._event_queue)
            self._sim_time = event.time
            self._process_event(event, scenario)
        
        # Compute results
        duration = time.time() - start_time
        
        # Analyze paths
        affected_topics = list(set(self._metrics.topic_throughput.keys()))
        reached_subscribers = [
            sub for sub, count in self._metrics.component_messages.items()
            if count > 0 and self.graph.components.get(sub, None) 
            and self.graph.components[sub].type == "Application"
            and sub != scenario.source_app
        ]
        
        # Compute component impacts (based on drop contribution)
        component_impacts = self._compute_component_impacts()
        
        return EventResult(
            scenario=scenario.description or f"Event simulation from {scenario.source_app}",
            source_app=scenario.source_app,
            duration=duration,
            metrics=self._metrics,
            affected_topics=affected_topics,
            reached_subscribers=reached_subscribers,
            component_impacts=component_impacts,
        )
    
    def simulate_all_publishers(self, scenario: EventScenario) -> Dict[str, EventResult]:
        """
        Run event simulation from all publisher applications.
        
        Returns:
            Dict mapping app_id to EventResult
        """
        results = {}
        
        # Find all publishers
        publishers = set()
        for topic_id in self.graph.topics:
            pubs = self.graph.get_publishers(topic_id)
            publishers.update(pubs)
        
        for pub_id in publishers:
            pub_scenario = EventScenario(
                source_app=pub_id,
                description=f"Event flow from {pub_id}",
                num_messages=scenario.num_messages,
                message_interval=scenario.message_interval,
                duration=scenario.duration,
                broker_latency=scenario.broker_latency,
                network_latency=scenario.network_latency,
                processing_latency=scenario.processing_latency,
                drop_probability=scenario.drop_probability,
            )
            results[pub_id] = self.simulate(pub_scenario)
        
        return results
    
    def _reset_state(self, seed: Optional[int] = None) -> None:
        """Reset simulation state."""
        self._event_queue = []
        self._messages = {}
        self._metrics = RuntimeMetrics()
        self._queues = defaultdict(list)
        self._sim_time = 0.0
        
        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random()
    
    def _schedule_event(self, event: Event) -> None:
        """Add event to the queue."""
        heapq.heappush(self._event_queue, event)
    
    def _process_event(self, event: Event, scenario: EventScenario) -> None:
        """Process a single event."""
        
        if event.event_type == EventType.PUBLISH:
            self._handle_publish(event, scenario)
        elif event.event_type == EventType.ROUTE:
            self._handle_route(event, scenario)
        elif event.event_type == EventType.DELIVER:
            self._handle_deliver(event, scenario)
        elif event.event_type == EventType.ACK:
            self._handle_ack(event, scenario)
        elif event.event_type == EventType.TIMEOUT:
            self._handle_timeout(event, scenario)
    
    def _handle_publish(self, event: Event, scenario: EventScenario) -> None:
        """Handle a publish event."""
        source = event.source
        topic_id = event.target
        msg_id = event.message_id
        
        # Check if source is active
        if not self.graph.is_active(source):
            self._record_drop(msg_id, source, topic_id, "source_failed")
            return
        
        # Check if topic is active
        if not self.graph.is_active(topic_id):
            self._record_drop(msg_id, topic_id, topic_id, "topic_unavailable")
            return
        
        # Create message
        topic_info = self.graph.topics.get(topic_id)
        message = Message(
            id=msg_id,
            source_app=source,
            topic_id=topic_id,
            size=event.data.get("size", 1024),
            priority=event.data.get("priority", 1),
            requires_ack=event.data.get("requires_ack", False),
            timestamp=self._sim_time,
        )
        self._messages[msg_id] = message
        self._metrics.messages_published += 1
        self._metrics.topic_throughput[topic_id] += 1
        
        # Random drop check
        if self._rng.random() < scenario.drop_probability:
            self._record_drop(msg_id, source, topic_id, "random_drop")
            return
        
        # Get routing brokers
        brokers = self.graph.get_routing_brokers(topic_id)
        
        if brokers:
            # Schedule route events to brokers
            for broker_id in brokers:
                if self.graph.is_active(broker_id):
                    self._schedule_event(Event(
                        time=self._sim_time + scenario.broker_latency,
                        event_type=EventType.ROUTE,
                        source=topic_id,
                        target=broker_id,
                        message_id=msg_id,
                    ))
        else:
            # Direct delivery (no broker)
            subscribers = self.graph.get_subscribers(topic_id)
            for sub_id in subscribers:
                if self.graph.is_active(sub_id) and sub_id != source:
                    self._schedule_event(Event(
                        time=self._sim_time + scenario.network_latency,
                        event_type=EventType.DELIVER,
                        source=topic_id,
                        target=sub_id,
                        message_id=msg_id,
                    ))
    
    def _handle_route(self, event: Event, scenario: EventScenario) -> None:
        """Handle a route event (message arrives at broker)."""
        broker_id = event.target
        msg_id = event.message_id
        
        message = self._messages.get(msg_id)
        if not message:
            return
        
        # Check if broker is active
        if not self.graph.is_active(broker_id):
            self._record_drop(msg_id, broker_id, message.topic_id, "broker_failed")
            return
        
        # Update message hops
        message.hops += 1
        self._metrics.component_messages[broker_id] += 1
        
        # Track queue depth
        self._queues[broker_id].append(msg_id)
        self._metrics.max_queue_depth[broker_id] = max(
            self._metrics.max_queue_depth[broker_id],
            len(self._queues[broker_id])
        )
        
        # Schedule delivery to subscribers
        topic_id = message.topic_id
        subscribers = self.graph.get_subscribers(topic_id)
        
        for sub_id in subscribers:
            if sub_id != message.source_app and self.graph.is_active(sub_id):
                # Check if subscriber's host node is active
                host_node = self.graph.get_host_node(sub_id)
                if host_node and not self.graph.is_active(host_node):
                    self._record_drop(msg_id, sub_id, topic_id, "host_node_failed")
                    continue
                
                self._schedule_event(Event(
                    time=self._sim_time + scenario.processing_latency + scenario.network_latency,
                    event_type=EventType.DELIVER,
                    source=broker_id,
                    target=sub_id,
                    message_id=msg_id,
                ))
        
        # Remove from queue
        if msg_id in self._queues[broker_id]:
            self._queues[broker_id].remove(msg_id)
    
    def _handle_deliver(self, event: Event, scenario: EventScenario) -> None:
        """Handle a deliver event (message arrives at subscriber)."""
        sub_id = event.target
        msg_id = event.message_id
        
        message = self._messages.get(msg_id)
        if not message:
            return
        
        # Check if already delivered to this subscriber
        if sub_id in message.delivered_to:
            return
        
        # Check if subscriber is active
        if not self.graph.is_active(sub_id):
            self._record_drop(msg_id, sub_id, message.topic_id, "subscriber_failed")
            return
        
        # Successful delivery
        message.delivered_to.add(sub_id)
        message.hops += 1
        
        latency = self._sim_time - message.timestamp
        
        self._metrics.messages_delivered += 1
        self._metrics.total_latency += latency
        self._metrics.min_latency = min(self._metrics.min_latency, latency)
        self._metrics.max_latency = max(self._metrics.max_latency, latency)
        self._metrics.latency_samples.append(latency)
        
        self._metrics.component_messages[sub_id] += 1
        self._metrics.component_latency[sub_id] += latency
    
    def _handle_ack(self, event: Event, scenario: EventScenario) -> None:
        """Handle an acknowledgment event."""
        # For future expansion: implement reliable delivery ACK tracking
        pass
    
    def _handle_timeout(self, event: Event, scenario: EventScenario) -> None:
        """Handle a timeout event."""
        msg_id = event.message_id
        message = self._messages.get(msg_id)
        if message and message.requires_ack and not message.delivered_to:
            self._record_drop(msg_id, event.source, message.topic_id, "timeout")
    
    def _record_drop(self, msg_id: str, component_id: str, topic_id: str, reason: str) -> None:
        """Record a message drop."""
        self._metrics.messages_dropped += 1
        self._metrics.component_drops[component_id] += 1
        self._metrics.topic_drops[topic_id] += 1
        
        message = self._messages.get(msg_id)
        if message:
            message.dropped_at = component_id
    
    def _compute_component_impacts(self) -> Dict[str, float]:
        """
        Compute impact score for each component based on simulation results.
        
        Impact is based on:
        - Message throughput (higher = more critical)
        - Drop contribution (higher = more problematic)
        """
        impacts = {}
        
        total_messages = max(self._metrics.messages_published, 1)
        total_drops = max(self._metrics.messages_dropped, 1)
        
        for comp_id in self.graph.components:
            msg_count = self._metrics.component_messages.get(comp_id, 0)
            drop_count = self._metrics.component_drops.get(comp_id, 0)
            
            # Throughput contribution (normalized)
            throughput_impact = msg_count / total_messages
            
            # Drop contribution (normalized)
            drop_impact = drop_count / total_drops if total_drops > 0 else 0
            
            # Combined impact (weighted)
            impacts[comp_id] = 0.6 * throughput_impact + 0.4 * drop_impact
        
        return impacts
    
    def _empty_result(self, scenario: EventScenario, reason: str) -> EventResult:
        """Create empty result for error cases."""
        return EventResult(
            scenario=f"{scenario.description} ({reason})",
            source_app=scenario.source_app,
            duration=0.0,
            metrics=RuntimeMetrics(),
        )


class AsyncEventSimulator:
    """
    Async version of EventSimulator for higher performance.
    Uses asyncio for concurrent event processing.
    """
    
    def __init__(self, graph: SimulationGraph):
        self.graph = graph
        self.sync_simulator = EventSimulator(graph)
    
    async def simulate(self, scenario: EventScenario) -> EventResult:
        """Run simulation asynchronously."""
        # Use asyncio.to_thread for CPU-bound work
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.sync_simulator.simulate, 
            scenario
        )
    
    async def simulate_batch(
        self, 
        scenarios: List[EventScenario]
    ) -> List[EventResult]:
        """Run multiple simulations concurrently."""
        tasks = [self.simulate(s) for s in scenarios]
        return await asyncio.gather(*tasks)