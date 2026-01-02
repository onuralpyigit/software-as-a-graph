"""
Event-Driven Simulator - Version 5.0

Discrete event simulation for pub-sub message flow.

Uses ORIGINAL edge types (PUBLISHES_TO, SUBSCRIBES_TO, etc.)
NOT derived DEPENDS_ON relationships.

Features:
- Message publication and delivery simulation
- QoS-aware message handling
- Latency and throughput modeling
- Component load tracking
- Failure injection during simulation
- Layer-specific statistics

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
import heapq
import logging
import random
import time as wall_time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict
import statistics as stats_module

from .simulation_graph import (
    SimulationGraph,
    Component,
    ComponentType,
    EdgeType,
    ComponentStatus,
    QoSPolicy,
)


# =============================================================================
# Enums
# =============================================================================

class EventType(Enum):
    """Types of simulation events."""
    PUBLISH = auto()       # App publishes message
    ROUTE = auto()         # Broker routes message
    DELIVER = auto()       # Message delivered to subscriber
    TIMEOUT = auto()       # Message timeout
    FAILURE = auto()       # Component fails
    RECOVERY = auto()      # Component recovers


class MessageStatus(Enum):
    """Status of a message in transit."""
    PENDING = "pending"
    ROUTING = "routing"
    DELIVERED = "delivered"
    FAILED = "failed"
    TIMEOUT = "timeout"
    DROPPED = "dropped"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(order=True)
class Event:
    """A simulation event."""
    time: float
    priority: int = field(compare=True, default=5)
    event_type: EventType = field(compare=False, default=EventType.PUBLISH)
    source: str = field(compare=False, default="")
    target: Optional[str] = field(compare=False, default=None)
    data: Dict[str, Any] = field(compare=False, default_factory=dict)
    event_id: int = field(compare=False, default=0)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "time": self.time,
            "type": self.event_type.name,
            "source": self.source,
            "target": self.target,
            "event_id": self.event_id,
        }


@dataclass
class Message:
    """A message in the pub-sub system."""
    message_id: str
    publisher: str
    topic: str
    payload_size: int = 100
    priority: int = 0
    qos: QoSPolicy = field(default_factory=QoSPolicy)
    created_at: float = 0.0
    delivered_at: Optional[float] = None
    status: MessageStatus = MessageStatus.PENDING
    path: List[str] = field(default_factory=list)
    
    @property
    def latency(self) -> Optional[float]:
        """Delivery latency in ms."""
        if self.delivered_at is not None:
            return self.delivered_at - self.created_at
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "publisher": self.publisher,
            "topic": self.topic,
            "status": self.status.value,
            "latency": self.latency,
            "path": self.path,
        }


@dataclass
class ComponentLoad:
    """Load statistics for a component."""
    component_id: str
    component_type: ComponentType = ComponentType.APPLICATION
    messages_processed: int = 0
    messages_dropped: int = 0
    total_latency: float = 0.0
    peak_queue_size: int = 0
    current_queue_size: int = 0
    
    @property
    def avg_latency(self) -> float:
        if self.messages_processed > 0:
            return self.total_latency / self.messages_processed
        return 0.0
    
    @property
    def drop_rate(self) -> float:
        total = self.messages_processed + self.messages_dropped
        if total > 0:
            return self.messages_dropped / total
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "type": self.component_type.value,
            "messages_processed": self.messages_processed,
            "messages_dropped": self.messages_dropped,
            "avg_latency": round(self.avg_latency, 4),
            "drop_rate": round(self.drop_rate, 4),
            "peak_queue_size": self.peak_queue_size,
        }


@dataclass
class SimulationStats:
    """Overall simulation statistics."""
    total_messages: int = 0
    delivered_messages: int = 0
    failed_messages: int = 0
    dropped_messages: int = 0
    timeout_messages: int = 0
    total_events: int = 0
    simulation_time: float = 0.0
    wall_clock_time: float = 0.0
    latencies: List[float] = field(default_factory=list)
    
    @property
    def delivery_rate(self) -> float:
        if self.total_messages > 0:
            return self.delivered_messages / self.total_messages
        return 0.0
    
    @property
    def avg_latency(self) -> float:
        if self.latencies:
            return stats_module.mean(self.latencies)
        return 0.0
    
    @property
    def p99_latency(self) -> float:
        if len(self.latencies) >= 100:
            sorted_lat = sorted(self.latencies)
            idx = int(len(sorted_lat) * 0.99)
            return sorted_lat[idx]
        elif self.latencies:
            return max(self.latencies)
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_messages": self.total_messages,
            "delivered_messages": self.delivered_messages,
            "failed_messages": self.failed_messages,
            "dropped_messages": self.dropped_messages,
            "timeout_messages": self.timeout_messages,
            "delivery_rate": round(self.delivery_rate, 4),
            "avg_latency_ms": round(self.avg_latency, 4),
            "p99_latency_ms": round(self.p99_latency, 4),
            "total_events": self.total_events,
            "simulation_time_ms": round(self.simulation_time, 2),
            "wall_clock_time_s": round(self.wall_clock_time, 4),
        }


@dataclass
class LayerStats:
    """Statistics for a specific layer."""
    layer: str
    layer_name: str
    messages_sent: int = 0
    messages_delivered: int = 0
    messages_failed: int = 0
    avg_latency: float = 0.0
    component_count: int = 0
    
    @property
    def delivery_rate(self) -> float:
        if self.messages_sent > 0:
            return self.messages_delivered / self.messages_sent
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "layer_name": self.layer_name,
            "messages_sent": self.messages_sent,
            "messages_delivered": self.messages_delivered,
            "messages_failed": self.messages_failed,
            "delivery_rate": round(self.delivery_rate, 4),
            "avg_latency": round(self.avg_latency, 4),
            "component_count": self.component_count,
        }


@dataclass
class SimulationResult:
    """Complete simulation result."""
    stats: SimulationStats
    component_loads: Dict[str, ComponentLoad] = field(default_factory=dict)
    layer_stats: Dict[str, LayerStats] = field(default_factory=dict)
    failed_components: Set[str] = field(default_factory=set)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_bottlenecks(self, threshold: int = 50) -> List[str]:
        """Get components with high queue sizes."""
        return [
            comp_id for comp_id, load in self.component_loads.items()
            if load.peak_queue_size > threshold
        ]
    
    def get_high_drop_components(self, threshold: float = 0.1) -> List[str]:
        """Get components with high drop rates."""
        return [
            comp_id for comp_id, load in self.component_loads.items()
            if load.drop_rate > threshold
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "stats": self.stats.to_dict(),
            "component_loads": {k: v.to_dict() for k, v in self.component_loads.items()},
            "layer_stats": {k: v.to_dict() for k, v in self.layer_stats.items()},
            "failed_components": list(self.failed_components),
            "bottlenecks": self.get_bottlenecks(),
            "high_drop_components": self.get_high_drop_components(),
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# Event Simulator
# =============================================================================

class EventSimulator:
    """
    Discrete event simulation for pub-sub message flow.
    
    Uses original edge types to trace message paths:
    - PUBLISHES_TO: App -> Topic
    - SUBSCRIBES_TO: App -> Topic (delivery direction)
    - ROUTES: Broker -> Topic
    
    Example:
        graph = SimulationGraph.from_json("system.json")
        simulator = EventSimulator(seed=42)
        
        result = simulator.run(
            graph,
            duration=5000,      # 5 seconds
            message_rate=100,   # 100 msg/sec
        )
        
        print(f"Delivered: {result.stats.delivery_rate:.2%}")
        print(f"Avg Latency: {result.stats.avg_latency:.2f}ms")
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        default_latency: float = 10.0,
        default_processing: float = 1.0,
        queue_capacity: int = 1000,
        timeout: float = 5000.0,
    ):
        """
        Initialize the event simulator.
        
        Args:
            seed: Random seed for reproducibility
            default_latency: Default network latency (ms)
            default_processing: Default processing time (ms)
            queue_capacity: Max queue size per component
            timeout: Message timeout (ms)
        """
        self.seed = seed
        self.rng = random.Random(seed)
        self.default_latency = default_latency
        self.default_processing = default_processing
        self.queue_capacity = queue_capacity
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
    
    def run(
        self,
        graph: SimulationGraph,
        duration: float = 1000.0,
        message_rate: float = 10.0,
        publishers: Optional[List[str]] = None,
        failure_schedule: Optional[List[Dict[str, Any]]] = None,
    ) -> SimulationResult:
        """
        Run the simulation.
        
        Args:
            graph: The simulation graph
            duration: Simulation duration (ms)
            message_rate: Messages per second
            publishers: Specific publishers (None = all apps)
            failure_schedule: List of {component_id, time, duration}
        
        Returns:
            SimulationResult with all statistics
        """
        # Reset state
        self._reset()
        self.graph = graph
        
        # Initialize component state
        for comp_id, comp in graph.components.items():
            self.component_status[comp_id] = ComponentStatus.HEALTHY
            self.component_loads[comp_id] = ComponentLoad(
                component_id=comp_id,
                component_type=comp.type,
            )
        
        # Get publishers
        if publishers is None:
            publishers = [
                c.id for c in graph.get_components_by_type(ComponentType.APPLICATION)
                if graph.get_topics_published_by(c.id)  # Has topics to publish to
            ]
        
        if not publishers:
            self.logger.warning("No publishers found")
            return self._build_result(graph)
        
        # Schedule failure events
        if failure_schedule:
            for failure in failure_schedule:
                self._schedule_failure(
                    failure["component_id"],
                    failure["time"],
                    failure.get("duration"),
                )
        
        # Schedule message generation
        self._schedule_messages(publishers, duration, message_rate)
        
        # Run simulation
        start_time = wall_time.time()
        self._run_loop(duration)
        wall_clock = wall_time.time() - start_time
        
        # Build result
        self.stats.simulation_time = duration
        self.stats.wall_clock_time = wall_clock
        
        return self._build_result(graph)
    
    def _reset(self) -> None:
        """Reset simulation state."""
        self.current_time = 0.0
        self.event_queue: List[Event] = []
        self.event_counter = 0
        self.messages: Dict[str, Message] = {}
        self.component_status: Dict[str, ComponentStatus] = {}
        self.component_loads: Dict[str, ComponentLoad] = {}
        self.stats = SimulationStats()
        self.failed_components: Set[str] = set()
    
    def _schedule_event(
        self,
        time: float,
        event_type: EventType,
        source: str,
        target: Optional[str] = None,
        priority: int = 5,
        data: Optional[Dict[str, Any]] = None,
    ) -> Event:
        """Schedule a new event."""
        self.event_counter += 1
        event = Event(
            time=time,
            priority=priority,
            event_type=event_type,
            source=source,
            target=target,
            data=data or {},
            event_id=self.event_counter,
        )
        heapq.heappush(self.event_queue, event)
        return event
    
    def _schedule_failure(
        self,
        component_id: str,
        at_time: float,
        duration: Optional[float] = None,
    ) -> None:
        """Schedule a component failure."""
        self._schedule_event(
            time=at_time,
            event_type=EventType.FAILURE,
            source=component_id,
            priority=0,
        )
        
        if duration is not None:
            self._schedule_event(
                time=at_time + duration,
                event_type=EventType.RECOVERY,
                source=component_id,
                priority=0,
            )
    
    def _schedule_messages(
        self,
        publishers: List[str],
        duration: float,
        message_rate: float,
    ) -> None:
        """Schedule message generation events."""
        if not publishers or message_rate <= 0:
            return
        
        # Convert to inter-arrival time (ms)
        inter_arrival = 1000.0 / message_rate
        
        current_time = 0.0
        while current_time < duration:
            # Pick random publisher
            publisher = self.rng.choice(publishers)
            
            # Get topics for this publisher
            topics = list(self.graph.get_topics_published_by(publisher))
            if not topics:
                current_time += inter_arrival
                continue
            
            topic = self.rng.choice(topics)
            
            # Create message
            msg_id = f"msg_{self.stats.total_messages + 1}"
            
            # Get QoS from edge
            qos = QoSPolicy()
            for edge in self.graph.get_outgoing_edges(publisher):
                if edge.target == topic and edge.edge_type == EdgeType.PUBLISHES_TO:
                    qos = edge.qos
                    break
            
            message = Message(
                message_id=msg_id,
                publisher=publisher,
                topic=topic,
                qos=qos,
                created_at=current_time,
                path=[publisher],
            )
            self.messages[msg_id] = message
            self.stats.total_messages += 1
            
            # Schedule publish event
            self._schedule_event(
                time=current_time,
                event_type=EventType.PUBLISH,
                source=publisher,
                target=topic,
                data={"message_id": msg_id},
            )
            
            # Schedule timeout
            self._schedule_event(
                time=current_time + self.timeout,
                event_type=EventType.TIMEOUT,
                source=msg_id,
                priority=10,
                data={"message_id": msg_id},
            )
            
            # Next arrival (exponential)
            current_time += self.rng.expovariate(1.0 / inter_arrival)
    
    def _run_loop(self, duration: float) -> None:
        """Main simulation loop."""
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            
            # Check if past duration
            if event.time > duration:
                break
            
            # Skip cancelled events
            if event.data.get("cancelled"):
                continue
            
            self.current_time = event.time
            self.stats.total_events += 1
            
            # Dispatch event
            if event.event_type == EventType.PUBLISH:
                self._handle_publish(event)
            elif event.event_type == EventType.ROUTE:
                self._handle_route(event)
            elif event.event_type == EventType.DELIVER:
                self._handle_deliver(event)
            elif event.event_type == EventType.TIMEOUT:
                self._handle_timeout(event)
            elif event.event_type == EventType.FAILURE:
                self._handle_failure(event)
            elif event.event_type == EventType.RECOVERY:
                self._handle_recovery(event)
    
    def _handle_publish(self, event: Event) -> None:
        """Handle message publication."""
        publisher = event.source
        topic = event.target
        msg_id = event.data.get("message_id")
        
        # Check publisher status
        if self.component_status.get(publisher) == ComponentStatus.FAILED:
            self._mark_message_failed(msg_id, "publisher_failed")
            return
        
        message = self.messages.get(msg_id)
        if not message:
            return
        
        message.path.append(topic)
        
        # Find broker for topic
        broker = self.graph.get_broker_for_topic(topic)
        
        if broker:
            # Route through broker
            latency = self._get_latency(publisher, broker)
            self._schedule_event(
                time=self.current_time + latency,
                event_type=EventType.ROUTE,
                source=broker,
                target=topic,
                data={"message_id": msg_id},
            )
            message.status = MessageStatus.ROUTING
        else:
            # Direct delivery
            self._schedule_deliveries(message, topic)
        
        # Update load
        load = self.component_loads.get(publisher)
        if load:
            load.messages_processed += 1
    
    def _handle_route(self, event: Event) -> None:
        """Handle broker routing."""
        broker = event.source
        topic = event.target
        msg_id = event.data.get("message_id")
        
        # Check broker status
        if self.component_status.get(broker) == ComponentStatus.FAILED:
            self._mark_message_failed(msg_id, "broker_failed")
            return
        
        message = self.messages.get(msg_id)
        if not message:
            return
        
        message.path.append(broker)
        
        # Check queue capacity
        load = self.component_loads.get(broker)
        if load:
            if load.current_queue_size >= self.queue_capacity:
                self._mark_message_dropped(msg_id, "queue_full")
                load.messages_dropped += 1
                return
            
            load.current_queue_size += 1
            load.peak_queue_size = max(load.peak_queue_size, load.current_queue_size)
        
        # Processing time
        proc_time = self._get_processing_time(broker)
        
        # Schedule deliveries after processing
        delivery_time = self.current_time + proc_time
        self._schedule_deliveries(message, topic, delivery_time)
        
        if load:
            load.messages_processed += 1
            load.current_queue_size -= 1
    
    def _handle_deliver(self, event: Event) -> None:
        """Handle message delivery."""
        subscriber = event.target
        msg_id = event.data.get("message_id")
        
        # Check subscriber status
        if self.component_status.get(subscriber) == ComponentStatus.FAILED:
            return  # Silent drop for this subscriber
        
        message = self.messages.get(msg_id)
        if not message or message.status == MessageStatus.DELIVERED:
            return
        
        # Check timeout
        if self.current_time - message.created_at > self.timeout:
            self._mark_message_timeout(msg_id)
            return
        
        # Deliver
        message.delivered_at = self.current_time
        message.status = MessageStatus.DELIVERED
        message.path.append(subscriber)
        
        self.stats.delivered_messages += 1
        if message.latency:
            self.stats.latencies.append(message.latency)
        
        # Update load
        load = self.component_loads.get(subscriber)
        if load:
            load.messages_processed += 1
            if message.latency:
                load.total_latency += message.latency
    
    def _handle_timeout(self, event: Event) -> None:
        """Handle message timeout."""
        msg_id = event.data.get("message_id")
        message = self.messages.get(msg_id)
        
        if message and message.status == MessageStatus.PENDING:
            self._mark_message_timeout(msg_id)
    
    def _handle_failure(self, event: Event) -> None:
        """Handle component failure."""
        component_id = event.source
        self.component_status[component_id] = ComponentStatus.FAILED
        self.failed_components.add(component_id)
    
    def _handle_recovery(self, event: Event) -> None:
        """Handle component recovery."""
        component_id = event.source
        self.component_status[component_id] = ComponentStatus.HEALTHY
        self.failed_components.discard(component_id)
    
    def _schedule_deliveries(
        self,
        message: Message,
        topic: str,
        base_time: Optional[float] = None,
    ) -> None:
        """Schedule delivery to all subscribers."""
        if base_time is None:
            base_time = self.current_time
        
        subscribers = self.graph.get_subscribers(topic)
        
        for subscriber in subscribers:
            if subscriber == message.publisher:
                continue
            
            latency = self._get_latency(topic, subscriber)
            
            self._schedule_event(
                time=base_time + latency,
                event_type=EventType.DELIVER,
                source=topic,
                target=subscriber,
                data={"message_id": message.message_id},
            )
    
    def _mark_message_failed(self, msg_id: str, reason: str) -> None:
        """Mark message as failed."""
        message = self.messages.get(msg_id)
        if message and message.status not in (MessageStatus.DELIVERED, MessageStatus.FAILED):
            message.status = MessageStatus.FAILED
            message.path.append(f"FAILED:{reason}")
            self.stats.failed_messages += 1
    
    def _mark_message_dropped(self, msg_id: str, reason: str) -> None:
        """Mark message as dropped."""
        message = self.messages.get(msg_id)
        if message:
            message.status = MessageStatus.DROPPED
            self.stats.dropped_messages += 1
    
    def _mark_message_timeout(self, msg_id: str) -> None:
        """Mark message as timed out."""
        message = self.messages.get(msg_id)
        if message and message.status not in (MessageStatus.DELIVERED,):
            message.status = MessageStatus.TIMEOUT
            self.stats.timeout_messages += 1
    
    def _get_latency(self, source: str, target: str) -> float:
        """Get network latency with jitter."""
        base = self.default_latency
        jitter = self.rng.uniform(0.9, 1.1)
        return base * jitter
    
    def _get_processing_time(self, component_id: str) -> float:
        """Get processing time with load factor."""
        base = self.default_processing
        jitter = self.rng.uniform(0.8, 1.2)
        
        load = self.component_loads.get(component_id)
        if load and load.current_queue_size > 0:
            load_factor = 1.0 + (load.current_queue_size / self.queue_capacity) * 0.5
            base *= load_factor
        
        return base * jitter
    
    def _build_result(self, graph: SimulationGraph) -> SimulationResult:
        """Build simulation result with layer statistics."""
        # Calculate layer stats
        layer_stats = {}
        
        for layer_key, layer_def in SimulationGraph.LAYER_DEFINITIONS.items():
            layer_types = set(layer_def["component_types"])
            
            # Count messages involving layer components
            layer_sent = 0
            layer_delivered = 0
            layer_failed = 0
            layer_latencies = []
            
            for msg in self.messages.values():
                # Check if publisher is in this layer
                pub_comp = graph.get_component(msg.publisher)
                if pub_comp and pub_comp.type in layer_types:
                    layer_sent += 1
                    
                    if msg.status == MessageStatus.DELIVERED:
                        layer_delivered += 1
                        if msg.latency:
                            layer_latencies.append(msg.latency)
                    elif msg.status in (MessageStatus.FAILED, MessageStatus.DROPPED, MessageStatus.TIMEOUT):
                        layer_failed += 1
            
            layer_stats[layer_key] = LayerStats(
                layer=layer_key,
                layer_name=layer_def["name"],
                messages_sent=layer_sent,
                messages_delivered=layer_delivered,
                messages_failed=layer_failed,
                avg_latency=stats_module.mean(layer_latencies) if layer_latencies else 0.0,
                component_count=sum(
                    len(graph._by_type[t]) for t in layer_types
                ),
            )
        
        return SimulationResult(
            stats=self.stats,
            component_loads=dict(self.component_loads),
            layer_stats=layer_stats,
            failed_components=set(self.failed_components),
        )


# =============================================================================
# Factory Functions
# =============================================================================

def run_event_simulation(
    graph: SimulationGraph,
    duration: float = 1000.0,
    message_rate: float = 10.0,
    seed: Optional[int] = None,
) -> SimulationResult:
    """
    Quick function to run event simulation.
    
    Args:
        graph: Simulation graph
        duration: Duration in ms
        message_rate: Messages per second
        seed: Random seed
    
    Returns:
        SimulationResult
    """
    simulator = EventSimulator(seed=seed)
    return simulator.run(graph, duration=duration, message_rate=message_rate)


def run_stress_test(
    graph: SimulationGraph,
    duration: float = 5000.0,
    peak_rate: float = 1000.0,
    seed: Optional[int] = None,
) -> SimulationResult:
    """
    Run stress test with high message rate.
    
    Args:
        graph: Simulation graph
        duration: Duration in ms
        peak_rate: Peak messages per second
        seed: Random seed
    
    Returns:
        SimulationResult
    """
    simulator = EventSimulator(
        seed=seed,
        queue_capacity=500,  # Lower capacity for stress
        timeout=2000.0,      # Shorter timeout
    )
    return simulator.run(graph, duration=duration, message_rate=peak_rate)
