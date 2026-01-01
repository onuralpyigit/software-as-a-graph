"""
Event-Driven Simulator - Version 5.0

Discrete event simulation for pub-sub message flow analysis.
Uses ORIGINAL edge types (NOT derived DEPENDS_ON).

Features:
- Message publication and delivery simulation
- QoS-aware message handling
- Latency and throughput modeling
- Component load tracking
- Event scheduling with priority queue
- Statistics collection

Event Types:
- PUBLISH: Application publishes message to topic
- ROUTE: Broker routes message
- DELIVER: Message delivered to subscriber
- TIMEOUT: Message delivery timeout
- FAILURE: Component failure event
- RECOVERY: Component recovery event

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
import heapq
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Dict, List, Set, Any, Optional, Tuple, 
    Callable, Iterator, TypeVar, Generic
)
from collections import defaultdict
import statistics

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
    """Types of simulation events"""
    PUBLISH = auto()      # App publishes message
    ROUTE = auto()        # Broker routes message
    DELIVER = auto()      # Message delivered to subscriber
    TIMEOUT = auto()      # Message timeout
    FAILURE = auto()      # Component fails
    RECOVERY = auto()     # Component recovers
    LOAD_CHECK = auto()   # Periodic load assessment
    CUSTOM = auto()       # User-defined event


class MessageStatus(Enum):
    """Status of a message in transit"""
    PENDING = "pending"
    ROUTING = "routing"
    DELIVERED = "delivered"
    FAILED = "failed"
    TIMEOUT = "timeout"
    DROPPED = "dropped"


class SimulationState(Enum):
    """State of the simulation"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(order=True)
class Event:
    """A simulation event"""
    time: float
    priority: int = field(compare=True)
    event_type: EventType = field(compare=False)
    source: str = field(compare=False)
    target: Optional[str] = field(default=None, compare=False)
    data: Dict[str, Any] = field(default_factory=dict, compare=False)
    event_id: int = field(default=0, compare=False)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "time": self.time,
            "type": self.event_type.name,
            "source": self.source,
            "target": self.target,
            "data": self.data,
            "event_id": self.event_id,
        }


@dataclass
class Message:
    """A message in the pub-sub system"""
    message_id: str
    publisher: str
    topic: str
    payload_size: int = 100  # bytes
    priority: int = 0
    qos: QoSPolicy = field(default_factory=QoSPolicy)
    created_at: float = 0.0
    delivered_at: Optional[float] = None
    status: MessageStatus = MessageStatus.PENDING
    path: List[str] = field(default_factory=list)
    
    @property
    def latency(self) -> Optional[float]:
        """Delivery latency in ms"""
        if self.delivered_at is not None:
            return self.delivered_at - self.created_at
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "publisher": self.publisher,
            "topic": self.topic,
            "payload_size": self.payload_size,
            "priority": self.priority,
            "status": self.status.value,
            "created_at": self.created_at,
            "delivered_at": self.delivered_at,
            "latency": self.latency,
            "path": self.path,
        }


@dataclass
class ComponentLoad:
    """Load statistics for a component"""
    component_id: str
    messages_processed: int = 0
    messages_dropped: int = 0
    total_latency: float = 0.0
    peak_queue_size: int = 0
    current_queue_size: int = 0
    
    @property
    def average_latency(self) -> float:
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
            "messages_processed": self.messages_processed,
            "messages_dropped": self.messages_dropped,
            "average_latency": round(self.average_latency, 4),
            "drop_rate": round(self.drop_rate, 4),
            "peak_queue_size": self.peak_queue_size,
        }


@dataclass
class SimulationStatistics:
    """Overall simulation statistics"""
    total_messages: int = 0
    delivered_messages: int = 0
    failed_messages: int = 0
    dropped_messages: int = 0
    timeout_messages: int = 0
    total_events: int = 0
    simulation_time: float = 0.0
    wall_clock_time: float = 0.0
    latencies: List[float] = field(default_factory=list)
    throughput_samples: List[float] = field(default_factory=list)
    
    @property
    def delivery_rate(self) -> float:
        if self.total_messages > 0:
            return self.delivered_messages / self.total_messages
        return 0.0
    
    @property
    def average_latency(self) -> float:
        if self.latencies:
            return statistics.mean(self.latencies)
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
    
    @property
    def average_throughput(self) -> float:
        if self.throughput_samples:
            return statistics.mean(self.throughput_samples)
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_messages": self.total_messages,
            "delivered_messages": self.delivered_messages,
            "failed_messages": self.failed_messages,
            "dropped_messages": self.dropped_messages,
            "timeout_messages": self.timeout_messages,
            "delivery_rate": round(self.delivery_rate, 4),
            "average_latency_ms": round(self.average_latency, 4),
            "p99_latency_ms": round(self.p99_latency, 4),
            "average_throughput": round(self.average_throughput, 4),
            "total_events": self.total_events,
            "simulation_time_ms": round(self.simulation_time, 2),
            "wall_clock_time_s": round(self.wall_clock_time, 4),
        }


@dataclass
class SimulationResult:
    """Complete simulation results"""
    statistics: SimulationStatistics
    component_loads: Dict[str, ComponentLoad]
    messages: List[Message]
    events_log: List[Event]
    failed_components: Set[str]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_critical_components(self, threshold: float = 0.1) -> List[str]:
        """Get components with drop rate above threshold"""
        return [
            comp_id for comp_id, load in self.component_loads.items()
            if load.drop_rate > threshold
        ]
    
    def get_bottlenecks(self, threshold: int = 100) -> List[str]:
        """Get components with peak queue above threshold"""
        return [
            comp_id for comp_id, load in self.component_loads.items()
            if load.peak_queue_size > threshold
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "statistics": self.statistics.to_dict(),
            "component_loads": {k: v.to_dict() for k, v in self.component_loads.items()},
            "message_count": len(self.messages),
            "sample_messages": [m.to_dict() for m in self.messages[:100]],
            "failed_components": list(self.failed_components),
            "critical_components": self.get_critical_components(),
            "bottlenecks": self.get_bottlenecks(),
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# Event Simulator
# =============================================================================

class EventSimulator:
    """
    Discrete event simulation for pub-sub message flow.
    
    Uses ORIGINAL edge types to trace message paths:
    - PUBLISHES_TO: App -> Topic
    - SUBSCRIBES_TO: App <- Topic (delivery direction)
    - ROUTES: Broker -> Topic
    """
    
    def __init__(
        self,
        graph: SimulationGraph,
        seed: Optional[int] = None,
        default_latency: float = 10.0,  # ms
        default_processing_time: float = 1.0,  # ms
        queue_capacity: int = 1000,
        timeout: float = 5000.0,  # ms
    ):
        """
        Initialize the event simulator.
        
        Args:
            graph: The simulation graph
            seed: Random seed for reproducibility
            default_latency: Default network latency in ms
            default_processing_time: Default processing time in ms
            queue_capacity: Max queue size per component
            timeout: Message timeout in ms
        """
        self.graph = graph
        self.rng = random.Random(seed)
        self.default_latency = default_latency
        self.default_processing_time = default_processing_time
        self.queue_capacity = queue_capacity
        self.timeout = timeout
        
        # Simulation state
        self.current_time: float = 0.0
        self.event_queue: List[Event] = []
        self.event_counter: int = 0
        self.state = SimulationState.IDLE
        
        # Component state
        self.component_status: Dict[str, ComponentStatus] = {}
        self.component_queues: Dict[str, List[Message]] = defaultdict(list)
        self.component_loads: Dict[str, ComponentLoad] = {}
        
        # Message tracking
        self.messages: Dict[str, Message] = {}
        self.events_log: List[Event] = []
        
        # Statistics
        self.stats = SimulationStatistics()
        
        # Initialize component states
        self._initialize_components()
        
        self._logger = logging.getLogger(__name__)
    
    def _initialize_components(self) -> None:
        """Initialize component states and loads"""
        for comp_id in self.graph.components:
            self.component_status[comp_id] = ComponentStatus.HEALTHY
            self.component_loads[comp_id] = ComponentLoad(component_id=comp_id)
    
    # =========================================================================
    # Event Management
    # =========================================================================
    
    def schedule_event(
        self,
        time: float,
        event_type: EventType,
        source: str,
        target: Optional[str] = None,
        priority: int = 5,
        data: Optional[Dict[str, Any]] = None,
    ) -> Event:
        """Schedule a new event"""
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
    
    def cancel_event(self, event_id: int) -> bool:
        """Cancel a scheduled event (marks as cancelled)"""
        for event in self.event_queue:
            if event.event_id == event_id:
                event.data["cancelled"] = True
                return True
        return False
    
    def pop_event(self) -> Optional[Event]:
        """Get next event from queue"""
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            if not event.data.get("cancelled"):
                return event
        return None
    
    # =========================================================================
    # Message Operations
    # =========================================================================
    
    def create_message(
        self,
        publisher: str,
        topic: str,
        payload_size: int = 100,
        priority: int = 0,
    ) -> Message:
        """Create a new message"""
        msg_id = f"msg_{self.stats.total_messages + 1}"
        
        # Get QoS from edge if available
        qos = QoSPolicy()
        for edge in self.graph.get_outgoing_edges(publisher):
            if edge.target == topic and edge.edge_type == EdgeType.PUBLISHES_TO:
                qos = edge.qos
                break
        
        message = Message(
            message_id=msg_id,
            publisher=publisher,
            topic=topic,
            payload_size=payload_size,
            priority=priority,
            qos=qos,
            created_at=self.current_time,
            path=[publisher],
        )
        
        self.messages[msg_id] = message
        self.stats.total_messages += 1
        
        return message
    
    def _get_processing_time(self, component_id: str) -> float:
        """Get processing time with jitter"""
        base = self.default_processing_time
        jitter = self.rng.uniform(0.8, 1.2)
        
        # Add load-based delay
        load = self.component_loads.get(component_id)
        if load and load.current_queue_size > 0:
            load_factor = 1.0 + (load.current_queue_size / self.queue_capacity) * 0.5
            base *= load_factor
        
        return base * jitter
    
    def _get_network_latency(self, source: str, target: str) -> float:
        """Get network latency between components"""
        base = self.default_latency
        jitter = self.rng.uniform(0.9, 1.1)
        return base * jitter
    
    # =========================================================================
    # Event Handlers
    # =========================================================================
    
    def _handle_publish(self, event: Event) -> None:
        """Handle message publication event"""
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
        
        # Find broker for this topic
        broker = self.graph.get_broker_for_topic(topic)
        
        if broker:
            # Schedule routing through broker
            latency = self._get_network_latency(publisher, broker)
            self.schedule_event(
                time=self.current_time + latency,
                event_type=EventType.ROUTE,
                source=broker,
                target=topic,
                priority=message.priority,
                data={"message_id": msg_id},
            )
            message.status = MessageStatus.ROUTING
        else:
            # Direct delivery (no broker)
            self._schedule_deliveries(message, topic)
        
        # Update publisher load
        load = self.component_loads[publisher]
        load.messages_processed += 1
    
    def _handle_route(self, event: Event) -> None:
        """Handle broker routing event"""
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
        load = self.component_loads[broker]
        if load.current_queue_size >= self.queue_capacity:
            self._mark_message_dropped(msg_id, "queue_full")
            load.messages_dropped += 1
            return
        
        # Process message
        processing_time = self._get_processing_time(broker)
        load.current_queue_size += 1
        load.peak_queue_size = max(load.peak_queue_size, load.current_queue_size)
        
        # Schedule delivery to subscribers after processing
        delivery_time = self.current_time + processing_time
        
        # Schedule deliveries
        self._schedule_deliveries(message, topic, delivery_time)
        
        # Update load
        load.messages_processed += 1
        load.current_queue_size -= 1
    
    def _handle_deliver(self, event: Event) -> None:
        """Handle message delivery event"""
        subscriber = event.target
        msg_id = event.data.get("message_id")
        
        # Check subscriber status
        if self.component_status.get(subscriber) == ComponentStatus.FAILED:
            # Message delivery failed but not entire message
            return
        
        message = self.messages.get(msg_id)
        if not message:
            return
        
        # Check timeout
        if self.current_time - message.created_at > self.timeout:
            self._mark_message_timeout(msg_id)
            return
        
        # Mark delivered
        message.delivered_at = self.current_time
        message.status = MessageStatus.DELIVERED
        message.path.append(subscriber)
        
        # Update statistics
        self.stats.delivered_messages += 1
        if message.latency:
            self.stats.latencies.append(message.latency)
        
        # Update subscriber load
        load = self.component_loads[subscriber]
        load.messages_processed += 1
        if message.latency:
            load.total_latency += message.latency
    
    def _handle_failure(self, event: Event) -> None:
        """Handle component failure event"""
        component_id = event.source
        self.component_status[component_id] = ComponentStatus.FAILED
        
        # Fail in-flight messages to this component
        for msg_id, message in self.messages.items():
            if message.status == MessageStatus.ROUTING:
                if component_id in message.path or message.topic == component_id:
                    self._mark_message_failed(msg_id, "component_failed")
    
    def _handle_recovery(self, event: Event) -> None:
        """Handle component recovery event"""
        component_id = event.source
        self.component_status[component_id] = ComponentStatus.HEALTHY
    
    def _schedule_deliveries(
        self,
        message: Message,
        topic: str,
        base_time: Optional[float] = None,
    ) -> None:
        """Schedule delivery events to all subscribers"""
        if base_time is None:
            base_time = self.current_time
        
        subscribers = self.graph.get_subscribers(topic)
        
        for subscriber in subscribers:
            if subscriber == message.publisher:
                continue  # Don't deliver to publisher
            
            latency = self._get_network_latency(topic, subscriber)
            
            self.schedule_event(
                time=base_time + latency,
                event_type=EventType.DELIVER,
                source=topic,
                target=subscriber,
                priority=message.priority,
                data={"message_id": message.message_id},
            )
    
    def _mark_message_failed(self, msg_id: str, reason: str) -> None:
        """Mark a message as failed"""
        message = self.messages.get(msg_id)
        if message and message.status not in (MessageStatus.DELIVERED, MessageStatus.FAILED):
            message.status = MessageStatus.FAILED
            message.path.append(f"FAILED:{reason}")
            self.stats.failed_messages += 1
    
    def _mark_message_dropped(self, msg_id: str, reason: str) -> None:
        """Mark a message as dropped"""
        message = self.messages.get(msg_id)
        if message:
            message.status = MessageStatus.DROPPED
            message.path.append(f"DROPPED:{reason}")
            self.stats.dropped_messages += 1
    
    def _mark_message_timeout(self, msg_id: str) -> None:
        """Mark a message as timed out"""
        message = self.messages.get(msg_id)
        if message and message.status not in (MessageStatus.DELIVERED,):
            message.status = MessageStatus.TIMEOUT
            self.stats.timeout_messages += 1
    
    # =========================================================================
    # Simulation Control
    # =========================================================================
    
    def run(
        self,
        duration: float,
        message_rate: float = 10.0,
        publishers: Optional[List[str]] = None,
        log_events: bool = False,
    ) -> SimulationResult:
        """
        Run the simulation.
        
        Args:
            duration: Simulation duration in ms
            message_rate: Messages per ms (across all publishers)
            publishers: Specific publishers to use (default: all apps)
            log_events: Whether to log all events
        
        Returns:
            SimulationResult with complete statistics
        """
        import time
        start_wall_time = time.time()
        
        self.state = SimulationState.RUNNING
        self._logger.info(f"Starting simulation for {duration}ms")
        
        # Get publishers
        if publishers is None:
            publishers = list(self.graph.get_component_ids_by_type(ComponentType.APPLICATION))
        
        # Schedule initial message publications
        self._schedule_message_generation(duration, message_rate, publishers)
        
        # Event handlers
        handlers = {
            EventType.PUBLISH: self._handle_publish,
            EventType.ROUTE: self._handle_route,
            EventType.DELIVER: self._handle_deliver,
            EventType.FAILURE: self._handle_failure,
            EventType.RECOVERY: self._handle_recovery,
        }
        
        # Main simulation loop
        while self.event_queue and self.state == SimulationState.RUNNING:
            event = self.pop_event()
            if not event:
                break
            
            # Check if past duration
            if event.time > duration:
                break
            
            self.current_time = event.time
            self.stats.total_events += 1
            
            if log_events:
                self.events_log.append(event)
            
            # Handle event
            handler = handlers.get(event.event_type)
            if handler:
                handler(event)
        
        # Finalize
        self.state = SimulationState.COMPLETED
        self.stats.simulation_time = duration
        self.stats.wall_clock_time = time.time() - start_wall_time
        
        # Calculate throughput
        if duration > 0:
            throughput = self.stats.delivered_messages / (duration / 1000.0)
            self.stats.throughput_samples.append(throughput)
        
        self._logger.info(
            f"Simulation complete: {self.stats.delivered_messages}/{self.stats.total_messages} "
            f"delivered ({self.stats.delivery_rate:.2%})"
        )
        
        return SimulationResult(
            statistics=self.stats,
            component_loads=self.component_loads,
            messages=list(self.messages.values()),
            events_log=self.events_log,
            failed_components={
                k for k, v in self.component_status.items() 
                if v == ComponentStatus.FAILED
            },
        )
    
    def _schedule_message_generation(
        self,
        duration: float,
        message_rate: float,
        publishers: List[str],
    ) -> None:
        """Schedule message generation events"""
        if not publishers:
            return
        
        # Calculate inter-arrival time
        inter_arrival = 1.0 / message_rate if message_rate > 0 else duration
        
        current_time = 0.0
        while current_time < duration:
            # Pick random publisher
            publisher = self.rng.choice(publishers)
            
            # Pick random topic from publisher's topics
            topics = list(self.graph.get_topics_published_by(publisher))
            if not topics:
                current_time += inter_arrival
                continue
            
            topic = self.rng.choice(topics)
            
            # Create and schedule message
            message = self.create_message(publisher, topic)
            
            self.schedule_event(
                time=current_time,
                event_type=EventType.PUBLISH,
                source=publisher,
                target=topic,
                priority=message.priority,
                data={"message_id": message.message_id},
            )
            
            # Next arrival (exponential distribution)
            current_time += self.rng.expovariate(1.0 / inter_arrival)
    
    def inject_failure(
        self,
        component_id: str,
        at_time: float,
        duration: Optional[float] = None,
    ) -> None:
        """
        Inject a component failure during simulation.
        
        Args:
            component_id: Component to fail
            at_time: When to trigger failure
            duration: How long failure lasts (None = permanent)
        """
        self.schedule_event(
            time=at_time,
            event_type=EventType.FAILURE,
            source=component_id,
            priority=0,  # High priority
        )
        
        if duration is not None:
            self.schedule_event(
                time=at_time + duration,
                event_type=EventType.RECOVERY,
                source=component_id,
                priority=0,
            )
    
    def reset(self) -> None:
        """Reset simulation state"""
        self.current_time = 0.0
        self.event_queue = []
        self.event_counter = 0
        self.state = SimulationState.IDLE
        self.messages = {}
        self.events_log = []
        self.stats = SimulationStatistics()
        self._initialize_components()


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
        message_rate: Messages per ms
        seed: Random seed
    
    Returns:
        SimulationResult
    """
    simulator = EventSimulator(graph, seed=seed)
    return simulator.run(duration, message_rate)


def run_stress_test(
    graph: SimulationGraph,
    duration: float = 5000.0,
    message_rate: float = 100.0,
    failure_rate: float = 0.01,
    seed: Optional[int] = None,
) -> SimulationResult:
    """
    Run stress test with random failures.
    
    Args:
        graph: Simulation graph
        duration: Duration in ms
        message_rate: Messages per ms
        failure_rate: Probability of failure per component per 1000ms
        seed: Random seed
    
    Returns:
        SimulationResult
    """
    rng = random.Random(seed)
    simulator = EventSimulator(graph, seed=seed)
    
    # Schedule random failures
    for comp_id in graph.components:
        if rng.random() < failure_rate:
            fail_time = rng.uniform(0, duration * 0.8)
            recovery_time = rng.uniform(100, 500)
            simulator.inject_failure(comp_id, fail_time, recovery_time)
    
    return simulator.run(duration, message_rate)
