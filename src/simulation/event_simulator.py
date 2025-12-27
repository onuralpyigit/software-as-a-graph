"""
Event-Driven Simulator - Version 4.0

Simulates message flow through a pub-sub system using discrete event simulation.
Works directly on the graph model without DEPENDS_ON relationships.

Features:
- Message publishing and delivery through topics
- Processing delays and queuing
- QoS level enforcement (at-most-once, at-least-once, exactly-once)
- Failure injection during simulation
- Load testing with configurable rates
- Chaos engineering scenarios
- Real-time metrics collection

Author: Software-as-a-Graph Research Project
Version: 4.0
"""

from __future__ import annotations
import heapq
import logging
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict

from .graph_model import SimulationGraph, Component, ComponentType, ConnectionType


# =============================================================================
# Enums
# =============================================================================

class EventType(Enum):
    """Types of simulation events"""
    MSG_PUBLISH = "msg_publish"
    MSG_BROKER_RECEIVE = "msg_broker_receive"
    MSG_DELIVER = "msg_deliver"
    MSG_ACK = "msg_ack"
    MSG_TIMEOUT = "msg_timeout"
    COMPONENT_FAIL = "component_fail"
    COMPONENT_RECOVER = "component_recover"
    SIM_END = "sim_end"


class MessageState(Enum):
    """Message lifecycle states"""
    PENDING = "pending"
    IN_TRANSIT = "in_transit"
    AT_BROKER = "at_broker"
    DELIVERED = "delivered"
    FAILED = "failed"
    TIMEOUT = "timeout"


class QoSLevel(Enum):
    """Quality of Service levels"""
    AT_MOST_ONCE = 0    # Fire and forget
    AT_LEAST_ONCE = 1   # Retry until ACK
    EXACTLY_ONCE = 2    # Dedup + retry


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(order=True)
class SimEvent:
    """A discrete simulation event"""
    time: float
    event_type: EventType = field(compare=False)
    data: Dict[str, Any] = field(compare=False, default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "time": round(self.time, 3),
            "type": self.event_type.value,
            "data": self.data,
        }


@dataclass
class Message:
    """A message in the system"""
    id: str
    publisher: str
    topic: str
    subscribers: List[str]
    payload_size: int
    qos: QoSLevel
    
    # Timing
    publish_time: float
    delivery_times: Dict[str, float] = field(default_factory=dict)
    
    # State
    state: MessageState = MessageState.PENDING
    retries: int = 0
    
    def latency(self, subscriber: str) -> float:
        """Get delivery latency for a subscriber"""
        if subscriber in self.delivery_times:
            return self.delivery_times[subscriber] - self.publish_time
        return -1


@dataclass
class ComponentStats:
    """Statistics for a component"""
    messages_sent: int = 0
    messages_received: int = 0
    messages_failed: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    processing_time_total: float = 0.0
    queue_depth_max: int = 0
    failures: int = 0

    def to_dict(self) -> Dict:
        return {
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "messages_failed": self.messages_failed,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "processing_time_total": round(self.processing_time_total, 2),
            "queue_depth_max": self.queue_depth_max,
            "failures": self.failures,
        }


@dataclass
class SimMetrics:
    """Overall simulation metrics"""
    messages_published: int = 0
    messages_delivered: int = 0
    messages_failed: int = 0
    messages_timeout: int = 0
    bytes_total: int = 0
    latencies: List[float] = field(default_factory=list)
    
    # Failures
    component_failures: int = 0
    cascade_failures: int = 0

    def delivery_rate(self) -> float:
        """Calculate successful delivery rate"""
        total = self.messages_published
        if total == 0:
            return 1.0
        return self.messages_delivered / total

    def avg_latency(self) -> float:
        """Calculate average latency"""
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    def p99_latency(self) -> float:
        """Calculate 99th percentile latency"""
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    def to_dict(self) -> Dict:
        return {
            "messages": {
                "published": self.messages_published,
                "delivered": self.messages_delivered,
                "failed": self.messages_failed,
                "timeout": self.messages_timeout,
                "delivery_rate": round(self.delivery_rate(), 4),
            },
            "latency": {
                "avg_ms": round(self.avg_latency(), 2),
                "p99_ms": round(self.p99_latency(), 2),
                "min_ms": round(min(self.latencies), 2) if self.latencies else 0,
                "max_ms": round(max(self.latencies), 2) if self.latencies else 0,
            },
            "throughput": {
                "bytes_total": self.bytes_total,
            },
            "failures": {
                "component_failures": self.component_failures,
                "cascade_failures": self.cascade_failures,
            },
        }


@dataclass
class SimulationResult:
    """Result of event-driven simulation"""
    simulation_id: str
    duration_ms: float
    real_time_ms: float
    speedup: float
    
    metrics: SimMetrics
    component_stats: Dict[str, ComponentStats]
    
    events_processed: int
    events_by_type: Dict[str, int]
    
    failures_injected: List[Dict[str, Any]]

    def to_dict(self) -> Dict:
        return {
            "simulation_id": self.simulation_id,
            "timing": {
                "simulated_duration_ms": round(self.duration_ms, 2),
                "real_time_ms": round(self.real_time_ms, 2),
                "speedup": round(self.speedup, 1),
            },
            "metrics": self.metrics.to_dict(),
            "component_stats": {
                k: v.to_dict() for k, v in self.component_stats.items()
            },
            "events": {
                "total_processed": self.events_processed,
                "by_type": self.events_by_type,
            },
            "failures_injected": self.failures_injected,
        }


# =============================================================================
# Event-Driven Simulator
# =============================================================================

class EventSimulator:
    """
    Discrete event simulator for pub-sub message flow.
    
    Simulates messages flowing through the pub-sub topology:
    Publisher -> Topic -> Broker -> Subscribers
    
    Works directly on SimulationGraph using native pub-sub connections.
    """

    def __init__(
        self,
        base_latency_ms: float = 1.0,
        latency_variance: float = 0.3,
        broker_latency_ms: float = 0.5,
        queue_capacity: int = 1000,
        timeout_ms: float = 5000,
        seed: Optional[int] = None,
    ):
        """
        Initialize the event simulator.
        
        Args:
            base_latency_ms: Base network latency per hop
            latency_variance: Variance in latency (0-1)
            broker_latency_ms: Additional broker processing latency
            queue_capacity: Maximum queue size per component
            timeout_ms: Message timeout threshold
            seed: Random seed for reproducibility
        """
        self.base_latency_ms = base_latency_ms
        self.latency_variance = latency_variance
        self.broker_latency_ms = broker_latency_ms
        self.queue_capacity = queue_capacity
        self.timeout_ms = timeout_ms
        
        self._rng = random.Random(seed)
        self._simulation_counter = 0
        self.logger = logging.getLogger(__name__)
        
        # State (reset per simulation)
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset simulation state"""
        self._event_queue: List[SimEvent] = []
        self._current_time: float = 0.0
        self._messages: Dict[str, Message] = {}
        self._component_queues: Dict[str, List[str]] = defaultdict(list)
        self._metrics = SimMetrics()
        self._component_stats: Dict[str, ComponentStats] = defaultdict(ComponentStats)
        self._events_processed: int = 0
        self._events_by_type: Dict[str, int] = defaultdict(int)
        self._failures_injected: List[Dict[str, Any]] = []
        self._graph: Optional[SimulationGraph] = None
        self._delivered_ids: Set[str] = set()  # For exactly-once dedup

    # =========================================================================
    # Main Simulation
    # =========================================================================

    def simulate(
        self,
        graph: SimulationGraph,
        duration_ms: float = 10000,
        message_rate: float = 100,
        qos: QoSLevel = QoSLevel.AT_LEAST_ONCE,
        failure_schedule: Optional[List[Dict[str, Any]]] = None,
    ) -> SimulationResult:
        """
        Run event-driven simulation.
        
        Args:
            graph: SimulationGraph to simulate on
            duration_ms: Simulation duration in milliseconds
            message_rate: Messages per second
            qos: Quality of service level
            failure_schedule: List of failure events to inject
        
        Returns:
            SimulationResult with metrics and statistics
        """
        self._simulation_counter += 1
        sim_id = f"event_{self._simulation_counter:05d}"
        real_start = datetime.now()
        
        self.logger.info(f"[{sim_id}] Starting simulation: {duration_ms}ms, {message_rate}/sec")
        
        # Reset state
        self._reset_state()
        self._graph = graph.copy()
        
        # Schedule message publications
        self._schedule_publications(duration_ms, message_rate, qos)
        
        # Schedule failures if provided
        if failure_schedule:
            self._schedule_failures(failure_schedule)
        
        # Schedule simulation end
        heapq.heappush(
            self._event_queue,
            SimEvent(time=duration_ms, event_type=EventType.SIM_END),
        )
        
        # Run simulation loop
        while self._event_queue:
            event = heapq.heappop(self._event_queue)
            self._current_time = event.time
            
            if event.event_type == EventType.SIM_END:
                break
            
            self._process_event(event)
            self._events_processed += 1
            self._events_by_type[event.event_type.value] += 1
        
        real_end = datetime.now()
        real_time_ms = (real_end - real_start).total_seconds() * 1000
        
        return SimulationResult(
            simulation_id=sim_id,
            duration_ms=duration_ms,
            real_time_ms=real_time_ms,
            speedup=duration_ms / real_time_ms if real_time_ms > 0 else 0,
            metrics=self._metrics,
            component_stats=dict(self._component_stats),
            events_processed=self._events_processed,
            events_by_type=dict(self._events_by_type),
            failures_injected=self._failures_injected,
        )

    # =========================================================================
    # Load Testing
    # =========================================================================

    def simulate_load_test(
        self,
        graph: SimulationGraph,
        duration_ms: float = 30000,
        initial_rate: float = 10,
        peak_rate: float = 500,
        ramp_time_ms: float = 10000,
        qos: QoSLevel = QoSLevel.AT_LEAST_ONCE,
    ) -> SimulationResult:
        """
        Run load testing simulation with ramping rate.
        
        Starts at initial_rate, ramps to peak_rate over ramp_time,
        then holds at peak_rate.
        """
        self._simulation_counter += 1
        sim_id = f"load_{self._simulation_counter:05d}"
        real_start = datetime.now()
        
        self.logger.info(f"[{sim_id}] Load test: {initial_rate} -> {peak_rate}/sec")
        
        self._reset_state()
        self._graph = graph.copy()
        
        # Schedule publications with ramping rate
        self._schedule_load_test_publications(
            duration_ms, initial_rate, peak_rate, ramp_time_ms, qos
        )
        
        heapq.heappush(
            self._event_queue,
            SimEvent(time=duration_ms, event_type=EventType.SIM_END),
        )
        
        while self._event_queue:
            event = heapq.heappop(self._event_queue)
            self._current_time = event.time
            
            if event.event_type == EventType.SIM_END:
                break
            
            self._process_event(event)
            self._events_processed += 1
            self._events_by_type[event.event_type.value] += 1
        
        real_end = datetime.now()
        real_time_ms = (real_end - real_start).total_seconds() * 1000
        
        return SimulationResult(
            simulation_id=sim_id,
            duration_ms=duration_ms,
            real_time_ms=real_time_ms,
            speedup=duration_ms / real_time_ms if real_time_ms > 0 else 0,
            metrics=self._metrics,
            component_stats=dict(self._component_stats),
            events_processed=self._events_processed,
            events_by_type=dict(self._events_by_type),
            failures_injected=self._failures_injected,
        )

    # =========================================================================
    # Chaos Engineering
    # =========================================================================

    def simulate_chaos(
        self,
        graph: SimulationGraph,
        duration_ms: float = 30000,
        message_rate: float = 100,
        failure_probability: float = 0.01,
        recovery_probability: float = 0.1,
        qos: QoSLevel = QoSLevel.AT_LEAST_ONCE,
    ) -> SimulationResult:
        """
        Run chaos engineering simulation with random failures.
        
        Components fail and recover probabilistically during simulation.
        """
        self._simulation_counter += 1
        sim_id = f"chaos_{self._simulation_counter:05d}"
        real_start = datetime.now()
        
        self.logger.info(f"[{sim_id}] Chaos: fail_prob={failure_probability}")
        
        self._reset_state()
        self._graph = graph.copy()
        
        # Schedule publications
        self._schedule_publications(duration_ms, message_rate, qos)
        
        # Schedule random failures throughout simulation
        self._schedule_chaos_failures(
            duration_ms, failure_probability, recovery_probability
        )
        
        heapq.heappush(
            self._event_queue,
            SimEvent(time=duration_ms, event_type=EventType.SIM_END),
        )
        
        while self._event_queue:
            event = heapq.heappop(self._event_queue)
            self._current_time = event.time
            
            if event.event_type == EventType.SIM_END:
                break
            
            self._process_event(event)
            self._events_processed += 1
            self._events_by_type[event.event_type.value] += 1
        
        real_end = datetime.now()
        real_time_ms = (real_end - real_start).total_seconds() * 1000
        
        return SimulationResult(
            simulation_id=sim_id,
            duration_ms=duration_ms,
            real_time_ms=real_time_ms,
            speedup=duration_ms / real_time_ms if real_time_ms > 0 else 0,
            metrics=self._metrics,
            component_stats=dict(self._component_stats),
            events_processed=self._events_processed,
            events_by_type=dict(self._events_by_type),
            failures_injected=self._failures_injected,
        )

    # =========================================================================
    # Event Scheduling
    # =========================================================================

    def _schedule_publications(
        self,
        duration_ms: float,
        message_rate: float,
        qos: QoSLevel,
    ) -> None:
        """Schedule message publications throughout simulation"""
        # Get active publishers
        publishers = [
            c for c in self._graph.get_components_by_type(ComponentType.APPLICATION)
            if c.is_active and self._graph.get_published_topics(c.id)
        ]
        
        if not publishers:
            self.logger.warning("No active publishers found")
            return
        
        # Calculate inter-arrival time
        interval_ms = 1000.0 / message_rate if message_rate > 0 else 1000
        
        current_time = 0.0
        while current_time < duration_ms:
            # Pick random publisher
            publisher = self._rng.choice(publishers)
            topics = self._graph.get_published_topics(publisher.id)
            
            if topics:
                topic = self._rng.choice(topics)
                
                # Schedule publish event
                heapq.heappush(
                    self._event_queue,
                    SimEvent(
                        time=current_time,
                        event_type=EventType.MSG_PUBLISH,
                        data={
                            "publisher": publisher.id,
                            "topic": topic,
                            "qos": qos.value,
                            "size": self._rng.randint(100, 10000),
                        },
                    ),
                )
            
            # Add jitter
            jitter = self._rng.uniform(-0.2, 0.2) * interval_ms
            current_time += interval_ms + jitter

    def _schedule_load_test_publications(
        self,
        duration_ms: float,
        initial_rate: float,
        peak_rate: float,
        ramp_time_ms: float,
        qos: QoSLevel,
    ) -> None:
        """Schedule publications with ramping rate"""
        publishers = [
            c for c in self._graph.get_components_by_type(ComponentType.APPLICATION)
            if c.is_active and self._graph.get_published_topics(c.id)
        ]
        
        if not publishers:
            return
        
        current_time = 0.0
        while current_time < duration_ms:
            # Calculate current rate based on ramp
            if current_time < ramp_time_ms:
                progress = current_time / ramp_time_ms
                rate = initial_rate + (peak_rate - initial_rate) * progress
            else:
                rate = peak_rate
            
            interval_ms = 1000.0 / rate if rate > 0 else 1000
            
            publisher = self._rng.choice(publishers)
            topics = self._graph.get_published_topics(publisher.id)
            
            if topics:
                topic = self._rng.choice(topics)
                heapq.heappush(
                    self._event_queue,
                    SimEvent(
                        time=current_time,
                        event_type=EventType.MSG_PUBLISH,
                        data={
                            "publisher": publisher.id,
                            "topic": topic,
                            "qos": qos.value,
                            "size": self._rng.randint(100, 10000),
                        },
                    ),
                )
            
            current_time += interval_ms * self._rng.uniform(0.8, 1.2)

    def _schedule_failures(self, schedule: List[Dict[str, Any]]) -> None:
        """Schedule predetermined failures"""
        for failure in schedule:
            heapq.heappush(
                self._event_queue,
                SimEvent(
                    time=failure.get("time_ms", 0),
                    event_type=EventType.COMPONENT_FAIL,
                    data={
                        "component": failure.get("component"),
                        "recover_time_ms": failure.get("recover_time_ms"),
                    },
                ),
            )

    def _schedule_chaos_failures(
        self,
        duration_ms: float,
        failure_prob: float,
        recovery_prob: float,
    ) -> None:
        """Schedule random failures for chaos simulation"""
        components = list(self._graph.components.keys())
        
        # Check every 100ms for failures
        check_interval = 100.0
        current_time = check_interval
        
        while current_time < duration_ms:
            for comp_id in components:
                comp = self._graph.components.get(comp_id)
                if not comp:
                    continue
                
                if comp.is_active and self._rng.random() < failure_prob:
                    # Schedule failure
                    heapq.heappush(
                        self._event_queue,
                        SimEvent(
                            time=current_time,
                            event_type=EventType.COMPONENT_FAIL,
                            data={"component": comp_id},
                        ),
                    )
                elif not comp.is_active and self._rng.random() < recovery_prob:
                    # Schedule recovery
                    heapq.heappush(
                        self._event_queue,
                        SimEvent(
                            time=current_time,
                            event_type=EventType.COMPONENT_RECOVER,
                            data={"component": comp_id},
                        ),
                    )
            
            current_time += check_interval

    # =========================================================================
    # Event Processing
    # =========================================================================

    def _process_event(self, event: SimEvent) -> None:
        """Process a simulation event"""
        if event.event_type == EventType.MSG_PUBLISH:
            self._handle_publish(event.data)
        
        elif event.event_type == EventType.MSG_BROKER_RECEIVE:
            self._handle_broker_receive(event.data)
        
        elif event.event_type == EventType.MSG_DELIVER:
            self._handle_deliver(event.data)
        
        elif event.event_type == EventType.MSG_TIMEOUT:
            self._handle_timeout(event.data)
        
        elif event.event_type == EventType.COMPONENT_FAIL:
            self._handle_failure(event.data)
        
        elif event.event_type == EventType.COMPONENT_RECOVER:
            self._handle_recovery(event.data)

    def _handle_publish(self, data: Dict) -> None:
        """Handle message publish event"""
        publisher_id = data["publisher"]
        topic_id = data["topic"]
        qos = QoSLevel(data["qos"])
        size = data["size"]
        
        # Check if publisher is active
        publisher = self._graph.components.get(publisher_id)
        if not publisher or not publisher.is_active:
            self._metrics.messages_failed += 1
            return
        
        # Check if topic is active
        topic = self._graph.components.get(topic_id)
        if not topic or not topic.is_active:
            self._metrics.messages_failed += 1
            return
        
        # Get subscribers
        subscribers = [
            s for s in self._graph.get_subscribers(topic_id)
            if self._graph.components.get(s, Component("", ComponentType.APPLICATION)).is_active
        ]
        
        if not subscribers:
            # No subscribers - message still counts as published
            self._metrics.messages_published += 1
            return
        
        # Create message
        msg_id = str(uuid.uuid4())[:8]
        msg = Message(
            id=msg_id,
            publisher=publisher_id,
            topic=topic_id,
            subscribers=subscribers,
            payload_size=size,
            qos=qos,
            publish_time=self._current_time,
        )
        self._messages[msg_id] = msg
        
        # Update stats
        self._metrics.messages_published += 1
        self._metrics.bytes_total += size
        self._component_stats[publisher_id].messages_sent += 1
        self._component_stats[publisher_id].bytes_sent += size
        
        # Get broker for topic
        broker_id = self._graph.get_broker_for_topic(topic_id)
        
        if broker_id:
            broker = self._graph.components.get(broker_id)
            if broker and broker.is_active:
                # Route through broker
                latency = self._calculate_latency()
                heapq.heappush(
                    self._event_queue,
                    SimEvent(
                        time=self._current_time + latency,
                        event_type=EventType.MSG_BROKER_RECEIVE,
                        data={"msg_id": msg_id, "broker": broker_id},
                    ),
                )
                msg.state = MessageState.IN_TRANSIT
            else:
                # Broker down - fail message
                msg.state = MessageState.FAILED
                self._metrics.messages_failed += 1
        else:
            # No broker - deliver directly to subscribers
            for sub in subscribers:
                latency = self._calculate_latency()
                heapq.heappush(
                    self._event_queue,
                    SimEvent(
                        time=self._current_time + latency,
                        event_type=EventType.MSG_DELIVER,
                        data={"msg_id": msg_id, "subscriber": sub},
                    ),
                )
            msg.state = MessageState.IN_TRANSIT
        
        # Schedule timeout
        heapq.heappush(
            self._event_queue,
            SimEvent(
                time=self._current_time + self.timeout_ms,
                event_type=EventType.MSG_TIMEOUT,
                data={"msg_id": msg_id},
            ),
        )

    def _handle_broker_receive(self, data: Dict) -> None:
        """Handle message arrival at broker"""
        msg_id = data["msg_id"]
        broker_id = data["broker"]
        
        msg = self._messages.get(msg_id)
        if not msg or msg.state not in (MessageState.IN_TRANSIT, MessageState.AT_BROKER):
            return
        
        broker = self._graph.components.get(broker_id)
        if not broker or not broker.is_active:
            msg.state = MessageState.FAILED
            self._metrics.messages_failed += 1
            return
        
        msg.state = MessageState.AT_BROKER
        self._component_stats[broker_id].messages_received += 1
        
        # Add broker processing delay
        broker_delay = self.broker_latency_ms * self._rng.uniform(0.8, 1.2)
        
        # Deliver to each subscriber
        for sub in msg.subscribers:
            sub_comp = self._graph.components.get(sub)
            if sub_comp and sub_comp.is_active:
                latency = self._calculate_latency() + broker_delay
                heapq.heappush(
                    self._event_queue,
                    SimEvent(
                        time=self._current_time + latency,
                        event_type=EventType.MSG_DELIVER,
                        data={"msg_id": msg_id, "subscriber": sub},
                    ),
                )

    def _handle_deliver(self, data: Dict) -> None:
        """Handle message delivery to subscriber"""
        msg_id = data["msg_id"]
        subscriber_id = data["subscriber"]
        
        msg = self._messages.get(msg_id)
        if not msg:
            return
        
        # Check for exactly-once dedup
        if msg.qos == QoSLevel.EXACTLY_ONCE:
            dedup_key = f"{msg_id}:{subscriber_id}"
            if dedup_key in self._delivered_ids:
                return
            self._delivered_ids.add(dedup_key)
        
        if msg.state in (MessageState.FAILED, MessageState.TIMEOUT):
            return
        
        subscriber = self._graph.components.get(subscriber_id)
        if not subscriber or not subscriber.is_active:
            # Retry for at-least-once
            if msg.qos in (QoSLevel.AT_LEAST_ONCE, QoSLevel.EXACTLY_ONCE):
                if msg.retries < 3:
                    msg.retries += 1
                    latency = self._calculate_latency() * (msg.retries + 1)
                    heapq.heappush(
                        self._event_queue,
                        SimEvent(
                            time=self._current_time + latency,
                            event_type=EventType.MSG_DELIVER,
                            data={"msg_id": msg_id, "subscriber": subscriber_id},
                        ),
                    )
            return
        
        # Successful delivery
        msg.delivery_times[subscriber_id] = self._current_time
        latency = msg.latency(subscriber_id)
        
        self._metrics.messages_delivered += 1
        self._metrics.latencies.append(latency)
        
        self._component_stats[subscriber_id].messages_received += 1
        self._component_stats[subscriber_id].bytes_received += msg.payload_size
        
        # Check if all subscribers received
        if len(msg.delivery_times) == len(msg.subscribers):
            msg.state = MessageState.DELIVERED

    def _handle_timeout(self, data: Dict) -> None:
        """Handle message timeout"""
        msg_id = data["msg_id"]
        msg = self._messages.get(msg_id)
        
        if not msg:
            return
        
        if msg.state not in (MessageState.DELIVERED, MessageState.FAILED, MessageState.TIMEOUT):
            msg.state = MessageState.TIMEOUT
            self._metrics.messages_timeout += 1

    def _handle_failure(self, data: Dict) -> None:
        """Handle component failure event"""
        comp_id = data["component"]
        comp = self._graph.components.get(comp_id)
        
        if not comp:
            return
        
        comp.is_active = False
        
        # Deactivate connections
        for conn in self._graph.get_outgoing(comp_id):
            conn.is_active = False
        for conn in self._graph.get_incoming(comp_id):
            conn.is_active = False
        
        self._metrics.component_failures += 1
        self._component_stats[comp_id].failures += 1
        
        self._failures_injected.append({
            "component": comp_id,
            "time_ms": self._current_time,
            "type": "failure",
        })
        
        # Schedule recovery if specified
        recover_time = data.get("recover_time_ms")
        if recover_time:
            heapq.heappush(
                self._event_queue,
                SimEvent(
                    time=self._current_time + recover_time,
                    event_type=EventType.COMPONENT_RECOVER,
                    data={"component": comp_id},
                ),
            )

    def _handle_recovery(self, data: Dict) -> None:
        """Handle component recovery event"""
        comp_id = data["component"]
        comp = self._graph.components.get(comp_id)
        
        if not comp:
            return
        
        comp.is_active = True
        
        # Reactivate connections
        for conn in self._graph.get_outgoing(comp_id):
            conn.is_active = True
        for conn in self._graph.get_incoming(comp_id):
            conn.is_active = True
        
        self._failures_injected.append({
            "component": comp_id,
            "time_ms": self._current_time,
            "type": "recovery",
        })

    def _calculate_latency(self) -> float:
        """Calculate latency with variance"""
        variance = self._rng.uniform(-self.latency_variance, self.latency_variance)
        return self.base_latency_ms * (1 + variance)