#!/usr/bin/env python3
"""
Event-Driven Simulator
=======================

Simulates message flow through a distributed pub-sub system using
discrete event simulation.

Features:
- Message publishing and delivery
- Processing delays and queuing
- Failure injection during simulation
- QoS level enforcement
- Load testing and chaos engineering
- Real-time metrics collection

Author: Software-as-a-Graph Research Project
"""

import heapq
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Callable
from collections import defaultdict

from .neo4j_loader import SimulationGraph, Component, ComponentType


class EventType(Enum):
    """Types of simulation events"""
    MESSAGE_PUBLISH = "message_publish"
    MESSAGE_ARRIVE = "message_arrive"
    MESSAGE_PROCESS = "message_process"
    MESSAGE_DELIVER = "message_deliver"
    MESSAGE_TIMEOUT = "message_timeout"
    COMPONENT_FAIL = "component_fail"
    COMPONENT_RECOVER = "component_recover"
    SIMULATION_END = "simulation_end"


class MessageState(Enum):
    """Message lifecycle states"""
    PENDING = "pending"
    IN_TRANSIT = "in_transit"
    PROCESSING = "processing"
    DELIVERED = "delivered"
    FAILED = "failed"
    TIMEOUT = "timeout"


class QoSLevel(Enum):
    """Quality of Service levels"""
    AT_MOST_ONCE = 0   # Fire and forget
    AT_LEAST_ONCE = 1  # With retry
    EXACTLY_ONCE = 2   # With dedup


@dataclass(order=True)
class SimEvent:
    """A discrete simulation event"""
    time: float
    event_type: EventType = field(compare=False)
    data: Dict[str, Any] = field(compare=False, default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'time': round(self.time, 3),
            'type': self.event_type.value,
            'data': self.data
        }


@dataclass
class Message:
    """A message in the system"""
    id: str
    source: str
    target: str
    payload_size: int  # bytes
    qos: QoSLevel
    created_at: float
    deadline: Optional[float] = None
    
    # Runtime state
    state: MessageState = MessageState.PENDING
    current_location: Optional[str] = None
    hops: List[str] = field(default_factory=list)
    delivered_at: Optional[float] = None
    failed_at: Optional[float] = None
    retries: int = 0
    
    def latency(self) -> Optional[float]:
        """Calculate end-to-end latency"""
        if self.delivered_at:
            return self.delivered_at - self.created_at
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'source': self.source,
            'target': self.target,
            'size': self.payload_size,
            'qos': self.qos.value,
            'state': self.state.value,
            'created_at': round(self.created_at, 3),
            'latency': round(self.latency(), 3) if self.latency() else None,
            'hops': self.hops,
            'retries': self.retries
        }


@dataclass
class ComponentStats:
    """Statistics for a component during simulation"""
    component_id: str
    messages_received: int = 0
    messages_sent: int = 0
    messages_dropped: int = 0
    bytes_processed: int = 0
    total_processing_time: float = 0.0
    queue_max: int = 0
    failures: int = 0
    recovery_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'component_id': self.component_id,
            'messages_received': self.messages_received,
            'messages_sent': self.messages_sent,
            'messages_dropped': self.messages_dropped,
            'bytes_processed': self.bytes_processed,
            'avg_processing_time': (
                round(self.total_processing_time / self.messages_received, 3)
                if self.messages_received > 0 else 0
            ),
            'queue_max': self.queue_max,
            'failures': self.failures
        }


@dataclass
class SimulationMetrics:
    """Aggregated simulation metrics"""
    # Message metrics
    messages_published: int = 0
    messages_delivered: int = 0
    messages_failed: int = 0
    messages_timeout: int = 0
    
    # Latency metrics
    latencies: List[float] = field(default_factory=list)
    
    # Throughput
    bytes_total: int = 0
    
    # Failures
    component_failures: int = 0
    cascade_failures: int = 0
    
    def delivery_rate(self) -> float:
        """Calculate delivery success rate"""
        total = self.messages_published
        if total == 0:
            return 0.0
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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'messages': {
                'published': self.messages_published,
                'delivered': self.messages_delivered,
                'failed': self.messages_failed,
                'timeout': self.messages_timeout,
                'delivery_rate': round(self.delivery_rate(), 4)
            },
            'latency': {
                'avg': round(self.avg_latency(), 3),
                'p99': round(self.p99_latency(), 3),
                'min': round(min(self.latencies), 3) if self.latencies else 0,
                'max': round(max(self.latencies), 3) if self.latencies else 0
            },
            'throughput': {
                'bytes_total': self.bytes_total,
                'messages_per_sec': 0  # Calculated in result
            },
            'failures': {
                'component_failures': self.component_failures,
                'cascade_failures': self.cascade_failures
            }
        }


@dataclass
class EventSimulationResult:
    """Result of event-driven simulation"""
    simulation_id: str
    duration_ms: float
    real_time_ms: float
    speedup: float
    
    metrics: SimulationMetrics
    component_stats: Dict[str, ComponentStats]
    
    # Event log (summary)
    events_processed: int
    events_by_type: Dict[str, int]
    
    # Failure injection results
    failures_injected: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        metrics_dict = self.metrics.to_dict()
        
        # Add throughput rate
        if self.duration_ms > 0:
            metrics_dict['throughput']['messages_per_sec'] = round(
                self.metrics.messages_published / (self.duration_ms / 1000), 2
            )
        
        return {
            'simulation_id': self.simulation_id,
            'timing': {
                'simulated_duration_ms': round(self.duration_ms, 2),
                'real_time_ms': round(self.real_time_ms, 2),
                'speedup': round(self.speedup, 1)
            },
            'metrics': metrics_dict,
            'component_stats': {
                k: v.to_dict() for k, v in self.component_stats.items()
            },
            'events': {
                'total_processed': self.events_processed,
                'by_type': self.events_by_type
            },
            'failures_injected': self.failures_injected
        }


class EventDrivenSimulator:
    """
    Discrete event simulator for pub-sub message flow.
    
    Simulates:
    - Message publishing and delivery
    - Processing delays and queuing
    - Component failures and recovery
    - Load patterns and chaos scenarios
    """
    
    def __init__(self,
                 base_latency_ms: float = 1.0,
                 latency_variance: float = 0.3,
                 queue_capacity: int = 1000,
                 timeout_ms: float = 5000,
                 seed: Optional[int] = None):
        """
        Initialize the event simulator.
        
        Args:
            base_latency_ms: Base processing latency per hop
            latency_variance: Variance in latency (0-1)
            queue_capacity: Maximum queue size per component
            timeout_ms: Message timeout threshold
            seed: Random seed for reproducibility
        """
        self.base_latency_ms = base_latency_ms
        self.latency_variance = latency_variance
        self.queue_capacity = queue_capacity
        self.timeout_ms = timeout_ms
        
        self._rng = random.Random(seed)
        self._simulation_counter = 0
        self.logger = logging.getLogger('EventSimulator')
        
        # Simulation state (reset per simulation)
        self._event_queue: List[SimEvent] = []
        self._current_time: float = 0.0
        self._messages: Dict[str, Message] = {}
        self._component_queues: Dict[str, List[str]] = {}
        self._metrics: SimulationMetrics = SimulationMetrics()
        self._component_stats: Dict[str, ComponentStats] = {}
        self._events_processed: int = 0
        self._events_by_type: Dict[str, int] = defaultdict(int)
        self._failures_injected: List[Dict[str, Any]] = []
        self._graph: Optional[SimulationGraph] = None
    
    def simulate(self,
                 graph: SimulationGraph,
                 duration_ms: float = 10000,
                 message_rate: float = 100,
                 qos: QoSLevel = QoSLevel.AT_LEAST_ONCE,
                 failure_schedule: Optional[List[Dict[str, Any]]] = None) -> EventSimulationResult:
        """
        Run event-driven simulation.
        
        Args:
            graph: SimulationGraph to simulate
            duration_ms: Simulation duration in milliseconds
            message_rate: Messages per second to generate
            qos: Default QoS level for messages
            failure_schedule: Scheduled failures [{time_ms, component, duration_ms}]
            
        Returns:
            EventSimulationResult with metrics
        """
        self._simulation_counter += 1
        sim_id = f"event_{self._simulation_counter:05d}"
        
        self.logger.info(f"[{sim_id}] Starting simulation: {duration_ms}ms, {message_rate} msg/s")
        
        start_real_time = datetime.now()
        
        # Initialize state
        self._reset_state(graph)
        
        # Schedule message generation
        self._schedule_messages(duration_ms, message_rate, qos)
        
        # Schedule failures if provided
        if failure_schedule:
            self._schedule_failures(failure_schedule)
        
        # Schedule simulation end
        heapq.heappush(self._event_queue, SimEvent(
            time=duration_ms,
            event_type=EventType.SIMULATION_END,
            data={}
        ))
        
        # Run simulation loop
        while self._event_queue:
            event = heapq.heappop(self._event_queue)
            
            if event.event_type == EventType.SIMULATION_END:
                break
            
            self._current_time = event.time
            self._process_event(event)
        
        end_real_time = datetime.now()
        real_time_ms = (end_real_time - start_real_time).total_seconds() * 1000
        speedup = duration_ms / real_time_ms if real_time_ms > 0 else 0
        
        self.logger.info(f"[{sim_id}] Complete: {self._events_processed} events, "
                        f"{real_time_ms:.0f}ms real time ({speedup:.0f}x speedup)")
        
        return EventSimulationResult(
            simulation_id=sim_id,
            duration_ms=duration_ms,
            real_time_ms=real_time_ms,
            speedup=speedup,
            metrics=self._metrics,
            component_stats=dict(self._component_stats),
            events_processed=self._events_processed,
            events_by_type=dict(self._events_by_type),
            failures_injected=self._failures_injected
        )
    
    def simulate_with_load_test(self,
                                 graph: SimulationGraph,
                                 duration_ms: float = 30000,
                                 initial_rate: float = 10,
                                 peak_rate: float = 500,
                                 ramp_time_ms: float = 10000) -> EventSimulationResult:
        """
        Simulate with ramping load pattern.
        
        Args:
            graph: SimulationGraph to simulate
            duration_ms: Total duration
            initial_rate: Starting message rate
            peak_rate: Peak message rate
            ramp_time_ms: Time to ramp up to peak
            
        Returns:
            EventSimulationResult with metrics
        """
        self._simulation_counter += 1
        sim_id = f"load_{self._simulation_counter:05d}"
        
        self.logger.info(f"[{sim_id}] Load test: {initial_rate} -> {peak_rate} msg/s")
        
        start_real_time = datetime.now()
        
        # Initialize state
        self._reset_state(graph)
        
        # Generate messages with ramping rate
        time = 0.0
        msg_id = 0
        
        while time < duration_ms:
            # Calculate current rate
            if time < ramp_time_ms:
                rate = initial_rate + (peak_rate - initial_rate) * (time / ramp_time_ms)
            elif time < duration_ms - ramp_time_ms:
                rate = peak_rate
            else:
                remaining = duration_ms - time
                rate = initial_rate + (peak_rate - initial_rate) * (remaining / ramp_time_ms)
            
            # Schedule next message
            interval = 1000 / rate if rate > 0 else 1000
            interval *= (0.5 + self._rng.random())  # Add variance
            
            # Create message
            apps = graph.get_by_type(ComponentType.APPLICATION)
            if len(apps) >= 2:
                src, dst = self._rng.sample(apps, 2)
                msg_id += 1
                self._schedule_single_message(
                    f"msg_{msg_id:06d}", src, dst, time, QoSLevel.AT_LEAST_ONCE
                )
            
            time += interval
        
        # Schedule end
        heapq.heappush(self._event_queue, SimEvent(
            time=duration_ms,
            event_type=EventType.SIMULATION_END
        ))
        
        # Run simulation
        while self._event_queue:
            event = heapq.heappop(self._event_queue)
            if event.event_type == EventType.SIMULATION_END:
                break
            self._current_time = event.time
            self._process_event(event)
        
        end_real_time = datetime.now()
        real_time_ms = (end_real_time - start_real_time).total_seconds() * 1000
        speedup = duration_ms / real_time_ms if real_time_ms > 0 else 0
        
        return EventSimulationResult(
            simulation_id=sim_id,
            duration_ms=duration_ms,
            real_time_ms=real_time_ms,
            speedup=speedup,
            metrics=self._metrics,
            component_stats=dict(self._component_stats),
            events_processed=self._events_processed,
            events_by_type=dict(self._events_by_type),
            failures_injected=self._failures_injected
        )
    
    def simulate_chaos(self,
                       graph: SimulationGraph,
                       duration_ms: float = 30000,
                       message_rate: float = 100,
                       failure_probability: float = 0.01,
                       recovery_probability: float = 0.1,
                       check_interval_ms: float = 1000) -> EventSimulationResult:
        """
        Simulate with random failures (chaos engineering).
        
        Args:
            graph: SimulationGraph to simulate
            duration_ms: Total duration
            message_rate: Message rate
            failure_probability: Probability of failure per check
            recovery_probability: Probability of recovery per check
            check_interval_ms: Interval between failure checks
            
        Returns:
            EventSimulationResult with metrics
        """
        self._simulation_counter += 1
        sim_id = f"chaos_{self._simulation_counter:05d}"
        
        self.logger.info(f"[{sim_id}] Chaos simulation: fail={failure_probability}, "
                        f"recover={recovery_probability}")
        
        # Build failure schedule with random failures
        failure_schedule = []
        failed_components: Set[str] = set()
        
        time = check_interval_ms
        while time < duration_ms:
            components = list(graph.components.keys())
            
            for comp in components:
                if comp in failed_components:
                    # Check for recovery
                    if self._rng.random() < recovery_probability:
                        failure_schedule.append({
                            'time_ms': time,
                            'component': comp,
                            'action': 'recover'
                        })
                        failed_components.discard(comp)
                else:
                    # Check for failure
                    if self._rng.random() < failure_probability:
                        failure_schedule.append({
                            'time_ms': time,
                            'component': comp,
                            'duration_ms': 0  # Permanent until recovery
                        })
                        failed_components.add(comp)
            
            time += check_interval_ms
        
        return self.simulate(
            graph,
            duration_ms=duration_ms,
            message_rate=message_rate,
            failure_schedule=failure_schedule
        )
    
    def _reset_state(self, graph: SimulationGraph):
        """Reset simulation state"""
        self._graph = graph.copy()
        self._event_queue = []
        self._current_time = 0.0
        self._messages = {}
        self._component_queues = {c: [] for c in graph.components}
        self._metrics = SimulationMetrics()
        self._component_stats = {
            c: ComponentStats(component_id=c) for c in graph.components
        }
        self._events_processed = 0
        self._events_by_type = defaultdict(int)
        self._failures_injected = []
    
    def _schedule_messages(self, duration_ms: float, rate: float, qos: QoSLevel):
        """Schedule message generation events"""
        if rate <= 0:
            return
        
        apps = self._graph.get_by_type(ComponentType.APPLICATION)
        if len(apps) < 2:
            self.logger.warning("Not enough applications to simulate messages")
            return
        
        interval = 1000 / rate
        time = 0.0
        msg_id = 0
        
        while time < duration_ms:
            msg_id += 1
            src, dst = self._rng.sample(apps, 2)
            self._schedule_single_message(f"msg_{msg_id:06d}", src, dst, time, qos)
            
            # Add variance to interval
            time += interval * (0.5 + self._rng.random())
    
    def _schedule_single_message(self, msg_id: str, src: str, dst: str,
                                   time: float, qos: QoSLevel):
        """Schedule a single message publication"""
        message = Message(
            id=msg_id,
            source=src,
            target=dst,
            payload_size=self._rng.randint(100, 10000),
            qos=qos,
            created_at=time,
            deadline=time + self.timeout_ms
        )
        self._messages[msg_id] = message
        
        heapq.heappush(self._event_queue, SimEvent(
            time=time,
            event_type=EventType.MESSAGE_PUBLISH,
            data={'message_id': msg_id}
        ))
    
    def _schedule_failures(self, schedule: List[Dict[str, Any]]):
        """Schedule failure events"""
        for failure in schedule:
            heapq.heappush(self._event_queue, SimEvent(
                time=failure['time_ms'],
                event_type=EventType.COMPONENT_FAIL,
                data={
                    'component': failure['component'],
                    'duration_ms': failure.get('duration_ms', 0)
                }
            ))
            
            # Schedule recovery if duration specified
            if failure.get('duration_ms', 0) > 0:
                heapq.heappush(self._event_queue, SimEvent(
                    time=failure['time_ms'] + failure['duration_ms'],
                    event_type=EventType.COMPONENT_RECOVER,
                    data={'component': failure['component']}
                ))
    
    def _process_event(self, event: SimEvent):
        """Process a single simulation event"""
        self._events_processed += 1
        self._events_by_type[event.event_type.value] += 1
        
        handlers = {
            EventType.MESSAGE_PUBLISH: self._handle_publish,
            EventType.MESSAGE_ARRIVE: self._handle_arrive,
            EventType.MESSAGE_PROCESS: self._handle_process,
            EventType.MESSAGE_DELIVER: self._handle_deliver,
            EventType.MESSAGE_TIMEOUT: self._handle_timeout,
            EventType.COMPONENT_FAIL: self._handle_fail,
            EventType.COMPONENT_RECOVER: self._handle_recover,
        }
        
        handler = handlers.get(event.event_type)
        if handler:
            handler(event)
    
    def _handle_publish(self, event: SimEvent):
        """Handle message publication"""
        msg_id = event.data['message_id']
        message = self._messages.get(msg_id)
        if not message:
            return
        
        self._metrics.messages_published += 1
        self._metrics.bytes_total += message.payload_size
        
        # Check if source is active
        src_comp = self._graph.components.get(message.source)
        if not src_comp or not src_comp.is_active:
            message.state = MessageState.FAILED
            message.failed_at = self._current_time
            self._metrics.messages_failed += 1
            return
        
        # Update stats
        stats = self._component_stats.get(message.source)
        if stats:
            stats.messages_sent += 1
        
        # Find path to target (via brokers)
        next_hop = self._find_next_hop(message.source, message.target)
        
        if next_hop:
            message.state = MessageState.IN_TRANSIT
            message.hops.append(message.source)
            
            # Schedule arrival at next hop
            latency = self._calculate_latency()
            heapq.heappush(self._event_queue, SimEvent(
                time=self._current_time + latency,
                event_type=EventType.MESSAGE_ARRIVE,
                data={'message_id': msg_id, 'location': next_hop}
            ))
            
            # Schedule timeout
            heapq.heappush(self._event_queue, SimEvent(
                time=message.deadline,
                event_type=EventType.MESSAGE_TIMEOUT,
                data={'message_id': msg_id}
            ))
        else:
            # No path to target
            message.state = MessageState.FAILED
            message.failed_at = self._current_time
            self._metrics.messages_failed += 1
    
    def _handle_arrive(self, event: SimEvent):
        """Handle message arrival at a component"""
        msg_id = event.data['message_id']
        location = event.data['location']
        
        message = self._messages.get(msg_id)
        if not message or message.state not in [MessageState.IN_TRANSIT, MessageState.PROCESSING]:
            return
        
        # Check if component is active
        comp = self._graph.components.get(location)
        if not comp or not comp.is_active:
            message.state = MessageState.FAILED
            message.failed_at = self._current_time
            self._metrics.messages_failed += 1
            return
        
        message.current_location = location
        message.hops.append(location)
        
        # Update stats
        stats = self._component_stats.get(location)
        if stats:
            stats.messages_received += 1
        
        # Add to queue
        queue = self._component_queues.get(location, [])
        if len(queue) >= self.queue_capacity:
            # Queue full - drop message
            message.state = MessageState.FAILED
            message.failed_at = self._current_time
            self._metrics.messages_failed += 1
            if stats:
                stats.messages_dropped += 1
            return
        
        queue.append(msg_id)
        if stats:
            stats.queue_max = max(stats.queue_max, len(queue))
        
        # Schedule processing
        process_time = self._calculate_latency() * (1 + 0.1 * len(queue))
        heapq.heappush(self._event_queue, SimEvent(
            time=self._current_time + process_time,
            event_type=EventType.MESSAGE_PROCESS,
            data={'message_id': msg_id, 'location': location}
        ))
    
    def _handle_process(self, event: SimEvent):
        """Handle message processing completion"""
        msg_id = event.data['message_id']
        location = event.data['location']
        
        message = self._messages.get(msg_id)
        if not message or message.state == MessageState.TIMEOUT:
            return
        
        # Remove from queue
        queue = self._component_queues.get(location, [])
        if msg_id in queue:
            queue.remove(msg_id)
        
        # Update stats
        stats = self._component_stats.get(location)
        if stats:
            stats.bytes_processed += message.payload_size
            stats.total_processing_time += self._calculate_latency()
        
        # Check if this is the destination
        if location == message.target:
            heapq.heappush(self._event_queue, SimEvent(
                time=self._current_time,
                event_type=EventType.MESSAGE_DELIVER,
                data={'message_id': msg_id}
            ))
        else:
            # Forward to next hop
            next_hop = self._find_next_hop(location, message.target)
            
            if next_hop:
                latency = self._calculate_latency()
                heapq.heappush(self._event_queue, SimEvent(
                    time=self._current_time + latency,
                    event_type=EventType.MESSAGE_ARRIVE,
                    data={'message_id': msg_id, 'location': next_hop}
                ))
            else:
                message.state = MessageState.FAILED
                message.failed_at = self._current_time
                self._metrics.messages_failed += 1
    
    def _handle_deliver(self, event: SimEvent):
        """Handle successful message delivery"""
        msg_id = event.data['message_id']
        message = self._messages.get(msg_id)
        
        if not message or message.state in [MessageState.DELIVERED, MessageState.TIMEOUT]:
            return
        
        message.state = MessageState.DELIVERED
        message.delivered_at = self._current_time
        
        latency = message.latency()
        if latency:
            self._metrics.latencies.append(latency)
        
        self._metrics.messages_delivered += 1
    
    def _handle_timeout(self, event: SimEvent):
        """Handle message timeout"""
        msg_id = event.data['message_id']
        message = self._messages.get(msg_id)
        
        if not message:
            return
        
        if message.state not in [MessageState.DELIVERED, MessageState.FAILED]:
            message.state = MessageState.TIMEOUT
            message.failed_at = self._current_time
            self._metrics.messages_timeout += 1
    
    def _handle_fail(self, event: SimEvent):
        """Handle component failure"""
        comp_id = event.data['component']
        
        comp = self._graph.components.get(comp_id)
        if comp:
            comp.is_active = False
            
            # Deactivate dependencies
            for dep in self._graph.get_outgoing(comp_id):
                dep.is_active = False
            for dep in self._graph.get_incoming(comp_id):
                dep.is_active = False
        
        self._metrics.component_failures += 1
        
        stats = self._component_stats.get(comp_id)
        if stats:
            stats.failures += 1
        
        self._failures_injected.append({
            'time': self._current_time,
            'component': comp_id,
            'action': 'fail'
        })
        
        self.logger.debug(f"Component failed: {comp_id} at t={self._current_time:.0f}ms")
    
    def _handle_recover(self, event: SimEvent):
        """Handle component recovery"""
        comp_id = event.data['component']
        
        comp = self._graph.components.get(comp_id)
        if comp:
            comp.is_active = True
            
            # Reactivate dependencies
            for dep in self._graph.get_outgoing(comp_id):
                target = self._graph.components.get(dep.target)
                if target and target.is_active:
                    dep.is_active = True
            for dep in self._graph.get_incoming(comp_id):
                source = self._graph.components.get(dep.source)
                if source and source.is_active:
                    dep.is_active = True
        
        self._failures_injected.append({
            'time': self._current_time,
            'component': comp_id,
            'action': 'recover'
        })
        
        self.logger.debug(f"Component recovered: {comp_id} at t={self._current_time:.0f}ms")
    
    def _find_next_hop(self, current: str, target: str) -> Optional[str]:
        """Find the next hop toward target"""
        # Direct connection?
        for dep in self._graph.get_outgoing(current):
            if dep.is_active and dep.target == target:
                target_comp = self._graph.components.get(target)
                if target_comp and target_comp.is_active:
                    return target
        
        # Via broker?
        for dep in self._graph.get_outgoing(current):
            if not dep.is_active:
                continue
            next_comp = self._graph.components.get(dep.target)
            if not next_comp or not next_comp.is_active:
                continue
            
            # If next is a broker, check if it can reach target
            if next_comp.type == ComponentType.BROKER:
                for broker_dep in self._graph.get_outgoing(dep.target):
                    if broker_dep.is_active and broker_dep.target == target:
                        return dep.target
        
        # BFS for any path
        visited = {current}
        queue = [(current, None)]
        
        while queue:
            node, first_hop = queue.pop(0)
            
            for dep in self._graph.get_outgoing(node):
                if not dep.is_active:
                    continue
                next_node = dep.target
                if next_node in visited:
                    continue
                
                next_comp = self._graph.components.get(next_node)
                if not next_comp or not next_comp.is_active:
                    continue
                
                next_first_hop = first_hop if first_hop else next_node
                
                if next_node == target:
                    return next_first_hop
                
                visited.add(next_node)
                queue.append((next_node, next_first_hop))
        
        return None
    
    def _calculate_latency(self) -> float:
        """Calculate random latency with variance"""
        variance = self.base_latency_ms * self.latency_variance
        return max(0.1, self.base_latency_ms + self._rng.uniform(-variance, variance))