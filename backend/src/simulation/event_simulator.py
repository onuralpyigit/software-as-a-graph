"""
Event-Driven Simulator

Simulates pub-sub message flow using discrete event simulation.
Tracks runtime metrics: throughput, latency, message drops, queue depths.

Works directly on RAW structural relationships (PUBLISHES_TO, SUBSCRIBES_TO,
ROUTES, RUNS_ON) without DEPENDS_ON derivation.

Message Flow:
    Publisher -[PUBLISHES_TO]-> Topic -[ROUTES]-> Broker -> Subscribers

Metrics:
    - Throughput: Messages published/delivered per second
    - Latency: End-to-end message delivery time
    - Drop Rate: Percentage of messages not delivered
    - Queue Depth: Messages waiting at each component
"""

from __future__ import annotations
import heapq
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Any, Optional
from enum import Enum
from collections import defaultdict

from .graph import SimulationGraph
from .models import ComponentState, EventType, TopicInfo, Event, Message, EventScenario, RuntimeMetrics, EventResult


class EventSimulator:
    """
    Discrete event simulator for pub-sub message flow.
    
    Simulates message propagation:
        Publisher -> Topic -> Broker(s) -> Subscribers
    
    Tracks runtime metrics including throughput, latency, and drops.
    
    Example:
        >>> graph = SimulationGraph(uri="bolt://localhost:7687")
        >>> sim = EventSimulator(graph)
        >>> result = sim.simulate(EventScenario(source_app="App1", num_messages=100))
        >>> print(f"Delivery rate: {result.metrics.delivery_rate}%")
    """
    
    def __init__(self, graph: SimulationGraph):
        """
        Initialize the event simulator.
        
        Args:
            graph: SimulationGraph instance with raw pub-sub relationships
        """
        self.graph = graph
        self.logger = logging.getLogger(__name__)
        
        # Event queue (min-heap by time)
        self._event_queue: List[Event] = []
        
        # Active messages
        self._messages: Dict[str, Message] = {}
        
        # Metrics
        self._metrics = RuntimeMetrics()
        
        # Random generator
        self._rng = random.Random()
        
        # Current simulation time
        self._sim_time = 0.0
        
        # Tracking
        self._affected_topics: Set[str] = set()
        self._reached_subscribers: Set[str] = set()
        self._brokers_used: Set[str] = set()
        self._drop_reasons: Dict[str, int] = defaultdict(int)
    
    def simulate(self, scenario: EventScenario) -> EventResult:
        """
        Run an event simulation.
        
        Args:
            scenario: Configuration for the simulation
            
        Returns:
            EventResult with metrics and analysis
        """
        start_time = time.time()
        
        # Reset state
        self._reset(scenario.seed)
        self.graph.reset()
        
        # Validate source
        if scenario.source_app not in self.graph.components:
            return self._empty_result(scenario, f"Source app '{scenario.source_app}' not found")
        
        # Get topics the source publishes to
        publishes_to, _ = self.graph.get_app_topics(scenario.source_app)
        if not publishes_to:
            return self._empty_result(scenario, f"Source app has no publications")
        
        self.logger.info(f"Starting event simulation: {scenario.source_app} -> {publishes_to}")
        
        # Schedule initial message publications
        for i in range(scenario.num_messages):
            msg_time = i * scenario.message_interval
            if msg_time > scenario.duration:
                break
            
            # Round-robin through topics
            topic_id = publishes_to[i % len(publishes_to)]
            topic_info = self.graph.topics.get(topic_id, TopicInfo(id=topic_id, name=topic_id))
            
            msg_id = f"msg_{i}"
            self._messages[msg_id] = Message(
                id=msg_id,
                source_app=scenario.source_app,
                topic_id=topic_id,
                size=scenario.message_size,
                priority=topic_info.priority_value,
                requires_ack=topic_info.requires_ack,
                created_at=msg_time,
            )
            
            # Schedule publish event
            self._schedule(Event(
                time=msg_time,
                event_type=EventType.PUBLISH,
                source=scenario.source_app,
                target=topic_id,
                message_id=msg_id,
            ))
        
        # Run simulation
        self._run_simulation(scenario)
        
        # Calculate duration
        wall_time = time.time() - start_time
        self._metrics.simulation_duration = self._sim_time
        
        # Build result
        return self._build_result(scenario, wall_time)
    
    def simulate_all_publishers(
        self,
        scenario_template: EventScenario
    ) -> Dict[str, EventResult]:
        """
        Run simulation for all publisher applications.
        
        Args:
            scenario_template: Base scenario configuration
            
        Returns:
            Dict mapping app_id to EventResult
        """
        results = {}
        
        # Find all publishers
        publishers = set()
        for topic_id in self.graph.topics:
            publishers.update(self.graph.get_publishers(topic_id))
        
        for app_id in publishers:
            scenario = EventScenario(
                source_app=app_id,
                description=scenario_template.description or f"Event sim: {app_id}",
                num_messages=scenario_template.num_messages,
                message_interval=scenario_template.message_interval,
                duration=scenario_template.duration,
                seed=scenario_template.seed,
                drop_probability=scenario_template.drop_probability,
            )
            
            results[app_id] = self.simulate(scenario)
        
        return results
    
    def _reset(self, seed: Optional[int] = None) -> None:
        """Reset simulator state."""
        self._event_queue = []
        self._messages = {}
        self._metrics = RuntimeMetrics()
        self._sim_time = 0.0
        self._affected_topics = set()
        self._reached_subscribers = set()
        self._brokers_used = set()
        self._drop_reasons = defaultdict(int)
        
        if seed is not None:
            self._rng.seed(seed)
    
    def _schedule(self, event: Event) -> None:
        """Schedule an event."""
        heapq.heappush(self._event_queue, event)
    
    def _run_simulation(self, scenario: EventScenario) -> None:
        """Process events until queue is empty or timeout."""
        while self._event_queue:
            event = heapq.heappop(self._event_queue)
            
            if event.time > scenario.duration:
                break
            
            self._sim_time = event.time
            self._process_event(event, scenario)
    
    def _process_event(self, event: Event, scenario: EventScenario) -> None:
        """Process a single event."""
        msg = self._messages.get(event.message_id)
        if msg is None or msg.dropped:
            return
        
        if event.event_type == EventType.PUBLISH:
            self._handle_publish(event, msg, scenario)
        elif event.event_type == EventType.ROUTE:
            self._handle_route(event, msg, scenario)
        elif event.event_type == EventType.DELIVER:
            self._handle_deliver(event, msg, scenario)
        elif event.event_type == EventType.DROP:
            self._handle_drop(event, msg, scenario)
    
    def _handle_publish(self, event: Event, msg: Message, scenario: EventScenario) -> None:
        """Handle message publication."""
        self._metrics.messages_published += 1
        self._affected_topics.add(msg.topic_id)
        
        # Update component metrics
        src_comp = self.graph.components.get(event.source)
        if src_comp:
            src_comp.messages_sent += 1
        
        # Check for random drop
        if self._rng.random() < scenario.drop_probability:
            self._drop_message(msg, "random_drop")
            return
        
        # Get routing brokers for the topic
        brokers = self.graph.get_routing_brokers(msg.topic_id)
        
        if not brokers:
            # No brokers - try direct delivery to subscribers
            subscribers = self.graph.get_subscribers(msg.topic_id)
            if not subscribers:
                self._drop_message(msg, "no_subscribers")
                return
            
            # Schedule direct delivery
            for sub in subscribers:
                delivery_time = self._sim_time + scenario.publish_latency + scenario.network_latency
                self._schedule(Event(
                    time=delivery_time,
                    event_type=EventType.DELIVER,
                    source=event.source,
                    target=sub,
                    message_id=msg.id,
                ))
        else:
            # Route through brokers
            for broker in brokers:
                self._brokers_used.add(broker)
                route_time = self._sim_time + scenario.publish_latency + scenario.network_latency
                self._schedule(Event(
                    time=route_time,
                    event_type=EventType.ROUTE,
                    source=msg.topic_id,
                    target=broker,
                    message_id=msg.id,
                ))
        
        msg.hops += 1
    
    def _handle_route(self, event: Event, msg: Message, scenario: EventScenario) -> None:
        """Handle message routing through broker."""
        broker_id = event.target
        
        # Check if broker is active
        if not self.graph.is_active(broker_id):
            self._drop_message(msg, "broker_failed")
            return
        
        # Update broker metrics
        broker_comp = self.graph.components.get(broker_id)
        if broker_comp:
            broker_comp.messages_routed += 1
        
        # Check for broker failure
        if self._rng.random() < scenario.broker_failure_prob:
            self.graph.fail_component(broker_id)
            self._drop_message(msg, "broker_failure_during_route")
            return
        
        # Get subscribers for the topic
        subscribers = self.graph.get_subscribers(msg.topic_id)
        
        if not subscribers:
            self._drop_message(msg, "no_active_subscribers")
            return
        
        # Schedule delivery to each subscriber
        for sub in subscribers:
            if sub in msg.delivered_to:
                continue
            
            delivery_time = self._sim_time + scenario.broker_latency + scenario.network_latency
            self._schedule(Event(
                time=delivery_time,
                event_type=EventType.DELIVER,
                source=broker_id,
                target=sub,
                message_id=msg.id,
            ))
        
        msg.hops += 1
    
    def _handle_deliver(self, event: Event, msg: Message, scenario: EventScenario) -> None:
        """Handle message delivery to subscriber."""
        subscriber_id = event.target
        
        # Check if already delivered to this subscriber
        if subscriber_id in msg.delivered_to:
            return
        
        # Check if subscriber is active
        if not self.graph.is_active(subscriber_id):
            self._drop_message(msg, "subscriber_failed")
            return
        
        # Check for timeout
        elapsed = self._sim_time - msg.created_at
        if elapsed > scenario.delivery_timeout:
            self._drop_message(msg, "delivery_timeout")
            return
        
        # Successful delivery
        msg.delivered_to.add(subscriber_id)
        msg.delivered_at = self._sim_time + scenario.subscribe_latency
        
        self._metrics.messages_delivered += 1
        self._reached_subscribers.add(subscriber_id)
        
        # Record latency
        latency = msg.delivered_at - msg.created_at
        self._metrics.total_latency += latency
        self._metrics.latencies.append(latency)
        self._metrics.min_latency = min(self._metrics.min_latency, latency)
        self._metrics.max_latency = max(self._metrics.max_latency, latency)
        
        # Update subscriber metrics
        sub_comp = self.graph.components.get(subscriber_id)
        if sub_comp:
            sub_comp.messages_received += 1
            sub_comp.total_latency += latency
        
        msg.hops += 1
    
    def _handle_drop(self, event: Event, msg: Message, scenario: EventScenario) -> None:
        """Handle message drop event."""
        reason = event.data.get("reason", "unknown")
        self._drop_message(msg, reason)
    
    def _drop_message(self, msg: Message, reason: str) -> None:
        """Drop a message and record reason."""
        if msg.dropped:
            return
        
        msg.dropped = True
        msg.drop_reason = reason
        self._metrics.messages_dropped += 1
        self._drop_reasons[reason] += 1
    
    def _empty_result(self, scenario: EventScenario, reason: str) -> EventResult:
        """Create an empty result for failed simulations."""
        return EventResult(
            source_app=scenario.source_app,
            scenario=reason,
            duration=0.0,
            metrics=RuntimeMetrics(),
            drop_reasons={"simulation_failed": 1},
        )
    
    def _build_result(self, scenario: EventScenario, wall_time: float) -> EventResult:
        """Build the final simulation result."""
        # Calculate component impacts (based on message handling)
        component_impacts = {}
        total_messages = self._metrics.messages_published
        
        if total_messages > 0:
            for comp_id, comp in self.graph.components.items():
                handled = comp.messages_sent + comp.messages_routed + comp.messages_received
                if handled > 0:
                    component_impacts[comp_id] = handled / total_messages
        
        return EventResult(
            source_app=scenario.source_app,
            scenario=scenario.description or f"Event simulation: {scenario.source_app}",
            duration=wall_time,
            metrics=self._metrics,
            affected_topics=list(self._affected_topics),
            reached_subscribers=list(self._reached_subscribers),
            brokers_used=list(self._brokers_used),
            component_impacts=component_impacts,
            failed_components=[
                c.id for c in self.graph.components.values()
                if c.state == ComponentState.FAILED
            ],
            drop_reasons=dict(self._drop_reasons),
            component_names={c.id: c.properties.get("name", c.id) for c in self.graph.components.values()},
        )