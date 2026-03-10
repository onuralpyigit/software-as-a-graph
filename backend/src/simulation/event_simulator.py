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

Poisson Failure Injection (EventScenario.failure_rate > 0):
    Components fail according to a homogeneous Poisson process with rate λ
    (failures per sim-second).  Inter-arrival times are Exp(λ); a uniformly
    random eligible component is chosen at each failure event.  Optional
    recovery: if mean_recovery_time > 0, a RECOVER_COMPONENT event is
    scheduled Exp(1/mean_recovery_time) sim-seconds after each failure.

    This is distinct from the per-event Bernoulli broker_failure_prob which
    fires independently on every routing event.  Use Poisson injection for
    time-correlated, memoryless failure modelling (M/M/1-style) and
    Bernoulli injection for per-message probabilistic drops.

Poisson Arrivals (EventScenario.poisson_arrivals = True):
    Replaces the fixed-interval message schedule with Poisson arrivals:
    inter-arrival times drawn from Exp(1 / message_interval).  Converts
    the simulator from a D/G/1 to an M/G/1 queue model.
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
from .models import (
    ComponentState,
    EventType,
    TopicInfo,
    Event,
    Message,
    EventScenario,
    RuntimeMetrics,
    EventResult,
)


class EventSimulator:
    """
    Discrete event simulator for pub-sub message flow.

    Simulates message propagation:
        Publisher -> Topic -> Broker(s) -> Subscribers

    Tracks runtime metrics including throughput, latency, and drops.

    Supports optional Poisson failure injection (EventScenario.failure_rate > 0)
    and Poisson message arrivals (EventScenario.poisson_arrivals = True).

    Example (standard):
        >>> graph = SimulationGraph(uri="bolt://localhost:7687")
        >>> sim = EventSimulator(graph)
        >>> result = sim.simulate(EventScenario(source_app="App1", num_messages=100))
        >>> print(f"Delivery rate: {result.metrics.delivery_rate}%")

    Example (Poisson failures, λ = 2 failures/sec, MTTF = 0.5 s):
        >>> scenario = EventScenario(
        ...     source_app="App1",
        ...     num_messages=500,
        ...     duration=10.0,
        ...     failure_rate=2.0,
        ...     mean_recovery_time=1.0,   # optional: components recover after ~1 s
        ...     failure_targets=["B0", "B1", "B2"],  # restrict to brokers
        ... )
        >>> result = sim.simulate(scenario)
        >>> print(result.poisson_failure_log)   # ordered timeline of failures/recoveries

    Example (Poisson arrivals, M/G/1 queue model):
        >>> scenario = EventScenario(
        ...     source_app="App1",
        ...     num_messages=200,
        ...     message_interval=0.02,   # mean inter-arrival = 20 ms
        ...     poisson_arrivals=True,
        ... )
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

        # [POISSON] Ordered log of Poisson failure/recovery events fired during
        # this simulation run.  Each entry: {"time", "component_id", "event"}.
        self._poisson_failure_log: List[Dict[str, Any]] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

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

        # ── Schedule message publications ──────────────────────────────────
        if scenario.poisson_arrivals:
            # [POISSON] Poisson arrival process: inter-arrival ~ Exp(1/message_interval)
            self._schedule_poisson_arrivals(scenario, publishes_to)
        else:
            # Standard: fixed deterministic inter-arrival
            self._schedule_deterministic_arrivals(scenario, publishes_to)

        # [POISSON] Schedule Poisson component-failure events
        if scenario.failure_rate > 0.0:
            self._schedule_poisson_failures(scenario)

        # Run simulation
        self._run_simulation(scenario)

        # Calculate duration
        wall_time = time.time() - start_time
        self._metrics.simulation_duration = self._sim_time

        # Build result
        return self._build_result(scenario, wall_time)

    def simulate_all_publishers(
        self,
        scenario_template: EventScenario,
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
        publishers: Set[str] = set()
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
                # [POISSON] propagate Poisson settings to per-publisher runs
                failure_rate=scenario_template.failure_rate,
                failure_targets=scenario_template.failure_targets,
                mean_recovery_time=scenario_template.mean_recovery_time,
                poisson_arrivals=scenario_template.poisson_arrivals,
            )
            results[app_id] = self.simulate(scenario)

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: scheduling helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _schedule_deterministic_arrivals(
        self,
        scenario: EventScenario,
        publishes_to: List[str],
    ) -> None:
        """Schedule messages at a fixed inter-arrival interval (original behaviour)."""
        for i in range(scenario.num_messages):
            msg_time = i * scenario.message_interval
            if msg_time > scenario.duration:
                break
            self._create_and_schedule_message(i, msg_time, scenario, publishes_to)

    def _schedule_poisson_arrivals(
        self,
        scenario: EventScenario,
        publishes_to: List[str],
    ) -> None:
        """
        [POISSON] Schedule messages with Poisson inter-arrivals.

        Inter-arrival times are drawn from Exp(1 / message_interval), making
        the mean inter-arrival equal to message_interval (same as the
        deterministic case) but with exponentially distributed spacing,
        converting the simulator to an M/G/1 queue model.
        """
        rate = 1.0 / max(scenario.message_interval, 1e-9)
        t = self._rng.expovariate(rate)
        for i in range(scenario.num_messages):
            if t > scenario.duration:
                break
            self._create_and_schedule_message(i, t, scenario, publishes_to)
            t += self._rng.expovariate(rate)

    def _create_and_schedule_message(
        self,
        idx: int,
        msg_time: float,
        scenario: EventScenario,
        publishes_to: List[str],
    ) -> None:
        """Create a Message object and enqueue its initial PUBLISH event."""
        # Round-robin through topics
        topic_id = publishes_to[idx % len(publishes_to)]
        topic_info = self.graph.topics.get(topic_id, TopicInfo(id=topic_id, name=topic_id))

        msg_id = f"msg_{idx}"
        self._messages[msg_id] = Message(
            id=msg_id,
            source_app=scenario.source_app,
            topic_id=topic_id,
            size=scenario.message_size,
            priority=topic_info.priority_value,
            requires_ack=topic_info.requires_ack,
            created_at=msg_time,
        )

        self._schedule(Event(
            time=msg_time,
            event_type=EventType.PUBLISH,
            source=scenario.source_app,
            target=topic_id,
            message_id=msg_id,
        ))

    # ─────────────────────────────────────────────────────────────────────────
    # [POISSON] Failure injection scheduling
    # ─────────────────────────────────────────────────────────────────────────

    def _schedule_poisson_failures(self, scenario: EventScenario) -> None:
        """
        Pre-schedule all Poisson component-failure events for this run.

        Uses the exponential inter-arrival property: given λ failures/sec,
        successive failure times form a Poisson process with inter-arrival
        times drawn from Exp(λ).  A uniformly random eligible target is
        chosen independently at each failure time.

        If mean_recovery_time > 0, a RECOVER_COMPONENT event is also
        enqueued, with delay drawn from Exp(1 / mean_recovery_time).

        This approach pre-generates the entire failure timeline before the
        main event loop starts, which is cleaner than re-scheduling inside
        the loop and avoids interleaving concerns.
        """
        # Build the pool of eligible failure targets
        targets: List[str] = self._build_failure_targets(scenario)
        if not targets:
            self.logger.warning(
                "Poisson failure injection requested but no eligible targets found; "
                "skipping failure scheduling."
            )
            return

        lambda_ = scenario.failure_rate  # failures per sim-second
        t = self._rng.expovariate(lambda_)

        failure_seq = 0
        while t < scenario.duration:
            target_id = self._rng.choice(targets)

            fail_event_id = f"__poisson_fail_{failure_seq}"
            self._schedule(Event(
                time=t,
                event_type=EventType.FAIL_COMPONENT,
                source="poisson_injector",
                target=target_id,
                message_id=fail_event_id,
                data={"sequence": failure_seq},
            ))

            # Optional recovery
            if scenario.mean_recovery_time > 0.0:
                recovery_rate = 1.0 / scenario.mean_recovery_time
                recovery_delay = self._rng.expovariate(recovery_rate)
                recovery_time = t + recovery_delay
                if recovery_time < scenario.duration:
                    self._schedule(Event(
                        time=recovery_time,
                        event_type=EventType.RECOVER_COMPONENT,
                        source="poisson_injector",
                        target=target_id,
                        message_id=f"__poisson_recover_{failure_seq}",
                        data={"sequence": failure_seq},
                    ))

            failure_seq += 1
            t += self._rng.expovariate(lambda_)

        self.logger.info(
            f"Scheduled {failure_seq} Poisson failure events "
            f"(λ={lambda_:.3f}/s, targets={len(targets)}, "
            f"recovery={'enabled' if scenario.mean_recovery_time > 0 else 'disabled'})"
        )

    def _build_failure_targets(self, scenario: EventScenario) -> List[str]:
        """
        Return the list of component IDs eligible for Poisson failure injection.

        If scenario.failure_targets is explicitly set, validate and use it;
        otherwise default to all components present in the graph.
        """
        all_ids = list(self.graph.components.keys())

        if scenario.failure_targets is not None:
            # Filter to only those that actually exist in this graph
            valid = [cid for cid in scenario.failure_targets if cid in self.graph.components]
            if not valid:
                self.logger.warning(
                    "None of the specified failure_targets exist in the graph; "
                    "falling back to all components."
                )
                return all_ids
            return valid

        return all_ids

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: simulation loop
    # ─────────────────────────────────────────────────────────────────────────

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
        # [POISSON] clear failure log
        self._poisson_failure_log = []

        if seed is not None:
            self._rng.seed(seed)

    def _schedule(self, event: Event) -> None:
        """Schedule an event."""
        heapq.heappush(self._event_queue, event)

    def _run_simulation(self, scenario: EventScenario) -> None:
        """Process events until queue is empty or time limit exceeded."""
        while self._event_queue:
            event = heapq.heappop(self._event_queue)

            if event.time > scenario.duration:
                break

            self._sim_time = event.time
            self._process_event(event, scenario)

    def _process_event(self, event: Event, scenario: EventScenario) -> None:
        """Dispatch a single event to the appropriate handler."""
        if event.event_type == EventType.PUBLISH:
            msg = self._messages.get(event.message_id)
            if msg is not None and not msg.dropped:
                self._handle_publish(event, msg, scenario)

        elif event.event_type == EventType.ROUTE:
            msg = self._messages.get(event.message_id)
            if msg is not None and not msg.dropped:
                self._handle_route(event, msg, scenario)

        elif event.event_type == EventType.DELIVER:
            msg = self._messages.get(event.message_id)
            if msg is not None and not msg.dropped:
                self._handle_deliver(event, msg, scenario)

        elif event.event_type == EventType.DROP:
            msg = self._messages.get(event.message_id)
            if msg is not None:
                self._handle_drop(event, msg, scenario)

        # [POISSON] new event types
        elif event.event_type == EventType.FAIL_COMPONENT:
            self._handle_poisson_fail(event)

        elif event.event_type == EventType.RECOVER_COMPONENT:
            self._handle_poisson_recover(event)

    # ─────────────────────────────────────────────────────────────────────────
    # [POISSON] Failure / recovery handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_poisson_fail(self, event: Event) -> None:
        """
        Handle a Poisson-injected component failure.

        Marks the target as FAILED in the simulation graph (identical to the
        effect of broker_failure_prob on a ROUTE event, but time-driven rather
        than per-message).  All subsequent ROUTE/DELIVER events destined for
        this component will drop with reason "broker_failed" or
        "subscriber_failed" via the existing is_active() checks — no
        additional logic required.
        """
        target_id = event.target
        comp = self.graph.components.get(target_id)
        if comp is None or comp.state == ComponentState.FAILED:
            return  # already gone — skip silently

        self.graph.fail_component(target_id)
        self.logger.debug(
            f"[Poisson] t={self._sim_time:.4f}s  FAIL  {target_id}  "
            f"(seq={event.data.get('sequence', '?')})"
        )
        self._poisson_failure_log.append({
            "time": round(self._sim_time, 6),
            "component_id": target_id,
            "event": "fail",
        })

    def _handle_poisson_recover(self, event: Event) -> None:
        """
        Handle a Poisson-injected component recovery.

        Restores the target to ACTIVE state so that subsequent routing/delivery
        events can succeed again.  Uses graph.recover_component() if available,
        otherwise directly sets the component state.
        """
        target_id = event.target
        comp = self.graph.components.get(target_id)
        if comp is None:
            return

        # Use a dedicated recover method if the graph exposes one,
        # otherwise fall back to direct state mutation.
        if hasattr(self.graph, "recover_component"):
            self.graph.recover_component(target_id)
        else:
            comp.state = ComponentState.ACTIVE

        self.logger.debug(
            f"[Poisson] t={self._sim_time:.4f}s  RECOVER  {target_id}  "
            f"(seq={event.data.get('sequence', '?')})"
        )
        self._poisson_failure_log.append({
            "time": round(self._sim_time, 6),
            "component_id": target_id,
            "event": "recover",
        })

    # ─────────────────────────────────────────────────────────────────────────
    # Existing message-flow handlers (unchanged)
    # ─────────────────────────────────────────────────────────────────────────

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
            # No brokers — try direct delivery to subscribers
            subscribers = self.graph.get_subscribers(msg.topic_id)
            if not subscribers:
                self._drop_message(msg, "no_subscribers")
                return

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

        # Check for per-message Bernoulli broker failure
        if self._rng.random() < scenario.broker_failure_prob:
            self.graph.fail_component(broker_id)
            self._drop_message(msg, "broker_failure_during_route")
            return

        # Get subscribers for the topic
        subscribers = self.graph.get_subscribers(msg.topic_id)
        if not subscribers:
            self._drop_message(msg, "no_active_subscribers")
            return

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

        if subscriber_id in msg.delivered_to:
            return

        if not self.graph.is_active(subscriber_id):
            self._drop_message(msg, "subscriber_failed")
            return

        elapsed = self._sim_time - msg.created_at
        if elapsed > scenario.delivery_timeout:
            self._drop_message(msg, "delivery_timeout")
            return

        # Successful delivery
        msg.delivered_to.add(subscriber_id)
        msg.delivered_at = self._sim_time + scenario.subscribe_latency

        self._metrics.messages_delivered += 1
        self._reached_subscribers.add(subscriber_id)

        latency = msg.delivered_at - msg.created_at
        self._metrics.total_latency += latency
        self._metrics.latencies.append(latency)
        self._metrics.min_latency = min(self._metrics.min_latency, latency)
        self._metrics.max_latency = max(self._metrics.max_latency, latency)

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

    # ─────────────────────────────────────────────────────────────────────────
    # Result building
    # ─────────────────────────────────────────────────────────────────────────

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
        component_impacts: Dict[str, float] = {}
        total_messages = self._metrics.messages_published

        if total_messages > 0:
            for comp_id, comp in self.graph.components.items():
                handled = comp.messages_sent + comp.messages_routed + comp.messages_received
                if handled > 0:
                    component_impacts[comp_id] = handled / total_messages

        successful_flows: Set[Tuple[str, str, str]] = set()
        for msg in self._messages.values():
            if msg.delivered_to:
                for sub_id in msg.delivered_to:
                    successful_flows.add((msg.source_app, msg.topic_id, sub_id))

        return EventResult(
            source_app=scenario.source_app,
            scenario=scenario.description or f"Event simulation: {scenario.source_app}",
            duration=wall_time,
            metrics=self._metrics,
            affected_topics=list(self._affected_topics),
            reached_subscribers=list(self._reached_subscribers),
            brokers_used=list(self._brokers_used),
            successful_flows=list(successful_flows),
            component_impacts=component_impacts,
            failed_components=[
                c.id for c in self.graph.components.values()
                if c.state == ComponentState.FAILED
            ],
            drop_reasons=dict(self._drop_reasons),
            component_names={
                c.id: c.properties.get("name", c.id)
                for c in self.graph.components.values()
            },
            # [POISSON] include the ordered failure/recovery timeline
            poisson_failure_log=list(self._poisson_failure_log),
        )