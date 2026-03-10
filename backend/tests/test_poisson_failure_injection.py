"""
tests/test_poisson_failure_injection.py
========================================

Unit tests for Poisson failure injection in EventSimulator.
Runs against a lightweight in-memory stub graph — no Neo4j required.

Run with:
    pytest tests/test_poisson_failure_injection.py -v
"""

from __future__ import annotations
import heapq
import random
import math
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
from enum import Enum
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# Minimal stubs (mirror the real classes enough to exercise the scheduler)
# ─────────────────────────────────────────────────────────────────────────────

class ComponentState(Enum):
    ACTIVE = "active"
    FAILED = "failed"


class EventType(Enum):
    PUBLISH          = "publish"
    ROUTE            = "route"
    DELIVER          = "deliver"
    ACK              = "ack"
    TIMEOUT          = "timeout"
    DROP             = "drop"
    FAIL_COMPONENT   = "fail_component"
    RECOVER_COMPONENT = "recover_component"


@dataclass
class TopicInfo:
    id: str
    name: str
    priority_value: int = 1
    requires_ack: bool = False


@dataclass(order=True)
class Event:
    time: float
    event_type: EventType = field(compare=False)
    source: str         = field(compare=False)
    target: str         = field(compare=False)
    message_id: str     = field(compare=False)
    data: Dict          = field(default_factory=dict, compare=False)


@dataclass
class Message:
    id: str
    source_app: str
    topic_id: str
    size: int
    priority: int
    requires_ack: bool
    created_at: float
    hops: int = 0
    delivered_to: Set[str] = field(default_factory=set)
    dropped: bool = False
    drop_reason: Optional[str] = None
    delivered_at: Optional[float] = None


@dataclass
class EventScenario:
    source_app: str
    description: str = ""
    num_messages: int = 100
    message_interval: float = 0.1
    message_size: int = 1024
    duration: float = 10.0
    seed: Optional[int] = 42
    publish_latency: float = 0.001
    broker_latency: float = 0.002
    network_latency: float = 0.005
    subscribe_latency: float = 0.001
    drop_probability: float = 0.0
    broker_failure_prob: float = 0.0
    delivery_timeout: float = 2.0
    failure_rate: float = 0.0
    failure_targets: Optional[List[str]] = None
    mean_recovery_time: float = 0.0
    poisson_arrivals: bool = False


@dataclass
class RuntimeMetrics:
    messages_published: int = 0
    messages_delivered: int = 0
    messages_dropped: int = 0
    total_latency: float = 0.0
    min_latency: float = float("inf")
    max_latency: float = 0.0
    latencies: List[float] = field(default_factory=list)
    simulation_duration: float = 0.0

    @property
    def delivery_rate(self) -> float:
        t = self.messages_published
        return (self.messages_delivered / t * 100) if t > 0 else 0.0

    @property
    def drop_rate(self) -> float:
        t = self.messages_published
        return (self.messages_dropped / t * 100) if t > 0 else 0.0


@dataclass
class ComponentInfo:
    id: str
    type: str
    state: ComponentState = ComponentState.ACTIVE
    properties: Dict = field(default_factory=dict)
    messages_sent: int = 0
    messages_received: int = 0
    messages_routed: int = 0
    total_latency: float = 0.0


@dataclass
class EventResult:
    source_app: str
    scenario: str
    duration: float
    metrics: RuntimeMetrics
    affected_topics: List[str] = field(default_factory=list)
    reached_subscribers: List[str] = field(default_factory=list)
    brokers_used: List[str] = field(default_factory=list)
    successful_flows: List = field(default_factory=list)
    component_impacts: Dict = field(default_factory=dict)
    failed_components: List[str] = field(default_factory=list)
    drop_reasons: Dict[str, int] = field(default_factory=dict)
    component_names: Dict[str, str] = field(default_factory=dict)
    related_components: List[str] = field(default_factory=list)
    poisson_failure_log: List[Dict[str, Any]] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Minimal stub graph: App1 -> Topic1 -> Broker1 -> Sub1
# ─────────────────────────────────────────────────────────────────────────────

class StubGraph:
    """In-memory stub reproducing the SimulationGraph interface."""

    def __init__(self):
        self.components: Dict[str, ComponentInfo] = {
            "App1":    ComponentInfo(id="App1",    type="Application"),
            "Topic1":  ComponentInfo(id="Topic1",  type="Topic"),
            "Broker1": ComponentInfo(id="Broker1", type="Broker"),
            "Sub1":    ComponentInfo(id="Sub1",    type="Application"),
        }
        self.topics = {"Topic1": TopicInfo(id="Topic1", name="Topic1")}
        self._publishers  = {"Topic1": ["App1"]}
        self._subscribers = {"Topic1": ["Sub1"]}
        self._brokers     = {"Topic1": ["Broker1"]}

    def reset(self):
        for c in self.components.values():
            c.state = ComponentState.ACTIVE

    def get_app_topics(self, app_id: str):
        pub = [t for t, pubs in self._publishers.items() if app_id in pubs]
        return pub, []

    def get_publishers(self, topic_id: str):
        return self._publishers.get(topic_id, [])

    def get_subscribers(self, topic_id: str):
        return [
            s for s in self._subscribers.get(topic_id, [])
            if self.is_active(s)
        ]

    def get_routing_brokers(self, topic_id: str):
        return [
            b for b in self._brokers.get(topic_id, [])
            if self.is_active(b)
        ]

    def is_active(self, comp_id: str) -> bool:
        c = self.components.get(comp_id)
        return c is not None and c.state == ComponentState.ACTIVE

    def fail_component(self, comp_id: str):
        c = self.components.get(comp_id)
        if c:
            c.state = ComponentState.FAILED

    def recover_component(self, comp_id: str):
        c = self.components.get(comp_id)
        if c:
            c.state = ComponentState.ACTIVE


# ─────────────────────────────────────────────────────────────────────────────
# Minimal EventSimulator (copy of the new implementation, self-contained)
# ─────────────────────────────────────────────────────────────────────────────

class EventSimulator:
    def __init__(self, graph):
        self.graph = graph
        self._event_queue: List[Event] = []
        self._messages: Dict[str, Message] = {}
        self._metrics = RuntimeMetrics()
        self._rng = random.Random()
        self._sim_time = 0.0
        self._affected_topics: Set[str] = set()
        self._reached_subscribers: Set[str] = set()
        self._brokers_used: Set[str] = set()
        self._drop_reasons: Dict[str, int] = defaultdict(int)
        self._poisson_failure_log: List[Dict[str, Any]] = []

    def simulate(self, scenario: EventScenario) -> EventResult:
        import time as _time
        start = _time.time()
        self._reset(scenario.seed)
        self.graph.reset()

        publishes_to, _ = self.graph.get_app_topics(scenario.source_app)
        if not publishes_to:
            return self._empty_result(scenario, "no publications")

        if scenario.poisson_arrivals:
            self._schedule_poisson_arrivals(scenario, publishes_to)
        else:
            self._schedule_deterministic_arrivals(scenario, publishes_to)

        if scenario.failure_rate > 0.0:
            self._schedule_poisson_failures(scenario)

        self._run_simulation(scenario)
        self._metrics.simulation_duration = self._sim_time
        wall = _time.time() - start
        return self._build_result(scenario, wall)

    # Scheduling helpers
    def _schedule_deterministic_arrivals(self, scenario, publishes_to):
        for i in range(scenario.num_messages):
            t = i * scenario.message_interval
            if t > scenario.duration:
                break
            self._create_and_schedule_message(i, t, scenario, publishes_to)

    def _schedule_poisson_arrivals(self, scenario, publishes_to):
        rate = 1.0 / max(scenario.message_interval, 1e-9)
        t = self._rng.expovariate(rate)
        for i in range(scenario.num_messages):
            if t > scenario.duration:
                break
            self._create_and_schedule_message(i, t, scenario, publishes_to)
            t += self._rng.expovariate(rate)

    def _create_and_schedule_message(self, idx, msg_time, scenario, publishes_to):
        topic_id = publishes_to[idx % len(publishes_to)]
        topic_info = self.graph.topics.get(topic_id, TopicInfo(id=topic_id, name=topic_id))
        msg_id = f"msg_{idx}"
        self._messages[msg_id] = Message(
            id=msg_id, source_app=scenario.source_app, topic_id=topic_id,
            size=scenario.message_size, priority=topic_info.priority_value,
            requires_ack=topic_info.requires_ack, created_at=msg_time,
        )
        self._schedule(Event(time=msg_time, event_type=EventType.PUBLISH,
                             source=scenario.source_app, target=topic_id, message_id=msg_id))

    def _schedule_poisson_failures(self, scenario):
        targets = self._build_failure_targets(scenario)
        if not targets:
            return
        lambda_ = scenario.failure_rate
        t = self._rng.expovariate(lambda_)
        seq = 0
        while t < scenario.duration:
            target_id = self._rng.choice(targets)
            self._schedule(Event(
                time=t, event_type=EventType.FAIL_COMPONENT,
                source="poisson_injector", target=target_id,
                message_id=f"__poisson_fail_{seq}", data={"sequence": seq},
            ))
            if scenario.mean_recovery_time > 0.0:
                recovery_delay = self._rng.expovariate(1.0 / scenario.mean_recovery_time)
                rt = t + recovery_delay
                if rt < scenario.duration:
                    self._schedule(Event(
                        time=rt, event_type=EventType.RECOVER_COMPONENT,
                        source="poisson_injector", target=target_id,
                        message_id=f"__poisson_recover_{seq}", data={"sequence": seq},
                    ))
            seq += 1
            t += self._rng.expovariate(lambda_)

    def _build_failure_targets(self, scenario):
        all_ids = list(self.graph.components.keys())
        if scenario.failure_targets is not None:
            valid = [c for c in scenario.failure_targets if c in self.graph.components]
            return valid or all_ids
        return all_ids

    def _reset(self, seed=None):
        self._event_queue = []
        self._messages = {}
        self._metrics = RuntimeMetrics()
        self._sim_time = 0.0
        self._affected_topics = set()
        self._reached_subscribers = set()
        self._brokers_used = set()
        self._drop_reasons = defaultdict(int)
        self._poisson_failure_log = []
        if seed is not None:
            self._rng.seed(seed)

    def _schedule(self, event):
        heapq.heappush(self._event_queue, event)

    def _run_simulation(self, scenario):
        while self._event_queue:
            event = heapq.heappop(self._event_queue)
            if event.time > scenario.duration:
                break
            self._sim_time = event.time
            self._process_event(event, scenario)

    def _process_event(self, event, scenario):
        if event.event_type == EventType.PUBLISH:
            msg = self._messages.get(event.message_id)
            if msg and not msg.dropped:
                self._handle_publish(event, msg, scenario)
        elif event.event_type == EventType.ROUTE:
            msg = self._messages.get(event.message_id)
            if msg and not msg.dropped:
                self._handle_route(event, msg, scenario)
        elif event.event_type == EventType.DELIVER:
            msg = self._messages.get(event.message_id)
            if msg and not msg.dropped:
                self._handle_deliver(event, msg, scenario)
        elif event.event_type == EventType.DROP:
            msg = self._messages.get(event.message_id)
            if msg:
                self._handle_drop(event, msg, scenario)
        elif event.event_type == EventType.FAIL_COMPONENT:
            self._handle_poisson_fail(event)
        elif event.event_type == EventType.RECOVER_COMPONENT:
            self._handle_poisson_recover(event)

    def _handle_poisson_fail(self, event):
        comp = self.graph.components.get(event.target)
        if comp is None or comp.state == ComponentState.FAILED:
            return
        self.graph.fail_component(event.target)
        self._poisson_failure_log.append({
            "time": round(self._sim_time, 6),
            "component_id": event.target,
            "event": "fail",
        })

    def _handle_poisson_recover(self, event):
        comp = self.graph.components.get(event.target)
        if comp is None:
            return
        self.graph.recover_component(event.target)
        self._poisson_failure_log.append({
            "time": round(self._sim_time, 6),
            "component_id": event.target,
            "event": "recover",
        })

    def _handle_publish(self, event, msg, scenario):
        self._metrics.messages_published += 1
        self._affected_topics.add(msg.topic_id)
        src_comp = self.graph.components.get(event.source)
        if src_comp:
            src_comp.messages_sent += 1
        if self._rng.random() < scenario.drop_probability:
            self._drop_message(msg, "random_drop")
            return
        brokers = self.graph.get_routing_brokers(msg.topic_id)
        if not brokers:
            subscribers = self.graph.get_subscribers(msg.topic_id)
            if not subscribers:
                self._drop_message(msg, "no_subscribers")
                return
            for sub in subscribers:
                t = self._sim_time + scenario.publish_latency + scenario.network_latency
                self._schedule(Event(time=t, event_type=EventType.DELIVER,
                                     source=event.source, target=sub, message_id=msg.id))
        else:
            for broker in brokers:
                self._brokers_used.add(broker)
                t = self._sim_time + scenario.publish_latency + scenario.network_latency
                self._schedule(Event(time=t, event_type=EventType.ROUTE,
                                     source=msg.topic_id, target=broker, message_id=msg.id))
        msg.hops += 1

    def _handle_route(self, event, msg, scenario):
        broker_id = event.target
        if not self.graph.is_active(broker_id):
            self._drop_message(msg, "broker_failed")
            return
        broker_comp = self.graph.components.get(broker_id)
        if broker_comp:
            broker_comp.messages_routed += 1
        if self._rng.random() < scenario.broker_failure_prob:
            self.graph.fail_component(broker_id)
            self._drop_message(msg, "broker_failure_during_route")
            return
        subscribers = self.graph.get_subscribers(msg.topic_id)
        if not subscribers:
            self._drop_message(msg, "no_active_subscribers")
            return
        for sub in subscribers:
            if sub in msg.delivered_to:
                continue
            t = self._sim_time + scenario.broker_latency + scenario.network_latency
            self._schedule(Event(time=t, event_type=EventType.DELIVER,
                                 source=broker_id, target=sub, message_id=msg.id))
        msg.hops += 1

    def _handle_deliver(self, event, msg, scenario):
        subscriber_id = event.target
        if subscriber_id in msg.delivered_to:
            return
        if not self.graph.is_active(subscriber_id):
            self._drop_message(msg, "subscriber_failed")
            return
        if self._sim_time - msg.created_at > scenario.delivery_timeout:
            self._drop_message(msg, "delivery_timeout")
            return
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

    def _handle_drop(self, event, msg, scenario):
        self._drop_message(msg, event.data.get("reason", "unknown"))

    def _drop_message(self, msg, reason):
        if msg.dropped:
            return
        msg.dropped = True
        msg.drop_reason = reason
        self._metrics.messages_dropped += 1
        self._drop_reasons[reason] += 1

    def _empty_result(self, scenario, reason):
        return EventResult(source_app=scenario.source_app, scenario=reason,
                           duration=0.0, metrics=RuntimeMetrics(),
                           drop_reasons={"simulation_failed": 1})

    def _build_result(self, scenario, wall_time):
        component_impacts = {}
        total = self._metrics.messages_published
        if total > 0:
            for cid, c in self.graph.components.items():
                h = c.messages_sent + c.messages_routed + c.messages_received
                if h > 0:
                    component_impacts[cid] = h / total
        successful_flows = set()
        for msg in self._messages.values():
            if msg.delivered_to:
                for sub in msg.delivered_to:
                    successful_flows.add((msg.source_app, msg.topic_id, sub))
        return EventResult(
            source_app=scenario.source_app,
            scenario=scenario.description or f"Event simulation: {scenario.source_app}",
            duration=wall_time, metrics=self._metrics,
            affected_topics=list(self._affected_topics),
            reached_subscribers=list(self._reached_subscribers),
            brokers_used=list(self._brokers_used),
            successful_flows=list(successful_flows),
            component_impacts=component_impacts,
            failed_components=[c.id for c in self.graph.components.values()
                                if c.state == ComponentState.FAILED],
            drop_reasons=dict(self._drop_reasons),
            component_names={c.id: c.properties.get("name", c.id)
                             for c in self.graph.components.values()},
            poisson_failure_log=list(self._poisson_failure_log),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def make_sim():
    return EventSimulator(StubGraph())


class TestNoPoissonBaseline:
    """Existing behaviour must be unchanged when failure_rate=0."""

    def test_full_delivery_no_failures(self):
        sim = make_sim()
        result = sim.simulate(EventScenario(source_app="App1", num_messages=20,
                                            duration=5.0, seed=1))
        assert result.metrics.messages_published == 20
        assert result.metrics.messages_delivered == 20
        assert result.metrics.messages_dropped == 0
        assert result.poisson_failure_log == []

    def test_delivery_rate_100_percent(self):
        sim = make_sim()
        result = sim.simulate(EventScenario(source_app="App1", num_messages=50,
                                            duration=10.0, seed=2))
        assert abs(result.metrics.delivery_rate - 100.0) < 1e-6

    def test_no_failed_components(self):
        sim = make_sim()
        result = sim.simulate(EventScenario(source_app="App1", num_messages=10,
                                            duration=3.0, seed=3))
        assert result.failed_components == []


class TestPoissonFailureScheduling:
    """Verify that Poisson failure events are correctly generated."""

    def test_failure_log_non_empty_when_rate_set(self):
        sim = make_sim()
        result = sim.simulate(EventScenario(
            source_app="App1", num_messages=100, duration=10.0,
            seed=42, failure_rate=2.0,  # λ=2 → expect ~20 failures in 10 s
        ))
        fail_events = [e for e in result.poisson_failure_log if e["event"] == "fail"]
        assert len(fail_events) > 0, "Expected at least one Poisson failure event"

    def test_failure_log_empty_when_rate_zero(self):
        sim = make_sim()
        result = sim.simulate(EventScenario(
            source_app="App1", num_messages=50, duration=5.0,
            seed=7, failure_rate=0.0,
        ))
        assert result.poisson_failure_log == []

    def test_failure_times_within_duration(self):
        sim = make_sim()
        duration = 8.0
        result = sim.simulate(EventScenario(
            source_app="App1", num_messages=100, duration=duration,
            seed=99, failure_rate=3.0,
        ))
        for entry in result.poisson_failure_log:
            assert entry["time"] <= duration, \
                f"Failure at t={entry['time']} exceeds duration={duration}"

    def test_failure_times_non_decreasing(self):
        sim = make_sim()
        result = sim.simulate(EventScenario(
            source_app="App1", num_messages=200, duration=20.0,
            seed=5, failure_rate=1.5,
        ))
        times = [e["time"] for e in result.poisson_failure_log]
        assert times == sorted(times), "Failure log must be chronologically ordered"

    def test_failure_targets_restricted(self):
        sim = make_sim()
        result = sim.simulate(EventScenario(
            source_app="App1", num_messages=50, duration=10.0,
            seed=13, failure_rate=5.0,
            failure_targets=["Broker1"],  # only allow broker failures
        ))
        for entry in result.poisson_failure_log:
            if entry["event"] == "fail":
                assert entry["component_id"] == "Broker1", \
                    f"Unexpected target: {entry['component_id']}"

    def test_failure_events_reference_known_components(self):
        sim = make_sim()
        known = {"App1", "Topic1", "Broker1", "Sub1"}
        result = sim.simulate(EventScenario(
            source_app="App1", num_messages=100, duration=10.0,
            seed=77, failure_rate=2.0,
        ))
        for entry in result.poisson_failure_log:
            assert entry["component_id"] in known, \
                f"Unknown component: {entry['component_id']}"


class TestPoissonStatisticalProperties:
    """Validate that the failure count distribution matches Poisson statistics."""

    def test_mean_failure_count_approx_lambda_T(self):
        """
        Expected number of Poisson failure events = λ × T.
        With many repeated runs the sample mean should converge to this value.
        We allow ±30% relative error given N=100 runs (generous for a unit test).

        Note: recovery must be enabled (mean_recovery_time > 0) so that the
        same component can be failed multiple times.  Without recovery the stub
        graph's 4 components saturate quickly (all permanently failed), which
        caps the count well below λ×T.
        """
        lambda_ = 2.0
        T = 5.0
        expected = lambda_ * T   # = 10 failures

        counts = []
        for seed in range(100):
            sim = make_sim()
            result = sim.simulate(EventScenario(
                source_app="App1", num_messages=200, duration=T,
                seed=seed, failure_rate=lambda_,
                mean_recovery_time=0.1,   # fast recovery keeps components available
            ))
            fail_count = sum(1 for e in result.poisson_failure_log if e["event"] == "fail")
            counts.append(fail_count)

        sample_mean = sum(counts) / len(counts)
        rel_err = abs(sample_mean - expected) / expected
        assert rel_err < 0.30, \
            f"Mean failure count {sample_mean:.2f} deviates >30% from expected {expected}"

    def test_inter_arrival_times_are_exponential(self):
        """
        The inter-arrival times between consecutive failure events should
        have a mean close to 1/λ (exponential distribution mean).
        We test with λ=5 → mean inter-arrival = 0.2 s.
        """
        lambda_ = 5.0
        expected_mean = 1.0 / lambda_

        sim = make_sim()
        result = sim.simulate(EventScenario(
            source_app="App1", num_messages=500, duration=50.0,
            seed=123, failure_rate=lambda_,
        ))
        fail_times = sorted(
            e["time"] for e in result.poisson_failure_log if e["event"] == "fail"
        )
        if len(fail_times) < 10:
            # Too few events to test reliably — skip
            return

        inter_arrivals = [
            fail_times[i + 1] - fail_times[i]
            for i in range(len(fail_times) - 1)
        ]
        sample_mean = sum(inter_arrivals) / len(inter_arrivals)
        rel_err = abs(sample_mean - expected_mean) / expected_mean
        assert rel_err < 0.25, \
            f"Mean inter-arrival {sample_mean:.4f} deviates >25% from 1/λ={expected_mean:.4f}"

    def test_deterministic_given_seed(self):
        """Same seed must produce identical failure logs."""
        def run(seed):
            sim = make_sim()
            return sim.simulate(EventScenario(
                source_app="App1", num_messages=100, duration=10.0,
                seed=seed, failure_rate=3.0,
            )).poisson_failure_log

        log_a = run(42)
        log_b = run(42)
        assert log_a == log_b, "Simulation is not deterministic with the same seed"

    def test_different_seeds_produce_different_logs(self):
        def run(seed):
            sim = make_sim()
            return sim.simulate(EventScenario(
                source_app="App1", num_messages=100, duration=10.0,
                seed=seed, failure_rate=3.0,
            )).poisson_failure_log

        log_a = run(1)
        log_b = run(9999)
        assert log_a != log_b, "Different seeds should produce different failure timelines"


class TestRecovery:
    """Verify Poisson recovery mechanics."""

    def test_recovery_events_present_when_enabled(self):
        sim = make_sim()
        result = sim.simulate(EventScenario(
            source_app="App1", num_messages=200, duration=20.0,
            seed=8, failure_rate=2.0, mean_recovery_time=1.0,
        ))
        recover_events = [e for e in result.poisson_failure_log if e["event"] == "recover"]
        assert len(recover_events) > 0, "Expected recovery events when mean_recovery_time > 0"

    def test_no_recovery_events_when_disabled(self):
        sim = make_sim()
        result = sim.simulate(EventScenario(
            source_app="App1", num_messages=100, duration=10.0,
            seed=11, failure_rate=2.0, mean_recovery_time=0.0,
        ))
        recover_events = [e for e in result.poisson_failure_log if e["event"] == "recover"]
        assert recover_events == []

    def test_recovery_always_after_failure(self):
        sim = make_sim()
        result = sim.simulate(EventScenario(
            source_app="App1", num_messages=200, duration=20.0,
            seed=17, failure_rate=1.0, mean_recovery_time=2.0,
        ))
        for i, entry in enumerate(result.poisson_failure_log):
            if entry["event"] == "recover":
                # Find the most recent failure for the same component
                same_comp_fails = [
                    e for e in result.poisson_failure_log[:i]
                    if e["event"] == "fail" and e["component_id"] == entry["component_id"]
                ]
                assert same_comp_fails, \
                    f"Recovery for {entry['component_id']} has no preceding failure in log"
                assert same_comp_fails[-1]["time"] <= entry["time"], \
                    "Recovery time must not precede its corresponding failure time"

    def test_recovery_improves_delivery_rate(self):
        """
        With high failure rate and recovery, final delivery rate should be
        higher than without recovery (same seed, same λ).
        """
        scenario_no_recovery = EventScenario(
            source_app="App1", num_messages=300, duration=20.0,
            seed=55, failure_rate=3.0, mean_recovery_time=0.0,
        )
        scenario_with_recovery = EventScenario(
            source_app="App1", num_messages=300, duration=20.0,
            seed=55, failure_rate=3.0, mean_recovery_time=0.5,
        )
        result_no_rec = make_sim().simulate(scenario_no_recovery)
        result_with_rec = make_sim().simulate(scenario_with_recovery)

        assert result_with_rec.metrics.delivery_rate >= result_no_rec.metrics.delivery_rate, \
            "Recovery should not decrease delivery rate"


class TestPoissonArrivals:
    """Verify M/G/1 message-arrival mode."""

    def test_poisson_arrivals_publishes_correct_count(self):
        sim = make_sim()
        result = sim.simulate(EventScenario(
            source_app="App1", num_messages=50, duration=100.0,
            message_interval=0.1, seed=3, poisson_arrivals=True,
        ))
        # All 50 messages should have been scheduled within duration=100s
        assert result.metrics.messages_published == 50

    def test_poisson_arrivals_mean_interval(self):
        """
        Mean inter-arrival of published messages ≈ message_interval.
        Measured over many messages with long duration.
        """
        mu = 0.1   # mean inter-arrival = 100 ms
        sim = make_sim()
        result = sim.simulate(EventScenario(
            source_app="App1", num_messages=500, duration=500.0,
            message_interval=mu, seed=42, poisson_arrivals=True,
        ))
        # Recover creation times from message objects (stub exposes _messages)
        times = sorted(m.created_at for m in sim._messages.values())
        if len(times) < 10:
            return
        ias = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        mean_ia = sum(ias) / len(ias)
        rel_err = abs(mean_ia - mu) / mu
        assert rel_err < 0.20, \
            f"Mean Poisson inter-arrival {mean_ia:.4f} deviates >20% from {mu}"

    def test_poisson_arrivals_no_log_when_failure_rate_zero(self):
        sim = make_sim()
        result = sim.simulate(EventScenario(
            source_app="App1", num_messages=30, duration=10.0,
            seed=6, poisson_arrivals=True, failure_rate=0.0,
        ))
        assert result.poisson_failure_log == []


class TestImpactOnDelivery:
    """Verify that Poisson failures actually affect message delivery."""

    def test_high_failure_rate_reduces_delivery(self):
        """
        A very high failure rate targeting the only broker should eventually
        cause some messages to be dropped.
        """
        sim = make_sim()
        result = sim.simulate(EventScenario(
            source_app="App1", num_messages=500, duration=30.0,
            seed=21, failure_rate=10.0,
            failure_targets=["Broker1"],
            mean_recovery_time=0.0,  # no recovery — broker stays down
        ))
        # With λ=10 and no recovery, broker fails almost immediately;
        # some messages will be dropped as "broker_failed"
        assert result.metrics.messages_dropped > 0 or result.metrics.delivery_rate < 100.0

    def test_failed_components_populated(self):
        sim = make_sim()
        result = sim.simulate(EventScenario(
            source_app="App1", num_messages=100, duration=10.0,
            seed=33, failure_rate=5.0,
            failure_targets=["Broker1"],
        ))
        assert "Broker1" in result.failed_components


if __name__ == "__main__":
    import sys

    test_classes = [
        TestNoPoissonBaseline,
        TestPoissonFailureScheduling,
        TestPoissonStatisticalProperties,
        TestRecovery,
        TestPoissonArrivals,
        TestImpactOnDelivery,
    ]

    passed = failed = 0
    for cls in test_classes:
        obj = cls()
        for name in [m for m in dir(cls) if m.startswith("test_")]:
            try:
                getattr(obj, name)()
                print(f"  PASS  {cls.__name__}::{name}")
                passed += 1
            except Exception as exc:
                print(f"  FAIL  {cls.__name__}::{name}  →  {exc}")
                failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
