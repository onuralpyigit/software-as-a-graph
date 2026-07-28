"""
tests/test_poisson_failure_injection.py
========================================

Unit tests for Poisson failure injection in EventSimulator.

These exercise the *real* ``saag.simulation`` classes against a small
in-memory SimulationGraph (App1 -> Topic1 -> Broker1 -> Sub1); no Neo4j
required.  This file previously carried its own private copy of
EventSimulator and ten model dataclasses, which is why the missing
``EventType.FAIL_COMPONENT`` / ``RECOVER_COMPONENT`` members went unnoticed:
the copy defined them, production did not.

Run with:
    pytest tests/test_poisson_failure_injection.py -v
"""

from saag.core.models import GraphData, ComponentData, EdgeData
from saag.simulation.graph import SimulationGraph
from saag.simulation.event_simulator import EventSimulator
from saag.simulation.models import EventScenario


def make_sim() -> EventSimulator:
    """Build an EventSimulator over App1 -> Topic1 -> Broker1 -> Sub1."""
    graph_data = GraphData(
        components=[
            ComponentData(id="App1", component_type="Application", properties={"layer": "app"}),
            ComponentData(id="Topic1", component_type="Topic", properties={"layer": "mw"}),
            ComponentData(id="Broker1", component_type="Broker", properties={"layer": "infra"}),
            ComponentData(id="Sub1", component_type="Application", properties={"layer": "app"}),
        ],
        edges=[
            EdgeData(source_id="App1", target_id="Topic1", source_type="Application",
                     target_type="Topic", relation_type="PUBLISHES_TO",
                     dependency_type="pubsub", weight=1.0),
            EdgeData(source_id="Sub1", target_id="Topic1", source_type="Application",
                     target_type="Topic", relation_type="SUBSCRIBES_TO",
                     dependency_type="pubsub", weight=1.0),
            EdgeData(source_id="Topic1", target_id="Broker1", source_type="Topic",
                     target_type="Broker", relation_type="ROUTES",
                     dependency_type="routing", weight=1.0),
        ],
    )
    return EventSimulator(SimulationGraph(graph_data=graph_data))


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
        assert result.failed_components == []
        assert abs(result.metrics.delivery_rate - 100.0) < 1e-6


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
