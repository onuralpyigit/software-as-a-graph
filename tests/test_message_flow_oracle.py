"""
Contract tests for I_dyn(v) — the message-flow impact oracle.

I_dyn(v) is the drop in delivered message rate that *surviving* consumers suffer
when v fails. It is the project's only behavioural ground truth: I*(v) and
I_comp(v) are both topological cascade engines over the same substrate, so the
agreement between those two cannot rule out a shared construction artifact.
That independence is only worth anything if the quantity measures what it claims,
which is what these tests pin down.

Both failure modes below were live defects, and both produced a plausible-looking
number rather than an error:

  * a silenced publisher's messages left the numerator and the denominator
    together, so publisher loss cancelled itself out and was invisible; and
  * the faulted node's own undrained queues counted as damage, which made the
    score track how much a node *consumed* rather than how much depended on it.

On atm_system the two together put rho(I_dyn, I*) at -0.25, driven by a +0.75
correlation with the faulted node's own subscription count.
"""
import pytest
pytest.importorskip("simpy")

import networkx as nx

from saag.simulation.message_flow_simulator import MessageFlowSimulator


def _impact(graph: nx.DiGraph, node: str, duration: float = 60.0) -> float:
    """I_dyn(node): delivery-rate loss to the components that survived."""
    result = MessageFlowSimulator(
        graph=graph,
        duration=duration,
        fault_node=node,
        fault_time=duration / 2.0,
        seed=42,
    ).run()
    event = result.fault_event
    assert event is not None, f"no fault event recorded for {node}"
    return event.delivery_rate_before - event.delivery_rate_after


def test_silenced_publisher_is_visible():
    """Killing a topic's only publisher must drive delivery to zero.

    Note this case alone does not discriminate: the buggy version also reported
    0.0, but from an empty denominator (no publisher left, so nothing expected)
    rather than from measured loss. test_redundant_publisher_loss_is_partial is
    the guard that separates the two.
    """
    g = nx.DiGraph()
    g.add_node("Pub", type="Application")
    g.add_node("SubA", type="Application")
    g.add_node("SubB", type="Application")
    g.add_node("/telemetry", type="Topic", frequency=20.0)

    g.add_edge("Pub", "/telemetry", type="PUBLISHES_TO")
    g.add_edge("SubA", "/telemetry", type="SUBSCRIBES_TO")
    g.add_edge("SubB", "/telemetry", type="SUBSCRIBES_TO")

    result = MessageFlowSimulator(
        graph=g, duration=60.0, fault_node="Pub", fault_time=30.0, seed=42,
    ).run()
    event = result.fault_event

    assert event.delivery_rate_before == pytest.approx(1.0, abs=0.02)
    assert event.delivery_rate_after == pytest.approx(0.0, abs=0.02)
    assert event.cascade_orphaned_topics == ["/telemetry"]
    assert sorted(event.cascade_impacted_subscribers) == ["SubA", "SubB"]


def test_redundant_publisher_loss_is_partial():
    """With two publishers sharing a topic, losing one is a partial loss.

    Guards the opposite error from the test above: demand that is still being
    served must not be counted as lost.
    """
    g = nx.DiGraph()
    g.add_node("PubA", type="Application")
    g.add_node("PubB", type="Application")
    g.add_node("Sub", type="Application")
    g.add_node("/telemetry", type="Topic", frequency=20.0)

    g.add_edge("PubA", "/telemetry", type="PUBLISHES_TO")
    g.add_edge("PubB", "/telemetry", type="PUBLISHES_TO")
    g.add_edge("Sub", "/telemetry", type="SUBSCRIBES_TO")

    result = MessageFlowSimulator(
        graph=g, duration=60.0, fault_node="PubA", fault_time=30.0, seed=42,
    ).run()
    event = result.fault_event

    # PubA carried half the topic's rate, so half the feed survives.
    assert event.delivery_rate_after == pytest.approx(0.5, abs=0.05)
    # The topic still has a live publisher, so it is not orphaned.
    assert event.cascade_orphaned_topics == []


def test_faulted_nodes_own_consumption_is_not_damage():
    """A pure consumer's failure harms nobody, however much it consumed.

    This is the +0.75 artifact: the faulted node's own undrained queues were
    counted as undelivered messages, so heavy subscribers scored as critical.
    """
    g = nx.DiGraph()
    g.add_node("Pub", type="Application")
    g.add_node("Greedy", type="Application")
    g.add_node("Other", type="Application")
    for i in range(5):
        g.add_node(f"/t{i}", type="Topic", frequency=20.0)
        g.add_edge("Pub", f"/t{i}", type="PUBLISHES_TO")
        g.add_edge("Greedy", f"/t{i}", type="SUBSCRIBES_TO")
    # One topic also feeds a component that survives the fault.
    g.add_edge("Other", "/t0", type="SUBSCRIBES_TO")

    greedy = _impact(g, "Greedy")
    publisher = _impact(g, "Pub")

    assert greedy == pytest.approx(0.0, abs=0.02), (
        f"a pure consumer scored {greedy:.4f}; its own lost feed is being "
        "counted as damage to the system"
    )
    assert publisher > greedy, (
        "the publisher every topic depends on must outrank a pure consumer"
    )


def test_components_the_engine_cannot_observe_are_unlabelled():
    """Brokers and Nodes are invisible here and must not be scored 0.0.

    This engine reads only PUBLISHES_TO / SUBSCRIBES_TO, so faulting a Broker
    (ROUTES) or a Node (RUNS_ON) is a no-op. Reporting that as impact 0.0 would
    make an unmeasured component indistinguishable from a harmless one — the
    same conflation tests/test_groundtruth_contract.py guards for I*(v).
    """
    g = nx.DiGraph()
    g.add_node("Pub", type="Application")
    g.add_node("Sub", type="Application")
    g.add_node("Broker1", type="Broker")
    g.add_node("Host1", type="Node")
    g.add_node("/telemetry", type="Topic", frequency=20.0)

    g.add_edge("Pub", "/telemetry", type="PUBLISHES_TO")
    g.add_edge("Sub", "/telemetry", type="SUBSCRIBES_TO")
    g.add_edge("Broker1", "/telemetry", type="ROUTES")
    g.add_edge("Pub", "Host1", type="RUNS_ON")

    result = MessageFlowSimulator(graph=g, duration=30.0, seed=42).run()

    assert result.labeler == "MessageFlowSimulator"
    assert sorted(result.labeled_node_ids) == ["Pub", "Sub"]
    for invisible in ("Broker1", "Host1", "/telemetry"):
        assert invisible in result.unlabeled_node_ids

    # Labelled and unlabelled must partition the graph: nothing silently dropped.
    assert set(result.labeled_node_ids) | set(result.unlabeled_node_ids) == set(g.nodes)
    assert not set(result.labeled_node_ids) & set(result.unlabeled_node_ids)

    round_tripped = result.to_dict()
    assert round_tripped["labeler"] == "MessageFlowSimulator"
    assert round_tripped["unlabeled_node_ids"] == result.unlabeled_node_ids


@pytest.mark.slow
def test_impact_tracks_dependency_not_consumption_on_a_real_scenario():
    """End-to-end discriminant check on atm_system.

    Asserts the direction of the two correlations rather than a target value:
    what matters is that the score is driven by what depends on a component,
    not by what that component happened to be reading.
    """
    import json
    from pathlib import Path

    from scipy.stats import spearmanr

    from cli.loso_evaluate import _build_graph_from_json

    topology = json.loads(Path("data/scenarios/atm_system.json").read_text())
    graph = _build_graph_from_json(topology)

    subscriptions = {}
    for src, _, data in graph.edges(data=True):
        if data.get("type") == "SUBSCRIBES_TO":
            subscriptions[src] = subscriptions.get(src, 0) + 1

    publishers = sorted({
        src for src, _, data in graph.edges(data=True)
        if data.get("type") == "PUBLISHES_TO"
    })

    impact, impacted_subscribers = [], []
    for node in publishers:
        result = MessageFlowSimulator(
            graph=graph, duration=60.0, fault_node=node, fault_time=30.0, seed=42,
        ).run()
        event = result.fault_event
        impact.append(event.delivery_rate_before - event.delivery_rate_after)
        impacted_subscribers.append(len(event.cascade_impacted_subscribers))

    own_consumption = [subscriptions.get(n, 0) for n in publishers]

    rho_consumption, _ = spearmanr(impact, own_consumption)
    rho_dependency, _ = spearmanr(impact, impacted_subscribers)

    # Before the accounting fix these were +0.75 and +0.05 respectively.
    assert rho_consumption < 0.4, (
        f"I_dyn still tracks the faulted node's own subscription count "
        f"(rho={rho_consumption:.3f}); it is measuring consumption, not dependency"
    )
    assert rho_dependency > rho_consumption, (
        f"I_dyn agrees with its own cascade annotation (rho={rho_dependency:.3f}) "
        f"less than with consumption (rho={rho_consumption:.3f})"
    )
