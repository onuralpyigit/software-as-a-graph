import pytest
from saag.prediction.data_preparation import extract_simulation_dict, extract_rm_scores_dict

def test_extract_simulation_dict_records_dict():
    raw = {
        "schema_version": "2.0",
        "graph_id": "test_graph",
        "total_nodes_injected": 2,
        "records": {
            "App1": {
                "node_id": "App1",
                "node_type": "Application",
                "node_name": "App One",
                "impact_score": 0.85,
                "cascade_depth": 3,
                "total_impacted_subscribers": 5
            },
            "App2": {
                "node_id": "App2",
                "node_type": "Application",
                "node_name": "App Two",
                "impact_score": 0.35,
                "cascade_depth": 1,
                "total_impacted_subscribers": 2
            }
        }
    }
    res = extract_simulation_dict(raw)
    # FaultInjector emits a single scalar I*(v), which maps onto composite /
    # reliability (reliability is itself the alpha-blend of fault-tolerance and
    # availability, so this scalar isn't emitted as a separate "availability"
    # key). maintainability must be ABSENT, not 0.0: an omitted key means
    # "unmeasured", a 0.0 means "measured as no impact". Emitting a fabricated
    # zero trained a head on a constant.
    assert res == {
        "App1": {
            "composite": 0.85,
            "reliability": 0.85,
        },
        "App2": {
            "composite": 0.35,
            "reliability": 0.35,
        }
    }

def test_extract_simulation_dict_records_list():
    raw = {
        "schema_version": "2.0",
        "records": [
            {
                "node_id": "App1",
                "impact_score": 0.75
            },
            {
                "id": "App2",
                "impact_score": 0.25
            }
        ]
    }
    res = extract_simulation_dict(raw)
    # As above: unmeasured dimensions are absent, not zero.
    assert res == {
        "App1": {
            "composite": 0.75,
            "reliability": 0.75,
        },
        "App2": {
            "composite": 0.25,
            "reliability": 0.25,
        }
    }

def test_extract_simulation_dict_legacy_list():
    raw = [
        {
            "target_id": "App1",
            "impact": {
                "composite_impact": 0.95,
                "reliability_impact": 0.85,
                "maintainability_impact": 0.75,
            }
        }
    ]
    res = extract_simulation_dict(raw)
    assert res == {
        "App1": {
            "composite": 0.95,
            "reliability": 0.85,
            "maintainability": 0.75,
        }
    }

def test_extract_rm_scores_dict_keys_by_id():
    # ComponentQuality has `id`, not `component_id` or `name` — a regression
    # guard for the bug where every key fell through to str(comp), leaving
    # networkx_to_hetero_data's name-keyed lookup (and therefore y_rm) all
    # zero. Uses a minimal duck-typed stand-in rather than the full
    # ComponentQuality/QualityScores dataclasses to keep this a pure
    # data_preparation unit test.
    from types import SimpleNamespace

    class _QualityResult:
        def __init__(self, components):
            self.components = components

    comp = SimpleNamespace(
        id="App1",
        scores=SimpleNamespace(
            overall=0.9, reliability=0.8, maintainability=0.7,
            fault_tolerance=0.65, availability=0.6,
        ),
    )
    res = extract_rm_scores_dict(_QualityResult([comp]))
    assert res == {
        "App1": {
            "overall": 0.9,
            "reliability": 0.8,
            "maintainability": 0.7,
            "fault_tolerance": 0.65,
            "availability": 0.6,
        }
    }


def test_fault_injector_does_not_mutate_graph():
    import networkx as nx
    from saag.simulation.fault_injector import FaultInjector

    # Create a graph without DEPENDS_ON edges
    g = nx.DiGraph()
    g.add_node("App1", type="Application")
    g.add_node("App2", type="Application")
    g.add_node("Topic1", type="Topic")
    g.add_edge("App1", "Topic1", type="PUBLISHES_TO")
    g.add_edge("App2", "Topic1", type="SUBSCRIBES_TO")

    # Record the original edges
    original_edges = set(g.edges())

    # Initialize FaultInjector, which dynamically derives DEPENDS_ON edges
    injector = FaultInjector(g)

    # Check that original graph's edges have not changed
    assert set(g.edges()) == original_edges

    # Check that FaultInjector's internal graph has the derived DEPENDS_ON edges
    internal_edges = set(injector.graph.edges())
    assert len(internal_edges) > len(original_edges)
    assert ("App2", "App1") in internal_edges
