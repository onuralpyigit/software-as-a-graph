import pytest
from saag.prediction.data_preparation import extract_simulation_dict

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
    assert res == {
        "App1": {
            "composite": 0.85,
            "reliability": 0.85,
            "maintainability": 0.0,
            "availability": 0.85,
            "security": 0.0
        },
        "App2": {
            "composite": 0.35,
            "reliability": 0.35,
            "maintainability": 0.0,
            "availability": 0.35,
            "security": 0.0
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
    assert res == {
        "App1": {
            "composite": 0.75,
            "reliability": 0.75,
            "maintainability": 0.0,
            "availability": 0.75,
            "security": 0.0
        },
        "App2": {
            "composite": 0.25,
            "reliability": 0.25,
            "maintainability": 0.0,
            "availability": 0.25,
            "security": 0.0
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
                "availability_impact": 0.65,
                "security_impact": 0.55
            }
        }
    ]
    res = extract_simulation_dict(raw)
    assert res == {
        "App1": {
            "composite": 0.95,
            "reliability": 0.85,
            "maintainability": 0.75,
            "availability": 0.65,
            "security": 0.55
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
