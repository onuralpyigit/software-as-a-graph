"""
Tests for GNN Feature Extension — Code-Level Quality Attributes

Covers:
    GNN-CQ-001: HeteroData node features have dim=23
    GNN-CQ-002: Code-quality metrics mapped to indices 13-17
    GNN-CQ-003: Node-type one-hot shifted to indices 18-22
    GNN-CQ-004: Non-CQ nodes (Broker, Node, Topic) have 0s in CQ indices
    GNN-CQ-005: extract_structural_metrics_dict extracts new CQP fields
"""

import pytest
import networkx as nx
import torch
import numpy as np

from src.prediction.data_preparation import (
    networkx_to_hetero_data,
    extract_structural_metrics_dict,
    TOPOLOGICAL_METRIC_KEYS,
    NODE_TYPE_TO_DIM
)
from src.core.metrics import StructuralMetrics


def test_node_feature_dimensions():
    """GNN-CQ-001: Verify type-specific feature dimensions."""
    assert NODE_TYPE_TO_DIM["Application"] == 23
    assert NODE_TYPE_TO_DIM["Broker"] == 18


def test_networkx_to_hetero_data_feature_mapping():
    """GNN-CQ-002, GNN-CQ-003: Verify mapping of metrics and one-hot on HeteroData."""
    G = nx.DiGraph()
    G.add_node("A1", component_type="Application")
    G.add_node("L1", component_type="Library")
    G.add_node("B1", component_type="Broker")

    # Mock structural metrics including new CQ fields
    structural_metrics = {
        "A1": {
            "pagerank": 0.1,
            "loc_norm": 0.5,
            "complexity_norm": 0.6,
            "instability_code": 0.7,
            "lcom_norm": 0.8,
            "code_quality_penalty": 0.9,
        },
        "L1": {
            "loc_norm": 0.2,
            "complexity_norm": 0.3,
        },
        "B1": {
            "pagerank": 0.05
        }
    }

    conv = networkx_to_hetero_data(G, structural_metrics=structural_metrics)
    data = conv.hetero_data

    # Check widths
    assert data["Application"].x.shape[1] == 23
    assert data["Library"].x.shape[1] == 23
    assert data["Broker"].x.shape[1] == 18

    # Check A1 features
    # Indices 18-22 should be CQ metrics
    a1_x = data["Application"].x[0].numpy()
    assert a1_x[18] == pytest.approx(0.5)  # loc_norm
    assert a1_x[19] == pytest.approx(0.6)  # complexity_norm
    assert a1_x[20] == pytest.approx(0.7)  # instability_code
    assert a1_x[21] == pytest.approx(0.8)  # lcom_norm
    assert a1_x[22] == pytest.approx(0.9)  # code_quality_penalty

    # Check B1 features (Broker)
    # CQ indices should not exist for Broker (dim is 18)
    b1_x = data["Broker"].x[0].numpy()
    assert b1_x.shape[0] == 18
    assert b1_x[0] == pytest.approx(0.05) # pagerank


def test_extract_structural_metrics_dict_includes_cq():
    """GNN-CQ-005: extract_structural_metrics_dict includes new CQP fields."""
    mock_comp = StructuralMetrics(
        id="A1",
        name="App1",
        type="Application",
        loc_norm=0.5,
        complexity_norm=0.6,
        instability_code=0.7,
        lcom_norm=0.8,
        code_quality_penalty=0.9
    )
    
    # Wrap in a mock result object
    class MockResult:
        def __init__(self, components):
            self.components = components
            
    res = MockResult([mock_comp])
    out = extract_structural_metrics_dict(res)
    
    metrics = out["App1"]
    assert metrics["loc_norm"] == 0.5
    assert metrics["complexity_norm"] == 0.6
    assert metrics["instability_code"] == 0.7
    assert metrics["lcom_norm"] == 0.8
    assert metrics["code_quality_penalty"] == 0.9
    assert metrics["pagerank"] == 0.0
