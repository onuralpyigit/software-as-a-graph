
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from api.main import app
from api.dependencies import get_simulation_service, get_simulate_graph_use_case

client = TestClient(app)

@pytest.fixture
def mock_event_result():
    result = MagicMock()
    result.to_dict.return_value = {
        "source_app": "app1",
        "scenario": "test",
        "duration_sec": 10.0,
        "metrics": {
            "messages_published": 100,
            "messages_delivered": 95,
            "messages_dropped": 5,
            "delivery_rate_percent": 95.0,
            "drop_rate_percent": 5.0,
            "avg_latency_ms": 1.2,
            "p99_latency_ms": 5.0,
            "throughput_per_sec": 9.5,
        },
        "affected_topics": ["t1"],
        "reached_subscribers": ["s1"],
        "brokers_used": ["b1"],
        "component_impacts": {"c1": 0.5},
        "failed_components": [],
        "drop_reasons": {},
        "related_components": [],
    }
    return result

@pytest.fixture
def mock_failure_result():
    result = MagicMock()
    result.target_id = "c1"
    result.target_type = "Application"
    result.impact.composite_impact = 0.8
    result.impact.fragmentation = 0.1
    
    # Mock to_dict for the presenter
    result.to_dict.return_value = {
        "target_id": "c1",
        "target_type": "Application",
        "scenario": "failure",
        "impact": {
            "reachability": {"initial_paths": 10, "remaining_paths": 2, "loss_percent": 80.0},
            "fragmentation": {"fragmentation_percent": 10.0},
            "throughput": {"loss_percent": 50.0},
            "flow_disruption": {"loss_percent": 30.0},
            "cascade": {"count": 2, "depth": 1},
            "composite_impact": 0.8,
            "reliability": {
                "cascade_reach": 0.2, "weighted_cascade_impact": 0.3, 
                "normalized_cascade_depth": 0.1, "reliability_impact": 0.25
            },
            "maintainability": {
                "change_reach": 0.1, "weighted_change_impact": 0.2, 
                "normalized_change_depth": 0.1, "maintainability_impact": 0.15
            },
            "availability": {
                "weighted_reachability_loss": 0.4, "weighted_fragmentation": 0.3, 
                "path_breaking_throughput_loss": 0.1, "availability_impact": 0.7,
                "ia_out": 0.5, "ia_in": 0.4
            },
            "vulnerability": {
                "attack_reach": 0.3, "weighted_attack_impact": 0.4, 
                "high_value_contamination": 0.2, "vulnerability_impact": 0.35
            },
        },
        "cascaded_failures": ["c2"],
        "cascade_sequence": [
            {"id": "c2", "type": "Application", "cause": "c1", "depth": 1}
        ],
        "layer_impacts": {"system": 0.8},
    }
    return result

def test_simulate_event(mock_event_result):
    mock_service = MagicMock()
    mock_service.run_event_simulation.return_value = mock_event_result
    
    app.dependency_overrides[get_simulation_service] = lambda: mock_service
    
    try:
        response = client.post("/api/v1/simulation/event", json={
            "source_app": "app1",
            "num_messages": 100,
            "duration": 10.0,
            "credentials": {
                "uri": "bolt://localhost:7687",
                "user": "neo4j",
                "password": "password"
            }
        })
        
        if response.status_code != 200:
            print(f"DEBUG: {response.json()}")
            
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["simulation_type"] == "event"
        assert data["result"]["source_app"] == "app1"
    finally:
        app.dependency_overrides = {}

def test_simulate_failure(mock_failure_result):
    mock_use_case = MagicMock()
    mock_use_case.execute.return_value = mock_failure_result
    
    app.dependency_overrides[get_simulate_graph_use_case] = lambda: mock_use_case
    
    try:
        response = client.post("/api/v1/simulation/failure", json={
            "target_id": "c1",
            "layer": "system",
            "credentials": {
                "uri": "bolt://localhost:7687",
                "user": "neo4j",
                "password": "password"
            }
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["simulation_type"] == "failure"
    finally:
        app.dependency_overrides = {}
