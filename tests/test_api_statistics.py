import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from api.main import app
from api.dependencies import get_statistics_service

client = TestClient(app)

@pytest.fixture
def mock_service():
    service = MagicMock()
    return service

def test_get_graph_stats(mock_service):
    # Setup mocks
    mock_service.get_graph_stats.return_value = {
        "total_nodes": 2,
        "total_edges": 1,
        "node_types": {"Service": 1, "Database": 1},
        "edge_types": {"CONNECTS_TO": 1}
    }
    
    app.dependency_overrides[get_statistics_service] = lambda: mock_service
    
    try:
        # Call endpoint
        response = client.post("/api/v1/stats", json={
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "password"
        })
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["stats"]["total_nodes"] == 2
        mock_service.get_graph_stats.assert_called_once()
    finally:
        app.dependency_overrides = {}


def test_get_connectivity_density_stats(mock_service):
    # Setup mocks
    mock_service.get_connectivity_density.return_value = {
        "density": 0.5,
        "total_nodes": 10,
        "total_edges": 5,
        "max_possible_edges": 90,
        "interpretation": "Test",
        "category": "test",
        "most_dense_components": []
    }
    
    app.dependency_overrides[get_statistics_service] = lambda: mock_service
    
    try:
        # Call endpoint
        response = client.post("/api/v1/stats/connectivity-density", json={
            "uri": "bolt://localhost:7687", 
            "user": "neo4j", 
            "password": "password"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["stats"]["density"] == 0.5
        mock_service.get_connectivity_density.assert_called_once()
    finally:
        app.dependency_overrides = {}


def test_get_message_flow_patterns(mock_service):
    # Setup mocks
    mock_service.get_message_flow_patterns.return_value = {
        "total_topics": 0,
        "total_brokers": 0,
        "total_applications": 2,
        "active_applications": 0,
        "avg_publishers_per_topic": 0.0,
        "avg_subscribers_per_topic": 0.0,
        "avg_topics_per_broker": 0.0,
        "interpretation": "Message flow appears well distributed",
        "category": "balanced",
        "health": "good",
        "hot_topics": [],
        "broker_utilization": [],
        "isolated_applications": [],
        "top_publishers": [],
        "top_subscribers": [],
    }

    app.dependency_overrides[get_statistics_service] = lambda: mock_service
    
    try:
        response = client.post("/api/v1/stats/message-flow-patterns", json={
            "uri": "bolt://localhost:7687", 
            "user": "neo4j", 
            "password": "password"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "stats" in data
        mock_service.get_message_flow_patterns.assert_called_once()
    finally:
        app.dependency_overrides = {}
