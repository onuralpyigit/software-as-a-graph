
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from api.main import app
from src.core.models import GraphData, ComponentData, EdgeData

client = TestClient(app)

@pytest.fixture
def mock_graph_data():
    return GraphData(
        components=[
            ComponentData(id="c1", component_type="Service", properties={"name": "Service A"}),
            ComponentData(id="c2", component_type="Database", properties={"name": "DB B"}),
        ],
        edges=[
            EdgeData(
                source_id="c1", 
                target_id="c2", 
                source_type="Service", 
                target_type="Database", 
                dependency_type="CONNECTS_TO", 
                relation_type="CONNECTS_TO", 
                weight=1.0
            )
        ]
    )

@patch("api.routers.statistics.StatisticsService")
def test_get_graph_stats(MockService, mock_graph_data):
    # Setup mocks
    mock_service = MockService.return_value
    mock_service.get_graph_stats.return_value = {
        "total_nodes": 2,
        "total_edges": 1,
        "total_structural_edges": 1,
        "component_types": {"Service": 1, "Database": 1},
        "edge_types": {"CONNECTS_TO": 1}
    }
    
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
    assert data["stats"]["total_edges"] == 1
    
    # Verify service called
    mock_service.get_graph_stats.assert_called_once()



@patch("api.routers.statistics.stats_logic")
@patch("api.routers.statistics.create_repository")
def test_get_connectivity_density_stats(mock_create_repo, mock_stats_logic, mock_graph_data):
    # Setup mocks
    mock_repo = mock_create_repo.return_value
    mock_repo.get_graph_data.return_value = mock_graph_data
    
    mock_stats_logic.get_connectivity_density.return_value = {
        "density": 0.5,
        "avg_degree": 1.0,
        "computation_time_ms": 10
    }
    
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
    
    mock_create_repo.assert_called_once()
    mock_stats_logic.get_connectivity_density.assert_called_once()


@patch("api.routers.statistics.create_repository")
def test_get_message_flow_patterns(mock_create_repo, mock_graph_data):
    # Setup mocks
    mock_repo = mock_create_repo.return_value
    mock_repo.get_graph_data.return_value = mock_graph_data
    mock_repo.get_statistics.return_value = {}

    response = client.post("/api/v1/stats/message-flow-patterns", json={
        "uri": "bolt://localhost:7687", 
        "user": "neo4j", 
        "password": "password"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    # The endpoint calculates stats based on graph data, checking strict structure might be brittle
    # but at least it should succeed
    assert "stats" in data
    mock_create_repo.assert_called_once()
