
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from api.main import app
from api.dependencies import get_repository, get_generation_service, get_client

client = TestClient(app)

@pytest.fixture
def mock_repo():
    return MagicMock()

@pytest.fixture
def mock_gen_service():
    return MagicMock()

def test_generate_graph(mock_gen_service):
    mock_gen_service.generate.return_value = {
        "nodes": [], "brokers": [], "topics": [], "applications": [], "metadata": {}
    }
    
    app.dependency_overrides[get_generation_service] = lambda: mock_gen_service
    
    try:
        response = client.post("/api/v1/graph/generate", json={
            "credentials": {"uri": "bolt://localhost", "user": "n", "password": "p"},
            "scale": "tiny",
            "seed": 42
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "graph_data" in data
        mock_gen_service.generate.assert_called_once()
    finally:
        app.dependency_overrides = {}

def test_import_graph():
    mock_client = MagicMock()
    mock_stats = MagicMock()
    mock_stats.to_dict.return_value = {"node_count": 10, "edge_count": 5}
    mock_client.import_topology.return_value = mock_stats
    
    app.dependency_overrides[get_client] = lambda: mock_client
    
    try:
        response = client.post("/api/v1/graph/import", json={
            "credentials": {"uri": "bolt://localhost", "user": "n", "password": "p"},
            "graph_data": {"nodes": [], "relationships": {}},
            "clear_database": True
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["stats"]["node_count"] == 10
        mock_client.import_topology.assert_called_once()
    finally:
        app.dependency_overrides = {}

def test_export_graph(mock_repo):
    mock_graph = MagicMock()
    mock_graph.components = []
    mock_graph.edges = []
    mock_repo.get_graph_data.return_value = mock_graph
    mock_repo.get_statistics.return_value = {"nodes": 0}
    
    app.dependency_overrides[get_repository] = lambda: mock_repo
    
    try:
        response = client.post("/api/v1/graph/export", json={
            "uri": "bolt://localhost", "user": "n", "password": "p"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "components" in data
        mock_repo.get_graph_data.assert_called_once()
    finally:
        app.dependency_overrides = {}
