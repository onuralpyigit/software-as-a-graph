import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from api.main import app
from api.dependencies import get_repository

client = TestClient(app)

@pytest.fixture
def mock_repo():
    repo = MagicMock()
    repo.get_statistics.return_value = {
        "total_nodes": 2,
        "total_edges": 1,
        "node_types": {"Service": 1, "Database": 1},
        "edge_types": {"CONNECTS_TO": 1}
    }
    repo.export_json.return_value = {
        "nodes": [],
        "edges": [],
        "nodes_by_type": {},
        "topics": [],
        "brokers": [],
        "applications": [],
        "libraries": []
    }
    return repo

def test_get_graph_stats(mock_repo):
    # Setup mocks
    app.dependency_overrides[get_repository] = lambda: mock_repo

    try:
        # Call endpoint
        response = client.post("/api/v1/stats/summary", json={
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "password"
        })

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["stats"]["total_nodes"] == 2
        mock_repo.get_statistics.assert_called_once()
    finally:
        app.dependency_overrides = {}


def test_get_statistics(mock_repo):
    # Setup mocks
    app.dependency_overrides[get_repository] = lambda: mock_repo

    try:
        # Call endpoint
        response = client.post("/api/v1/stats/", json={
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "password"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "stats" in data
        mock_repo.export_json.assert_called_once()
    finally:
        app.dependency_overrides = {}