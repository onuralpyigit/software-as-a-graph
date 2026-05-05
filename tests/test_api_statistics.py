import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from api.main import app
from api.dependencies import get_repository

client = TestClient(app)


def test_get_graph_stats():
    mock_repo = MagicMock()
    mock_repo.get_statistics.return_value = {
        "application_count": 5,
        "broker_count": 2,
        "node_count": 3,
        "topic_count": 10,
        "library_count": 1,
    }

    app.dependency_overrides[get_repository] = lambda: mock_repo

    try:
        response = client.post("/api/v1/stats/summary", json={
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "password",
        })

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["stats"]["application_count"] == 5
        mock_repo.get_statistics.assert_called_once()
    finally:
        app.dependency_overrides = {}

