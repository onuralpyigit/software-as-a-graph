"""
Tests for the validation API router and its presenter.

The presenter's key names are consumed directly by the web UI, so these tests
pin the response shape rather than the metric values.
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from api.main import app
from api.presenters.validation_presenter import (
    build_pipeline_response, build_targets_response, serialize_layer,
)
from saag.validation.models import (
    LayerValidationResult, PipelineResult, ValidationTargets,
)

client = TestClient(app)

CREDENTIALS = {"credentials": {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "pw"}}


@pytest.fixture
def layer_result():
    return LayerValidationResult(
        layer="system",
        layer_name="System Layer",
        predicted_components=10,
        simulated_components=9,
        matched_components=8,
        spearman=0.8123,
        f1_score=0.7654,
        precision=0.9012,
        recall=0.6543,
        top_5_overlap=0.6,
        rmse=0.1234,
        passed=True,
    )


class TestValidationPresenter:

    def test_layer_carries_the_data_block_the_frontend_reads(self, layer_result):
        payload = serialize_layer(layer_result)
        assert payload["data"] == {
            "predicted_components": 10,
            "simulated_components": 9,
            "matched_components": 8,
        }

    def test_layer_summary_key_set_is_stable(self, layer_result):
        summary = serialize_layer(layer_result)["summary"]
        for key in ("passed", "spearman", "f1_score", "precision", "recall", "top_5_overlap", "rmse"):
            assert key in summary, f"web UI reads summary.{key}"
        assert summary["spearman"] == 0.8123
        assert summary["passed"] is True

    def test_pipeline_response_envelope(self, layer_result):
        result = PipelineResult(
            timestamp="2026-01-01T00:00:00",
            layers={"system": layer_result},
            total_components=10,
            layers_passed=1,
            all_passed=True,
            targets=ValidationTargets(),
        )
        response = build_pipeline_response(result)

        assert response["success"] is True
        assert response["result"]["summary"] == {
            "total_components": 10,
            "layers_validated": 1,
            "layers_passed": 1,
            "all_passed": True,
        }
        assert "system" in response["result"]["layers"]
        assert response["result"]["targets"]["spearman"] == 0.70

    def test_targets_response_exposes_only_scalars(self):
        targets = build_targets_response(ValidationTargets())["targets"]
        assert all(isinstance(v, (int, float)) for v in targets.values())
        # Non-scalar configuration must not leak into the API payload
        assert "dimension_weights" not in targets
        assert "node_type_rho" not in targets

    def test_targets_response_keeps_fields_the_web_ui_reads(self):
        targets = build_targets_response(ValidationTargets())["targets"]
        for key in ("spearman", "f1_score", "precision", "recall", "top_5_overlap", "rmse_max"):
            assert key in targets, f"web UI reads targets.{key}"


class TestValidationEndpoints:

    def test_get_targets(self):
        response = client.get("/api/v1/validation/targets")
        assert response.status_code == 200
        assert response.json()["success"] is True
        assert "spearman" in response.json()["targets"]

    def test_get_layers(self):
        response = client.get("/api/v1/validation/layers")
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_quick_validation_needs_both_sides(self):
        response = client.post(
            "/api/v1/validation/quick",
            json={**CREDENTIALS, "predicted_data": {"a": 1.0}},
        )
        assert response.status_code == 400

    def test_quick_validation_runs_without_a_repository(self):
        """The endpoint is pure statistics — it must not open a Neo4j connection."""
        predicted = {f"c{i}": float(i) for i in range(10)}
        actual = {f"c{i}": float(i) * 2 for i in range(10)}

        with patch("api.routers.validation.create_repository") as create_repo:
            response = client.post(
                "/api/v1/validation/quick",
                json={**CREDENTIALS, "predicted_data": predicted, "actual_data": actual},
            )

        assert response.status_code == 200
        create_repo.assert_not_called()
        result = response.json()["result"]
        assert result["data_alignment"]["matched_count"] == 10
        assert result["overall"]["metrics"]["correlation"]["spearman"] == 1.0

    def test_run_pipeline_delegates_to_the_service(self, layer_result):
        pipeline = PipelineResult(
            timestamp="2026-01-01T00:00:00",
            layers={"system": layer_result},
            total_components=10,
            layers_passed=1,
            all_passed=True,
            targets=ValidationTargets(),
        )
        service = MagicMock()
        service.validate_layers.return_value = pipeline

        with patch("api.routers.validation.create_repository"), \
             patch("api.routers.validation._validation_service", return_value=service):
            response = client.post(
                "/api/v1/validation/run-pipeline",
                json={**CREDENTIALS, "layers": ["system"]},
            )

        assert response.status_code == 200
        service.validate_layers.assert_called_once_with(layers=["system"])
        assert response.json()["result"]["layers"]["system"]["summary"]["spearman"] == 0.8123
