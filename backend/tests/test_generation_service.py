import hashlib
import json
import pytest
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tools.generation import GenerationService, load_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canonical_sha256(data: dict) -> str:
    """Return the SHA-256 of the JSON-canonical form (sorted keys, no whitespace)."""
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Golden-file constants for scenario_08_tiny_regression.yaml (seed 8008)
# Update these values by running:
#   python -c "
#     import sys, json, hashlib; sys.path.insert(0,'backend')
#     from tools.generation import load_config, GenerationService
#     from pathlib import Path
#     d = GenerationService(config=load_config(Path('input/scenario_08_tiny_regression.yaml'))).generate()
#     print(hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest())
#   "
# ---------------------------------------------------------------------------

_SCENARIO_08_YAML = project_root / "input" / "scenario_08_tiny_regression.yaml"

_GOLDEN_SHA256 = "5ce27fd923ef4252123d4d371ddecc8635302ab43996397c385d30c2bd4aa7ad"

_GOLDEN_ENTITY_COUNTS = {
    "nodes": 3,
    "brokers": 2,
    "topics": 8,
    "applications": 12,
    "libraries": 4,
}

_GOLDEN_RELATIONSHIP_COUNTS = {
    "runs_on": 14,
    "routes": 10,
    "publishes_to": 30,
    "subscribes_to": 37,
    "uses": 13,
    "connects_to": 2,
}


class TestGenerationService:
    def test_generate_tiny(self):
        gen = GenerationService(scale="tiny", seed=1)
        data = gen.generate()

        assert len(data["nodes"]) == 2
        assert len(data["brokers"]) == 1
        assert len(data["topics"]) == 5
        assert len(data["applications"]) == 5


class TestScenario08GoldenFile:
    """Regression tests that pin the scenario_08 generator output.

    These tests enforce two properties:
      1. Determinism — running the generator twice with the same config and
         seed must produce byte-for-byte identical canonical JSON.
      2. Stability — the canonical SHA-256 must match the stored golden value,
         catching any generator change that silently shifts the output.

    When a generator change is *intentional* (e.g. a new field is added),
    re-run the command in the module docstring above, paste the new hash into
    _GOLDEN_SHA256, update _GOLDEN_ENTITY_COUNTS / _GOLDEN_RELATIONSHIP_COUNTS
    if the counts changed, and commit both together.
    """

    @pytest.fixture(scope="class")
    def scenario_08_data(self):
        config = load_config(_SCENARIO_08_YAML)
        return GenerationService(config=config).generate()

    def test_entity_counts(self, scenario_08_data):
        """Entity counts must match the golden values."""
        data = scenario_08_data
        for entity, expected in _GOLDEN_ENTITY_COUNTS.items():
            actual = len(data[entity])
            assert actual == expected, (
                f"{entity}: expected {expected}, got {actual}"
            )

    def test_relationship_counts(self, scenario_08_data):
        """Relationship counts must match the golden values."""
        rels = scenario_08_data["relationships"]
        for rel, expected in _GOLDEN_RELATIONSHIP_COUNTS.items():
            actual = len(rels[rel])
            assert actual == expected, (
                f"relationships.{rel}: expected {expected}, got {actual}"
            )

    def test_metadata(self, scenario_08_data):
        """Seed and generation_mode must be preserved in the output metadata."""
        meta = scenario_08_data["metadata"]
        assert meta["seed"] == 8008
        assert meta["generation_mode"] == "statistical"

    def test_all_applications_have_code_metrics(self, scenario_08_data):
        """Every application must carry a code_metrics dict (regression for GEN-10)."""
        for app in scenario_08_data["applications"]:
            assert "code_metrics" in app, f"app {app['id']} missing code_metrics"

    def test_all_applications_have_system_hierarchy(self, scenario_08_data):
        """Every application must carry a system_hierarchy dict (regression for GEN-11)."""
        for app in scenario_08_data["applications"]:
            assert "system_hierarchy" in app, (
                f"app {app['id']} missing system_hierarchy"
            )
            hier = app["system_hierarchy"]
            for key in ("component_name", "config_item_name", "domain_name", "system_name"):
                assert key in hier, f"app {app['id']} system_hierarchy missing {key!r}"

    def test_determinism(self):
        """Two independent runs with the same config must produce identical output."""
        config = load_config(_SCENARIO_08_YAML)
        run_a = GenerationService(config=config).generate()
        run_b = GenerationService(config=config).generate()
        assert _canonical_sha256(run_a) == _canonical_sha256(run_b), (
            "Generator is not deterministic: two runs with seed=8008 produced different output"
        )

    def test_golden_checksum(self, scenario_08_data):
        """Canonical SHA-256 must match the stored golden value.

        A mismatch means a generator change altered the scenario_08 output.
        If the change is intentional, update _GOLDEN_SHA256 in this file.
        """
        actual = _canonical_sha256(scenario_08_data)
        assert actual == _GOLDEN_SHA256, (
            f"scenario_08 output has changed.\n"
            f"  expected: {_GOLDEN_SHA256}\n"
            f"  actual:   {actual}\n"
            "If this change is intentional, update _GOLDEN_SHA256 and the "
            "golden count dicts in test_generation_service.py."
        )
