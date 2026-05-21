import hashlib
import json
import pytest
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

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
#     d = GenerationService(config=load_config(Path('data/scenario_08_tiny_regression.yaml'))).generate()
#     print(hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest())
#   "
# ---------------------------------------------------------------------------

_SCENARIO_08_YAML = project_root / "data" / "scenarios" / "scenario_08_tiny_regression.yaml"

_GOLDEN_SHA256 = "6df351bd76bb6c039748b3d790e9199800799020d211d23a1439173b08c19104"

_GOLDEN_ENTITY_COUNTS = {
    "nodes": 3,
    "brokers": 2,
    "topics": 8,
    "applications": 12,
    "libraries": 4,
}

_GOLDEN_RELATIONSHIP_COUNTS = {
    "runs_on": 14,
    "routes": 8,
    "publishes_to": 23,
    "subscribes_to": 29,
    "uses": 13,
    "connects_to": 1,
}


class TestGenerationService:
    def test_generate_tiny(self):
        gen = GenerationService(scale="tiny", seed=1)
        data = gen.generate()

        assert len(data["nodes"]) == 2
        assert len(data["brokers"]) == 1
        assert len(data["topics"]) == 5
        assert len(data["applications"]) == 5

    def test_connection_density(self):
        # 1. Test connection_density = 0.0 (only the fallback/guard edge should be present)
        gen_sparse = GenerationService(scale="medium", seed=42, connection_density=0.0)
        data_sparse = gen_sparse.generate()
        connects_sparse = data_sparse["relationships"]["connects_to"]
        assert len(connects_sparse) == 1

        # 2. Test connection_density = 1.0 (should have a complete graph of connects_to edges)
        gen_dense = GenerationService(scale="medium", seed=42, connection_density=1.0)
        data_dense = gen_dense.generate()
        connects_dense = data_dense["relationships"]["connects_to"]
        num_nodes = len(data_dense["nodes"])
        expected_edges = num_nodes * (num_nodes - 1) // 2
        assert len(connects_dense) == expected_edges


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
            for key in ("csc_name", "csci_name", "css_name", "csms_name"):
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


class TestTopicDerivedFields:
    """Regression for topic frequency and criticality derived from QoS during generation."""

    def _cases():
        # (reliability, durability, priority) -> (expected_criticality, expected_freq)
        # Thresholds: ≤0.19 minimal, ≤0.43 low, ≤0.64 medium, ≤1.00 high, >1.00 critical
        return [
            # RELIABLE × MEDIUM  → combined=0.330 bin 5 → 20 Hz
            ("RELIABLE",    "VOLATILE",   "MEDIUM",   "medium",   20.0),
            # RELIABLE × HIGH   → combined=0.660 bin 10 → 100 Hz
            ("RELIABLE",    "VOLATILE",   "HIGH",     "high",    100.0),
        ]

    @pytest.mark.parametrize("rel,dur,pri,exp_crit,exp_hz", _cases())
    def test_derived_fields_match_qos(self, rel, dur, pri, exp_crit, exp_hz):
        from saag.core.models import Topic, QoSPolicy, CRITICALITY_THRESHOLDS, TOPIC_FREQUENCY_HZ
        t = Topic(id=f"T-rel{rel}-pri{pri}", name=f"test.{rel.lower()}.{pri.lower()}",
                  size=1024, qos=QoSPolicy(reliability=rel, durability=dur,
                                            transport_priority=pri))

        # 1. Frequency must match the bin-lookup table used in __post_init__
        r = QoSPolicy.RELIABILITY_SCORES.get(rel, 0.0)
        p = QoSPolicy.PRIORITY_SCORES.get(pri, 0.0)
        combined = r * p
        bin_idx = int(combined * len(TOPIC_FREQUENCY_HZ))
        bin_idx = max(0, min(bin_idx, len(TOPIC_FREQUENCY_HZ) - 1))
        expected_hz = float(TOPIC_FREQUENCY_HZ[bin_idx])
        assert t.frequency == expected_hz

        # 2. Criticality must match the threshold table in __post_init__
        qos_score = t.qos.calculate_weight()
        expected_crit = "critical"
        for threshold, label in CRITICALITY_THRESHOLDS:
            if qos_score <= threshold:
                expected_crit = label
                break
        assert t.criticality == expected_crit

    def test_to_dict_contains_derived_fields(self):
        from saag.core.models import Topic, QoSPolicy
        t = Topic(id="T0", name="X", size=256,
                  qos=QoSPolicy(reliability="RELIABLE", durability="PERSISTENT",
                                transport_priority="HIGH"))
        d = t.to_dict()
        assert "frequency" in d, "to_dict() missing 'frequency'"
        assert "criticality" in d, "to_dict() missing 'criticality'"
        assert isinstance(d["frequency"], (int, float))
        assert d["criticality"] in {"critical", "high", "medium", "low", "minimal"}

    def test_generated_topics_have_derived_fields(self):
        gen = GenerationService(scale="tiny", seed=7)
        data = gen.generate()
        for topic in data["topics"]:
            assert "frequency" in topic, f"Topic {topic.get('id')!r} missing 'frequency'"
            assert "criticality" in topic, f"Topic {topic.get('id')!r} missing 'criticality'"
            assert topic["criticality"] in {"critical", "high", "medium", "low", "minimal"}, (
                f"Topic {topic.get('id')!r}: invalid criticality {topic['criticality']!r}"
            )
            assert isinstance(topic["frequency"], (int, float))
            assert topic["frequency"] > 0.0
