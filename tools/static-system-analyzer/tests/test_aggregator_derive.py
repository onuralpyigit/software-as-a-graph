"""Tests for QoS-derived topic attributes and random application attributes
added in the aggregator service."""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipeline.aggregator.models import Topic
from pipeline.aggregator.parsers import SystemRepoParser
from pipeline.aggregator.service import (
    _APP_HOTSTANDBY_OPTIONS,
    _APP_PRIORITY_OPTIONS,
    _create_apps_libs_and_relations,
    _create_topics,
    _derive_topic_criticality,
    _derive_topic_frequency,
)


# --- _derive_topic_frequency ------------------------------------------------

def test_frequency_max_for_reliable_urgent():
    # combined = 1.0 * 1.0 -> last bin
    assert _derive_topic_frequency("RELIABLE", "URGENT") == 200.0


def test_frequency_zero_score_yields_first_bin():
    # BEST_EFFORT -> reliability score 0 -> combined 0 -> first bin (1.0 Hz)
    assert _derive_topic_frequency("BEST_EFFORT", "URGENT") == 1.0
    assert _derive_topic_frequency("RELIABLE", "LOW") == 1.0


def test_frequency_reliable_high():
    # combined = 1.0 * 0.66 = 0.66 -> bin_idx = int(0.66*16) = 10 -> 100.0
    assert _derive_topic_frequency("RELIABLE", "HIGH") == 100.0


def test_frequency_unknown_values_default_to_zero():
    assert _derive_topic_frequency("NOT_FOUND", "NOT_FOUND") == 1.0


# --- _derive_topic_criticality ----------------------------------------------

def test_criticality_minimal_for_all_lowest():
    # score 0.0 -> first threshold (0.00, minimal)
    assert _derive_topic_criticality("VOLATILE", "BEST_EFFORT", "LOW") == "minimal"


def test_criticality_critical_for_all_highest():
    # score = 0.3 + 0.4 + 0.3 = 1.0 -> critical
    assert _derive_topic_criticality("PERSISTENT", "RELIABLE", "URGENT") == "critical"


def test_criticality_medium_band():
    # score = 0.40*1.0 (PERSISTENT) = 0.40 -> <= 0.43 -> medium
    assert _derive_topic_criticality("PERSISTENT", "BEST_EFFORT", "LOW") == "medium"


def test_criticality_high_band():
    # score = 0.40 (dur) + 0.30 (rel) = 0.70 -> > 0.64 -> critical? No: 0.70 > 0.64
    # thresholds ascending; 0.70 > 1.00 is False so 'critical' via <= 1.00 -> 'critical'
    # Use a value landing in high band: dur PERSISTENT(0.4) + pri HIGH(0.66*0.3=0.198)=0.598 -> high
    assert _derive_topic_criticality("PERSISTENT", "BEST_EFFORT", "HIGH") == "high"


def test_criticality_unknown_values_default_minimal():
    assert _derive_topic_criticality("NOT_FOUND", "NOT_FOUND", "NOT_FOUND") == "minimal"


# --- _create_topics integration ---------------------------------------------

def test_create_topics_adds_frequency_and_criticality():
    topic_set = {
        Topic(
            name="T-high",
            size=100,
            durability="PERSISTENT",
            reliability="RELIABLE",
            transport_priority="URGENT",
        ),
        Topic(
            name="T-low",
            size=100,
            durability="VOLATILE",
            reliability="BEST_EFFORT",
            transport_priority="LOW",
        ),
    }
    topics, _ = _create_topics(topic_set, dds_mask=False)
    by_name = {t["name"]: t for t in topics}

    assert by_name["T-high"]["frequency"] == 200.0
    assert by_name["T-high"]["criticality"] == "critical"
    assert by_name["T-low"]["frequency"] == 1.0
    assert by_name["T-low"]["criticality"] == "minimal"


# --- application priority / hotstandby ---------------------------------------

def _build_apps():
    app_node_rel = [("App-0", "Node-0"), ("App-1", "Node-1"), ("App-2", "Node-0")]
    apps, *_ = _create_apps_libs_and_relations(
        app_node_relations=app_node_rel,
        csv_data=[],
        topic_map={},
        app_role_map={},
        app_criticality_map={},
        dds_mask=False,
    )
    return apps


def test_apps_have_priority_and_hotstandby_fields():
    for app in _build_apps():
        assert app["priority"] in _APP_PRIORITY_OPTIONS
        assert isinstance(app["hotstandby"], bool)
        assert app["hotstandby"] in _APP_HOTSTANDBY_OPTIONS


def test_app_priority_options_are_low_medium_high():
    assert set(_APP_PRIORITY_OPTIONS) == {"LOW", "MEDIUM", "HIGH"}


def test_app_attributes_are_reproducible_by_name():
    first = {a["name"]: (a["priority"], a["hotstandby"]) for a in _build_apps()}
    second = {a["name"]: (a["priority"], a["hotstandby"]) for a in _build_apps()}
    assert first == second


# --- application role (multi-role support) ----------------------------------

def _apps_with_roles(app_role_map):
    app_node_rel = [("App-0", "Node-0"), ("App-1", "Node-1"), ("App-2", "Node-0")]
    apps, *_ = _create_apps_libs_and_relations(
        app_node_relations=app_node_rel,
        csv_data=[],
        topic_map={},
        app_role_map=app_role_map,
        app_criticality_map={},
        dds_mask=False,
    )
    return {a["name"]: a["role"] for a in apps}


def test_app_role_defaults_to_not_found_list_when_missing():
    roles_by_app = _apps_with_roles({})
    for name, roles in roles_by_app.items():
        assert isinstance(roles, list), f"{name}: role must be a list"
        assert roles == ["NOT_FOUND"]


def test_app_role_preserves_single_role_as_list():
    roles_by_app = _apps_with_roles({"App-0": ["publisher"]})
    assert roles_by_app["App-0"] == ["publisher"]
    assert roles_by_app["App-1"] == ["NOT_FOUND"]


def test_app_role_preserves_multiple_roles():
    mapping = {
        "App-0": ["publisher", "monitor"],
        "App-2": ["publisher", "logger", "monitor"],
    }
    roles_by_app = _apps_with_roles(mapping)
    assert roles_by_app["App-0"] == ["publisher", "monitor"]
    assert roles_by_app["App-2"] == ["publisher", "logger", "monitor"]
    assert roles_by_app["App-1"] == ["NOT_FOUND"]


def test_app_role_is_copied_not_aliased():
    shared = ["publisher", "monitor"]
    mapping = {"App-0": shared}
    roles_by_app = _apps_with_roles(mapping)
    roles_by_app["App-0"].append("mutated")
    assert shared == ["publisher", "monitor"]


def test_system_repo_parser_get_app_role_relation_returns_lists():
    parser = SystemRepoParser(platform_name="mock-platform")
    result = parser.get_app_role_relation()
    assert isinstance(result, dict)
    assert result, "mock data should not be empty"
    for app_name, roles in result.items():
        assert isinstance(app_name, str)
        assert isinstance(roles, list)
        assert all(isinstance(r, str) for r in roles)
        assert roles, f"{app_name}: role list must not be empty"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
