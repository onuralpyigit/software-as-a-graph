"""Unit tests for TrafficSimulator backed by IGraphRepository (MemoryRepository)."""

from __future__ import annotations

import pytest

from saag.infrastructure.memory_repo import MemoryRepository
from saag.simulation.traffic_simulator import TrafficSimulator


def _sample_topology():
    return {
        "nodes": [{"id": "N0", "name": "host-0"}],
        "brokers": [
            {"id": "B0", "name": "broker-primary"},
            {"id": "B1", "name": "broker-secondary"},
        ],
        "topics": [
            {
                "id": "T0",
                "name": "/sensor/imu",
                "size": 256,
                "frequency": 20.0,
                "criticality": "HIGH",
                "deadline_ms": 50.0,
                "history_depth": 15,
                "qos": {
                    "reliability": "RELIABLE",
                    "durability": "TRANSIENT_LOCAL",
                    "transport_priority": "HIGH",
                },
            },
            {
                "id": "T1",
                "name": "/telemetry/log",
                "size": 0,
                "qos": {
                    "reliability": "BEST_EFFORT",
                    "durability": "VOLATILE",
                    "transport_priority": "LOW",
                },
            },
        ],
        "applications": [
            {"id": "A0", "name": "ImuDriver", "role": "pub", "criticality": "HIGH"},
            {"id": "A1", "name": "FusionNode", "role": "pubsub", "hotstandby": True},
            {"id": "A2", "name": "Logger", "role": "sub"},
        ],
        "libraries": [],
        "relationships": {
            "runs_on": [{"from": c, "to": "N0"} for c in ("A0", "A1", "A2", "B0", "B1")],
            "routes": [
                {"from": "B0", "to": "T0"},
                {"from": "B1", "to": "T0"},
                {"from": "B0", "to": "T1"},
            ],
            "publishes_to": [
                {"from": "A0", "to": "T0"},
                {"from": "A1", "to": "T1"},
            ],
            "subscribes_to": [
                {"from": "A1", "to": "T0"},
                {"from": "A2", "to": "T0"},
                {"from": "A2", "to": "T1"},
            ],
            "connects_to": [],
            "uses": [],
        },
    }


@pytest.fixture
def traffic_sim():
    repo = MemoryRepository()
    repo.save_graph(_sample_topology(), clear=True)
    return TrafficSimulator(repo)


def test_get_all_apps_returns_sorted_apps_with_pub_sub_topics(traffic_sim):
    apps = traffic_sim.get_all_apps()
    assert [a["name"] for a in apps] == ["FusionNode", "ImuDriver", "Logger"]

    by_id = {a["id"]: a for a in apps}
    assert by_id["A0"]["pub_topic_ids"] == ["T0"]
    assert by_id["A0"]["sub_topic_ids"] == []
    assert by_id["A1"]["pub_topic_ids"] == ["T1"]
    assert by_id["A1"]["sub_topic_ids"] == ["T0"]
    assert by_id["A1"]["hotstandby"] is True
    assert by_id["A2"]["sub_topic_ids"] == ["T0", "T1"]


def test_get_all_topics_preserves_qos_and_counts(traffic_sim):
    topics = traffic_sim.get_all_topics()
    assert [t["id"] for t in topics] == ["T0", "T1"]

    t0 = topics[0]
    assert t0["publisher_count"] == 1
    assert t0["subscriber_count"] == 2
    assert t0["broker_ids"] == ["B0", "B1"]
    assert t0["broker_names"] == ["broker-primary", "broker-secondary"]
    assert t0["qos_reliability"] == "RELIABLE"
    assert t0["qos_durability"] == "TRANSIENT_LOCAL"
    assert t0["qos_transport_priority"] == "HIGH"
    assert t0["size"] == 256
    assert t0["frequency"] == pytest.approx(20.0)
    assert t0["deadline_ms"] == pytest.approx(50.0)
    assert t0["history_depth"] == 15


def test_simulate_computes_rates_bandwidth_and_broker_usage(traffic_sim):
    res = traffic_sim.simulate(
        topic_ids=["T0", "T1"],
        frequency_hz=5.0,
        duration_sec=10.0,
        message_size_bytes=512,
        per_topic_params={"T1": {"frequency_hz": 2.0, "message_size_bytes": 100}},
    )

    by_topic = {t["topic_id"]: t for t in res["per_topic"]}
    # T0: 1 pub * 20 Hz = 20 in/s; 2 subs -> 40 out/s; size 256
    assert by_topic["T0"]["msgs_published_per_sec"] == pytest.approx(20.0)
    assert by_topic["T0"]["msgs_delivered_per_sec"] == pytest.approx(40.0)
    assert by_topic["T0"]["bandwidth_total_bps"] == pytest.approx(60.0 * 256)

    # T1: overridden to 2 Hz, 100 bytes; 1 pub, 1 sub
    assert by_topic["T1"]["msgs_published_per_sec"] == pytest.approx(2.0)
    assert by_topic["T1"]["msgs_delivered_per_sec"] == pytest.approx(2.0)
    assert by_topic["T1"]["bandwidth_total_bps"] == pytest.approx(4.0 * 100)

    # B0 routes both T0 and T1; B1 routes only T0
    brokers = {b["broker_id"]: b for b in res["broker_usage"]}
    assert brokers["B0"]["topics_routed"] == ["T0", "T1"]
    assert brokers["B1"]["topics_routed"] == ["T0"]
    assert brokers["B0"]["msgs_total_per_sec"] > brokers["B1"]["msgs_total_per_sec"]
    assert res["summary"]["brokers_involved"] == 2


def test_simulate_empty_topic_list(traffic_sim):
    res = traffic_sim.simulate(topic_ids=[], frequency_hz=10.0, duration_sec=5.0)
    assert res["summary"]["selected_topics"] == 0
    assert res["per_topic"] == []
    assert res["broker_usage"] == []
