"""Round-trip guard for the Topic temporal-contract properties.

``_flatten_topic()`` emitting a property is not the same as the repository
persisting it.  When ``deadline_ms`` / ``history_depth`` were added, the
flattener started emitting them but ``Neo4jRepository._import_topics()`` still
SET only ``topic_frequency`` and ``topic_criticality``, so both fields were
silently dropped on import — and ``TrafficSimulator``'s new reads of them came
back null on every topic.  The unit tests passed throughout, because they only
covered flatten/reconstruct.
"""

import pytest

from saag.core.utils import serialization
from saag.infrastructure.neo4j_repo import Neo4jRepository

#: The Topic properties that must survive a full write→read cycle.
CONTRACT_PROPERTIES = ("topic_criticality", "qos_deadline_ms", "qos_history_depth")

_GRAPH = {
    "nodes": [{"id": "N0", "name": "host-0"}],
    "brokers": [],
    "applications": [],
    "libraries": [],
    "topics": [
        {
            "id": "T0",
            "name": "/control/cmd",
            "size": 512,
            "qos": {
                "reliability": "RELIABLE",
                "durability": "TRANSIENT_LOCAL",
                "transport_priority": "HIGH",
            },
            "frequency": 50.0,
            "criticality": "HIGH",
            "deadline_ms": 25.0,
            "history_depth": 50,
        }
    ],
    "relationships": {
        "runs_on": [], "routes": [], "publishes_to": [],
        "subscribes_to": [], "connects_to": [], "uses": [],
    },
}


def test_flattened_topic_properties_are_all_persisted():
    """Every property the flattener emits must appear in the import Cypher.

    A pure-static check, so it runs without a database: it catches the exact
    drift above — a new flattened field that no SET clause ever writes.
    """
    flat = serialization.flatten_component(_GRAPH["topics"][0], "Topic")
    import inspect
    cypher = inspect.getsource(Neo4jRepository._import_topics)
    missing = [
        key for key in CONTRACT_PROPERTIES
        if key in flat and f'"{key}"' not in cypher and f"row.{key}" not in cypher
    ]
    assert not missing, (
        f"_flatten_topic() emits {missing} but _import_topics() never writes "
        "them — they are dropped on import"
    )


@pytest.mark.integration
def test_topic_contract_survives_neo4j_roundtrip():
    """Write a topic through the repository and read its properties back."""
    from saag.infrastructure import create_repository

    try:
        repo = create_repository(
            uri="bolt://localhost:7687", user="neo4j", password="password"
        )
        repo.save_graph(_GRAPH, clear=True)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Neo4j unavailable: {exc}")

    try:
        with repo.driver.session() as session:
            record = session.run(
                "MATCH (t:Topic {id: 'T0'}) RETURN properties(t) AS props"
            ).single()
        assert record is not None, "topic T0 was not persisted"
        props = record["props"]

        assert props.get("topic_criticality") == "HIGH"
        assert props.get("qos_deadline_ms") == pytest.approx(25.0)
        assert props.get("qos_history_depth") == 50
    finally:
        repo.close()
