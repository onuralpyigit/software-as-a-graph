
import pytest
from saag.infrastructure.memory_repo import MemoryRepository
from saag.core.models import GraphData, ComponentData

def test_memory_repository_fidelity_lossy():
    """
    Verify that MemoryRepository.export_json() filters out unsupported fields,
    matching the behavior of the real persistence layer.
    """
    repo = MemoryRepository()
    
    # Create input with "garbage" fields
    input_data = {
        "metadata": {
            "seed": 42,
            "extra_garbage_meta": "should be gone"
        },
        "nodes": [
            {
                "id": "node-1",
                "name": "Node 1",
                "cpu_cores": 4,
                "extra_garbage_field": "secret_data"
            }
        ],
        "brokers": [],
        "topics": [],
        "applications": [],
        "libraries": [],
        "relationships": {
            "runs_on": [], "routes": [], "publishes_to": [],
            "subscribes_to": [], "connects_to": [], "uses": []
        }
    }
    
    # Save to memory repo
    repo.save_graph(input_data, clear=True)
    
    # Export from memory repo
    exported = repo.export_json()
    
    # Assertions for PARITY
    # 1. Metadata should be normalized (garbage meta gone)
    assert "seed" in exported["metadata"]
    assert exported["metadata"]["seed"] == 42
    assert "extra_garbage_meta" not in exported["metadata"]
    
    # 2. Node should be normalized (garbage field gone)
    node = exported["nodes"][0]
    assert node["id"] == "node-1"
    assert node["cpu_cores"] == 4
    assert "extra_garbage_field" not in node
    
    # 3. Verify internal state is also normalized (shows that normalization happens on save)
    assert "extra_garbage_field" not in repo.data["nodes"][0]

def test_reconstruction_consistency():
    """
    Verify that the reconstruction logic correctly handles nested objects like code_metrics.
    """
    repo = MemoryRepository()
    input_data = {
        "metadata": {},
        "nodes": [], "brokers": [], "topics": [], "libraries": [],
        "applications": [
            {
                "id": "app-1",
                "role": ["worker"],
                "code_metrics": {
                    "size": {"total_loc": 1000},
                    "quality": {"bugs": 5}
                },
                "extra": "ignore"
            }
        ],
        "relationships": {}
    }
    
    repo.save_graph(input_data, clear=True)
    exported = repo.export_json()
    
    app = exported["applications"][0]
    assert app["id"] == "app-1"
    assert app["role"] == ["worker"]
    assert "code_metrics" in app
    assert app["code_metrics"]["size"]["total_loc"] == 1000
    assert app["code_metrics"]["quality"]["bugs"] == 5
    assert "extra" not in app


def test_repository_fidelity_parity_all_rules():
    """
    Verify weight calculation, Step 1.5 pass, and DEPENDS_ON derivation parity
    between MemoryRepository and Neo4jRepository.
    """
    # Define a complex topology with libraries, applications, topics, nodes, brokers
    graph_data = {
        "metadata": {
            "seed": 1234
        },
        "nodes": [
            {"id": "node_a", "name": "Node A"},
            {"id": "node_b", "name": "Node B"}
        ],
        "brokers": [
            {"id": "broker_1", "name": "Broker 1"}
        ],
        "topics": [
            {
                "id": "topic_critical",
                "size": 1024,
                "qos": {"reliability": "RELIABLE", "durability": "PERSISTENT", "transport_priority": "URGENT"}
            },
            {
                "id": "topic_standard",
                "size": 256,
                "qos": {"reliability": "BEST_EFFORT", "durability": "VOLATILE", "transport_priority": "MEDIUM"}
            }
        ],
        "applications": [
            {"id": "app_pub", "name": "Publisher App"},
            {"id": "app_sub", "name": "Subscriber App"},
            {"id": "app_isolated_uses_lib", "name": "Isolated App"}  # No direct topics, uses lib_1
        ],
        "libraries": [
            {"id": "lib_1", "name": "Shared Library"}
        ],
        "relationships": {
            "publishes_to": [
                {"from": "app_pub", "to": "topic_critical"},
                {"from": "lib_1", "to": "topic_standard"}
            ],
            "subscribes_to": [
                {"from": "app_sub", "to": "topic_critical"},
                {"from": "lib_1", "to": "topic_standard"}
            ],
            "uses": [
                {"from": "app_isolated_uses_lib", "to": "lib_1"}
            ],
            "routes": [
                {"from": "broker_1", "to": "topic_standard"}
            ],
            "runs_on": [
                {"from": "app_pub", "to": "node_a"},
                {"from": "app_sub", "to": "node_b"},
                {"from": "broker_1", "to": "node_a"}
            ]
        }
    }

    # Instantiate both repositories
    mem_repo = MemoryRepository()
    mem_repo.save_graph(graph_data, clear=True)
    mem_repo.derive_dependencies()

    # Get data from memory repository
    mem_data = mem_repo.get_graph_data(include_raw=True)

    # Let's inspect the weights in memory repo
    comp_by_id_mem = {c.id: c for c in mem_data.components}
    
    # Topic Standard: QoS(BEST_EFFORT/VOLATILE/MEDIUM) ~= 0.0462, Size = 256B,
    #   no explicit frequency (TOPIC_DEFAULT_FREQUENCY_HZ fallback) -> w ~= 0.10473
    # Topic Critical: QoS = 1.0, Size = 1024B, no explicit frequency -> w ~= 0.83504
    # (values via compute_topic_weight, TOPIC_SIZE_ENVELOPE_BYTES = 2**20,
    # QoSPolicy.W_RELIABILITY/W_DURABILITY/W_PRIORITY = 0.24/0.62/0.14)
    assert comp_by_id_mem["topic_standard"].weight == pytest.approx(0.10473, abs=0.001)
    assert comp_by_id_mem["topic_critical"].weight == pytest.approx(0.83504, abs=0.001)

    # App Pub: connected to topic_critical. Weight = 0.83504
    # App Sub: connected to topic_critical. Weight = 0.83504
    assert comp_by_id_mem["app_pub"].weight == pytest.approx(0.83504, abs=0.001)
    assert comp_by_id_mem["app_sub"].weight == pytest.approx(0.83504, abs=0.001)

    # Lib 1: connected to topic_standard (0.10473), dg_in = 1.
    # Smooth saturating boost: 1 - (1 - 0.10473) * 2^(-0.15) ~= 0.19325
    assert comp_by_id_mem["lib_1"].weight == pytest.approx(0.19325, abs=0.001)

    # App Isolated Uses Lib: updated to max(lib_1.weight) = 0.19325
    assert comp_by_id_mem["app_isolated_uses_lib"].weight == pytest.approx(0.19325, abs=0.001)

    # Broker 1: routes topic_standard. Weight = 0.10473
    assert comp_by_id_mem["broker_1"].weight == pytest.approx(0.10473, abs=0.001)

    # Node A: hosts app_pub (0.83504) and broker_1 (0.10473) -> max = 0.83504
    # Node B: hosts app_sub (0.83504) -> max = 0.83504
    assert comp_by_id_mem["node_a"].weight == pytest.approx(0.83504, abs=0.001)
    assert comp_by_id_mem["node_b"].weight == pytest.approx(0.83504, abs=0.001)

    # Let's inspect DEPENDS_ON relationships
    mem_dep_data = mem_repo.get_graph_data()
    edges_by_type = {}
    for e in mem_dep_data.edges:
        edges_by_type.setdefault(e.dependency_type, []).append(e)

    # Rule 1 app_to_app:
    assert "app_to_app" in edges_by_type
    app_to_app = edges_by_type["app_to_app"]
    assert len(app_to_app) == 3
    
    app_to_app_map = {(e.source_id, e.target_id): e for e in app_to_app}
    assert ("app_sub", "app_pub") in app_to_app_map
    e1 = app_to_app_map[("app_sub", "app_pub")]
    assert e1.weight == pytest.approx(0.83504, abs=0.001)
    assert e1.path_count == 1

    assert ("app_isolated_uses_lib", "lib_1") in app_to_app_map
    e2 = app_to_app_map[("app_isolated_uses_lib", "lib_1")]
    assert e2.weight == pytest.approx(0.10473, abs=0.001)
    assert e2.path_count == 1

    assert ("lib_1", "app_isolated_uses_lib") in app_to_app_map
    e3 = app_to_app_map[("lib_1", "app_isolated_uses_lib")]
    assert e3.weight == pytest.approx(0.10473, abs=0.001)
    assert e3.path_count == 1

    # Rule 3 node_to_node:
    assert "node_to_node" in edges_by_type
    node_to_node = edges_by_type["node_to_node"]
    assert len(node_to_node) == 1
    assert node_to_node[0].source_id == "node_b"
    assert node_to_node[0].target_id == "node_a"
    assert node_to_node[0].weight == pytest.approx(0.83504, abs=0.001)

    # Rule 5 app_to_lib:
    # Harmonic mean between app_isolated_uses_lib (0.19325) and lib_1 (0.19325) = 0.19325
    assert "app_to_lib" in edges_by_type
    app_to_lib = edges_by_type["app_to_lib"]
    assert len(app_to_lib) == 1
    assert app_to_lib[0].source_id == "app_isolated_uses_lib"
    assert app_to_lib[0].target_id == "lib_1"
    assert app_to_lib[0].weight == pytest.approx(0.19325, abs=0.001)

    # Now let's run comparison with Neo4jRepository if Neo4j is available
    from saag.infrastructure.neo4j_repo import Neo4jRepository
    neo_repo = Neo4jRepository(uri="bolt://localhost:7687", user="neo4j", password="password")
    try:
        neo_repo._run_query("RETURN 1")
    except Exception:
        # Skip Neo4j parity check if Neo4j is not available
        neo_repo.close()
        return

    try:
        neo_repo.save_graph(graph_data, clear=True)
        neo_repo.derive_dependencies()
        neo_data = neo_repo.get_graph_data(include_raw=True)

        # Compare components
        comp_by_id_neo = {c.id: c for c in neo_data.components}
        assert len(comp_by_id_mem) == len(comp_by_id_neo)
        for cid, mem_c in comp_by_id_mem.items():
            neo_c = comp_by_id_neo[cid]
            assert mem_c.component_type == neo_c.component_type
            assert mem_c.weight == pytest.approx(neo_c.weight, abs=0.001)

        # Compare edges
        edges_by_key_mem = {(e.source_id, e.target_id, e.dependency_type, e.relation_type): e for e in mem_data.edges}
        edges_by_key_neo = {(e.source_id, e.target_id, e.dependency_type, e.relation_type): e for e in neo_data.edges}
        assert len(edges_by_key_mem) == len(edges_by_key_neo)
        for key, mem_e in edges_by_key_mem.items():
            assert key in edges_by_key_neo
            neo_e = edges_by_key_neo[key]
            assert mem_e.weight == pytest.approx(neo_e.weight, abs=0.001)
            # path_count is only set/compared on DEPENDS_ON edges
            if mem_e.relation_type == "DEPENDS_ON":
                assert mem_e.path_count == neo_e.path_count

    finally:
        neo_repo.close()


def test_memory_repository_statistics_schema_parity():
    """MemoryRepository.get_statistics() matches Neo4jRepository key names."""
    repo = MemoryRepository()
    repo.save_graph(
        {
            "nodes": [{"id": "N1"}],
            "network_devices": [{"id": "SW1", "role": "switch"}],
            "brokers": [{"id": "B1"}],
            "topics": [{"id": "T1"}],
            "applications": [{"id": "A1"}],
            "libraries": [{"id": "L1"}],
            "relationships": {
                "publishes_to": [{"from": "A1", "to": "T1"}],
                "routes": [{"from": "B1", "to": "T1"}],
                "runs_on": [{"from": "A1", "to": "N1"}, {"from": "B1", "to": "N1"}],
                "connects_to": [{"from": "N1", "to": "SW1"}],
                "uses": [{"from": "A1", "to": "L1"}],
            },
        },
        clear=True,
    )
    repo.derive_dependencies()
    stats = repo.get_statistics()

    assert stats["library_count"] == 1
    assert "librarie_count" not in stats
    assert stats["networkdevice_count"] == 1
    assert stats["node_count"] == 1
    assert stats["broker_count"] == 1
    assert stats["topic_count"] == 1
    assert stats["application_count"] == 1
    assert stats["total_nodes"] == 6
    assert stats["publishes_to_count"] == 1
    assert stats["uses_count"] == 1
    assert stats["app_to_lib_count"] == 1
    assert stats["app_to_broker_count"] == 1
    assert stats["node_to_broker_count"] == 1


def test_isolated_app_to_lib_does_not_crash_finalization():
    """Topic-free lib->lib (weight 0.01) and isolated app->lib both finalize without crashing."""
    repo = MemoryRepository()
    repo.save_graph(
        {
            "nodes": [],
            "brokers": [],
            "topics": [],
            "applications": [{"id": "A_isolated"}],
            "libraries": [{"id": "L_a"}, {"id": "L_b"}, {"id": "L_util"}],
            "relationships": {
                "uses": [
                    {"from": "L_a", "to": "L_b"},
                    {"from": "A_isolated", "to": "L_util"},
                ],
            },
        },
        clear=True,
    )
    repo.derive_dependencies()
    from saag.core.models import compute_library_weight

    deps = {(e.source_id, e.target_id): e for e in repo.get_graph_data().edges}
    assert deps[("L_a", "L_b")].dependency_type == "app_to_lib"
    assert deps[("L_a", "L_b")].weight == pytest.approx(0.01)
    assert deps[("A_isolated", "L_util")].dependency_type == "app_to_lib"
    assert deps[("A_isolated", "L_util")].weight == pytest.approx(compute_library_weight(0.01, 1))


def test_rule1_multipath_distinct_equal_weight_topics_union():
    """Distinct mediating topics across direct & library paths union even when weights are equal."""
    from saag.core.models import compute_effective_edge_weight

    repo = MemoryRepository()
    repo.save_graph(
        {
            "nodes": [],
            "brokers": [],
            "topics": [
                {"id": "T1", "size": 256, "qos": {"reliability": "RELIABLE", "durability": "VOLATILE", "transport_priority": "HIGH"}},
                {"id": "T2", "size": 256, "qos": {"reliability": "RELIABLE", "durability": "VOLATILE", "transport_priority": "HIGH"}},
            ],
            "applications": [{"id": "PUB"}, {"id": "SUB"}],
            "libraries": [{"id": "L1"}],
            "relationships": {
                "publishes_to": [{"from": "PUB", "to": "T1"}, {"from": "PUB", "to": "T2"}],
                "subscribes_to": [{"from": "SUB", "to": "T1"}, {"from": "L1", "to": "T2"}],
                "uses": [{"from": "SUB", "to": "L1"}],
            },
        },
        clear=True,
    )
    repo.derive_dependencies()
    comps = {c.id: c.weight for c in repo.get_graph_data().components}
    expected = compute_effective_edge_weight([comps["T1"], comps["T2"]])
    edge = next(
        e for e in repo.get_graph_data().edges
        if (e.source_id, e.target_id, e.dependency_type) == ("SUB", "PUB", "app_to_app")
    )
    assert edge.path_count == 2
    assert edge.weight == pytest.approx(expected, abs=1e-9)


