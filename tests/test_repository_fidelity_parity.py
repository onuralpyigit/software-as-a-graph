
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
                "role": "worker",
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
    assert app["role"] == "worker"
    assert "code_metrics" in app
    assert app["code_metrics"]["size"]["total_loc"] == 1000
    assert app["code_metrics"]["quality"]["bugs"] == 5
    assert "extra" not in app
