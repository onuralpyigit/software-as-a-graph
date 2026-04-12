
import sys
import os
import json
import numpy as np

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from api.statistics import DescriptiveStats, CategoricalStats, extract_cross_cutting_data, compute_all_extras_statistics
from api.presenters.statistics_presenter import serialise_numpy

def test_serialization():
    print("Testing serialization...")
    data = {
        "arr": np.array([1, 2, 3]),
        "int": np.int64(10),
        "float": np.float64(3.14),
        "nested": {"set": {1, 2, 3}}
    }
    serialised = serialise_numpy(data)
    print(f"Serialised: {serialised}")
    json.dumps(serialised) # Should not raise error
    print("Serialization test passed.")

def test_statistics_logic():
    print("Testing statistics logic...")
    raw_data = {
        "nodes": [{"id": "n1", "name": "Node 1"}],
        "applications": [{"id": "a1", "name": "App 1", "criticality": True}],
        "topics": [{"id": "t1", "name": "Topic 1", "size": 100, "qos": {"durability": "High"}}],
        "libraries": [],
        "relationships": {
            "runs_on": [{"from": "a1", "to": "n1"}],
            "publishes_to": [{"from": "a1", "to": "t1"}],
            "subscribes_to": [],
            "uses": []
        }
    }
    cc = extract_cross_cutting_data(raw_data)
    
    # Define weight function
    def weight_fn(_, val):
        return 3.0 if val == "High" else 1.0
        
    stats = compute_all_extras_statistics(cc, risk_weight_fn=weight_fn)
    print(f"Computed stats keys: {stats.keys()}")
    assert "qos_risk" in stats
    print("Statistics logic test passed.")

if __name__ == "__main__":
    try:
        test_serialization()
        test_statistics_logic()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)
