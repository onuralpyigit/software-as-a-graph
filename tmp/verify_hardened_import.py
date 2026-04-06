import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from src.infrastructure.neo4j_repo import create_repository

logging.basicConfig(level=logging.INFO)

def test_hardened_import():
    repo = create_repository()
    
    # Payload with missing entity in relationships
    # App A exists, but Topic T99 does NOT exist in the topics list.
    bad_data = {
        "nodes": [],
        "brokers": [],
        "topics": [
            {"id": "T1", "name": "Topic 1"}
        ],
        "applications": [
            {"id": "A1", "name": "App 1"}
        ],
        "libraries": [],
        "relationships": {
            "publishes_to": [
                {"source": "A1", "target": "T99"} # T99 MISSING!
            ]
        }
    }
    
    print("--- Test 1: Import with missing target entity ---")
    try:
        # This SHOULD fail because T99 is missing
        repo.save_graph(bad_data, clear=True)
        print("FAILED: save_graph should have raised ValueError for missing T99")
    except ValueError as e:
        print(f"SUCCESS: Caught expected error: {e}")
        
    # Verify that nothing was imported (rollback check)
    stats = repo.driver.execute_query("MATCH (n) RETURN count(n) as count")[0][0]['count']
    if stats == 0:
        print("SUCCESS: Database is empty (Rollback confirmed)")
    else:
        print(f"FAILED: Database contains {stats} nodes. Rollback FAILED.")

    print("\n--- Test 2: Import with missing source entity ---")
    bad_data_2 = {
        "nodes": [],
        "brokers": [],
        "topics": [{"id": "T1"}],
        "applications": [],
        "libraries": [],
        "relationships": {
            "publishes_to": [{"source": "A_MISSING", "target": "T1"}]
        }
    }
    try:
        repo.save_graph(bad_data_2, clear=True)
        print("FAILED: save_graph should have raised ValueError for missing A_MISSING")
    except ValueError as e:
        print(f"SUCCESS: Caught expected error: {e}")

    print("\n--- Test 3: Normal Import ---")
    good_data = {
        "nodes": [],
        "brokers": [],
        "topics": [{"id": "T1"}],
        "applications": [{"id": "A1"}],
        "libraries": [],
        "relationships": {
            "publishes_to": [{"source": "A1", "target": "T1"}]
        }
    }
    try:
        repo.save_graph(good_data, clear=True)
        print("SUCCESS: Normal import worked.")
        stats = repo.driver.execute_query("MATCH (n) RETURN count(n) as count")[0][0]['count']
        rels = repo.driver.execute_query("MATCH ()-[r]->() RETURN count(r) as count")[0][0]['count']
        print(f"Nodes: {stats}, Rels: {rels}")
    except Exception as e:
        print(f"FAILED: Normal import failed: {e}")

    repo.close()

if __name__ == "__main__":
    test_hardened_import()
