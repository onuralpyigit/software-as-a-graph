import sys
import os
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from saag import Client
from src.usecases.models import ImportStats

def main():
    # Mock Neo4j credentials or just use a dummy client since we won't call save_graph if we match logic correctly
    # Actually ModelGraphUseCase.execute(graph_data, clear, dry_run)
    # If I pass dry_run=True, it won't touch the DB.
    
    client = Client(neo4j_uri="bolt://localhost:7687", user="neo4j", password="password")
    
    print("Testing empty dict import (dry_run=True)...")
    try:
        # This used to raise ValueError because if not graph_data: was True for {}
        result = client.import_topology(graph_data={}, dry_run=True)
        print(f"Success! Result type: {type(result)}")
        print(f"Result details: {result.to_dict()}")
        
        if not isinstance(result, ImportStats):
            print("ERROR: Result is not an ImportStats object!")
            sys.exit(1)
            
    except ValueError as e:
        print(f"FAILED: ValueError raised: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"FAILED: Unexpected error: {e}")
        sys.exit(1)

    print("Verification complete.")

if __name__ == "__main__":
    main()
