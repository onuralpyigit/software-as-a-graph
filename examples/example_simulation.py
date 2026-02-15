"""
Example: Running Simulations Programmatically
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from src.core import create_repository
from src.simulation import SimulationService

def main():
    # Make sure Neo4j is running with data imported
    try:
        # Initialize repository
        repo = create_repository()
        
        # Initialize Service
        sim = SimulationService(repo)
        
        try:
            # 1. Failure Sim (Simulate Node Failure)
            print("Simulating Node Failure (Node-0)...")
            
            # Get graph data directly from repository to find component IDs
            graph_data = repo.get_graph_data()
            
            # Filter for actual nodes (starts with N typically, or check type if possible)
            nodes = [c.id for c in graph_data.components if c.component_type == "Node"]
            
            if nodes:
                target_id = nodes[0]
                print(f"Targeting: {target_id}")
                
                # run_failure_simulation takes component_id (str) and layer context
                fail_res = sim.run_failure_simulation(target_id, layer="infra")
                # fail_res is FailureResult
                print(f"Cascaded Failures: {len(fail_res.cascaded_failures)}")
            else:
                print("No nodes found to simulate failure.")
            
            # 2. Event Sim (Simulate App Publishing)
            print("\nSimulating Event...")
            apps = [c.id for c in graph_data.components if c.component_type == "Application"]
            if apps:
                source_id = apps[0]
                print(f"Source: {source_id}")
                
                # run_event_simulation takes source_id (str)
                event_res = sim.run_event_simulation(source_id)
                # event_res is EventResult
                print(f"Reached Subscribers: {len(event_res.reached_subscribers)}")
            else:
                print("No applications found to simulate events.")
                
        finally:
            repo.close()
            
    except Exception as e:
        print(f"Error: {e}")
        print("Note: Ensure you have run 'bin/import_graph.py' first.")

if __name__ == "__main__":
    main()