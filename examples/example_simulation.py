"""
Example: Running Simulations Programmatically
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.simulation import Simulator

def main():
    # Make sure Neo4j is running with data imported
    try:
        with Simulator() as sim:
            # 1. Failure Sim (Simulate Node Failure)
            print("Simulating Node Failure (Node-0)...")
            fail_res = sim.run_failure_sim("N0") # ID from generator usually N0, N1...
            print(f"Cascaded Failures: {len(fail_res.cascaded_failures)}")
            
            # 2. Event Sim (Simulate App Publishing)
            print("\nSimulating Event (App-0)...")
            event_res = sim.run_event_sim("A0")
            print(f"Reached Subscribers: {len(event_res.reached_subscribers)}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Note: Ensure you have run 'import_graph.py' first.")

if __name__ == "__main__":
    main()