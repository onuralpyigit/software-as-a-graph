"""
Example: Generating Graph Data Programmatically
"""
import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from src.generation import generate_graph

def main():
    try:
        print("Generating 'tiny' graph...")
        # Method 1: Functional API
        # Scales: tiny, small, medium, large, xlarge
        graph_data = generate_graph(scale="tiny", seed=123)
        
        # Count components
        n_nodes = len(graph_data.get("nodes", []))
        n_apps = len(graph_data.get("applications", []))
        n_brokers = len(graph_data.get("brokers", []))
        n_topics = len(graph_data.get("topics", []))
        
        print(f"Generated Graph Stats:")
        print(f"  Nodes (Infrastructure): {n_nodes}")
        print(f"  Applications: {n_apps}")
        print(f"  Brokers: {n_brokers}")
        print(f"  Topics: {n_topics}")
        
        # Save the output
        output_file = Path("example_graph.json")
        with open(output_file, "w") as f:
            json.dump(graph_data, f, indent=2)
            
        print(f"\nSaved to {output_file.absolute()}")
        print("You can now run examples/example_import.py to import this file.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
