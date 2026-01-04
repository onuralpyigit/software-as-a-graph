"""
Example: Generating Dashboard Programmatically
"""
import sys
import os
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.visualization.visualizer import GraphVisualizer

def main():
    output_file = "example_dashboard.html"
    
    try:
        print("Generating Dashboard...")
        # Note: Requires running Neo4j instance with imported data
        with GraphVisualizer() as viz:
            path = viz.generate_dashboard(output_file)
            print(f"Dashboard saved to: {os.path.abspath(path)}")
            
    except Exception as e:
        print(f"Failed to generate dashboard: {e}")
        print("Check if Neo4j is running.")

if __name__ == "__main__":
    main()