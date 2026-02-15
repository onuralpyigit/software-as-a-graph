"""
Example: Generating Dashboard Programmatically
"""
import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from src.core import create_repository
from src.analysis import AnalysisService
from src.simulation import SimulationService
from src.validation import ValidationService
from src.visualization import VisualizationService

def main():
    output_file = "example_dashboard.html"
    
    try:
        print("Generating Dashboard...")
        # Note: Requires running Neo4j instance with imported data
        
        # Initialize dependencies
        repo = create_repository()
        analysis = AnalysisService(repo)
        simulation = SimulationService(repo)
        validation = ValidationService(analysis, simulation)
        
        # Initialize Visualization Service
        viz = VisualizationService(
            analysis_service=analysis,
            simulation_service=simulation,
            validation_service=validation,
            repository=repo
        )
        
        try:
            # Generate dashboard
            path = viz.generate_dashboard(
                output_file=output_file,
                layers=["app", "system"], # Customize layers
                include_network=True
            )
            print(f"Dashboard saved to: {os.path.abspath(path)}")
            
        finally:
            repo.close()
            
    except Exception as e:
        print(f"Failed to generate dashboard: {e}")
        print("Check if Neo4j is running.")

if __name__ == "__main__":
    main()