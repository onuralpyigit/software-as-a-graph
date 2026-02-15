"""
Example usage of the Refactored Analysis Module
"""
import sys
from pathlib import Path

# Add project root to sys.path
# examples/ -> root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from src.core import create_repository
from src.analysis import AnalysisService

def main():
    # Ensure Neo4j is running or this will fail
    try:
        # Initialize Repository
        repo = create_repository()
        
        # Initialize Service
        analyzer = AnalysisService(repo)
        
        try:
            # 1. Analyze Applications only
            print("Analyzing Applications...")
            app_res = analyzer.analyze_layer("app")
            # app_res is LayerAnalysisResult
            print(f"Found {len(app_res.quality.components)} apps")
            
            # 2. Analyze Full System
            print("\nAnalyzing Full System...")
            full_res = analyzer.analyze_layer("system")
            
            # 3. Access Critical Components
            crit = [c for c in full_res.quality.components 
                   if str(c.levels.overall.value).upper() == 'CRITICAL']
            print(f"Critical Components detected: {len(crit)}")
            
            # 4. Access Critical Edges
            crit_edges = [e for e in full_res.quality.edges 
                         if str(e.level).upper() == 'CRITICAL']
            print(f"Critical Dependencies detected: {len(crit_edges)}")

        finally:
            repo.close()
            
    except Exception as e:
        print(f"Failed to run example: {e}")
        print("Ensure Neo4j is running and populated (run bin/import_graph.py first)")

if __name__ == "__main__":
    main()