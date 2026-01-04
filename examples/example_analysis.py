"""
Example usage of the Refactored Analysis Module
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.analysis.analyzer import GraphAnalyzer

def main():
    # Ensure Neo4j is running or this will fail
    try:
        with GraphAnalyzer() as analyzer:
            # 1. Analyze Applications only
            print("Analyzing Applications...")
            app_res = analyzer.analyze_by_type("Application")
            print(f"Found {len(app_res['quality'].components)} apps")
            
            # 2. Analyze Full System
            print("\nAnalyzing Full System...")
            full_res = analyzer.analyze_full_system()
            
            # 3. Access Critical Components
            crit = [c for c in full_res['quality'].components 
                   if c.level.value == 'critical']
            print(f"Critical Components detected: {len(crit)}")
            
            # 4. Access Critical Edges
            crit_edges = [e for e in full_res['quality'].edges 
                         if e.level.value == 'critical']
            print(f"Critical Dependencies detected: {len(crit_edges)}")
            
    except Exception as e:
        print(f"Failed to run example: {e}")
        print("Ensure Neo4j is running and populated (run import_graph.py first)")

if __name__ == "__main__":
    main()