"""
Example: Running Validation Programmatically
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.validation.pipeline import ValidationPipeline

def main():
    try:
        print("Starting Validation Pipeline...")
        # Note: Requires Neo4j to be running and populated
        with ValidationPipeline() as pipeline:
            result = pipeline.run()
            
            print(f"Validation Status: {'PASS' if result.overall.passed else 'FAIL'}")
            print(f"Correlation: {result.overall.correlation.spearman:.3f}")
            
            # Check specific type
            if "Application" in result.by_type:
                app_res = result.by_type["Application"]
                print(f"App Validation: Rho={app_res.correlation.spearman:.3f}")
                
    except Exception as e:
        print(f"Execution failed: {e}")
        print("Ensure Neo4j is running and data is imported.")

if __name__ == "__main__":
    main()