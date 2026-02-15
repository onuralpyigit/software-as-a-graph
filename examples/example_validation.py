"""
Example: Running Validation Programmatically
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from src.core import create_repository
from src.analysis import AnalysisService
from src.simulation import SimulationService
from src.validation import ValidationService

def main():
    try:
        print("Starting Validation Pipeline...")
        # Note: Requires Neo4j to be running and populated
        
        # Initialize dependencies
        repo = create_repository()
        analysis = AnalysisService(repo)
        simulation = SimulationService(repo)
        
        # Initialize Validation Service
        validation = ValidationService(analysis, simulation)
        
        try:
            # Run validation for app and system layers
            result = validation.validate_layers(["app", "system"])
            # result is PipelineResult
            
            print(f"Validation Status: {'PASS' if result.all_passed else 'FAIL'}")
            
            # Check detailed results for 'app' layer
            if "app" in result.layers:
                app_layer = result.layers["app"]
                # app_layer is LayerValidationResult
                
                if app_layer.validation_result:
                    val_res = app_layer.validation_result
                    # val_res is ValidationResult
                    
                    print(f"App Layer Spearman Rho: {val_res.overall.correlation.spearman:.3f}")
                    print(f"App Layer Passed: {val_res.passed}")
            
            # Check detailed results for 'system' layer
            if "system" in result.layers:
                sys_layer = result.layers["system"]
                print(f"System Layer RMSE: {sys_layer.rmse:.3f}")

        finally:
            repo.close()
                
    except Exception as e:
        print(f"Execution failed: {e}")
        print("Ensure Neo4j is running and data is imported.")

if __name__ == "__main__":
    main()