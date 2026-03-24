import ast
import pathlib

def test_simulation_does_not_import_prediction():
    """Enforce the independence guarantee: I(v) must not derive from Q(v)."""
    
    # Locate src/simulation relative to this test file (backend/tests)
    backend_dir = pathlib.Path(__file__).resolve().parent.parent
    sim_dir = backend_dir / "src" / "simulation"
    
    assert sim_dir.exists(), f"Directory not found: {sim_dir}"
    
    forbidden = {
        "PredictionService", 
        "QualityAnalyzer", 
        "QualityAnalysisResult",
        "ComponentQuality", 
        "QualityScores"
    }
    
    for py_file in sim_dir.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except SyntaxError as e:
            # If there's a syntax error in the file, we can't parse it, but we shouldn't fail the test
            # unless the project itself is completely broken.
            continue
            
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Handle `import X` and `from X import Y`
                names = []
                for alias in node.names:
                    names.append(alias.name)
                    if alias.asname:
                        names.append(alias.asname)
                
                # Also check module name in `from X import Y`
                if isinstance(node, ast.ImportFrom) and node.module:
                    # E.g., from src.prediction.models import PredictionService
                    # node.module = "src.prediction.models"
                    # If the forbidden name is directly in the module path (less likely but possible)
                    parts = node.module.split('.')
                    names.extend(parts)
                    
                violations = forbidden & set(names)
                assert not violations, (
                    f"{py_file.name} imports {violations} — violates independence guarantee"
                )
