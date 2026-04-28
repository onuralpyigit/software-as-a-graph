"""
Enforce the Analyze/Predict separation guarantee.

PredictGraphUseCase must never import from the simulation module — prediction
is a purely topology-driven, pre-deployment operation.  Importing simulation
symbols would couple Step 3 to Step 4, breaking the independence claim.
"""
import ast
import pathlib


def _parse_imports(py_file: pathlib.Path) -> list[tuple[str, set[str]]]:
    """Return (filename, {all imported names}) for each parseable file."""
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
    except SyntaxError:
        return []

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                imported.add(alias.name)
                if alias.asname:
                    imported.add(alias.asname)
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.update(node.module.split("."))

    return [(py_file.name, imported)]


def test_prediction_usecase_does_not_import_simulation():
    """PredictGraphUseCase must not reference simulation symbols."""
    backend_dir = pathlib.Path(__file__).resolve().parent.parent
    predict_uc = backend_dir / "src" / "usecases" / "predict_graph.py"

    assert predict_uc.exists(), f"File not found: {predict_uc}"

    forbidden = {
        "simulation",
        "FailureSimulator",
        "SimulationService",
        "CascadeSimulator",
        "FailureSimulationResult",
    }

    for name, imported in _parse_imports(predict_uc):
        violations = forbidden & imported
        assert not violations, (
            f"{name} imports {violations} — PredictGraphUseCase must not "
            "depend on the simulation module (breaks Analyze/Predict separation)"
        )


def test_prediction_module_does_not_import_simulation():
    """No file in src/prediction/ may import from src/simulation/."""
    backend_dir = pathlib.Path(__file__).resolve().parent.parent
    pred_dir = backend_dir / "src" / "prediction"

    assert pred_dir.exists(), f"Directory not found: {pred_dir}"

    forbidden = {
        "simulation",
        "FailureSimulator",
        "SimulationService",
        "CascadeSimulator",
        "FailureSimulationResult",
        "simulate",
    }

    for py_file in pred_dir.rglob("*.py"):
        for name, imported in _parse_imports(py_file):
            violations = forbidden & imported
            assert not violations, (
                f"src/prediction/{name} imports {violations} — prediction "
                "module must not depend on simulation (breaks Step 3/Step 4 independence)"
            )
