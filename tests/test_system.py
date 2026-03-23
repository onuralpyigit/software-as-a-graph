import pytest
import subprocess
import sys
from pathlib import Path

# Provide absolute paths relative to this script
PROJECT_ROOT = Path(__file__).resolve().parent.parent

@pytest.mark.e2e
def test_cli_smoke_end_to_end(tmp_path):
    """
    True CLI smoke tests using the subprocess approach.
    Runs the full end-to-end pipeline example on a tiny graph
    to verify all CLI stages are correctly integrated.
    """
    example_script = PROJECT_ROOT / "examples" / "example_end_to_end.py"
    
    # We use a very small scale and skip visualization to speed up tests
    cmd = [
        sys.executable, str(example_script),
        "--scale", "tiny",
        "--output-dir", str(tmp_path),
        "--skip-viz"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # If Neo4j is not available, the script may fail at connection
    # Standard practice is to assert success unless Neo4j is definitively absent
    # We will check if the failure is just due to neo4j unavailable
    if result.returncode != 0 and "ServiceUnavailable" in result.stderr:
        pytest.skip("Neo4j is not running - skipping true CLI e2e test.")
        
    assert result.returncode == 0, f"End-to-end simulation failed.\\nSTDOUT:\\n{result.stdout}\\nSTDERR:\\n{result.stderr}"
    
    # Verify the output graph was created
    outputs = list(tmp_path.glob("e2e_graph_tiny_seed*.json"))
    assert len(outputs) > 0, "Expected JSON output was not found in the output directory"
