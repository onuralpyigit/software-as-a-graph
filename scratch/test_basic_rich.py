import sys
from pathlib import Path

# Fix path to import saag and bin
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bin.common.console import ConsoleDisplay

def test_basic():
    console = ConsoleDisplay()
    console.print_header("Structural Graph Analysis")
    console.print_step("Connecting to Neo4j...")
    console.print_success("Connected successfully!")
    console.print_warning("High memory usage detected on server.")
    console.print_error("Failed to fetch some metadata.")

if __name__ == "__main__":
    test_basic()
