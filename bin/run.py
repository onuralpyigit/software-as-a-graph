#!/usr/bin/env python3
"""
Software-as-a-Graph Pipeline Orchestrator

A lightweight CLI orchestrator that executes the end-to-end pipeline 
by delegating to specialized CLI scripts.

Pipeline Stages:
    1. Generate   - Create synthetic graph data (optional)
    2. Import     - Build graph model in Neo4j
    3. Analyze    - Compute structural metrics via Neo4j
    4. Simulate   - Run failure simulations via Neo4j
    5. Validate   - Compare predictions vs simulation via Neo4j
    6. Visualize  - Generate dashboard from Neo4j data
"""
import sys
from pathlib import Path

# Add project root to path for imports if needed
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import logging
import subprocess
import time
from typing import List, Optional

# =============================================================================
# Configuration & Utils
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

def print_header(title: str) -> None:
    print(f"\n{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD} {title} {Colors.RESET}")
    print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}")

def print_step(message: str) -> None:
    print(f"{Colors.BLUE}→{Colors.RESET} {message}")

def print_success(message: str) -> None:
    print(f"{Colors.GREEN}✓{Colors.RESET} {message}")

def print_error(message: str) -> None:
    print(f"{Colors.RED}✗{Colors.RESET} {message}")

def run_script(script_name: str, args: List[str], cwd: Path) -> bool:
    """
    Run a Python script as a subprocess.
    """
    script_path = cwd / "bin" / script_name
    if not script_path.exists():
        # Fallback to current directory if bin/ doesn't exist (e.g., running from inside bin)
        script_path = cwd / script_name
        if not script_path.exists():
             print_error(f"Script not found: {script_name}")
             return False

    cmd = [sys.executable, str(script_path)] + args
    cmd_str = " ".join(cmd)
    
    print_step(f"Executing: {Colors.GRAY}{script_name}{Colors.RESET}")
    
    try:
        start_time = time.time()
        # We pass stdout/stderr to sys.stdout/sys.stderr to allow real-time output
        result = subprocess.run(cmd, cwd=str(cwd), check=False)
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print_success(f"{script_name} completed in {duration:.2f}s")
            return True
        else:
            print_error(f"{script_name} failed with exit code {result.returncode}")
            return False
            
    except Exception as e:
        print_error(f"Failed to execute {script_name}: {e}")
        return False

# =============================================================================
# Main Orchestrator
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Software-as-a-Graph Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Stages
    stages = parser.add_argument_group("Pipeline Stages")
    stages.add_argument("--all", "-a", action="store_true", help="Run complete pipeline")
    stages.add_argument("--generate", "-g", action="store_true", help="Generate synthetic data")
    stages.add_argument("--import", "-i", dest="do_import", action="store_true", help="Import data to Neo4j")
    stages.add_argument("--analyze", "-A", action="store_true", help="Analyze graph in Neo4j")
    stages.add_argument("--simulate", "-s", action="store_true", help="Run simulations")
    stages.add_argument("--validate", "-V", action="store_true", help="Validate results")
    stages.add_argument("--visualize", "-z", action="store_true", help="Generate dashboard")

    # Options
    options = parser.add_argument_group("Options")
    options.add_argument("--config", help="Graph generation config file (YAML)")
    options.add_argument("--scale", default="medium", choices=["tiny", "small", "medium", "large", "xlarge"], help="Graph scale (default: medium)")
    options.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    options.add_argument("--input", default="output/system.json", help="Input JSON file for import")
    options.add_argument("--output-dir", default="output", help="Output directory for reports")
    options.add_argument("--layer", default="app", help="Layers to process (comma-separated)")
    
    # Neo4j
    neo4j = parser.add_argument_group("Neo4j Connection")
    neo4j.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI")
    neo4j.add_argument("--user", default="neo4j", help="Neo4j user")
    neo4j.add_argument("--password", default="password", help="Neo4j password")
    
    args = parser.parse_args()

    # Determine project root
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Common Neo4j Args
    neo4j_args = ["--uri", args.uri, "--user", args.user, "--password", args.password]
    
    # Process Flags
    run_all = args.all
    tasks = []
    
    if run_all or args.generate:
        print_header("Stage 1: Generation")
        gen_args = ["--output", args.input]
        if args.config:
            gen_args.extend(["--config", args.config])
        else:
            gen_args.extend(["--scale", args.scale, "--seed", str(args.seed)])
            
        if not run_script("generate_graph.py", gen_args, project_root):
             return 1

    if run_all or args.do_import:
        print_header("Stage 2: Import")
        import_args = ["--input", args.input, "--clear"] + neo4j_args
        if not run_script("import_graph.py", import_args, project_root):
             return 1

    if run_all or args.analyze:
         print_header("Stage 3: Analysis")
         # analyze_graph.py supports --layer "app,infra" or --all. If args.layer is passed, we use --layer.
         # For consistency with run.py arguments:
         if args.layer:
             an_args = ["--layer", args.layer, "--output", str(output_dir / "analysis_results.json")]
         else:
             an_args = ["--all", "--output", str(output_dir / "analysis_results.json")]
             
         an_args += neo4j_args
         if not run_script("analyze_graph.py", an_args, project_root):
             return 1

    if run_all or args.simulate:
        print_header("Stage 4: Simulation")
        # simulate_graph.py takes --layer (singular) and --report (for multiple layers report)
        # We will use --report mode to generate a comprehensive report for the requested layers
        # simulate_graph.py --report --layers "app,infra"
        sim_args = ["report", "--layer", args.layer, "--output", str(output_dir / "simulation_report.json")]
        sim_args += neo4j_args
        if not run_script("simulate_graph.py", sim_args, project_root):
             return 1

    if run_all or args.validate:
        print_header("Stage 5: Validation")
        val_args = ["--layer", args.layer, "--output", str(output_dir / "validation_results.json")]
        val_args += neo4j_args
        if not run_script("validate_graph.py", val_args, project_root):
             return 1

    if run_all or args.visualize:
        print_header("Stage 6: Visualization")
        viz_args = ["--layers", args.layer, "--output", str(output_dir / "dashboard.html")]
        viz_args += neo4j_args
        # If running as part of a pipeline, we probably don't want to auto-open unless specified.
        # But run.py didn't have --open argument in this simple version, let's just generate.
        # If user wants to open, they can use visualize_graph.py directly or we could add --open.
        if not run_script("visualize_graph.py", viz_args, project_root):
             return 1

    print_header("Pipeline Complete")
    print(f"\n{Colors.GREEN}All requested stages executed successfully.{Colors.RESET}")
    print(f"Outputs located in: {output_dir}")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Pipeline interrupted by user.{Colors.RESET}")
        sys.exit(130)
