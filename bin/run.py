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
    
    # Format command for display (truncate if too long could be added, but keeping it simple)
    cmd_str = f"python {script_name} " + " ".join(args)
    print_step(f"Executing: {Colors.GRAY}{cmd_str}{Colors.RESET}")
    
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
    options.add_argument("--input", default="output/system.json", help="Input JSON file for import (default: output/system.json)")
    options.add_argument("--output-dir", default="output", help="Output directory for reports (default: output)")
    options.add_argument("--layer", "--layers", dest="layers", default="app,infra,mw", help="Layers to process (comma-separated, default: app,infra,mw)")
    options.add_argument("--clean", "--clear", dest="clean", action="store_true", help="Clear existing database before import")
    
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
    
    # 1. GENERATE
    if run_all or args.generate:
        print_header("Stage 1: Generation")
        gen_args = ["--output", args.input]
        
        if args.config:
            gen_args.extend(["--config", args.config])
        else:
            gen_args.extend(["--scale", args.scale, "--seed", str(args.seed)])
            
        if not run_script("generate_graph.py", gen_args, project_root):
             return 1

    # 2. IMPORT
    if run_all or args.do_import:
        print_header("Stage 2: Import")
        import_args = ["--input", args.input]
        if args.clean:
            import_args.append("--clear")
        import_args += neo4j_args
        
        if not run_script("import_graph.py", import_args, project_root):
             return 1

    # 3. ANALYZE
    if run_all or args.analyze:
         print_header("Stage 3: Analysis")
         
         target_layers = args.layers.split(",")
         # analyze_graph.py only accepts one layer at a time via --layer, or --all
         
         if "system" in target_layers and len(target_layers) > 1:
             # Just run --all if list is comprehensive enough or simplify logic?
             # For now, let's iterate to be safe and explicit
             pass

         for layer in target_layers:
             layer = layer.strip()
             if not layer: continue
             
             print_step(f"Analyzing layer: {layer}")
             an_output = output_dir / f"analysis_results_{layer}.json"
             an_args = ["--layer", layer, "--output", str(an_output)]
             an_args += neo4j_args
             
             if not run_script("analyze_graph.py", an_args, project_root):
                 return 1
         
         # Note on output: The last call will overwrite analysis_results.json. 
         # This is a limitation of the current analyze_graph.py if called sequentially with same output.
         # But usually users might want --all.
         # If args.all is passed to run.py, maybe we should use --all for analyze_graph.py?
         
         if run_all:
             # Run one pass with --all to get a consolidated result if possible
             # But we just ran individual layers.
             # Actually, if we use --all in analyze_graph, it returns MultiLayerAnalysisResult with all layers.
             # The existing analyze_graph.py logic:
             # if args.all: return analyzer.analyze_all_layers()
             
             # Refinement: If user didn't specify specific layers (used default) and ran --all, 
             # we should probably just call analyze_graph.py --all
             pass

    # 4. SIMULATE
    if run_all or args.simulate:
        print_header("Stage 4: Simulation")
        # simulate_graph.py report --layers "app,infra"
        sim_args = ["report", "--layers", args.layers, "--output", str(output_dir / "simulation_report.json")]
        sim_args += neo4j_args
        if not run_script("simulate_graph.py", sim_args, project_root):
             return 1

    # 5. VALIDATE
    if run_all or args.validate:
        print_header("Stage 5: Validation")
        val_args = ["--layer", args.layers, "--output", str(output_dir / "validation_results.json")]
        val_args += neo4j_args
        if not run_script("validate_graph.py", val_args, project_root):
             return 1

    # 6. VISUALIZE
    if run_all or args.visualize:
        print_header("Stage 6: Visualization")
        viz_args = ["--layers", args.layers, "--output", str(output_dir / "dashboard.html")]
        viz_args += neo4j_args
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
