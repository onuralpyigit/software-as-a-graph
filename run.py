#!/usr/bin/env python3
"""
Software-as-a-Graph Pipeline Runner

Orchestrates the complete end-to-end workflow:
1. Generate Data (Optional) -> JSON
2. Build Model (Import)     -> Neo4j
3. Analyze Model            -> Neo4j -> Results
4. Simulate Failures        -> Neo4j -> Results
5. Validate Model           -> Neo4j -> Results
6. Visualize Results        -> Neo4j -> Dashboard

Usage:
    # Full pipeline (Generate new data and run everything)
    python run.py --generate --import --analyze --simulate --validate --visualize

    # Run only analysis and visualization on existing Neo4j data
    python run.py --analyze --visualize

    # Reset DB with specific file and run pipeline
    python run.py --import --input input/system.json --analyze --simulate

Author: Software-as-a-Graph Research Project
"""

import argparse
import sys
import subprocess
import time
import logging
import random
from pathlib import Path
from typing import List, Optional

# Configuration
DEFAULT_NEO4J_URI = "bolt://localhost:7687"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASS = "password"
DEFAULT_INPUT_FILE = "input/system.json"
DEFAULT_OUTPUT_DIR = "output"

# Colors for terminal output
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Pipeline")

def print_step(step_name: str):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}STEP: {step_name}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

def run_command(command: List[str], description: str) -> bool:
    """Run a CLI command via subprocess."""
    logger.info(f"Starting: {description}")
    logger.debug(f"Command: {' '.join(command)}")
    
    start_time = time.time()
    try:
        # Use sys.executable to ensure the same python environment is used
        full_cmd = [sys.executable] + command
        
        result = subprocess.run(
            full_cmd,
            check=False,
            text=True
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n{GREEN}✓ {description} completed in {duration:.2f}s{RESET}")
            return True
        else:
            print(f"\n{RED}✗ {description} failed with exit code {result.returncode}{RESET}")
            return False
            
    except Exception as e:
        print(f"\n{RED}✗ Execution error: {e}{RESET}")
        return False

def get_random_node_id(uri, user, password, label="Application"):
    """
    Helper to fetch a random node ID from Neo4j for simulation.
    Requires neo4j driver to be installed in the environment.
    """
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            result = session.run(f"MATCH (n:{label}) RETURN n.id AS id, rand() as r ORDER BY r LIMIT 1")
            record = result.single()
            if record:
                return record["id"]
    except ImportError:
        logger.warning("neo4j driver not found in run.py environment. Using default ID 'A0'.")
    except Exception as e:
        logger.warning(f"Could not fetch random node: {e}. Using default ID 'A0'.")
    return "A0"

def main():
    parser = argparse.ArgumentParser(description="Run Software-as-a-Graph Pipeline")
    
    # Workflow flags
    flow = parser.add_argument_group("Workflow Steps")
    flow.add_argument("--all", action="store_true", help="Run all steps")
    flow.add_argument("--generate", action="store_true", help="1. Generate synthetic graph data")
    flow.add_argument("--import-data", dest="do_import", action="store_true", help="2. Import data into Neo4j")
    flow.add_argument("--analyze", action="store_true", help="3. Analyze graph structure and quality")
    flow.add_argument("--simulate", action="store_true", help="4. Simulate system failures")
    flow.add_argument("--validate", action="store_true", help="5. Validate predictions vs simulation")
    flow.add_argument("--visualize", action="store_true", help="6. Generate visualization dashboard")
    
    # Configuration
    config = parser.add_argument_group("Configuration")
    config.add_argument("--uri", default=DEFAULT_NEO4J_URI, help="Neo4j URI")
    config.add_argument("--user", default=DEFAULT_NEO4J_USER, help="Neo4j User")
    config.add_argument("--password", default=DEFAULT_NEO4J_PASS, help="Neo4j Password")
    config.add_argument("--input", default=DEFAULT_INPUT_FILE, help="Input JSON file path")
    config.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for artifacts")
    config.add_argument("--scale", default="medium", help="Graph scale for generation (if --generate used)")
    
    args = parser.parse_args()
    
    # Check if any step is selected, otherwise print help
    steps_selected = [args.all, args.generate, args.do_import, args.analyze, args.simulate, args.validate, args.visualize]
    if not any(steps_selected):
        parser.print_help()
        return

    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Common Args
    neo4j_args = ["--uri", args.uri, "--password", args.password]
    # Note: Scripts might support --user, but standardizing on URI/Password for simplicity in this runner
    # If scripts require --user, append it:
    neo4j_args += ["--user", args.user] 

    # 1. Generate Data
    if args.all or args.generate:
        print_step("Generate Graph Data")
        cmd = ["generate_graph.py", "--scale", args.scale, "--output", args.input]
        if not run_command(cmd, "Data Generation"):
            return

    # 2. Import Data (Build Graph Model)
    if args.all or args.do_import:
        print_step("Build Graph Model (Import to Neo4j)")
        cmd = ["import_graph.py", "--input", args.input, "--clear"] + neo4j_args
        if not run_command(cmd, "Graph Import"):
            return

    # 3. Analyze Model
    if args.all or args.analyze:
        print_step("Analyze Graph Model")
        # Runs analysis and optionally classifies components
        cmd = ["analyze_graph.py"] + neo4j_args
        if not run_command(cmd, "Structural & Quality Analysis"):
            return

    # 4. Simulate Failures
    if args.all or args.simulate:
        print_step("Simulate System Failures")
        
        # Determine a target node for the simulation demo
        target_node = get_random_node_id(args.uri, args.user, args.password, "Application")
        print(f"Selected target for simulation demo: {target_node}")
        
        # Run Failure Simulation
        cmd_fail = ["simulate_graph.py", "--failure", target_node] + neo4j_args
        run_command(cmd_fail, f"Failure Simulation (Target: {target_node})")
        
        # Run Event Simulation
        cmd_event = ["simulate_graph.py", "--event", target_node] + neo4j_args
        run_command(cmd_event, f"Event Simulation (Source: {target_node})")

    # 5. Validate Results
    if args.all or args.validate:
        print_step("Validate Analysis vs Simulation")
        # Validate predictions against simulation results
        validation_output = f"{args.output_dir}/validation_results.json"
        cmd = ["validate_graph.py", "--output", validation_output] + neo4j_args
        if not run_command(cmd, "Validation Pipeline"):
            return

    # 6. Visualize Results
    if args.all or args.visualize:
        print_step("Visualize & Report")
        dashboard_path = f"{args.output_dir}/dashboard.html"
        # Generate overview dashboard
        cmd = ["visualize_graph.py", "--output", dashboard_path] + neo4j_args
        if not run_command(cmd, "Dashboard Generation"):
            return
        
        print(f"\n{GREEN}Dashboard successfully generated at: {dashboard_path}{RESET}")

    print(f"\n{GREEN}{'='*60}")
    print(f"PIPELINE EXECUTION COMPLETE")
    print(f"{'='*60}{RESET}\n")

if __name__ == "__main__":
    main()