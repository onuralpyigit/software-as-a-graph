#!/usr/bin/env python3
"""
Software-as-a-Graph Pipeline Runner

Orchestrates the complete end-to-end workflow:
1. Generate Data (Optional) -> JSON
2. Build Model (Import)     -> Neo4j
3. Analyze Model            -> Neo4j -> Analysis Results
4. Simulate Failures        -> Neo4j -> Simulation Results
5. Validate Model           -> Neo4j -> Validation Report
6. Visualize Results        -> Neo4j -> Dashboard HTML

This script acts as the master controller, invoking the specialized CLI tools
for each stage of the graph data science lifecycle.

Usage:
    # Full pipeline
    python run.py --all

    # Specific stages
    python run.py --import-data --analyze --visualize

Author: Software-as-a-Graph Research Project
"""

import argparse
import sys
import subprocess
import time
import logging
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any

# --- Configuration ---
DEFAULT_CONFIG = {
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "password",
    "input_file": "input/system.json",
    "output_dir": "output",
    "scale": "medium"
}

# --- ANSI Colors ---
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Pipeline")

class PipelineRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.python_exe = sys.executable
        self.output_path = Path(args.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Base arguments for all Neo4j-connected scripts
        self.neo4j_args = [
            "--uri", args.uri,
            "--user", args.user,
            "--password", args.password
        ]

    def run(self):
        """Execute the pipeline stages based on arguments."""
        self._print_banner()

        # 0. Pre-flight Checks
        if self._requires_neo4j():
            if not self._check_connection():
                logger.error("Aborting pipeline: Neo4j is unreachable.")
                sys.exit(1)

        # 1. Generate
        if self.args.all or self.args.generate:
            self._run_step("Generate Graph Data", [
                "generate_graph.py",
                "--scale", self.args.scale,
                "--output", self.args.input
            ])

        # 2. Import
        if self.args.all or self.args.do_import:
            self._run_step("Import to Neo4j", [
                "import_graph.py",
                "--input", self.args.input,
                "--clear"
            ] + self.neo4j_args)

        # 3. Analyze
        if self.args.all or self.args.analyze:
            self._run_step("Analyze Graph Model", [
                "analyze_graph.py"
            ] + self.neo4j_args)

        # 4. Simulate
        if self.args.all or self.args.simulate:
            self._run_simulation_step()

        # 5. Validate
        if self.args.all or self.args.validate:
            output_file = self.output_path / "validation_results.json"
            self._run_step("Validate Model", [
                "validate_graph.py",
                "--output", str(output_file)
            ] + self.neo4j_args)

        # 6. Visualize
        if self.args.all or self.args.visualize:
            dashboard_file = self.output_path / "dashboard.html"
            self._run_step("Generate Dashboard", [
                "visualize_graph.py",
                "--output", str(dashboard_file),
                "--no-browser" 
            ] + self.neo4j_args)
            print(f"\n{GREEN}Dashboard available at: {dashboard_file.absolute()}{RESET}")

        print(f"\n{GREEN}{'='*60}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*60}{RESET}\n")

    def _run_step(self, name: str, command: List[str]) -> bool:
        """Helper to run a CLI script."""
        print(f"\n{BLUE}{'-'*60}")
        print(f"STEP: {name}")
        print(f"{'-'*60}{RESET}")
        
        full_cmd = [self.python_exe] + command
        start_time = time.time()
        
        try:
            # We allow stdout to flow to the terminal so user sees progress of sub-scripts
            result = subprocess.run(full_cmd, check=False)
            
            duration = time.time() - start_time
            if result.returncode == 0:
                print(f"{GREEN}✓ Completed in {duration:.2f}s{RESET}")
                return True
            else:
                print(f"{RED}✗ Failed with code {result.returncode}{RESET}")
                sys.exit(result.returncode) # Fail fast
                
        except Exception as e:
            print(f"{RED}✗ Execution Error: {e}{RESET}")
            sys.exit(1)

    def _run_simulation_step(self):
        """Special handling for simulation to pick smart targets."""
        print(f"\n{BLUE}{'-'*60}")
        print(f"STEP: System Simulation")
        print(f"{'-'*60}{RESET}")

        # 1. Event Simulation (Find a Publisher)
        source_node = self._get_smart_target("Application", criteria="publisher")
        print(f"Running Event Simulation from Source: {BOLD}{source_node}{RESET}")
        self._run_step("Event Sim", [
            "simulate_graph.py", "--event", source_node
        ] + self.neo4j_args)

        # 2. Failure Simulation (Find a Central Hub or Critical Node)
        # We try to find a Broker or a highly connected Node
        target_node = self._get_smart_target("Broker", criteria="hub")
        if target_node == "N/A":
             target_node = self._get_smart_target("Node", criteria="hub")
             
        print(f"Running Failure Simulation on Target: {BOLD}{target_node}{RESET}")
        self._run_step("Failure Sim", [
            "simulate_graph.py", "--failure", target_node
        ] + self.neo4j_args)

    def _get_smart_target(self, label: str, criteria: str = "random") -> str:
        """
        Connects to Neo4j to find a relevant node ID for simulation
        rather than picking blindly.
        """
        try:
            from neo4j import GraphDatabase
            
            query = ""
            if criteria == "publisher":
                # Find an app that actually publishes to something
                query = f"MATCH (n:{label})-[:PUBLISHES_TO]->() RETURN n.id as id LIMIT 1"
            elif criteria == "hub":
                # Find a node with high degree
                query = f"MATCH (n:{label}) WITH n, count{{(n)--()}} as degree ORDER BY degree DESC LIMIT 1 RETURN n.id as id"
            else:
                query = f"MATCH (n:{label}) RETURN n.id as id LIMIT 1"

            with GraphDatabase.driver(self.args.uri, auth=(self.args.user, self.args.password)) as driver:
                with driver.session() as session:
                    result = session.run(query)
                    record = result.single()
                    if record:
                        return record["id"]
        except ImportError:
            logger.warning("neo4j driver not installed. Using default ID.")
        except Exception as e:
            logger.warning(f"Could not query Neo4j for target: {e}")
        
        return "A0" if label == "Application" else "B0"

    def _check_connection(self) -> bool:
        """Verify Neo4j connectivity."""
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(self.args.uri, auth=(self.args.user, self.args.password))
            driver.verify_connectivity()
            driver.close()
            logger.info("Neo4j connection verified.")
            return True
        except ImportError:
            logger.warning("neo4j library not found. Skipping connection check (scripts might fail).")
            return True # Assume user knows what they are doing
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def _requires_neo4j(self) -> bool:
        """Check if any selected step requires Neo4j."""
        steps = [self.args.do_import, self.args.analyze, self.args.simulate, self.args.validate, self.args.visualize]
        return self.args.all or any(steps)

    def _print_banner(self):
        print(f"""{BLUE}
   _____       ______                                  
  / ___/____ _/ ____/      Run Pipeline               
  \__ \/ __ `/ / __        v2.0                       
 ___/ / /_/ / /_/ /                                   
/____/\__,_/\____/                                    
{RESET}""")
        print(f"Configuration:")
        print(f"  URI:    {self.args.uri}")
        print(f"  Input:  {self.args.input}")
        print(f"  Output: {self.args.output_dir}")
        print("")

def main():
    parser = argparse.ArgumentParser(description="Software-as-a-Graph Pipeline Runner")
    
    # Actions
    g = parser.add_argument_group("Pipeline Stages")
    g.add_argument("--all", action="store_true", help="Run full end-to-end pipeline")
    g.add_argument("--generate", action="store_true", help="1. Generate Synthetic Data")
    g.add_argument("--import-data", dest="do_import", action="store_true", help="2. Import Data to Neo4j")
    g.add_argument("--analyze", action="store_true", help="3. Analyze Graph Model")
    g.add_argument("--simulate", action="store_true", help="4. Simulate Failures & Events")
    g.add_argument("--validate", action="store_true", help="5. Validate Model Accuracy")
    g.add_argument("--visualize", action="store_true", help="6. Generate Dashboard")

    # Config
    c = parser.add_argument_group("Configuration")
    c.add_argument("--uri", default=DEFAULT_CONFIG["uri"], help="Neo4j URI")
    c.add_argument("--user", default=DEFAULT_CONFIG["user"], help="Neo4j User")
    c.add_argument("--password", default=DEFAULT_CONFIG["password"], help="Neo4j Password")
    c.add_argument("--input", default=DEFAULT_CONFIG["input_file"], help="Input JSON path")
    c.add_argument("--output-dir", default=DEFAULT_CONFIG["output_dir"], help="Output directory")
    c.add_argument("--scale", default=DEFAULT_CONFIG["scale"], help="Generation scale (tiny, small, medium, large)")

    args = parser.parse_args()

    # Default to printing help if no action provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    runner = PipelineRunner(args)
    runner.run()

if __name__ == "__main__":
    main()