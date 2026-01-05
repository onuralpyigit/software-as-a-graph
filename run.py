#!/usr/bin/env python3
"""
Software-as-a-Graph Pipeline Runner

Orchestrates the complete PhD research methodology workflow:
1. Data Generation (Synthetic Topologies)
2. Model Construction (Graph Import & Dependency Derivation)
3. Structural Analysis (Centrality, Criticality Scoring)
4. Failure Simulation (Cascade Propagation & Impact Assessment)
5. Model Validation (Prediction vs. Ground Truth Correlation)
6. Visualization (Multi-Layer Dashboarding)

Usage:
    # Run full end-to-end pipeline with default settings
    python run.py --all

    # Run specific stages with custom simulation parameters
    python run.py --import-data --analyze --simulate --cascade-threshold 0.4

Author: Software-as-a-Graph Research Project
"""

import argparse
import sys
import subprocess
import time
import logging
from pathlib import Path
from typing import List, Optional

# --- Configuration & Defaults ---
DEFAULT_CONFIG = {
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "password",
    "input_file": "input/system.json",
    "output_dir": "output",
    "scale": "medium"
}

# --- ANSI Colors for CLI ---
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
        
        # Common Neo4j arguments
        self.neo4j_args = [
            "--uri", args.uri,
            "--user", args.user,
            "--password", args.password
        ]

    def run(self):
        """Execute the pipeline stages based on provided arguments."""
        self._print_banner()

        # 0. Connectivity Check
        if self._requires_neo4j():
            if not self._check_connection():
                logger.error("Aborting pipeline: Neo4j is unreachable.")
                sys.exit(1)

        # 1. Generate Data
        if self.args.all or self.args.generate:
            self._run_stage_generate()

        # 2. Import & Build Model
        if self.args.all or self.args.do_import:
            self._run_stage_import()

        # 3. Analyze Model (Topological & Quality)
        if self.args.all or self.args.analyze:
            self._run_stage_analyze()

        # 4. Simulate Failures (Demonstration)
        if self.args.all or self.args.simulate:
            self._run_stage_simulate()

        # 5. Validate Model (Statistical Correlation)
        if self.args.all or self.args.validate:
            self._run_stage_validate()

        # 6. Visualize Results (Dashboard)
        if self.args.all or self.args.visualize:
            self._run_stage_visualize()

        self._print_footer()

    # --- Stage Implementations ---

    def _run_stage_generate(self):
        """Stage 1: Generate Synthetic Graph Data"""
        self._run_subprocess("Generate Graph Data", [
            "generate_graph.py",
            "--scale", self.args.scale,
            "--output", self.args.input,
            "--seed", str(self.args.seed)
        ])

    def _run_stage_import(self):
        """Stage 2: Import Data to Neo4j"""
        cmd = [
            "import_graph.py",
            "--input", self.args.input,
            "--clear"
        ] + self.neo4j_args
        self._run_subprocess("Import & Build Model", cmd)

    def _run_stage_analyze(self):
        """Stage 3: Analyze Graph Model"""
        # runs analyze_graph.py which triggers Structural & Quality Analyzers
        self._run_subprocess("Analyze Graph Model", ["analyze_graph.py"] + self.neo4j_args)

    def _run_stage_simulate(self):
        """Stage 4: Simulate Failures (Demo)"""
        print(f"\n{BLUE}{'-'*60}")
        print(f"STEP: System Simulation (Demo)")
        print(f"{'-'*60}{RESET}")
        
        # 4a. Event Simulation (Data Flow)
        source_node = self._get_smart_target("Application", criteria="publisher")
        print(f"Simulating Event propagation from publisher: {BOLD}{source_node}{RESET}")
        self._run_subprocess("Event Simulation", [
            "simulate_graph.py", 
            "--event", source_node
        ] + self.neo4j_args)

        # 4b. Failure Simulation (Cascade)
        # Find a critical hub to fail for maximum impact demonstration
        target_node = self._get_smart_target("Broker", criteria="hub")
        if target_node == "N/A":
             target_node = self._get_smart_target("Node", criteria="hub")
             
        print(f"Simulating Failure cascade from critical hub: {BOLD}{target_node}{RESET}")
        print(f"Params: Threshold={self.args.cascade_threshold}, Prob={self.args.cascade_prob}")
        
        self._run_subprocess("Failure Simulation", [
            "simulate_graph.py", 
            "--failure", target_node,
            "--threshold", str(self.args.cascade_threshold),
            "--probability", str(self.args.cascade_prob),
            "--depth", str(self.args.cascade_depth)
        ] + self.neo4j_args)

    def _run_stage_validate(self):
        """Stage 5: Validate Model"""
        output_file = self.output_path / "validation_results.json"
        cmd = [
            "validate_graph.py",
            "--output", str(output_file),
            "--target-spearman", str(self.args.target_spearman),
            "--target-f1", str(self.args.target_f1)
        ] + self.neo4j_args
        self._run_subprocess("Validate Model", cmd)

    def _run_stage_visualize(self):
        """Stage 6: Visualize Results"""
        dashboard_file = self.output_path / "dashboard.html"
        cmd = [
            "visualize_graph.py",
            "--output", str(dashboard_file),
            "--no-browser" 
        ] + self.neo4j_args
        
        self._run_subprocess("Generate Dashboard", cmd)
        print(f"\n{GREEN}>> Dashboard available at: {dashboard_file.absolute()}{RESET}")

    # --- Helpers ---

    def _run_subprocess(self, name: str, command: List[str]):
        """Helper to run a CLI script safely."""
        print(f"\n{BLUE}[Running] {name}...{RESET}")
        
        full_cmd = [self.python_exe] + command
        start_time = time.time()
        
        try:
            result = subprocess.run(full_cmd, check=False)
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"{GREEN}✓ {name} completed ({duration:.2f}s){RESET}")
            else:
                print(f"{RED}✗ {name} failed with code {result.returncode}{RESET}")
                sys.exit(result.returncode)
                
        except Exception as e:
            print(f"{RED}✗ Execution Error: {e}{RESET}")
            sys.exit(1)

    def _get_smart_target(self, label: str, criteria: str = "random") -> str:
        """
        Connects to Neo4j to find a relevant node ID for simulation
        (e.g., a highly connected Broker) rather than a random ID.
        """
        try:
            from neo4j import GraphDatabase
            
            query = ""
            if criteria == "publisher":
                # Find an app that has outgoing PUBLISHES_TO edges
                query = f"MATCH (n:{label})-[:PUBLISHES_TO]->() RETURN n.id as id LIMIT 1"
            elif criteria == "hub":
                # Find the node with the highest degree
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
            pass # neo4j driver might not be installed in the env running the wrapper
        except Exception:
            pass 
        
        # Fallback defaults if query fails
        return "A1" if label == "Application" else "B1"

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
            logger.warning("Neo4j python driver not found. Skipping connection check.")
            return True 
        except Exception as e:
            logger.error(f"Could not connect to Neo4j: {e}")
            return False

    def _requires_neo4j(self) -> bool:
        """Check if any selected step requires Neo4j."""
        steps = [self.args.do_import, self.args.analyze, self.args.simulate, self.args.validate, self.args.visualize]
        return self.args.all or any(steps)

    def _print_banner(self):
        print(f"""{BLUE}
   _____       ______                                  
  / ___/____ _/ ____/      Software-as-a-Graph        
  \__ \/ __ `/ / __        Research Pipeline v2.0     
 ___/ / /_/ / /_/ /                                   
/____/\__,_/\____/                                    
{RESET}""")
        print(f"Configuration:")
        print(f"  URI:    {self.args.uri}")
        print(f"  Input:  {self.args.input}")
        print(f"  Output: {self.args.output_dir}")
        print(f"  Scale:  {self.args.scale}")
        print("")

    def _print_footer(self):
        print(f"\n{GREEN}{'='*60}")
        print(f"PIPELINE EXECUTION COMPLETE")
        print(f"{'='*60}{RESET}\n")

def main():
    parser = argparse.ArgumentParser(description="Software-as-a-Graph Pipeline Runner")
    
    # --- Pipeline Actions ---
    g = parser.add_argument_group("Pipeline Stages")
    g.add_argument("--all", action="store_true", help="Run full end-to-end research pipeline")
    g.add_argument("--generate", action="store_true", help="1. Generate Synthetic Data")
    g.add_argument("--import-data", dest="do_import", action="store_true", help="2. Import Data to Neo4j")
    g.add_argument("--analyze", action="store_true", help="3. Analyze Graph Model (Structure & Quality)")
    g.add_argument("--simulate", action="store_true", help="4. Simulate Failures (Demo)")
    g.add_argument("--validate", action="store_true", help="5. Validate Model (Stats & Correlation)")
    g.add_argument("--visualize", action="store_true", help="6. Generate Dashboard")

    # --- Configuration ---
    c = parser.add_argument_group("System Configuration")
    c.add_argument("--uri", default=DEFAULT_CONFIG["uri"], help="Neo4j URI")
    c.add_argument("--user", default=DEFAULT_CONFIG["user"], help="Neo4j User")
    c.add_argument("--password", default=DEFAULT_CONFIG["password"], help="Neo4j Password")
    c.add_argument("--input", default=DEFAULT_CONFIG["input_file"], help="Input JSON path")
    c.add_argument("--output-dir", default=DEFAULT_CONFIG["output_dir"], help="Output directory")
    
    # --- Generation Params ---
    p = parser.add_argument_group("Generation Parameters")
    p.add_argument("--scale", default=DEFAULT_CONFIG["scale"], choices=["tiny", "small", "medium", "large", "xlarge"], help="Graph scale")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # --- Simulation Params ---
    s = parser.add_argument_group("Simulation Parameters")
    s.add_argument("--cascade-threshold", type=float, default=0.5, help="Dependency weight threshold for cascade")
    s.add_argument("--cascade-prob", type=float, default=0.7, help="Probability of cascade propagation")
    s.add_argument("--cascade-depth", type=int, default=5, help="Maximum cascade depth")

    # --- Validation Params ---
    v = parser.add_argument_group("Validation Targets")
    v.add_argument("--target-spearman", type=float, default=0.70, help="Target Spearman Correlation")
    v.add_argument("--target-f1", type=float, default=0.80, help="Target F1 Score")

    args = parser.parse_args()

    # Default to printing help if no action provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    runner = PipelineRunner(args)
    runner.run()

if __name__ == "__main__":
    main()