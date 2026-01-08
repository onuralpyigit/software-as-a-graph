#!/usr/bin/env python3
"""
Software-as-a-Graph Pipeline Runner

Orchestrates the complete PhD research methodology workflow using the CLI tools:
1. Data Generation (Optional): Creates synthetic graph topologies.
2. Model Construction: Imports data to Neo4j and derives dependencies.
3. Analysis: Structural and Quality analysis for all graph layers.
4. Simulation: Failure impact assessment and event propagation.
5. Validation: Statistical comparison of Prediction vs. Reality.
6. Visualization: Generates a comprehensive HTML dashboard.

Usage:
    # Run full end-to-end pipeline
    python run.py --all

    # Run specific stages (e.g., just Analyze and Visualize)
    python run.py --analyze --visualize

Author: Software-as-a-Graph Research Project
"""

import argparse
import sys
import subprocess
import time
import logging
import shutil
from pathlib import Path
from typing import List

# --- Configuration & Defaults ---
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
        self.project_root = Path(__file__).parent.resolve()
        
        # Setup Output Directory
        self.output_path = Path(args.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Common Neo4j arguments for subprocesses
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
                print(f"{YELLOW}Hint: Ensure your Docker container is running (docker-compose up -d){RESET}")
                sys.exit(1)

        # 1. Generate Data (Optional)
        if self.args.generate:
            self._run_stage_generate()

        # 2. Import & Build Model
        if self.args.all or self.args.do_import:
            self._run_stage_import()

        # 3. Analyze Model (Multi-Layer)
        if self.args.all or self.args.analyze:
            self._run_stage_analyze()

        # 4. Simulate Failures (Report & Demo)
        if self.args.all or self.args.simulate:
            self._run_stage_simulate()

        # 5. Validate Model (Statistical)
        if self.args.all or self.args.validate:
            self._run_stage_validate()

        # 6. Visualize Results (Dashboard)
        if self.args.all or self.args.visualize:
            self._run_stage_visualize()

        self._print_footer()

    # --- Stage Implementations ---

    def _run_stage_generate(self):
        """Stage 1: Generate Synthetic Graph Data"""
        script = self.project_root / "generate_graph.py"
        cmd = [
            str(script),
            "--scale", self.args.scale,
            "--output", self.args.input,
            "--seed", str(self.args.seed)
        ]
        self._run_subprocess("Data Generation", cmd)

    def _run_stage_import(self):
        """Stage 2: Import Data to Neo4j"""
        script = self.project_root / "import_graph.py"
        cmd = [
            str(script),
            "--input", self.args.input,
            "--clear"
        ] + self.neo4j_args
        self._run_subprocess("Graph Import & Model Build", cmd)

    def _run_stage_analyze(self):
        """Stage 3: Analyze Graph Model"""
        # Analyzes all layers (Application, Infrastructure, Complete)
        script = self.project_root / "analyze_graph.py"
        output_file = self.output_path / "analysis_results.json"
        
        cmd = [
            str(script),
            "--all", # Trigger multi-layer batch analysis
            "--output", str(output_file)
        ] + self.neo4j_args
        self._run_subprocess("Multi-Layer Structural Analysis", cmd)

    def _run_stage_simulate(self):
        """Stage 4: Simulate Failures"""
        script = self.project_root / "simulate_graph.py"
        
        # 4a. System Evaluation Report (Exhaustive Simulation)
        report_file = self.output_path / "simulation_report.json"
        print(f"\n{BLUE}>> Generating System Evaluation Report...{RESET}")
        cmd_report = [
            str(script),
            "--report",
            "--output", str(report_file)
        ] + self.neo4j_args
        self._run_subprocess("System Failure Simulation", cmd_report)

        # 4b. Single Event Propagation Demo (Visual feedback)
        # We pick a random Application node to act as a publisher
        source_node = self._get_smart_target("Application", criteria="publisher")
        if source_node:
            print(f"\n{BLUE}>> Running Event Propagation Demo (Source: {source_node})...{RESET}")
            cmd_event = [
                str(script), 
                "--event", source_node
            ] + self.neo4j_args
            subprocess.run([self.python_exe] + cmd_event, check=False)

    def _run_stage_validate(self):
        """Stage 5: Validate Model"""
        script = self.project_root / "validate_graph.py"
        output_file = self.output_path / "validation_report.json"
        
        cmd = [
            str(script),
            "--all", # Validate all layers
            "--output", str(output_file),
            "--target-spearman", str(self.args.target_spearman),
            "--target-f1", str(self.args.target_f1)
        ] + self.neo4j_args
        self._run_subprocess("Statistical Model Validation", cmd)

    def _run_stage_visualize(self):
        """Stage 6: Visualize Results"""
        script = self.project_root / "visualize_graph.py"
        dashboard_file = self.output_path / "dashboard.html"
        
        cmd = [
            str(script),
            "--output", str(dashboard_file),
            "--no-browser" 
        ] + self.neo4j_args
        
        self._run_subprocess("Dashboard Generation", cmd)
        print(f"\n{GREEN}{BOLD}>> Dashboard available at: {dashboard_file.absolute()}{RESET}")

    # --- Helpers ---

    def _run_subprocess(self, name: str, command: List[str]):
        """Helper to run a CLI script safely."""
        print(f"\n{BLUE}[Running] {name}...{RESET}")
        
        full_cmd = [self.python_exe] + command
        start_time = time.time()
        
        try:
            # We assume scripts are well-behaved and exit with 0 on success
            result = subprocess.run(full_cmd, check=False)
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"{GREEN}✓ {name} completed ({duration:.2f}s){RESET}")
            else:
                print(f"{RED}✗ {name} failed with code {result.returncode}{RESET}")
                # We exit the pipeline if a critical stage fails
                sys.exit(result.returncode)
                
        except KeyboardInterrupt:
            print(f"\n{YELLOW}Pipeline interrupted by user.{RESET}")
            sys.exit(130)
        except Exception as e:
            print(f"{RED}✗ Execution Error: {e}{RESET}")
            sys.exit(1)

    def _get_smart_target(self, label: str, criteria: str = "random") -> str:
        """
        Connects to Neo4j to find a relevant node ID for simulation demos.
        """
        try:
            from neo4j import GraphDatabase
            
            query = ""
            if criteria == "publisher":
                query = f"MATCH (n:{label})-[:PUBLISHES_TO]->() RETURN n.id as id LIMIT 1"
            elif criteria == "hub":
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
        return "A1" if label == "Application" else "N1"

    def _check_connection(self) -> bool:
        """Verify Neo4j connectivity."""
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(self.args.uri, auth=(self.args.user, self.args.password))
            driver.verify_connectivity()
            driver.close()
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
  \__ \/ __ `/ / __        Research Pipeline v2.1     
 ___/ / /_/ / /_/ /                                   
/____/\__,_/\____/                                    
{RESET}""")
        print(f"Configuration:")
        print(f"  URI:    {self.args.uri}")
        print(f"  Input:  {self.args.input}")
        print(f"  Output: {self.args.output_dir}")
        print(f"  Mode:   {'ALL (End-to-End)' if self.args.all else 'Selective'}")
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
    g.add_argument("--generate", action="store_true", help="1. Generate Synthetic Data (Default: Skip if input exists)")
    g.add_argument("--import-data", dest="do_import", action="store_true", help="2. Import Data to Neo4j")
    g.add_argument("--analyze", action="store_true", help="3. Analyze Graph Model (Structure & Quality)")
    g.add_argument("--simulate", action="store_true", help="4. Simulate Failures (Report & Demo)")
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