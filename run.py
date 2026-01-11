#!/usr/bin/env python3
"""
Software-as-a-Graph Pipeline Orchestrator

This script executes the complete research methodology pipeline by orchestrating 
the following CLI tools:
1. Data Generation (generate_graph.py) - Optional
2. Model Construction (import_graph.py)
3. Structural Analysis (analyze_graph.py)
4. Failure Simulation (simulate_graph.py)
5. Statistical Validation (validate_graph.py)
6. Dashboard Visualization (visualize_graph.py)

Usage:
    python run.py --all
    python run.py --generate --import-data --analyze
    python run.py --visualize --output-dir ./my_results

Author: Software-as-a-Graph Research Project
"""

import argparse
import sys
import subprocess
import time
import logging
import shutil
from pathlib import Path
from typing import List, Optional

# --- Configuration & Defaults ---
DEFAULTS = {
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "password",
    "input_file": "input/system.json",
    "output_dir": "output",
    "scale": "medium",
    "seed": 42
}

# --- ANSI Colors for Terminal Output ---
COLORS = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "RED": "\033[91m",
    "BOLD": "\033[1m",
    "RESET": "\033[0m"
}

def print_c(msg, color="RESET", bold=False):
    """Helper to print colored messages."""
    style = COLORS.get("BOLD", "") if bold else ""
    code = COLORS.get(color, COLORS["RESET"])
    reset = COLORS["RESET"]
    print(f"{style}{code}{msg}{reset}")

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
        self.output_path = Path(args.output_dir).resolve()
        self.input_file = Path(args.input).resolve()
        
        # Common Neo4j arguments passed to most scripts
        self.neo4j_args = [
            "--uri", args.uri,
            "--user", args.user,
            "--password", args.password
        ]

    def _verify_scripts_exist(self):
        """Ensure all required scripts are present before starting."""
        required_scripts = [
            "generate_graph.py", "import_graph.py", "analyze_graph.py",
            "simulate_graph.py", "validate_graph.py", "visualize_graph.py"
        ]
        missing = []
        for script in required_scripts:
            if not (self.project_root / script).exists():
                missing.append(script)
        
        if missing:
            print_c(f"Error: Missing script files: {', '.join(missing)}", "RED")
            sys.exit(1)

    def _check_neo4j_connection(self) -> bool:
        """Check if Neo4j is reachable using the provided credentials."""
        print_c("Checking Neo4j connection...", "BLUE")
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(self.args.uri, auth=(self.args.user, self.args.password))
            driver.verify_connectivity()
            driver.close()
            print_c("Neo4j connection successful.", "GREEN")
            return True
        except ImportError:
            logger.warning("neo4j python driver not found. Skipping pre-check.")
            return True
        except Exception as e:
            print_c(f"Failed to connect to Neo4j: {e}", "RED")
            print_c("Hint: Check if the database is running and credentials are correct.", "YELLOW")
            return False

    def setup(self):
        """Prepare output directory and check prerequisites."""
        self._verify_scripts_exist()
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        if self._requires_neo4j():
            if not self._check_neo4j_connection():
                sys.exit(1)

    def run(self):
        """Orchestrate the pipeline stages."""
        self.setup()
        start_total = time.time()
        
        print_c("\n=== Starting Software-as-a-Graph Pipeline ===", "HEADER", bold=True)
        print(f"Input:  {self.input_file}")
        print(f"Output: {self.output_path}")

        # --- Stage 1: Generate Data ---
        if self.args.generate:
            self._run_stage_generate()
        elif not self.input_file.exists() and (self.args.do_import or self.args.all):
            print_c(f"Input file not found at {self.input_file}", "RED")
            print_c("Use --generate to create synthetic data or provide a valid --input path.", "YELLOW")
            sys.exit(1)

        # --- Stage 2: Import & Build Model ---
        if self.args.all or self.args.do_import:
            self._run_stage_import()

        # --- Stage 3: Analyze Model ---
        if self.args.all or self.args.analyze:
            self._run_stage_analyze()

        # --- Stage 4: Simulate Failures ---
        if self.args.all or self.args.simulate:
            self._run_stage_simulate()

        # --- Stage 5: Validate Model ---
        if self.args.all or self.args.validate:
            self._run_stage_validate()

        # --- Stage 6: Visualize Results ---
        if self.args.all or self.args.visualize:
            self._run_stage_visualize()

        elapsed = time.time() - start_total
        print_c(f"\n=== Pipeline Completed in {elapsed:.2f}s ===", "HEADER", bold=True)
        print(f"Results available in: {self.output_path}")

    # =========================================================================
    # Stage Implementations
    # =========================================================================

    def _run_stage_generate(self):
        """Generate Synthetic Graph Data."""
        self._print_stage_header("1. Data Generation")
        cmd = [
            str(self.project_root / "generate_graph.py"),
            "--scale", self.args.scale,
            "--output", str(self.input_file),
            "--seed", str(self.args.seed)
        ]
        self._exec_subprocess(cmd, "Generating graph topology")

    def _run_stage_import(self):
        """Import Data to Neo4j."""
        self._print_stage_header("2. Model Construction (Import)")
        cmd = [
            str(self.project_root / "import_graph.py"),
            "--input", str(self.input_file),
            "--clear" # Always clear DB for a fresh pipeline run
        ] + self.neo4j_args
        self._exec_subprocess(cmd, "Importing data and building graph model")

    def _run_stage_analyze(self):
        """Analyze Graph Model (Multi-Layer)."""
        self._print_stage_header("3. Graph Analysis")
        output_file = self.output_path / "analysis_results.json"
        cmd = [
            str(self.project_root / "analyze_graph.py"),
            "--all",  # Analyze all layers
            "--output", str(output_file)
        ] + self.neo4j_args
        self._exec_subprocess(cmd, "Calculating structural and quality metrics")

    def _run_stage_simulate(self):
        """Simulate Failures and Events."""
        self._print_stage_header("4. System Simulation")
        script = self.project_root / "simulate_graph.py"
        
        # 4a. Exhaustive Failure Report
        report_file = self.output_path / "simulation_report.json"
        cmd_report = [
            str(script),
            "--report",
            "--output", str(report_file)
        ] + self.neo4j_args
        self._exec_subprocess(cmd_report, "Running exhaustive failure simulation")

        # 4b. Event Propagation Demo (Visual feedback in console)
        # Find a suitable node to publish a message
        source_node = self._get_node_from_db("Application", criteria="publisher")
        if source_node:
            print_c(f"\n>> Running Live Event Demo (Source: {source_node})", "BLUE")
            cmd_event = [
                str(script), 
                "--event", source_node
            ] + self.neo4j_args
            subprocess.run([self.python_exe] + cmd_event)
        else:
            logger.warning("Could not find a publisher node for event demo.")

    def _run_stage_validate(self):
        """Validate Model vs Simulation."""
        self._print_stage_header("5. Statistical Validation")
        output_file = self.output_path / "validation_report.json"
        cmd = [
            str(self.project_root / "validate_graph.py"),
            "--all",
            "--output", str(output_file),
            "--spearman", str(self.args.target_spearman),
            "--f1", str(self.args.target_f1)
        ] + self.neo4j_args
        self._exec_subprocess(cmd, "Validating predictions against ground truth")

    def _run_stage_visualize(self):
        """Generate HTML Dashboard."""
        self._print_stage_header("6. Visualization")
        dashboard_file = self.output_path / "dashboard.html"
        cmd = [
            str(self.project_root / "visualize_graph.py"),
            "--output", str(dashboard_file),
            "--layer", "all", # Explicitly request all layers
            "--no-browser"
        ] + self.neo4j_args
        self._exec_subprocess(cmd, "Generating interactive dashboard")
        print_c(f"\nDashboard ready: {dashboard_file}", "GREEN", bold=True)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _exec_subprocess(self, cmd: List[str], desc: str):
        """Execute a subprocess with timing and error handling."""
        print(f"[{desc}]...")
        start = time.time()
        try:
            full_cmd = [self.python_exe] + cmd
            # We allow stdout to pass through for user feedback
            result = subprocess.run(full_cmd, check=True)
            duration = time.time() - start
            print(f"✓ Done ({duration:.2f}s)")
        except subprocess.CalledProcessError as e:
            print_c(f"✗ Failed (Exit Code: {e.returncode})", "RED")
            sys.exit(e.returncode)
        except Exception as e:
            print_c(f"✗ Execution Error: {e}", "RED")
            sys.exit(1)

    def _print_stage_header(self, title):
        print(f"\n{COLORS['BLUE']}{'='*60}{COLORS['RESET']}")
        print(f"{COLORS['BOLD']}{title}{COLORS['RESET']}")
        print(f"{COLORS['BLUE']}{'='*60}{COLORS['RESET']}")

    def _requires_neo4j(self) -> bool:
        """Determine if Neo4j is needed based on selected flags."""
        return (self.args.all or 
                self.args.do_import or 
                self.args.analyze or 
                self.args.simulate or 
                self.args.validate or 
                self.args.visualize)

    def _get_node_from_db(self, label: str, criteria: str = "random") -> Optional[str]:
        """Fetch a specific node ID from Neo4j for demo purposes."""
        try:
            from neo4j import GraphDatabase
            
            # Query selection logic
            query = f"MATCH (n:{label}) RETURN n.id as id LIMIT 1"
            if criteria == "publisher":
                query = f"MATCH (n:{label})-[:PUBLISHES_TO]->() RETURN n.id as id LIMIT 1"
            
            with GraphDatabase.driver(self.args.uri, auth=(self.args.user, self.args.password)) as driver:
                with driver.session() as session:
                    record = session.run(query).single()
                    if record:
                        return record["id"]
        except ImportError:
            pass # Driver missing
        except Exception as e:
            logger.debug(f"Failed to query Neo4j: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Software-as-a-Graph Pipeline Orchestrator")
    
    # Action Flags
    g = parser.add_argument_group("Pipeline Stages")
    g.add_argument("--all", action="store_true", help="Run the complete end-to-end pipeline")
    g.add_argument("--generate", action="store_true", help="Step 1: Generate synthetic graph data")
    g.add_argument("--import-data", dest="do_import", action="store_true", help="Step 2: Import data into Neo4j")
    g.add_argument("--analyze", action="store_true", help="Step 3: Analyze graph model")
    g.add_argument("--simulate", action="store_true", help="Step 4: Simulate failures & events")
    g.add_argument("--validate", action="store_true", help="Step 5: Validate model predictions")
    g.add_argument("--visualize", action="store_true", help="Step 6: Generate dashboard")

    # Config Flags
    c = parser.add_argument_group("Configuration")
    c.add_argument("--input", default=DEFAULTS["input_file"], help=f"Input JSON path (default: {DEFAULTS['input_file']})")
    c.add_argument("--output-dir", default=DEFAULTS["output_dir"], help=f"Output directory (default: {DEFAULTS['output_dir']})")
    c.add_argument("--scale", default=DEFAULTS["scale"], choices=["tiny", "small", "medium", "large", "xlarge"], help="Graph scale for generation")
    c.add_argument("--seed", type=int, default=DEFAULTS["seed"], help="Random seed")

    # Neo4j Flags
    n = parser.add_argument_group("Neo4j Connection")
    n.add_argument("--uri", default=DEFAULTS["uri"], help="Neo4j URI")
    n.add_argument("--user", default=DEFAULTS["user"], help="Neo4j User")
    n.add_argument("--password", default=DEFAULTS["password"], help="Neo4j Password")

    # Validation Thresholds
    v = parser.add_argument_group("Validation Targets")
    v.add_argument("--target-spearman", type=float, default=0.70, help="Target Spearman Correlation (default: 0.70)")
    v.add_argument("--target-f1", type=float, default=0.80, help="Target F1 Score (default: 0.80)")

    args = parser.parse_args()

    # If no arguments provided, print help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    runner = PipelineRunner(args)
    runner.run()

if __name__ == "__main__":
    main()