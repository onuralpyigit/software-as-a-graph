#!/usr/bin/env python3
"""
Software-as-a-Graph Pipeline Orchestrator

End-to-end pipeline for graph-based modeling and analysis of
distributed publish-subscribe systems.

Pipeline Stages:
    1. Generate   - Create synthetic graph data (optional)
    2. Import     - Build graph model in Neo4j
    3. Analyze    - Compute structural metrics and quality scores
    4. Simulate   - Run failure simulation for impact assessment
    5. Validate   - Compare predictions against simulation results
    6. Visualize  - Generate interactive HTML dashboard

Layers:
    app      - Application layer
    infra    - Infrastructure layer
    mw-app   - Middleware-Application layer
    mw-infra - Middleware-Infrastructure layer
    system   - Complete system

Usage:
    python run.py --all                          # Full pipeline
    python run.py --generate --import --analyze  # Partial pipeline
    python run.py --layer app                    # Specific layers
    python run.py --programmatic                 # Direct API calls

Author: Software-as-a-Graph Research Project
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    # Neo4j connection
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    
    # Paths
    input_file: str = "output/system.json"
    output_dir: str = "output"
    scripts_dir: str = "scripts"
    
    # Generation
    scale: str = "medium"
    seed: int = 42
    
    # Layers to process
    layers: List[str] = field(default_factory=lambda: ["app", "infra", "system"])
    
    # Validation targets
    target_spearman: float = 0.70
    target_f1: float = 0.80
    target_precision: float = 0.80
    target_recall: float = 0.80
    
    # Options
    verbose: bool = False
    quiet: bool = False
    open_browser: bool = False


# Layer definitions
LAYER_DEFINITIONS = {
    "app": {"name": "Application Layer", "icon": "📱"},
    "infra": {"name": "Infrastructure Layer", "icon": "🖥️"},
    "mw-app": {"name": "Middleware-Application Layer", "icon": "🔗"},
    "mw-infra": {"name": "Middleware-Infrastructure Layer", "icon": "⚙️"},
    "system": {"name": "Complete System", "icon": "🌐"},
}


# =============================================================================
# Terminal Styling
# =============================================================================

class Colors:
    """ANSI color codes."""
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def colored(text: str, color: str, bold: bool = False) -> str:
    """Apply color to text."""
    style = Colors.BOLD if bold else ""
    return f"{style}{color}{text}{Colors.RESET}"


def print_header(title: str, char: str = "═", width: int = 70) -> None:
    """Print a formatted header."""
    line = char * width
    print(f"\n{colored(line, Colors.CYAN)}")
    print(f"{colored(f' {title} '.center(width), Colors.CYAN, bold=True)}")
    print(f"{colored(line, Colors.CYAN)}")


def print_stage(stage: int, total: int, title: str, icon: str = "▶") -> None:
    """Print a stage header."""
    print(f"\n{colored(f'{icon} Stage {stage}/{total}: {title}', Colors.BLUE, bold=True)}")
    print(f"  {colored('─' * 50, Colors.GRAY)}")


def print_step(message: str, indent: int = 2) -> None:
    """Print a step message."""
    print(f"{' ' * indent}→ {message}")


def print_success(message: str, indent: int = 2) -> None:
    """Print a success message."""
    print(f"{' ' * indent}{colored('✓', Colors.GREEN, bold=True)} {message}")


def print_warning(message: str, indent: int = 2) -> None:
    """Print a warning message."""
    print(f"{' ' * indent}{colored('⚠', Colors.YELLOW)} {message}")


def print_error(message: str, indent: int = 2) -> None:
    """Print an error message."""
    print(f"{' ' * indent}{colored('✗', Colors.RED, bold=True)} {message}")


def print_result(label: str, value: Any, passed: bool = None, indent: int = 4) -> None:
    """Print a result line."""
    if passed is not None:
        icon = colored("✓", Colors.GREEN) if passed else colored("✗", Colors.RED)
        print(f"{' ' * indent}{icon} {label}: {value}")
    else:
        print(f"{' ' * indent}{label}: {value}")


# =============================================================================
# Stage Results
# =============================================================================

@dataclass
class StageResult:
    """Result from a pipeline stage."""
    stage: str
    success: bool
    duration: float
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""
    timestamp: str
    config: PipelineConfig
    stages: List[StageResult] = field(default_factory=list)
    total_duration: float = 0.0
    success: bool = True
    
    def add_stage(self, result: StageResult) -> None:
        """Add a stage result."""
        self.stages.append(result)
        if not result.success:
            self.success = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "total_duration": self.total_duration,
            "success": self.success,
            "stages": [
                {
                    "stage": s.stage,
                    "success": s.success,
                    "duration": s.duration,
                    "message": s.message,
                    "data": s.data,
                }
                for s in self.stages
            ],
        }


# =============================================================================
# Pipeline Runner
# =============================================================================

class PipelineRunner:
    """
    Orchestrates the end-to-end pipeline execution.
    
    Supports two execution modes:
        - Subprocess: Calls CLI scripts via subprocess
        - Programmatic: Direct API calls (faster, more detailed)
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.project_root = Path(__file__).parent.resolve()
        self.output_path = Path(config.output_dir).resolve()
        self.input_file = Path(config.input_file).resolve()
        
        self.logger = logging.getLogger("Pipeline")
        
        # Execution mode
        self._use_programmatic = False
        
        # Lazy-loaded modules
        self._analyzer = None
        self._simulator = None
        self._validator = None
    
    # =========================================================================
    # Setup & Validation
    # =========================================================================
    
    def setup(self) -> bool:
        """Initialize pipeline, verify prerequisites."""
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Verify scripts exist
        missing = self._verify_scripts()
        if missing:
            print_error(f"Missing scripts: {', '.join(missing)}")
            return False
        
        # Check Neo4j connection
        if not self._check_neo4j():
            return False
        
        return True
    
    def _verify_scripts(self) -> List[str]:
        """Verify all required scripts exist."""
        scripts_dir = self.project_root / self.config.scripts_dir
        required = [
            "generate_graph.py",
            "import_graph.py",
            "analyze_graph.py",
            "simulate_graph.py",
            "validate_graph.py",
            "visualize_graph.py",
        ]
        
        missing = []
        for script in required:
            if not (scripts_dir / script).exists():
                # Also check project root
                if not (self.project_root / script).exists():
                    missing.append(script)
        
        return missing
    
    def _check_neo4j(self) -> bool:
        """Check Neo4j connectivity."""
        print_step("Checking Neo4j connection...")
        
        try:
            from neo4j import GraphDatabase
            
            driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.user, self.config.password)
            )
            driver.verify_connectivity()
            driver.close()
            
            print_success("Neo4j connection successful")
            return True
        
        except ImportError:
            print_warning("neo4j driver not installed, skipping pre-check")
            return True
        
        except Exception as e:
            print_error(f"Neo4j connection failed: {e}")
            print_warning("Ensure Neo4j is running and credentials are correct")
            return False
    
    def _get_script_path(self, script: str) -> Path:
        """Get path to a script."""
        scripts_dir = self.project_root / self.config.scripts_dir
        if (scripts_dir / script).exists():
            return scripts_dir / script
        return self.project_root / script
    
    def _neo4j_args(self) -> List[str]:
        """Get common Neo4j connection arguments."""
        return [
            "--uri", self.config.uri,
            "--user", self.config.user,
            "--password", self.config.password,
        ]
    
    # =========================================================================
    # Subprocess Execution
    # =========================================================================
    
    def _run_subprocess(
        self,
        script: str,
        args: List[str],
        description: str
    ) -> Tuple[bool, str, str]:
        """Run a script via subprocess."""
        script_path = self._get_script_path(script)
        cmd = [sys.executable, str(script_path)] + args
        
        if self.config.verbose:
            print_step(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
            )
            
            if result.returncode == 0:
                return True, result.stdout, result.stderr
            else:
                return False, result.stdout, result.stderr
        
        except Exception as e:
            return False, "", str(e)
    
    # =========================================================================
    # Programmatic Execution
    # =========================================================================
    
    @property
    def analyzer(self):
        """Lazy-load analyzer."""
        if self._analyzer is None:
            try:
                from src.analysis import GraphAnalyzer
                self._analyzer = GraphAnalyzer(
                    uri=self.config.uri,
                    user=self.config.user,
                    password=self.config.password,
                )
            except ImportError:
                pass
        return self._analyzer
    
    @property
    def simulator(self):
        """Lazy-load simulator."""
        if self._simulator is None:
            try:
                from src.simulation import Simulator
                self._simulator = Simulator(
                    uri=self.config.uri,
                    user=self.config.user,
                    password=self.config.password,
                )
            except ImportError:
                pass
        return self._simulator
    
    @property
    def validator(self):
        """Lazy-load validator."""
        if self._validator is None:
            try:
                from src.validation import Validator, ValidationTargets
                targets = ValidationTargets(
                    spearman=self.config.target_spearman,
                    f1_score=self.config.target_f1,
                    precision=self.config.target_precision,
                    recall=self.config.target_recall,
                )
                self._validator = Validator(targets)
            except ImportError:
                pass
        return self._validator
    
    # =========================================================================
    # Pipeline Stages
    # =========================================================================
    
    def run_generate(self) -> StageResult:
        """Stage 1: Generate synthetic graph data."""
        start = time.time()
        
        print_step(f"Generating {self.config.scale} scale graph...")
        
        # Ensure input directory exists
        self.input_file.parent.mkdir(parents=True, exist_ok=True)
        
        args = [
            "--scale", self.config.scale,
            "--seed", str(self.config.seed),
            "--output", str(self.input_file),
        ]
        
        success, stdout, stderr = self._run_subprocess(
            "generate_graph.py", args, "Generating graph data"
        )
        
        if success:
            print_success(f"Graph data saved to: {self.input_file}")
        else:
            print_error(f"Generation failed: {stderr}")
        
        return StageResult(
            stage="generate",
            success=success,
            duration=time.time() - start,
            message=f"Generated {self.config.scale} graph",
            data={"output_file": str(self.input_file)},
        )
    
    def run_import(self) -> StageResult:
        """Stage 2: Import data into Neo4j."""
        start = time.time()
        
        if not self.input_file.exists():
            print_error(f"Input file not found: {self.input_file}")
            return StageResult(
                stage="import",
                success=False,
                duration=time.time() - start,
                message="Input file not found",
                errors=[f"File not found: {self.input_file}"],
            )
        
        print_step(f"Importing from: {self.input_file}")
        print_step("Clearing existing data...")
        
        args = [
            "--input", str(self.input_file),
            "--clear",
        ] + self._neo4j_args()
        
        success, stdout, stderr = self._run_subprocess(
            "import_graph.py", args, "Importing graph data"
        )
        
        if success:
            # Parse component counts from output
            print_success("Graph model built in Neo4j")
        else:
            print_error(f"Import failed: {stderr}")
        
        return StageResult(
            stage="import",
            success=success,
            duration=time.time() - start,
            message="Graph imported to Neo4j",
            data={"input_file": str(self.input_file)},
        )
    
    def run_analyze(self, programmatic: bool = False) -> StageResult:
        """Stage 3: Analyze graph model."""
        start = time.time()
        
        layers_str = ",".join(self.config.layers)
        print_step(f"Analyzing layers: {layers_str}")
        
        analysis_results = {}
        
        if programmatic and self.analyzer:
            # Programmatic execution
            for layer in self.config.layers:
                layer_def = LAYER_DEFINITIONS.get(layer, {"name": layer, "icon": "📊"})
                print_step(f"  {layer_def['icon']} {layer_def['name']}...")
                
                try:
                    result = self.analyzer.analyze_layer(layer)
                    
                    analysis_results[layer] = {
                        "nodes": result.structural.graph_summary.nodes,
                        "edges": result.structural.graph_summary.edges,
                        "density": result.structural.graph_summary.density,
                        "critical_count": len([
                            c for c in result.quality.components
                            if hasattr(c.levels.overall, 'name') and c.levels.overall.name == "CRITICAL"
                        ]),
                        "problems": len(result.problems),
                    }
                    
                    print_result(
                        f"{layer}",
                        f"N={result.structural.graph_summary.nodes}, "
                        f"E={result.structural.graph_summary.edges}, "
                        f"Problems={len(result.problems)}"
                    )
                
                except Exception as e:
                    print_error(f"Analysis failed for {layer}: {e}")
                    analysis_results[layer] = {"error": str(e)}
            
            success = all("error" not in r for r in analysis_results.values())
        
        else:
            # Subprocess execution
            output_file = self.output_path / "analysis_results.json"
            
            args = [
                "--layer", layers_str,
                "--output", str(output_file),
            ] + self._neo4j_args()

            if self.config.verbose:
                args.append("--verbose")
            
            success, stdout, stderr = self._run_subprocess(
                "analyze_graph.py", args, "Running analysis"
            )
            
            if success:
                print_success(f"Analysis results saved to: {output_file}")
                analysis_results["output_file"] = str(output_file)
            else:
                print_error(f"Analysis failed: {stderr}")
        
        return StageResult(
            stage="analyze",
            success=success,
            duration=time.time() - start,
            message=f"Analyzed {len(self.config.layers)} layers",
            data=analysis_results,
        )
    
    def run_simulate(self, programmatic: bool = False) -> StageResult:
        """Stage 4: Run failure simulation."""
        start = time.time()
        
        layers_str = ",".join(self.config.layers)
        print_step(f"Simulating failures for layers: {layers_str}")
        
        simulation_results = {}
        
        if programmatic and self.simulator:
            # Programmatic execution
            for layer in self.config.layers:
                layer_def = LAYER_DEFINITIONS.get(layer, {"name": layer, "icon": "📊"})
                print_step(f"  {layer_def['icon']} {layer_def['name']}...")
                
                try:
                    # Run exhaustive failure simulation
                    results = self.simulator.run_failure_simulation_exhaustive(layer=layer)
                    
                    if results:
                        avg_impact = sum(r.impact.composite_impact for r in results) / len(results)
                        max_impact = max(r.impact.composite_impact for r in results)
                        
                        simulation_results[layer] = {
                            "components_tested": len(results),
                            "avg_impact": avg_impact,
                            "max_impact": max_impact,
                        }
                        
                        print_result(
                            f"{layer}",
                            f"Tested={len(results)}, "
                            f"AvgImpact={avg_impact:.3f}, "
                            f"MaxImpact={max_impact:.3f}"
                        )
                    else:
                        simulation_results[layer] = {"components_tested": 0}
                
                except Exception as e:
                    print_error(f"Simulation failed for {layer}: {e}")
                    simulation_results[layer] = {"error": str(e)}
            
            success = all("error" not in r for r in simulation_results.values())
        
        else:
            # Subprocess execution
            output_file = self.output_path / "simulation_results.json"
            
            args = [
                "--layer", layers_str,
                "--exhaustive",
                "--output", str(output_file),
            ] + self._neo4j_args()
            
            success, stdout, stderr = self._run_subprocess(
                "simulate_graph.py", args, "Running simulation"
            )
            
            if success:
                print_success(f"Simulation results saved to: {output_file}")
                simulation_results["output_file"] = str(output_file)
            else:
                print_error(f"Simulation failed: {stderr}")
        
        return StageResult(
            stage="simulate",
            success=success,
            duration=time.time() - start,
            message=f"Simulated failures for {len(self.config.layers)} layers",
            data=simulation_results,
        )
    
    def run_validate(self, programmatic: bool = False) -> StageResult:
        """Stage 5: Validate analysis vs simulation."""
        start = time.time()
        
        layers_str = ",".join(self.config.layers)
        print_step(f"Validating predictions for layers: {layers_str}")
        print_step(f"Targets: ρ≥{self.config.target_spearman}, F1≥{self.config.target_f1}")
        
        validation_results = {}
        all_passed = True
        
        if programmatic and self.analyzer and self.simulator and self.validator:
            # Programmatic execution
            for layer in self.config.layers:
                layer_def = LAYER_DEFINITIONS.get(layer, {"name": layer, "icon": "📊"})
                print_step(f"  {layer_def['icon']} {layer_def['name']}...")
                
                try:
                    # Get predictions from analysis
                    analysis = self.analyzer.analyze_layer(layer)
                    predicted = {c.id: c.scores.overall for c in analysis.quality.components}
                    types = {c.id: c.type for c in analysis.quality.components}
                    
                    # Get actual from simulation
                    sim_results = self.simulator.run_failure_simulation_exhaustive(layer=layer)
                    actual = {r.target_id: r.impact.composite_impact for r in sim_results}
                    
                    # Validate
                    val_result = self.validator.validate(
                        predicted_scores=predicted,
                        actual_scores=actual,
                        component_types=types,
                        layer=layer,
                    )
                    
                    passed = val_result.passed
                    if not passed:
                        all_passed = False
                    
                    validation_results[layer] = {
                        "spearman": val_result.overall.correlation.spearman,
                        "f1_score": val_result.overall.classification.f1_score,
                        "precision": val_result.overall.classification.precision,
                        "recall": val_result.overall.classification.recall,
                        "passed": passed,
                    }
                    
                    status = colored("PASS", Colors.GREEN) if passed else colored("FAIL", Colors.RED)
                    print_result(
                        f"{layer}",
                        f"ρ={val_result.overall.correlation.spearman:.3f}, "
                        f"F1={val_result.overall.classification.f1_score:.3f} [{status}]",
                        passed=passed
                    )
                
                except Exception as e:
                    print_error(f"Validation failed for {layer}: {e}")
                    validation_results[layer] = {"error": str(e)}
                    all_passed = False
            
            success = True  # Stage succeeded even if validation failed
        
        else:
            # Subprocess execution
            output_file = self.output_path / "validation_results.json"
            
            args = [
                "--layer", layers_str,
                "--spearman", str(self.config.target_spearman),
                "--f1", str(self.config.target_f1),
                "--output", str(output_file),
            ] + self._neo4j_args()
            
            success, stdout, stderr = self._run_subprocess(
                "validate_graph.py", args, "Running validation"
            )
            
            if success:
                print_success(f"Validation results saved to: {output_file}")
                validation_results["output_file"] = str(output_file)
                # Try to read results
                if output_file.exists():
                    try:
                        with open(output_file) as f:
                            data = json.load(f)
                            all_passed = data.get("all_passed", False)
                    except:
                        pass
            else:
                print_error(f"Validation failed: {stderr}")
        
        validation_results["all_passed"] = all_passed
        
        return StageResult(
            stage="validate",
            success=success,
            duration=time.time() - start,
            message=f"Validation {'PASSED' if all_passed else 'FAILED'}",
            data=validation_results,
        )
    
    def run_visualize(self) -> StageResult:
        """Stage 6: Generate dashboard visualization."""
        start = time.time()
        
        layers_str = ",".join(self.config.layers)
        output_file = self.output_path / "dashboard.html"
        
        print_step(f"Generating dashboard for layers: {layers_str}")
        print_step(f"Output: {output_file}")
        
        args = [
            "--layers", layers_str,
            "--output", str(output_file),
        ] + self._neo4j_args()
        
        if self.config.open_browser:
            args.append("--open")
        
        success, stdout, stderr = self._run_subprocess(
            "visualize_graph.py", args, "Generating dashboard"
        )
        
        if success:
            print_success(f"Dashboard generated: {output_file}")
            if self.config.open_browser:
                print_step("Opening in browser...")
        else:
            print_error(f"Visualization failed: {stderr}")
        
        return StageResult(
            stage="visualize",
            success=success,
            duration=time.time() - start,
            message="Dashboard generated",
            data={"output_file": str(output_file)},
        )
    
    # =========================================================================
    # Pipeline Execution
    # =========================================================================
    
    def run(
        self,
        generate: bool = False,
        do_import: bool = False,
        analyze: bool = False,
        simulate: bool = False,
        validate: bool = False,
        visualize: bool = False,
        all_stages: bool = False,
        programmatic: bool = False,
    ) -> PipelineResult:
        """
        Execute the pipeline.
        
        Args:
            generate: Run generation stage
            do_import: Run import stage
            analyze: Run analysis stage
            simulate: Run simulation stage
            validate: Run validation stage
            visualize: Run visualization stage
            all_stages: Run all stages
            programmatic: Use direct API calls instead of subprocess
            
        Returns:
            PipelineResult with all stage results
        """
        result = PipelineResult(
            timestamp=datetime.now().isoformat(),
            config=self.config,
        )
        
        # Determine which stages to run
        if all_stages:
            generate = do_import = analyze = simulate = validate = visualize = True
        
        # Count stages
        stages = []
        if generate:
            stages.append("generate")
        if do_import:
            stages.append("import")
        if analyze:
            stages.append("analyze")
        if simulate:
            stages.append("simulate")
        if validate:
            stages.append("validate")
        if visualize:
            stages.append("visualize")
        
        total_stages = len(stages)
        
        if total_stages == 0:
            print_warning("No stages selected. Use --help for usage.")
            return result
        
        # Setup
        if not self.setup():
            result.success = False
            return result
        
        start_total = time.time()
        
        # Print header
        if not self.config.quiet:
            print_header("SOFTWARE-AS-A-GRAPH PIPELINE", "═")
            print(f"\n  {colored('Mode:', Colors.CYAN)} {'Programmatic' if programmatic else 'Subprocess'}")
            print(f"  {colored('Layers:', Colors.CYAN)} {', '.join(self.config.layers)}")
            print(f"  {colored('Output:', Colors.CYAN)} {self.output_path}")
        
        current_stage = 0
        
        # Stage 1: Generate
        if generate:
            current_stage += 1
            print_stage(current_stage, total_stages, "Data Generation", "🔧")
            stage_result = self.run_generate()
            result.add_stage(stage_result)
            
            if not stage_result.success:
                print_error("Pipeline aborted due to generation failure")
                return result
        
        # Stage 2: Import
        if do_import:
            current_stage += 1
            print_stage(current_stage, total_stages, "Model Construction", "📥")
            stage_result = self.run_import()
            result.add_stage(stage_result)
            
            if not stage_result.success:
                print_error("Pipeline aborted due to import failure")
                return result
        
        # Stage 3: Analyze
        if analyze:
            current_stage += 1
            print_stage(current_stage, total_stages, "Graph Analysis", "📊")
            stage_result = self.run_analyze(programmatic=programmatic)
            result.add_stage(stage_result)
        
        # Stage 4: Simulate
        if simulate:
            current_stage += 1
            print_stage(current_stage, total_stages, "Failure Simulation", "💥")
            stage_result = self.run_simulate(programmatic=programmatic)
            result.add_stage(stage_result)
        
        # Stage 5: Validate
        if validate:
            current_stage += 1
            print_stage(current_stage, total_stages, "Statistical Validation", "✅")
            stage_result = self.run_validate(programmatic=programmatic)
            result.add_stage(stage_result)
        
        # Stage 6: Visualize
        if visualize:
            current_stage += 1
            print_stage(current_stage, total_stages, "Dashboard Visualization", "📈")
            stage_result = self.run_visualize()
            result.add_stage(stage_result)
        
        # Summary
        result.total_duration = time.time() - start_total
        
        if not self.config.quiet:
            self._print_summary(result)
        
        # Save results
        results_file = self.output_path / "pipeline_results.json"
        with open(results_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        return result
    
    def _print_summary(self, result: PipelineResult) -> None:
        """Print pipeline summary."""
        print_header("PIPELINE COMPLETE", "─")
        
        # Stage summary
        print(f"\n  {colored('Stage Summary:', Colors.BOLD, bold=True)}")
        for stage in result.stages:
            status = colored("✓ PASS", Colors.GREEN) if stage.success else colored("✗ FAIL", Colors.RED)
            print(f"    {stage.stage.capitalize():<12} {status:<20} ({stage.duration:.2f}s)")
        
        # Overall status
        print(f"\n  {colored('Total Duration:', Colors.CYAN)} {result.total_duration:.2f}s")
        
        overall_status = (
            colored("✓ SUCCESS", Colors.GREEN, bold=True)
            if result.success
            else colored("✗ FAILED", Colors.RED, bold=True)
        )
        print(f"  {colored('Overall Status:', Colors.CYAN)} {overall_status}")
        
        print(f"\n  {colored('Results:', Colors.CYAN)} {self.output_path}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Software-as-a-Graph Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Layers:
  app      Application layer (Applications only)
  infra    Infrastructure layer (Nodes only)
  mw-app   Middleware-Application (Applications + Brokers)
  mw-infra Middleware-Infrastructure (Nodes + Brokers)
  system   Complete system (all components)

Examples:
  %(prog)s --all                             # Run complete pipeline
  %(prog)s --generate --import --analyze     # Generate, import, analyze
  %(prog)s --analyze --simulate --validate   # Analysis and validation only
  %(prog)s --all --layers app,system         # Full pipeline, specific layers
  %(prog)s --all --programmatic              # Direct API calls (faster)
        """
    )
    
    # Pipeline stages
    stages = parser.add_argument_group("Pipeline Stages")
    stages.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run complete end-to-end pipeline"
    )
    stages.add_argument(
        "--generate", "-g",
        action="store_true",
        help="Stage 1: Generate synthetic graph data"
    )
    stages.add_argument(
        "--import", dest="do_import",
        action="store_true",
        help="Stage 2: Import data into Neo4j"
    )
    stages.add_argument(
        "--analyze", "-A",
        action="store_true",
        help="Stage 3: Analyze graph model"
    )
    stages.add_argument(
        "--simulate", "-s",
        action="store_true",
        help="Stage 4: Run failure simulation"
    )
    stages.add_argument(
        "--validate", "-V",
        action="store_true",
        help="Stage 5: Validate predictions"
    )
    stages.add_argument(
        "--visualize", "-z",
        action="store_true",
        help="Stage 6: Generate dashboard"
    )
    
    # Layer selection
    layer_group = parser.add_argument_group("Layer Selection")
    layer_group.add_argument(
        "--layer", "-l",
        default="app",
        help="layer (default: app)"
    )
    
    # Neo4j connection
    neo4j = parser.add_argument_group("Neo4j Connection")
    neo4j.add_argument(
        "--uri",
        default="bolt://localhost:7687",
        help="Neo4j URI (default: bolt://localhost:7687)"
    )
    neo4j.add_argument(
        "--user", "-u",
        default="neo4j",
        help="Neo4j username (default: neo4j)"
    )
    neo4j.add_argument(
        "--password", "-p",
        default="password",
        help="Neo4j password (default: password)"
    )
    
    # Paths
    paths = parser.add_argument_group("Paths")
    paths.add_argument(
        "--input", "-i",
        default="input/system.json",
        help="Input JSON file (default: input/system.json)"
    )
    paths.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Output directory (default: output)"
    )
    paths.add_argument(
        "--scripts-dir",
        default="scripts",
        help="Scripts directory (default: scripts)"
    )
    
    # Generation options
    gen = parser.add_argument_group("Generation Options")
    gen.add_argument(
        "--scale",
        default="medium",
        choices=["tiny", "small", "medium", "large", "xlarge"],
        help="Graph scale (default: medium)"
    )
    gen.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    # Validation targets
    targets = parser.add_argument_group("Validation Targets")
    targets.add_argument(
        "--target-spearman",
        type=float,
        default=0.70,
        help="Target Spearman correlation (default: 0.70)"
    )
    targets.add_argument(
        "--target-f1",
        type=float,
        default=0.80,
        help="Target F1 score (default: 0.80)"
    )
    
    # Execution options
    exec_opts = parser.add_argument_group("Execution Options")
    exec_opts.add_argument(
        "--programmatic",
        action="store_true",
        help="Use direct API calls instead of subprocess"
    )
    exec_opts.add_argument(
        "--open-browser",
        action="store_true",
        help="Open dashboard in browser after generation"
    )
    
    # Output options
    output = parser.add_argument_group("Output Options")
    output.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    output.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Check if any stage selected
    has_stage = any([
        args.all,
        args.generate,
        args.do_import,
        args.analyze,
        args.simulate,
        args.validate,
        args.visualize,
    ])
    
    if not has_stage:
        print("No pipeline stages selected.")
        print("Use --all for complete pipeline or --help for options.")
        return 1
    
    # Parse layers
    layers = [args.layer]
    valid_layers = [l for l in layers if l in LAYER_DEFINITIONS]
    
    if not valid_layers:
        print(f"No valid layers specified: {args.layers}")
        print(f"Valid layers: {', '.join(LAYER_DEFINITIONS.keys())}")
        return 1
    
    # Create config
    config = PipelineConfig(
        uri=args.uri,
        user=args.user,
        password=args.password,
        input_file=args.input,
        output_dir=args.output_dir,
        scripts_dir=args.scripts_dir,
        scale=args.scale,
        seed=args.seed,
        layers=valid_layers,
        target_spearman=args.target_spearman,
        target_f1=args.target_f1,
        verbose=args.verbose,
        quiet=args.quiet,
        open_browser=args.open_browser,
    )
    
    # Run pipeline
    runner = PipelineRunner(config)
    
    try:
        result = runner.run(
            generate=args.generate,
            do_import=args.do_import,
            analyze=args.analyze,
            simulate=args.simulate,
            validate=args.validate,
            visualize=args.visualize,
            all_stages=args.all,
            programmatic=args.programmatic,
        )
        
        return 0 if result.success else 1
    
    except KeyboardInterrupt:
        print(f"\n{colored('Pipeline interrupted.', Colors.YELLOW)}")
        return 130
    
    except Exception as e:
        logging.exception("Pipeline failed")
        print_error(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())