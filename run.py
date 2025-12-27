#!/usr/bin/env python3
"""
Software-as-a-Graph: End-to-End Pipeline
==========================================

Complete pipeline for graph-based modeling and analysis of
distributed publish-subscribe systems.

Pipeline Steps:
    1. GENERATE  - Create realistic pub-sub system graph
    2. ANALYZE   - Calculate criticality using graph algorithms
    3. SIMULATE  - Run failure simulations for actual impact
    4. VALIDATE  - Compare predictions with simulation results
    5. VISUALIZE - Generate interactive multi-layer visualizations

This script runs entirely without Neo4j, using pure Python graph analysis.

Usage:
    # Full pipeline with defaults
    python run.py
    
    # Specific scenario and scale
    python run.py --scenario financial --scale large
    
    # Use existing graph file
    python run.py --input my_system.json
    
    # Quick demo
    python run.py --quick
    
    # Skip specific steps
    python run.py --skip-generate --input system.json

Author: Ibrahim Onuralp Yigit
Research: Graph-Based Modeling and Analysis of Distributed Pub-Sub Systems
Publication: IEEE RASSE 2025
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

# Add src to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))


# =============================================================================
# Terminal Output
# =============================================================================

class Colors:
    """ANSI color codes"""
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    @classmethod
    def disable(cls):
        for attr in dir(cls):
            if not attr.startswith('_') and attr != 'disable':
                setattr(cls, attr, '')


def use_colors() -> bool:
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty() and not os.getenv('NO_COLOR')


def print_header(text: str) -> None:
    print(f"\n{Colors.PURPLE}{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.PURPLE}{Colors.BOLD}{text:^70}{Colors.END}")
    print(f"{Colors.PURPLE}{Colors.BOLD}{'='*70}{Colors.END}\n")


def print_step(num: str, title: str) -> None:
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'─'*70}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}  STEP {num}: {title}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'─'*70}{Colors.END}\n")


def print_success(text: str) -> None:
    print(f"  {Colors.GREEN}✓{Colors.END} {text}")


def print_error(text: str) -> None:
    print(f"  {Colors.RED}✗{Colors.END} {text}", file=sys.stderr)


def print_warning(text: str) -> None:
    print(f"  {Colors.YELLOW}⚠{Colors.END} {text}")


def print_info(text: str) -> None:
    print(f"  {Colors.BLUE}ℹ{Colors.END} {text}")


def print_detail(text: str) -> None:
    print(f"    {Colors.DIM}{text}{Colors.END}")


def print_metric(name: str, value: Any, target: Optional[str] = None) -> None:
    if target:
        print(f"    {name}: {Colors.BOLD}{value}{Colors.END} (target: {target})")
    else:
        print(f"    {name}: {Colors.BOLD}{value}{Colors.END}")


def format_time(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f}ms"
    elif ms < 60000:
        return f"{ms/1000:.1f}s"
    else:
        return f"{ms/60000:.1f}m"


# =============================================================================
# Pipeline Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    # Graph generation
    scenario: str = "iot"
    scale: str = "medium"
    seed: int = 42
    inject_antipatterns: bool = False
    
    # Input/Output
    input_file: Optional[Path] = None
    output_dir: Path = Path("./output")
    
    # Pipeline control
    skip_generate: bool = False
    skip_simulate: bool = False
    skip_validate: bool = False
    skip_visualize: bool = False
    
    # Simulation options
    enable_cascade: bool = True
    cascade_threshold: float = 0.5
    event_duration_ms: int = 10000
    event_rate: int = 100
    
    # Validation targets
    spearman_target: float = 0.70
    f1_target: float = 0.90
    
    # Output options
    verbose: bool = False
    quiet: bool = False


# =============================================================================
# Pipeline Results
# =============================================================================

@dataclass
class PipelineResults:
    """Pipeline execution results"""
    timestamp: datetime
    config: PipelineConfig
    
    # Step results
    graph_file: Optional[Path] = None
    graph_stats: Optional[Dict] = None
    
    analysis_results: Optional[Dict] = None
    criticality: Optional[Dict[str, Dict]] = None
    
    simulation_results: Optional[Dict] = None
    failure_results: Optional[Dict] = None
    event_results: Optional[Dict] = None
    
    validation_results: Optional[Dict] = None
    validation_status: str = "not_run"
    
    visualization_files: List[Path] = None
    
    # Timing
    step_times: Dict[str, float] = None
    total_time_ms: float = 0
    
    def __post_init__(self):
        if self.visualization_files is None:
            self.visualization_files = []
        if self.step_times is None:
            self.step_times = {}


# =============================================================================
# Pipeline Steps
# =============================================================================

def step_generate(config: PipelineConfig, results: PipelineResults) -> bool:
    """Step 1: Generate graph data"""
    from src.core import generate_graph
    
    print_step("1/5", "GENERATE - Create Pub-Sub System Graph")
    
    if config.skip_generate:
        if config.input_file and config.input_file.exists():
            print_info(f"Using existing graph: {config.input_file}")
            results.graph_file = config.input_file
            return True
        else:
            print_error("No input file specified or file not found")
            return False
    
    start = time.time()
    
    print_info(f"Generating {config.scale} {config.scenario} system...")
    print_detail(f"Scenario: {config.scenario}")
    print_detail(f"Scale: {config.scale}")
    print_detail(f"Seed: {config.seed}")
    if config.inject_antipatterns:
        print_detail("Anti-patterns: enabled")
    
    try:
        # Generate graph
        graph_data = generate_graph(
            scale=config.scale,
            scenario=config.scenario,
            seed=config.seed,
            antipatterns=["god_topic", "single_point_of_failure"] if config.inject_antipatterns else None,
        )
        
        # Save to file
        config.output_dir.mkdir(parents=True, exist_ok=True)
        graph_file = config.output_dir / f"{config.scenario}_{config.scale}_graph.json"
        
        with open(graph_file, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        # Extract stats
        results.graph_file = graph_file
        
        # Count total relationships
        relationships = graph_data.get("relationships", {})
        total_connections = sum(
            len(v) if isinstance(v, list) else 0
            for v in relationships.values()
        ) if isinstance(relationships, dict) else len(relationships)
        
        results.graph_stats = {
            "applications": len(graph_data.get("applications", [])),
            "topics": len(graph_data.get("topics", [])),
            "brokers": len(graph_data.get("brokers", [])),
            "nodes": len(graph_data.get("nodes", [])),
            "total_components": (
                len(graph_data.get("applications", [])) +
                len(graph_data.get("topics", [])) +
                len(graph_data.get("brokers", [])) +
                len(graph_data.get("nodes", []))
            ),
            "total_connections": total_connections,
        }
        
        elapsed = (time.time() - start) * 1000
        results.step_times["generate"] = elapsed
        
        print_success(f"Graph generated: {graph_file}")
        print_detail(f"Components: {results.graph_stats['total_components']}")
        print_detail(f"Connections: {results.graph_stats['total_connections']}")
        print_detail(f"Time: {format_time(elapsed)}")
        
        return True
        
    except Exception as e:
        print_error(f"Generation failed: {e}")
        if config.verbose:
            import traceback
            traceback.print_exc()
        return False


def step_analyze(config: PipelineConfig, results: PipelineResults) -> bool:
    """Step 2: Analyze graph to compute criticality scores"""
    from src.simulation import SimulationGraph
    from src.validation import GraphAnalyzer
    
    print_step("2/5", "ANALYZE - Calculate Criticality Scores")
    
    if not results.graph_file:
        print_error("No graph file available")
        return False
    
    start = time.time()
    
    print_info("Loading graph model...")
    
    try:
        # Load graph
        graph = SimulationGraph.from_json(results.graph_file)
        
        print_info("Computing centrality metrics...")
        print_detail("Methods: betweenness, degree, message_path, composite")
        
        # Run analysis
        analyzer = GraphAnalyzer(graph)
        all_metrics = analyzer.analyze_all()
        
        # Build criticality with levels
        composite = all_metrics["composite"]
        scores = sorted(composite.values())
        n = len(scores)
        
        def get_percentile(p):
            idx = int(n * p / 100)
            return scores[min(idx, n - 1)] if n > 0 else 0
        
        p90 = get_percentile(90)
        p75 = get_percentile(75)
        p50 = get_percentile(50)
        p25 = get_percentile(25)
        
        criticality = {}
        level_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "minimal": 0}
        
        for comp_id, score in composite.items():
            if score >= p90:
                level = "critical"
            elif score >= p75:
                level = "high"
            elif score >= p50:
                level = "medium"
            elif score >= p25:
                level = "low"
            else:
                level = "minimal"
            
            criticality[comp_id] = {"score": score, "level": level}
            level_counts[level] += 1
        
        results.analysis_results = all_metrics
        results.criticality = criticality
        
        # Save analysis results
        analysis_file = config.output_dir / "analysis_results.json"
        with open(analysis_file, 'w') as f:
            json.dump({
                "metrics": {k: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in all_metrics.items()},
                "criticality": criticality,
                "distribution": level_counts,
            }, f, indent=2)
        
        elapsed = (time.time() - start) * 1000
        results.step_times["analyze"] = elapsed
        
        print_success("Analysis complete")
        print_detail(f"Critical: {level_counts['critical']}, High: {level_counts['high']}, "
                    f"Medium: {level_counts['medium']}, Low: {level_counts['low']}")
        
        # Show top critical
        top_critical = sorted(criticality.items(), key=lambda x: -x[1]["score"])[:5]
        print_info("Top critical components:")
        for comp, data in top_critical:
            print_detail(f"{comp}: {data['score']:.4f} ({data['level']})")
        
        print_detail(f"Time: {format_time(elapsed)}")
        
        return True
        
    except Exception as e:
        print_error(f"Analysis failed: {e}")
        if config.verbose:
            import traceback
            traceback.print_exc()
        return False


def step_simulate(config: PipelineConfig, results: PipelineResults) -> bool:
    """Step 3: Simulate failures and events"""
    from src.simulation import SimulationGraph, FailureSimulator, EventSimulator
    
    print_step("3/5", "SIMULATE - Run Failure & Event Simulation")
    
    if config.skip_simulate:
        print_info("Simulation skipped")
        return True
    
    if not results.graph_file:
        print_error("No graph file available")
        return False
    
    start = time.time()
    
    try:
        # Load graph
        graph = SimulationGraph.from_json(results.graph_file)
        
        # Failure simulation
        print_info("Running failure simulation campaign...")
        print_detail(f"Cascade: {'enabled' if config.enable_cascade else 'disabled'}")
        print_detail(f"Threshold: {config.cascade_threshold}")
        
        fail_sim = FailureSimulator(
            cascade_threshold=config.cascade_threshold,
            seed=config.seed,
        )
        
        batch = fail_sim.simulate_all_failures(
            graph,
            enable_cascade=config.enable_cascade,
        )
        
        # Extract results
        failure_impacts = {
            r.primary_failures[0]: r.impact.impact_score
            for r in batch.results
        }
        
        results.failure_results = {
            "total_simulations": len(batch.results),
            "critical_components": batch.critical_components[:10],
            "impacts": failure_impacts,
        }
        
        print_success(f"Failure simulation complete: {len(batch.results)} components tested")
        
        # Show top impactful
        print_info("Highest impact failures:")
        for comp, impact in batch.critical_components[:5]:
            print_detail(f"{comp}: {impact:.4f}")
        
        # Event simulation
        print_info("Running event-driven simulation...")
        print_detail(f"Duration: {config.event_duration_ms}ms")
        print_detail(f"Rate: {config.event_rate} msg/sec")
        
        event_sim = EventSimulator(seed=config.seed)
        event_result = event_sim.simulate(
            graph,
            duration_ms=config.event_duration_ms,
            message_rate=config.event_rate,
        )
        
        results.event_results = event_result.metrics.to_dict()
        
        print_success(f"Event simulation complete")
        print_detail(f"Published: {event_result.metrics.messages_published}")
        print_detail(f"Delivered: {event_result.metrics.messages_delivered}")
        print_detail(f"Delivery rate: {event_result.metrics.delivery_rate():.1%}")
        print_detail(f"Avg latency: {event_result.metrics.avg_latency():.2f}ms")
        
        # Save simulation results
        sim_file = config.output_dir / "simulation_results.json"
        with open(sim_file, 'w') as f:
            json.dump({
                "failure_simulation": {
                    "total": len(batch.results),
                    "top_critical": batch.critical_components[:20],
                },
                "event_simulation": results.event_results,
            }, f, indent=2)
        
        elapsed = (time.time() - start) * 1000
        results.step_times["simulate"] = elapsed
        
        print_detail(f"Time: {format_time(elapsed)}")
        
        return True
        
    except Exception as e:
        print_error(f"Simulation failed: {e}")
        if config.verbose:
            import traceback
            traceback.print_exc()
        return False


def step_validate(config: PipelineConfig, results: PipelineResults) -> bool:
    """Step 4: Validate predictions against simulation"""
    from src.simulation import SimulationGraph
    from src.validation import ValidationPipeline, ValidationTargets
    
    print_step("4/5", "VALIDATE - Statistical Validation")
    
    if config.skip_validate:
        print_info("Validation skipped")
        return True
    
    if not results.graph_file:
        print_error("No graph file available")
        return False
    
    start = time.time()
    
    try:
        # Load graph
        graph = SimulationGraph.from_json(results.graph_file)
        
        print_info("Running validation pipeline...")
        print_detail(f"Spearman target: ≥{config.spearman_target}")
        print_detail(f"F1 target: ≥{config.f1_target}")
        
        # Create targets
        targets = ValidationTargets(
            spearman=config.spearman_target,
            f1=config.f1_target,
        )
        
        # Run validation
        pipeline = ValidationPipeline(
            targets=targets,
            cascade_threshold=config.cascade_threshold,
            seed=config.seed,
        )
        
        result = pipeline.run(graph, analysis_method="composite")
        
        v = result.validation
        results.validation_results = v.to_dict()
        results.validation_status = v.status.value
        
        # Display results
        status_color = Colors.GREEN if v.status.value == "passed" else \
                      Colors.YELLOW if v.status.value == "partial" else Colors.RED
        
        print_success(f"Validation complete: {status_color}{v.status.value.upper()}{Colors.END}")
        
        print_info("Correlation metrics:")
        spearman_color = Colors.GREEN if v.correlation.spearman >= config.spearman_target else Colors.RED
        print_detail(f"Spearman ρ: {spearman_color}{v.correlation.spearman:.4f}{Colors.END} (target: ≥{config.spearman_target})")
        print_detail(f"Pearson r: {v.correlation.pearson:.4f}")
        print_detail(f"Kendall τ: {v.correlation.kendall:.4f}")
        
        print_info("Classification metrics:")
        f1_color = Colors.GREEN if v.classification.f1 >= config.f1_target else Colors.RED
        print_detail(f"F1-Score: {f1_color}{v.classification.f1:.4f}{Colors.END} (target: ≥{config.f1_target})")
        print_detail(f"Precision: {v.classification.precision:.4f}")
        print_detail(f"Recall: {v.classification.recall:.4f}")
        
        print_info("Ranking metrics:")
        print_detail(f"Top-5 overlap: {v.ranking.top_k_overlap.get(5, 0):.4f}")
        print_detail(f"Top-10 overlap: {v.ranking.top_k_overlap.get(10, 0):.4f}")
        print_detail(f"NDCG: {v.ranking.ndcg:.4f}")
        
        # Save validation results
        val_file = config.output_dir / "validation_results.json"
        with open(val_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        elapsed = (time.time() - start) * 1000
        results.step_times["validate"] = elapsed
        
        print_detail(f"Time: {format_time(elapsed)}")
        
        return True
        
    except Exception as e:
        print_error(f"Validation failed: {e}")
        if config.verbose:
            import traceback
            traceback.print_exc()
        return False


def step_visualize(config: PipelineConfig, results: PipelineResults) -> bool:
    """Step 5: Generate visualizations"""
    from src.simulation import SimulationGraph
    from src.visualization import GraphRenderer, DashboardGenerator, RenderConfig, DashboardConfig, ColorScheme
    
    print_step("5/5", "VISUALIZE - Generate Interactive Visualizations")
    
    if config.skip_visualize:
        print_info("Visualization skipped")
        return True
    
    if not results.graph_file:
        print_error("No graph file available")
        return False
    
    start = time.time()
    
    try:
        # Load graph
        graph = SimulationGraph.from_json(results.graph_file)
        
        print_info("Generating visualizations...")
        
        viz_files = []
        
        # 1. Basic graph visualization
        print_detail("Creating network graph...")
        renderer = GraphRenderer(RenderConfig(
            title=f"{config.scenario.upper()} System - Network Graph",
        ))
        html = renderer.render(graph, results.criticality)
        
        graph_file = config.output_dir / "graph_visualization.html"
        with open(graph_file, 'w') as f:
            f.write(html)
        viz_files.append(graph_file)
        print_success(f"Network graph: {graph_file.name}")
        
        # 2. Multi-layer view
        print_detail("Creating multi-layer view...")
        renderer_ml = GraphRenderer(RenderConfig(
            title=f"{config.scenario.upper()} System - Multi-Layer Architecture",
        ))
        html_ml = renderer_ml.render_multi_layer(graph, results.criticality)
        
        ml_file = config.output_dir / "multi_layer_view.html"
        with open(ml_file, 'w') as f:
            f.write(html_ml)
        viz_files.append(ml_file)
        print_success(f"Multi-layer view: {ml_file.name}")
        
        # 3. Criticality view
        print_detail("Creating criticality view...")
        renderer_crit = GraphRenderer(RenderConfig(
            title=f"{config.scenario.upper()} System - Criticality Analysis",
            color_scheme=ColorScheme.CRITICALITY,
        ))
        html_crit = renderer_crit.render(graph, results.criticality)
        
        crit_file = config.output_dir / "criticality_view.html"
        with open(crit_file, 'w') as f:
            f.write(html_crit)
        viz_files.append(crit_file)
        print_success(f"Criticality view: {crit_file.name}")
        
        # 4. Dashboard
        print_detail("Creating comprehensive dashboard...")
        dashboard = DashboardGenerator(DashboardConfig(
            title=f"{config.scenario.upper()} System Analysis Dashboard",
        ))
        
        html_dash = dashboard.generate(
            graph=graph,
            criticality=results.criticality,
            validation=results.validation_results,
            simulation=results.event_results,
            analysis={"composite": {k: v["score"] for k, v in (results.criticality or {}).items()}} if results.criticality else None,
        )
        
        dash_file = config.output_dir / "dashboard.html"
        with open(dash_file, 'w') as f:
            f.write(html_dash)
        viz_files.append(dash_file)
        print_success(f"Dashboard: {dash_file.name}")
        
        results.visualization_files = viz_files
        
        elapsed = (time.time() - start) * 1000
        results.step_times["visualize"] = elapsed
        
        print_detail(f"Time: {format_time(elapsed)}")
        
        return True
        
    except Exception as e:
        print_error(f"Visualization failed: {e}")
        if config.verbose:
            import traceback
            traceback.print_exc()
        return False


# =============================================================================
# Pipeline Execution
# =============================================================================

def run_pipeline(config: PipelineConfig) -> PipelineResults:
    """Run the complete pipeline"""
    results = PipelineResults(
        timestamp=datetime.now(),
        config=config,
    )
    
    pipeline_start = time.time()
    
    # Run steps
    steps = [
        ("generate", step_generate),
        ("analyze", step_analyze),
        ("simulate", step_simulate),
        ("validate", step_validate),
        ("visualize", step_visualize),
    ]
    
    success = True
    for name, step_fn in steps:
        if not step_fn(config, results):
            print_error(f"Pipeline failed at step: {name}")
            success = False
            break
    
    results.total_time_ms = (time.time() - pipeline_start) * 1000
    
    return results


def print_summary(results: PipelineResults) -> None:
    """Print pipeline summary"""
    print_header("PIPELINE COMPLETE")
    
    config = results.config
    
    # Status
    if results.validation_status == "passed":
        print(f"  {Colors.GREEN}{Colors.BOLD}All steps completed successfully! ✓{Colors.END}")
    elif results.validation_status == "partial":
        print(f"  {Colors.YELLOW}{Colors.BOLD}Pipeline complete (partial validation){Colors.END}")
    else:
        print(f"  {Colors.BLUE}{Colors.BOLD}Pipeline complete{Colors.END}")
    
    print()
    
    # Generated files
    print(f"  {Colors.CYAN}Generated Files:{Colors.END}")
    print()
    
    if results.graph_file:
        print(f"  {Colors.BOLD}Graph Data:{Colors.END}")
        print(f"    📊 {results.graph_file}")
    
    print()
    print(f"  {Colors.BOLD}Analysis:{Colors.END}")
    print(f"    📈 {config.output_dir}/analysis_results.json")
    
    if results.failure_results:
        print()
        print(f"  {Colors.BOLD}Simulation:{Colors.END}")
        print(f"    💥 {config.output_dir}/simulation_results.json")
    
    if results.validation_results:
        print()
        print(f"  {Colors.BOLD}Validation:{Colors.END}")
        print(f"    ✅ {config.output_dir}/validation_results.json")
    
    if results.visualization_files:
        print()
        print(f"  {Colors.BOLD}Visualizations:{Colors.END}")
        for vf in results.visualization_files:
            icon = "📊" if "dashboard" in vf.name else "🔀" if "multi" in vf.name else "🎯" if "critical" in vf.name else "🎨"
            print(f"    {icon} {vf}")
    
    # Timing
    print()
    print(f"  {Colors.CYAN}Timing:{Colors.END}")
    for step, ms in results.step_times.items():
        print(f"    {step.capitalize():12} {format_time(ms):>10}")
    print(f"    {Colors.BOLD}{'Total':12} {format_time(results.total_time_ms):>10}{Colors.END}")
    
    # Validation summary
    if results.validation_results:
        print()
        v = results.validation_results
        corr = v.get("correlation", {}).get("spearman", {})
        cls = v.get("classification", {})
        
        print(f"  {Colors.CYAN}Validation Summary:{Colors.END}")
        
        spearman = corr.get("coefficient", 0)
        f1 = cls.get("f1", 0)
        
        sp_color = Colors.GREEN if spearman >= 0.7 else Colors.YELLOW if spearman >= 0.5 else Colors.RED
        f1_color = Colors.GREEN if f1 >= 0.9 else Colors.YELLOW if f1 >= 0.7 else Colors.RED
        
        print(f"    Spearman: {sp_color}{spearman:.4f}{Colors.END}")
        print(f"    F1-Score: {f1_color}{f1:.4f}{Colors.END}")
    
    # Open dashboard hint
    if results.visualization_files:
        print()
        print(f"  {Colors.CYAN}{'─'*60}{Colors.END}")
        print()
        print(f"  {Colors.BOLD}To view the dashboard:{Colors.END}")
        print()
        
        dash_file = config.output_dir / "dashboard.html"
        
        if sys.platform == "darwin":
            print(f"    open {dash_file}")
        elif sys.platform == "linux":
            print(f"    xdg-open {dash_file}")
        else:
            print(f"    start {dash_file}")
    
    print()
    print(f"  {Colors.GREEN}{Colors.BOLD}Pipeline Complete! 🎉{Colors.END}")
    print()


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Software-as-a-Graph: End-to-End Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline with defaults
    python run.py
    
    # Specific scenario and scale
    python run.py --scenario financial --scale large
    
    # Quick demo (small scale)
    python run.py --quick
    
    # Use existing graph file
    python run.py --input my_system.json
    
    # Skip specific steps
    python run.py --skip-simulate --skip-validate
    
    # With anti-patterns for testing
    python run.py --scenario iot --antipatterns

Scenarios:
    iot          - IoT sensor network
    financial    - Financial trading platform
    microservices - Cloud microservices
    ros2         - ROS 2 robotic system

Scales:
    small   - ~30 components, ~80 connections
    medium  - ~80 components, ~250 connections
    large   - ~200 components, ~600 connections

Pipeline Steps:
    1. GENERATE  - Create pub-sub system graph
    2. ANALYZE   - Calculate criticality scores
    3. SIMULATE  - Run failure simulations
    4. VALIDATE  - Statistical validation
    5. VISUALIZE - Generate visualizations
        """,
    )
    
    # Graph generation
    gen_group = parser.add_argument_group("Graph Generation")
    gen_group.add_argument(
        "--scenario", "-s", type=str, default="iot",
        choices=["iot", "financial", "microservices", "ros2"],
        help="System scenario (default: iot)",
    )
    gen_group.add_argument(
        "--scale", type=str, default="medium",
        choices=["small", "medium", "large"],
        help="System scale (default: medium)",
    )
    gen_group.add_argument(
        "--antipatterns", action="store_true",
        help="Inject anti-patterns for testing",
    )
    gen_group.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    
    # Input/Output
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        "--input", "-i", type=Path,
        help="Use existing graph file (skips generation)",
    )
    io_group.add_argument(
        "--output", "-o", type=Path, default=Path("./output"),
        help="Output directory (default: ./output)",
    )
    
    # Pipeline control
    ctrl_group = parser.add_argument_group("Pipeline Control")
    ctrl_group.add_argument(
        "--quick", "-q", action="store_true",
        help="Quick demo with small scale",
    )
    ctrl_group.add_argument(
        "--skip-generate", action="store_true",
        help="Skip graph generation",
    )
    ctrl_group.add_argument(
        "--skip-simulate", action="store_true",
        help="Skip simulation step",
    )
    ctrl_group.add_argument(
        "--skip-validate", action="store_true",
        help="Skip validation step",
    )
    ctrl_group.add_argument(
        "--skip-visualize", action="store_true",
        help="Skip visualization step",
    )
    
    # Simulation options
    sim_group = parser.add_argument_group("Simulation Options")
    sim_group.add_argument(
        "--cascade", action="store_true", default=True,
        help="Enable cascade propagation (default)",
    )
    sim_group.add_argument(
        "--no-cascade", dest="cascade", action="store_false",
        help="Disable cascade propagation",
    )
    sim_group.add_argument(
        "--cascade-threshold", type=float, default=0.5,
        help="Cascade threshold (default: 0.5)",
    )
    sim_group.add_argument(
        "--event-duration", type=int, default=10000,
        help="Event simulation duration in ms (default: 10000)",
    )
    sim_group.add_argument(
        "--event-rate", type=int, default=100,
        help="Event simulation rate (default: 100 msg/sec)",
    )
    
    # Validation options
    val_group = parser.add_argument_group("Validation Options")
    val_group.add_argument(
        "--spearman-target", type=float, default=0.70,
        help="Spearman correlation target (default: 0.70)",
    )
    val_group.add_argument(
        "--f1-target", type=float, default=0.90,
        help="F1-score target (default: 0.90)",
    )
    
    # Output options
    out_group = parser.add_argument_group("Output Options")
    out_group.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output",
    )
    out_group.add_argument(
        "--quiet", action="store_true",
        help="Minimal output",
    )
    out_group.add_argument(
        "--no-color", action="store_true",
        help="Disable colors",
    )
    
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    # Handle colors
    if args.no_color or not use_colors():
        Colors.disable()
    
    # Build config
    config = PipelineConfig(
        scenario=args.scenario,
        scale="small" if args.quick else args.scale,
        seed=args.seed,
        inject_antipatterns=args.antipatterns,
        input_file=args.input,
        output_dir=args.output,
        skip_generate=args.skip_generate or (args.input is not None),
        skip_simulate=args.skip_simulate,
        skip_validate=args.skip_validate,
        skip_visualize=args.skip_visualize,
        enable_cascade=args.cascade,
        cascade_threshold=args.cascade_threshold,
        event_duration_ms=args.event_duration,
        event_rate=args.event_rate,
        spearman_target=args.spearman_target,
        f1_target=args.f1_target,
        verbose=args.verbose,
        quiet=args.quiet,
    )
    
    # Print header
    if not args.quiet:
        print_header("Software-as-a-Graph Pipeline")
        print(f"  {Colors.DIM}Graph-Based Modeling and Analysis of Distributed Pub-Sub Systems{Colors.END}")
        print(f"  {Colors.DIM}IEEE RASSE 2025{Colors.END}")
        print()
        print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Scenario:  {config.scenario}")
        print(f"  Scale:     {config.scale}")
        print(f"  Output:    {config.output_dir}")
    
    try:
        # Run pipeline
        results = run_pipeline(config)
        
        # Print summary
        if not args.quiet:
            print_summary(results)
        
        # Exit code based on validation
        if results.validation_status == "passed":
            return 0
        elif results.validation_status == "partial":
            return 0  # Still success
        elif results.validation_status == "failed":
            return 1
        else:
            return 0
    
    except KeyboardInterrupt:
        print_warning("\nPipeline interrupted")
        return 130
    
    except Exception as e:
        print_error(f"Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
