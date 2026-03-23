"""
End-to-End Pipeline: Software-as-a-Graph
=========================================
Runs the complete 6-step analysis pipeline in a single script:

  Step 1  Model Construction   — "What does my system architecture look like as a graph?"
  Step 2  Structural Analysis   — "Which components are topologically central or represent single points of failure?"
  Step 3  Criticality Prediction — "Which components are most 'at risk' from a quality perspective?" (RMAV + GNN)
  Step 4  Failure Simulation    — "If a component fails, how far does the damage spread?" (I(v) Ground Truth)
  Step 5  Statistical Validation — "Can we trust the topological predictions against simulated reality?"
  Step 6  Visualization          — "How do we communicate these risks to stakeholders?"

Prerequisites:
  • Python 3.9+ virtual environment with requirements.txt installed
  • Neo4j 5.x running at bolt://localhost:7687  (user=neo4j, password=password)

Run from the project root:
    python examples/example_end_to_end.py

To use a different scale:
    python examples/example_end_to_end.py --scale small --layer app --seed 77
"""
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from tools.generation import generate_graph, SCALE_PRESETS
from src.adapters import create_repository
from src.analysis import AnalysisService
from src.simulation import SimulationService
from src.validation import ValidationService
from src.visualization import VisualizationService
from src.prediction import PredictionService, GNNService, extract_structural_metrics_dict, extract_rmav_scores_dict, extract_simulation_dict


# ──────────────────────────────────────────────
# CLI args
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Software-as-a-Graph — end-to-end pipeline")
    p.add_argument("--scale", default="tiny",
                   choices=list(SCALE_PRESETS.keys()),
                   help="Graph size preset (default: tiny)")
    p.add_argument("--layer", default="app",
                   choices=["app", "infra", "mw", "system"],
                   help="Primary analysis/validation layer (default: app)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    p.add_argument("--neo4j-uri",     default="bolt://localhost:7687")
    p.add_argument("--neo4j-user",    default="neo4j")
    p.add_argument("--neo4j-password",default="password")
    p.add_argument("--output-dir",    default=str(ROOT / "output"),
                   help="Directory for all output files")
    p.add_argument("--skip-viz", action="store_true",
                   help="Skip dashboard generation (faster)")
    return p.parse_args()


# ──────────────────────────────────────────────
# Pretty printing helpers
# ──────────────────────────────────────────────

STEP_TIMINGS: dict = {}


def step_header(n: int, title: str, question: str) -> None:
    print(f"\n{'━'*75}")
    print(f"  Step {n}/6 — {title}")
    print(f"  Q: {question}")
    print(f"{'━'*75}")
    STEP_TIMINGS[n] = time.time()


def step_done(n: int) -> None:
    elapsed = time.time() - STEP_TIMINGS.get(n, time.time())
    print(f"  ✓ Done  ({elapsed:.1f}s)")


def print_kv(key: str, value, width: int = 26) -> None:
    print(f"  {key:<{width}}: {value}")


def pass_fail(value: float, threshold: float) -> str:
    return "✓ PASS" if value >= threshold else "✗ FAIL"


# ──────────────────────────────────────────────
# Step functions
# ──────────────────────────────────────────────

def step1_model_construction(scale: str, seed: int, output_dir: Path, neo4j_cfg: dict) -> tuple[Path, any]:
    """Step 1: Construct the graph model from topology specification."""
    step_header(1, "Graph Model Construction", "What does my system architecture look like as a graph?")

    # 1.1 Generate
    print(f"  [1.1] Generating '{scale}' topology (seed={seed})...")
    graph_data = generate_graph(scale=scale, seed=seed)
    
    out_path = output_dir / f"e2e_graph_{scale}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(graph_data, f, indent=2)
    
    # 1.2 Import
    print(f"  [1.2] Importing into Neo4j at {neo4j_cfg['uri']}...")
    repo = create_repository(
        uri=neo4j_cfg['uri'], 
        user=neo4j_cfg['user'], 
        password=neo4j_cfg['password']
    )
    repo.save_graph(graph_data, clear=True)
    
    stats = repo.get_statistics()
    print_kv("Applications",          len(graph_data.get("applications", [])))
    print_kv("Infrastructure nodes",  len(graph_data.get("nodes", [])))
    print_kv("Neo4j node count",       stats.get("node_count", "?"))
    print_kv("Relationship count",     stats.get("relationship_count", stats.get("total_count", "?")))

    step_done(1)
    return out_path, repo


def step2_structural_analysis(repo, layer: str) -> dict:
    """Step 2: Compute topological metrics (PageRank, Betweenness, etc.)."""
    step_header(2, "Structural Analysis", "Which components are topologically central or represent SPOFs?")

    analyzer = AnalysisService(repo)
    result   = analyzer.analyze_layer(layer)
    struct   = result.structural
    
    print_kv("Analyzed Layer", layer)
    print_kv("Components",     len(struct.components))
    
    # Show top-3 by PageRank and Betweenness
    top_pr = struct.get_top_by_metric("pagerank", n=3)
    top_bt = struct.get_top_by_metric("betweenness", n=3)
    
    print("\n  Top-3 by Topological Centrality:")
    for i, c in enumerate(top_pr):
        print(f"    {i+1}. {c.id[:30]:<32} PageRank: {c.pagerank:.4f}")
        
    ap_count = len(struct.get_articulation_points())
    print_kv("\nArticulation Points (SPOFs)", ap_count)

    step_done(2)
    return {"result": result}


def step3_criticality_prediction(repo, output_dir: Path, layer: str, analysis_res: dict, sim_results: Optional[list] = None) -> dict:
    """Step 3: Map structural metrics to RMAV scores and GNN predictions."""
    step_header(3, "Criticality Prediction", "Which components are most 'at risk' from a quality perspective?")

    # 3.1 Rule-based RMAV Scoring
    print("  [3.1] Computing Rule-based RMAV Quality Scores...")
    result = analysis_res["result"]
    # In this framework, AnalysisService.analyze_layer often includes RMAV if quality analyzer is plugged in
    # Here we assume it's already there or we trigger it.
    from src.prediction.analyzer import QualityAnalyzer
    quality_analyzer = QualityAnalyzer()
    quality_res = quality_analyzer.analyze(result.structural)
    result.quality = quality_res
    
    comps = quality_res.components
    print_kv("Criticality Levels", f"{len([c for c in comps if c.levels.overall.value == 'CRITICAL'])} Critical, {len([c for c in comps if c.levels.overall.value == 'HIGH'])} High")

    # Top-5 Table
    print(f"\n  Top 5 components by RMAV Prediction Q(v):")
    print(f"  {'ID':<28} {'Q':>6} {'R':>6} {'M':>6} {'A':>6} {'V':>6}  Level")
    print(f"  {'-'*75}")
    for c in comps[:5]:
        s = c.scores
        print(f"  {c.id[:27]:<28} {s.overall:>6.3f} {s.reliability:>6.3f} {s.maintainability:>6.3f} {s.availability:>6.3f} {s.vulnerability:>6.3f}  {c.levels.overall.value.upper()}")

    # 3.2 Learning-based GNN Prediction (Optional - requires sim_results for training)
    gnn_summary = {}
    if sim_results:
        print(f"\n  [3.2] Training GNN for layer='{layer}' (Refining predictions)...")
        structural_dict = extract_structural_metrics_dict(result.structural)
        rmav_dict       = extract_rmav_scores_dict(result.quality)
        simulation_dict = extract_simulation_dict(sim_results)

        gnn_service = GNNService(
            checkpoint_dir=str(output_dir / "gnn_checkpoint"),
        )
        train_result = gnn_service.train(
            graph=result.graph,
            structural_metrics=structural_dict,
            simulation_results=simulation_dict,
            rmav_scores=rmav_dict,
            num_epochs=10,
        )
        gnn_summary = {"trained": True, "top": train_result.top_critical_nodes(3)}
        print_kv("GNN Refinement", "Complete")

    step_done(3)
    return {"quality": quality_res, "gnn": gnn_summary, "analysis_result": result}


def step4_simulate(repo, layer: str) -> list:
    """Step 4: Run exhaustive failure simulation to measure ground-truth impact."""
    step_header(4, "Failure Simulation", "If a component fails, how far does the damage spread?")

    sim_service = SimulationService(repo)
    print(f"  Simulating failures in layer='{layer}'...")
    
    results = sim_service.run_failure_simulation_exhaustive(layer=layer)
    ranked  = sorted(results, key=lambda r: r.impact.composite_impact, reverse=True)

    print_kv("Components simulated", len(results))
    avg_i = sum(r.impact.composite_impact for r in results) / max(len(results), 1)
    print_kv("Avg composite I(v)",    f"{avg_i:.4f}")
    
    if ranked:
        print_kv("Worst-case Impact", f"{ranked[0].impact.composite_impact:.4f} ({ranked[0].target_id})")

    step_done(4)
    return results


def step5_validate(repo, layer: str) -> dict:
    """Step 5: Statistically compare predictions (Q) against simulations (I)."""
    step_header(5, "Statistical Validation", "Can we trust the topological predictions against simulated reality?")

    analysis   = AnalysisService(repo)
    prediction = PredictionService()
    simulation = SimulationService(repo)
    validator  = ValidationService(analysis, prediction, simulation, ndcg_k=10)

    layer_res = validator.validate_single_layer(layer)
    vr = layer_res.validation_result
    
    if vr:
        sp = vr.overall.correlation.spearman
        f1 = vr.overall.classification.f1_score
        print_kv("Spearman ρ", f"{sp:.4f}  ({pass_fail(sp, 0.75)})")
        print_kv("F1-Score",   f"{f1:.4f}  ({pass_fail(f1, 0.70)})")
    
    print_kv("Validation Result", "PASS ✓" if layer_res.passed else "FAIL ✗")

    step_done(5)
    return {
        "passed":   layer_res.passed,
        "spearman": vr.overall.correlation.spearman if vr else 0.0,
        "f1":       vr.overall.classification.f1_score if vr else 0.0,
    }


# Formerly Step 6, now integrated into Step 3


def step6_visualize(repo, output_dir: Path, layer: str) -> Optional[str]:
    """Step 6: Generate interactive HTML dashboard."""
    step_header(6, "Dashboard Visualization", "How do we communicate these risks to stakeholders?")

    analysis   = AnalysisService(repo)
    prediction = PredictionService()
    simulation = SimulationService(repo)
    validation = ValidationService(analysis, prediction, simulation)
    viz        = VisualizationService(
        analysis_service=analysis,
        prediction_service=prediction,
        simulation_service=simulation,
        validation_service=validation,
        repository=repo,
    )

    out_file = str(output_dir / "e2e_dashboard.html")
    path = viz.generate_dashboard(
        output_file=out_file,
        layers=[layer, "system"],
        include_network=True,
    )

    print_kv("Dashboard File", out_file)
    print(f"\n  Open in browser:  xdg-open {os.path.abspath(path)}")

    step_done(6)
    return path


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_start = time.time()

    print("\n" + "━" * 65)
    print("  Software-as-a-Graph — End-to-End Pipeline")
    print("━" * 65)
    print(f"  Scale  : {args.scale}")
    print(f"  Layer  : {args.layer}")
    print(f"  Seed   : {args.seed}")
    print(f"  Neo4j  : {args.neo4j_uri}")
    print(f"  Output : {output_dir}")

    repo = None
    try:
        # ── Step 1: Model Construction ────────────────────────────────
        json_path, repo = step1_model_construction(
            args.scale, args.seed, output_dir,
            neo4j_cfg={
                "uri": args.neo4j_uri,
                "user": args.neo4j_user,
                "password": args.neo4j_password
            }
        )

        # ── Step 2: Structural Analysis ───────────────────────────────
        struct_summary = step2_structural_analysis(repo, args.layer)

        # ── Step 3: Criticality Prediction (Initial) ──────────────────
        # We run simulation first to allow GNN training in Step 3
        # In a real "decision" flow, you might predict first, then simulate to validate
        
        # ── Step 4: Failure Simulation ────────────────────────────────
        _sim_results = step4_simulate(repo, args.layer)

        # Now complete Step 3 with GNN
        predict_summary = step3_criticality_prediction(
            repo, output_dir, args.layer, struct_summary, _sim_results
        )

        # ── Step 5: Statistical Validation ────────────────────────────
        validation_summary = step5_validate(repo, args.layer)

        # ── Step 6: Visualization ─────────────────────────────────────
        if not args.skip_viz:
            step6_visualize(repo, output_dir, args.layer)
        
        # ── DECISION SUPPORT REPORT ──────────────────────────────────
        print("\n" + "━" * 75)
        print("  DECISION SUPPORT REPORT")
        print("━" * 75)
        
        passed = validation_summary.get("passed", False)
        print(f"  [1] System Deployment Safety:  {'RAISE CAUTION ⚠️' if not passed else 'GO / PROCEED ✅'}")
        print(f"      (Validation Confidence ρ={validation_summary.get('spearman', 0):.2f})")
        
        critical_comps = [c for c in predict_summary['quality'].components if c.levels.overall.value == 'CRITICAL']
        worst_id = critical_comps[0].id if critical_comps else "None"
        print(f"  [2] High-Priority Redundancy:  Component '{worst_id}'")
        print(f"      (Target for immediate replication or failover testing)")
        
        spof_nodes = struct_summary['result'].structural.get_articulation_points()
        print(f"  [3] Architectural Bottlenecks: {len(spof_nodes)} SPOFs detected")
        if spof_nodes:
            print(f"      Recommended: Decouple direct dependencies on {spof_nodes[0].id[:20]}...")

    except KeyboardInterrupt:
        print("\n\n[Interrupted by user]")
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if repo is not None:
            repo.close()

    # ── Final summary ─────────────────────────────────────────────────
    elapsed = time.time() - pipeline_start
    print(f"\n{'━'*65}")
    print("  Pipeline Complete")
    print(f"{'━'*65}")
    print(f"  Total time   : {elapsed:.1f}s")
    if 'validation_summary' in dir() or 'validation_summary' in locals():
        passed = validation_summary.get("passed", False)
        sp     = validation_summary.get("spearman", 0.0)
        f1     = validation_summary.get("f1", 0.0)
        print(f"  Validation   : {'PASS ✓' if passed else 'FAIL ✗'}")
        print(f"  Spearman ρ   : {sp:.4f}")
        print(f"  F1 Score     : {f1:.4f}")
    print(f"  Output dir   : {output_dir.absolute()}")
    print()


if __name__ == "__main__":
    main()
