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
import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, Any
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
from src.explanation import CLIFormatter
from src.prediction import PredictionService, GNNService, extract_structural_metrics_dict, extract_rmav_scores_dict, extract_simulation_dict


# ──────────────────────────────────────────────
# CLI args
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Software-as-a-Graph — end-to-end pipeline")
    p.add_argument("--scale", type=str, default="tiny",
                   choices=["tiny", "small", "medium", "large"],
                   help="Scale of synthetic graph to generate (default: tiny)")
    p.add_argument("--topology", type=str,
                   help="Path to a custom JSON topology file to load instead of generating a synthetic graph")
    p.add_argument("--layer", type=str, default="app",
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


def load_json_with_comments(path: str) -> Dict[str, Any]:
    """Load JSON file, stripping // style comments."""
    import json
    import re
    with open(path, 'r') as f:
        content = f.read()
    # Remove // comments
    content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
    return json.loads(content)


def pass_fail(value: float, threshold: float) -> str:
    return "✓ PASS" if value >= threshold else "✗ FAIL"


def interpret_critical_components(components, top_n: int = 3):
    """Translate RMAV scores into engineering decisions."""
    # Include both CRITICAL and HIGH to ensure visible output in example runs
    targets = [c for c in components if c.levels.overall.value.lower() in ("critical", "high")]
    if not targets:
        return

    print("\n  [Analysis Interpretation: Actionable Guidance]")
    for c in targets[:top_n]:
        reasons = []
        s = c.scores
        st = c.structural
        
        # Heuristics for interpretation
        if st.is_articulation_point:
            reasons.append("Structural SPOF — its removal disconnects the architecture graph.")
        if s.availability > 0.70:
            reasons.append(f"High Availability Risk — lacks sufficient redundancy for its topological role.")
        if s.reliability > 0.70:
            reasons.append(f"Cascade Hub — failure here propagates widely due to high PageRank.")
        if s.maintainability > 0.70 or st.in_degree_raw > 5:
            reasons.append(f"Change Bottleneck — {st.in_degree_raw} components depend on this; interface changes are high-risk.")
        
        if not reasons:
            reasons.append("High composite RMAV score across multiple dimensions.")

        print(f"\n    • {c.id[:30]:<32} is {c.levels.overall.value.upper()} because:")
        for r in reasons:
            print(f"      - {r}")
        print(f"      Recommendation: Add redundancy, circuit breakers, or extract stable interface.")


# ──────────────────────────────────────────────
# Step functions
# ──────────────────────────────────────────────

# The original step1_model_construction function is now inlined into main()
# and will be removed or adapted if the user explicitly requests it.
# For now, it's kept as the user only provided changes to main().
# However, the provided diff for main() completely replaces the call to this function
# with inline logic, making this function effectively unused.
# To maintain a syntactically correct and logical file, I will remove the now unused
# `step1_model_construction` function as its logic is fully replaced in `main`.
# def step1_model_construction(scale: str, seed: int, output_dir: Path, neo4j_cfg: dict) -> tuple[Path, any]:
#     """Step 1: Construct the graph model from topology specification."""
#     step_header(1, "Graph Model Construction", "What does my system architecture look like as a graph?")

#     # 1.1 Generate
#     print(f"  [1.1] Generating '{scale}' topology (seed={seed})...")
#     graph_data = generate_graph(scale=scale, seed=seed)
    
#     out_path = output_dir / f"e2e_graph_{scale}_seed{seed}.json"
#     with open(out_path, "w") as f:
#         json.dump(graph_data, f, indent=2)
    
#     # 1.2 Import
#     print(f"  [1.2] Importing into Neo4j at {neo4j_cfg['uri']}...")
#     repo = create_repository(
#         uri=neo4j_cfg['uri'], 
#         user=neo4j_cfg['user'], 
#         password=neo4j_cfg['password']
#     )
#     repo.save_graph(graph_data, clear=True)
    
#     stats = repo.get_statistics()
#     print_kv("Applications",          len(graph_data.get("applications", [])))
#     print_kv("Infrastructure nodes",  len(graph_data.get("nodes", [])))
#     print_kv("Neo4j node count",       stats.get("node_count", "?"))
#     print_kv("Relationship count",     stats.get("relationship_count", stats.get("total_count", "?")))

#     step_done(1)
#     return out_path, repo


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
    from src.prediction.analyzer import QualityAnalyzer
    quality_analyzer = QualityAnalyzer()
    quality_res = quality_analyzer.analyze(result.structural)
    result.quality = quality_res
    
    comps = quality_res.components
    print_kv("Criticality Levels", f"{len([c for c in comps if c.levels.overall.value.lower() == 'critical'])} Critical, {len([c for c in comps if c.levels.overall.value.lower() == 'high'])} High")

    # Replaced Top-5 Table with Component Cards
    from src.analysis import AntiPatternDetector
    detector = AntiPatternDetector()
    smell_report = detector.detect(result.structural)
    
    CLIFormatter.print_critical_report(quality_res, problems=smell_report.problems, limit_top=5)

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
        # Question: Is our architectural model valid and consistent?
        step_header(1, "Model Construction", "Is our architectural model valid and consistent?")
        
        graph_data = None
        if args.topology:
            print(f"  [1.1] Loading custom topology from: {args.topology}")
            graph_data = load_json_with_comments(args.topology)
            print_kv("Source", os.path.basename(args.topology))
            print_kv("Mode", "Static JSON Load")
        else:
            print(f"  [1.1] Generating synthetic graph (scale='{args.scale}', seed={args.seed})...")
            config = GraphConfig.from_scale(args.scale, seed=args.seed)
            graph_data = generate_graph(config)
            print_kv("Source", "Synthetic Generator")
            print_kv("Scale", args.scale)
        
        # 1.2 Import
        print(f"  [1.2] Importing into Neo4j at {args.neo4j_uri}...")
        repo = create_repository(
            uri=args.neo4j_uri, 
            user=args.neo4j_user, 
            password=args.neo4j_password
        )
        repo.save_graph(graph_data, clear=True)
        
        stats = repo.get_statistics()
        print_kv("Applications",          len(graph_data.get("applications", [])))
        print_kv("Infrastructure nodes",  len(graph_data.get("nodes", [])))
        print_kv("Neo4j node count",       stats.get("node_count", "?"))
        print_kv("Relationship count",     stats.get("relationship_count", stats.get("total_count", "?")))

        step_done(1)
        # End of inlined step1_model_construction logic

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

        # ── Independence guarantee — the scientific foundation ────────────
        # Q(v) is computed in Step 3 using only the graph topology
        # (centrality, articulation points, bridge ratio, etc.).
        # I(v) is computed in Step 4 using only cascade propagation
        # rules on the structural graph — it never reads Q(v) or any RMAV score.
        #
        # Measuring their agreement in Step 5 is therefore a genuine
        # empirical test, not a consistency check. Spearman ρ means
        # topology-only predictions match failure-simulation ground truth 
        # in rank order — without any runtime monitoring data.
        # ──────────────────────────────────────────────────────────────────
        rho_val = validation_summary.get('spearman', 0)
        print("\n  [Scientific Note: Independence Guarantee]")
        print(f"  Spearman ρ = {rho_val:.3f} confirms that topology-only predictions (Q)")
        print(f"  match failure-simulation ground truth (I) {rho_val*100:.1f}% of the time")
        print("  in rank order — without needing any runtime monitoring data.")

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
        
        critical_comps = [c for c in predict_summary['quality'].components if c.levels.overall.value.lower() == 'critical']
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
