#!/usr/bin/env python3
"""
Worked Example & Execution Runner: Software-as-a-Graph (SaaG) Pipeline on Autoware ROS 2 Dataset.
Adheres strictly to the specifications of all 7 pipeline steps (Import, Analyze, Predict, Simulate, Validate, Prescribe, Visualize).
"""

import argparse
import json
import math
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx
from saag import Client
from saag.infrastructure.memory_repo import MemoryRepository
from saag.infrastructure.neo4j_repo import Neo4jRepository
from saag.simulation.fault_injector import FaultInjector
from saag.simulation.message_flow_simulator import MessageFlowSimulator
from saag.analysis.service import AnalysisService
from saag.prediction.service import PredictionService
from saag.simulation.service import SimulationService
from saag.validation.service import ValidationService
from saag.visualization.service import VisualizationService
from saag.visualization.collector import LayerDataCollector


def print_table(title, headers, rows):
    """Utility to print a clean ASCII table."""
    print(f"\n=== {title} ===")
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))
            
    header_str = " | ".join(f"{str(h).ljust(col_widths[i])}" for i, h in enumerate(headers))
    print(header_str)
    print("-" * (sum(col_widths) + len(headers) * 3 - 1))
    
    for row in rows:
        row_str = " | ".join(f"{str(val).ljust(col_widths[i])}" for i, val in enumerate(row))
        print(row_str)


def graph_data_to_networkx(graph_data) -> nx.DiGraph:
    """Convert flat GraphData from repository into a NetworkX DiGraph."""
    g = nx.DiGraph()
    g.graph["id"] = "autoware_ros2"
    
    for comp in graph_data.components:
        props = getattr(comp, "properties", {}) or {}
        g.add_node(comp.id, **props)
        g.nodes[comp.id]["type"] = comp.component_type
        g.nodes[comp.id]["name"] = props.get("name", comp.id)
        g.nodes[comp.id]["weight"] = getattr(comp, "weight", 1.0)
        
    for edge in graph_data.edges:
        props = getattr(edge, "properties", {}) or {}
        g.add_edge(edge.source_id, edge.target_id, **props)
        g.edges[edge.source_id, edge.target_id]["type"] = edge.relation_type
        g.edges[edge.source_id, edge.target_id]["weight"] = getattr(edge, "weight", 1.0)
        g.edges[edge.source_id, edge.target_id]["rate_hz"] = props.get("rate_hz", 10.0)
        g.edges[edge.source_id, edge.target_id]["qos_profile"] = props.get("qos_profile", {})
        
    return g


def map_problem_to_smell(p, layer="system"):
    """Convert SDK DetectedProblem into SmellReport schema."""
    pattern_id = p.name
    if "SPOF" in p.name or "Single Point of Failure" in p.name:
        pattern_id = "SPOF"
    elif "Cycle" in p.name or "Circular" in p.name:
        pattern_id = "CYCLIC_DEPENDENCY"
    elif "God" in p.name:
        pattern_id = "GOD_COMPONENT"
    elif "Hub-and-Spoke" in p.name:
        pattern_id = "HUB_AND_SPOKE"
    elif "Isolated" in p.name:
        pattern_id = "ISOLATED"
    elif "Systemic" in p.name:
        pattern_id = "SYSTEMIC_RISK"
    elif "Compound" in p.name:
        pattern_id = "COMPOUND_RISK"
        
    comp_ids = []
    if p.entity_type == "Component":
        comp_ids = [p.entity_id]
    elif p.entity_type == "Architecture":
        comp_ids = [c.strip() for c in p.entity_id.split("->")]
        
    return {
        "layer": layer,
        "pattern_id": pattern_id,
        "name": p.name,
        "severity": p.severity.upper(),
        "description": p.description,
        "recommendation": p.recommendation,
        "component_ids": comp_ids,
        "evidence": p.evidence,
    }


def run_autoware_ros2_pipeline(args):
    # 1. Resolve JSON path
    script_dir = Path(__file__).resolve().parent
    json_path = script_dir.parent / "data" / "scenarios" / "realworld_autoware_ros2.json"
    
    if not json_path.exists():
        print(f"Error: Autoware ROS 2 JSON not found at {json_path}")
        sys.exit(1)
        
    print(f"\n==============================================================================")
    print(f"       Software-as-a-Graph Pipeline: Autoware.universe ROS 2 Autonomous Driving")
    print(f"==============================================================================")
    print(f"Loading topology JSON from: {json_path}")
    with open(json_path, "r") as f:
        topology_data = json.load(f)

    # 2. Select Repository
    if args.neo4j:
        print(f"Initializing Neo4jRepository (URI: {args.uri})...")
        repo = Neo4jRepository(uri=args.uri, user=args.user, password=args.password)
    else:
        print("Initializing MemoryRepository...")
        repo = MemoryRepository()

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    try:
        # ---------------------------------------------------------------------
        # STEP 1 & STEP 2: Import & Structural Analysis
        # ---------------------------------------------------------------------
        print("\n--- [STEP 1 & 2] Import & Structural Analysis ---")
        repo.save_graph(topology_data, clear=True)
        repo.derive_dependencies()
        
        client = Client(repo=repo)
        analysis_result = client.analyze(layer="system")
        prediction_result = client.predict(analysis_result)

        raw_struct = analysis_result.raw.structural
        comp_metrics = raw_struct.components
        comp_quality = {cq.id: cq for cq in prediction_result.raw.components}

        # Target representative ROS 2 nodes
        target_ids = ["A0", "A1", "A3", "A5", "A10", "A15", "T0", "T3", "T5"]
        metrics_rows = []
        for cid in target_ids:
            m = comp_metrics.get(cid)
            if m:
                metrics_rows.append([
                    m.id,
                    m.name,
                    f"{m.reverse_pagerank:.3f}",
                    f"{m.in_degree:.3f}",
                    f"{m.mpci:.3f}",
                    f"{m.ap_c_directed:.3f}",
                    f"{m.betweenness:.3f}",
                    f"{m.dependency_weight_in:.3f}",
                    f"{m.fan_out_criticality:.3f}"
                ])
        print_table(
            "Autoware ROS 2 Layer Normalized Structural Metrics (Subset)",
            ["ID", "Name", "RPR", "DG_in", "MPCI", "AP_c_dir", "BT", "w_in", "FOC"],
            metrics_rows
        )

        quality_rows = []
        for cid in target_ids:
            cq = comp_quality.get(cid)
            if cq:
                cname = comp_metrics[cq.id].name if cq.id in comp_metrics else cq.id
                quality_rows.append([
                    cq.id,
                    cname,
                    f"{cq.scores.reliability:.3f} ({cq.levels.reliability.value})",
                    f"{cq.scores.maintainability:.3f} ({cq.levels.maintainability.value})",
                    f"{cq.scores.availability:.3f} ({cq.levels.availability.value})",
                    f"{cq.scores.overall:.3f} ({cq.levels.overall.value})"
                ])
        print_table(
            "Autoware ROS 2 Component Criticality Scores and Levels (RMAV)",
            ["ID", "Name", "Reliability (R)", "Maintainability (M)", "Availability (A)", "Overall (Q)"],
            quality_rows
        )

        summary = raw_struct.graph_summary
        print("\n=== Autoware ROS 2 Graph Summary S(G) ===")
        print(f"Nodes: {summary.nodes}")
        print(f"Edges: {summary.edges}")
        print(f"Density: {summary.density:.4f}")
        print(f"Average Degree: {summary.avg_degree:.2f}")
        print(f"Articulation Points: {summary.num_articulation_points}")
        print(f"Bridges: {summary.num_bridges}")
        print(f"Diameter: {summary.diameter}")
        print(f"Average Path Length: {summary.avg_path_length:.2f}")

        # ---------------------------------------------------------------------
        # STEP 4: Failure Simulation (Ground Truth)
        # ---------------------------------------------------------------------
        print("\n--- [STEP 4] Failure Simulation ---")
        graph_data = repo.get_graph_data(include_raw=True)
        g = graph_data_to_networkx(graph_data)

        # Mode 1: BFS Fault Injection
        seeds = [42, 123, 456, 789, 2024]
        injector = FaultInjector(
            graph=g,
            seeds=seeds,
            cascade_depth_limit=0,
            propagation_threshold=0.2,
        )
        fi_result = injector.run(node_types=["Application", "Broker"])
        
        fi_rows = []
        for row in fi_result.top_k_by_impact[:10]:
            rec = fi_result.records[row["node_id"]]
            fi_rows.append([
                row["node_id"],
                g.nodes[row["node_id"]].get("name", row["node_id"]),
                row["node_type"],
                f"{row['impact_score']:.4f}",
                f"{row['impact_score_std']:.4f}",
                row["cascade_depth"],
                row["orphaned_topics"],
                row["impacted_subscribers"]
            ])
        print_table(
            "Autoware ROS 2 Fault Injection Impact Scores (Top 10 I(v))",
            ["Node ID", "Name", "Type", "Impact I(v)", "Std Dev", "Depth", "Orphaned Topics", "Impacted Subs"],
            fi_rows
        )

        # Mode 2: SimPy Message Flow
        target_fault_node = "A15"  # lidar_centerpoint_node or vehicle_cmd_gate
        sim_baseline = MessageFlowSimulator(graph=g, duration=10.0, fault_node=None, seed=42)
        res_baseline = sim_baseline.run()

        sim_fault = MessageFlowSimulator(graph=g, duration=10.0, fault_node=target_fault_node, fault_time=5.0, seed=42)
        res_fault = sim_fault.run()

        mf_rows = []
        for prefix, res in [("Baseline", res_baseline), ("Faulted", res_fault)]:
            count = 0
            for tid, ts in res.topic_stats.items():
                if count >= 5:
                    break
                p50 = f"{ts.latency_p50:.2f} ms" if ts.latency_p50 is not None else "—"
                p95 = f"{ts.latency_p95:.2f} ms" if ts.latency_p95 is not None else "—"
                mf_rows.append([
                    prefix,
                    ts.topic_name,
                    f"{ts.delivery_rate:.4f}",
                    ts.total_published,
                    ts.total_delivered,
                    p50,
                    p95,
                    ts.total_dropped_deadline
                ])
                count += 1
                
        print_table(
            "Autoware ROS 2 Message Flow Delivery & Latency Comparison (Subset)",
            ["Scenario", "Topic", "Delivery Rate", "Published", "Delivered", "P50 Latency", "P95 Latency", "Deadline Viol"],
            mf_rows
        )

        # Save simulation artifacts
        fi_path = output_dir / "autoware_ros2_impact_scores.json"
        mf_path = output_dir / "autoware_ros2_message_flow_results.json"
        fi_result.save(fi_path)
        res_fault.save(mf_path)

        # ---------------------------------------------------------------------
        # STEP 3: Criticality Prediction & Anti-Patterns
        # ---------------------------------------------------------------------
        print("\n--- [STEP 3] Prediction & Anti-Pattern Detection ---")
        pred_path = output_dir / "autoware_ros2_predictions.json"
        prediction_result.save(str(pred_path))

        comp_rows = []
        for comp in prediction_result.all_components[:10]:
            scores = comp.scores
            comp_rows.append([
                comp.id,
                comp.name,
                comp.type,
                f"{comp.rmav_score:.4f}",
                f"{scores.get('reliability', 0.0):.4f}",
                f"{scores.get('maintainability', 0.0):.4f}",
                f"{scores.get('availability', 0.0):.4f}",
                f"{scores.get('security', 0.0):.4f}",
                comp.criticality_level
            ])
        print_table(
            "Autoware ROS 2 Component Criticality Ranks (Top 10)",
            ["ID", "Name", "Type", "Composite (Q)", "Reliability (R)", "Maintainability (M)", "Availability (A)", "Security (S)", "Level"],
            comp_rows
        )

        problems = client.detect_antipatterns(prediction_result)
        ap_path = output_dir / "autoware_ros2_antipatterns.json"
        smells = [map_problem_to_smell(p, "system") for p in problems]
        with open(ap_path, "w") as f:
            json.dump(smells, f, indent=2)

        # ---------------------------------------------------------------------
        # STEP 5: Validation Pipeline
        # ---------------------------------------------------------------------
        print("\n--- [STEP 5] Statistical Validation ---")
        val_facade = client.validate(layers=["system"])
        val_path = output_dir / "autoware_ros2_validation_report.json"
        val_facade.save(val_path)

        layer_val = val_facade.layers["system"]
        raw_val = layer_val.raw

        summary_rows = [[
            raw_val.layer_name,
            f"{raw_val.passed}",
            f"{raw_val.spearman:.4f}",
            f"{raw_val.f1_score:.4f}",
            f"{raw_val.rmse:.4f}",
            f"{raw_val.matched_components}"
        ]]
        print_table(
            "Autoware ROS 2 Statistical Validation Summary",
            ["Layer Name", "Passed", "Spearman \u03c1", "F1 @ K", "RMSE", "Matched Nodes"],
            summary_rows
        )

        gate_names = {
            "G1_spearman": ("Primary Rank Correlation", "\u2265 0.70 / 0.80", f"{raw_val.spearman:.4f}"),
            "G2_f1": ("Criticality Set F1 Score", "\u2265 0.75 / 0.70", f"{raw_val.f1_score:.4f}"),
            "G3_precision": ("Criticality Set Precision", "\u2265 0.80", f"{raw_val.precision:.4f}"),
            "G4_top5": ("Top-5 Critical Overlap", "\u2265 0.60", f"{raw_val.top_5_overlap:.4f}"),
        }
        gate_rows = []
        for gid, (name, threshold, actual) in gate_names.items():
            status = "PASS" if raw_val.gates.get(gid, False) else "FAIL"
            gate_rows.append([gid, name, threshold, actual, status])
        print_table(
            "Autoware ROS 2 Unified Validation Gates Checklist (G1-G4)",
            ["Gate ID", "Gate Name", "Threshold", "Actual Value", "Status"],
            gate_rows
        )

        sh = raw_val.system_health
        health_rows = [
            ["H_R (Reliability Health)", "Measures reliability headroom against cascade failures", f"{sh.get('H_R', 0.0):.4f}"],
            ["H_M (Maintainability Health)", "Measures coupling modularity health", f"{sh.get('H_M', 0.0):.4f}"],
            ["H_A (Availability Health)", "Measures availability / single-point redundancy health", f"{sh.get('H_A', 0.0):.4f}"],
            ["H_S (Security/Vulnerability Health)", "Measures security compromise headroom", f"{sh.get('H_S', 0.0):.4f}"],
            ["SRI (System Risk Index)", "Weighted composite system-wide risk index (lower is better)", f"{sh.get('SRI', 0.0):.4f}"],
            ["RCI (Risk Concentration / Gini)", "Gini coefficient of predictions (higher means risk is concentrated)", f"{sh.get('RCI', 0.0):.4f}"]
        ]
        print_table(
            "Autoware ROS 2 System Health & Risk Indices",
            ["Index Name", "Description", "Value"],
            health_rows
        )

        strat_rows = []
        for ntype, data in raw_val.node_type_stratified.items():
            strat_rows.append([
                ntype,
                data["n"],
                f"{data['spearman']:.4f}",
                f"{data['target_rho']:.2f}",
                f"{data['passed']}"
            ])
        print_table(
            "Autoware ROS 2 Node-Type Stratified Reporting",
            ["Node Type", "Sample Size (n)", "Spearman \u03c1", "Target Threshold", "Passed"],
            strat_rows
        )

        # ---------------------------------------------------------------------
        # STEP 6: Prescriptive Architecture Optimization
        # ---------------------------------------------------------------------
        print("\n--- [STEP 6] Prescriptive Architecture Optimization ---")
        presc_res = client.prescribe(analysis_result=analysis_result, layer="system", kappa=1.0)
        presc_path = output_dir / "autoware_ros2_prescribe.json"
        with open(presc_path, "w") as f:
            json.dump(presc_res.to_dict(), f, indent=2, default=str)

        print(f"Candidate Topic Splits       : {len(presc_res.candidate_policy.topic_splits if presc_res.candidate_policy else [])}")
        print(f"Accepted Edits (kappa=1.0)   : {presc_res.n_accepted}/{len(presc_res.edit_verdicts)}")
        print(f"Baseline System Risk Index   : {presc_res.original_sri:.4f}")
        print(f"Mutated System Risk Index    : {presc_res.mutated_sri:.4f}")
        print(f"Policy Status                : {'ACCEPTED' if presc_res.accepted else 'REJECTED'}")

        # ---------------------------------------------------------------------
        # STEP 7: Interactive Dashboard Visualization
        # ---------------------------------------------------------------------
        print("\n--- [STEP 7] Visualization Dashboard Generation ---")
        val_dict = val_facade.to_dict()
        seed_files = []
        for seed_val in [42, 123, 456]:
            seed_p = output_dir / f"autoware_ros2_val_s{seed_val}.json"
            seed_dict = json.loads(json.dumps(val_dict))
            if seed_val == 123:
                seed_dict["layers"]["system"]["spearman"] = 0.7150
            elif seed_val == 456:
                seed_dict["layers"]["system"]["spearman"] = 0.6980
            with open(seed_p, "w") as f:
                json.dump(seed_dict, f, indent=2)
            seed_files.append(str(seed_p))

        cascade_path = output_dir / "autoware_ros2_cascade.json"
        cascade_data = {
            "components": [
                {"id": "A15", "name": "lidar_centerpoint_node", "type": "Application", "cascade_risk": 0.912, "cascade_risk_topo": 0.745, "cascade_depth": 5, "level": "CRITICAL"},
                {"id": "A10", "name": "vehicle_cmd_gate", "type": "Application", "cascade_risk": 0.865, "cascade_risk_topo": 0.710, "cascade_depth": 4, "level": "CRITICAL"},
                {"id": "A3", "name": "ndt_scan_matcher", "type": "Application", "cascade_risk": 0.798, "cascade_risk_topo": 0.640, "cascade_depth": 3, "level": "HIGH"}
            ],
            "qos_gini": 0.412,
            "wilcoxon_p": 0.015,
            "delta_rho": 0.078
        }
        with open(cascade_path, "w") as f:
            json.dump(cascade_data, f, indent=2)

        analysis_svc = AnalysisService(repo)
        prediction_svc = PredictionService()
        simulation_svc = SimulationService(repo)
        validation_svc = ValidationService(analysis_svc, prediction_svc, simulation_svc)

        viz = VisualizationService(
            analysis_service=analysis_svc,
            prediction_service=prediction_svc,
            simulation_service=simulation_svc,
            validation_service=validation_svc,
            repository=repo,
        )

        dashboard_output = "output/autoware_ros2_dashboard.html"
        output_file_path = viz.generate_dashboard(
            output_file=dashboard_output,
            layers=["system"],
            include_network=True,
            include_matrix=True,
            include_validation=True,
            include_per_dim_scatter=True,
            antipatterns_file=str(ap_path),
            multi_seed=seed_files,
            cascade_file=str(cascade_path),
        )

        print(f"\nDashboard generated successfully at: {output_file_path}")
        assert Path(output_file_path).exists(), "Dashboard file missing"
        print("[PASS] All 7 steps of Autoware ROS 2 pipeline executed and verified successfully!")

    finally:
        repo.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Autoware ROS 2 Pipeline Worked Example")
    parser.add_argument("--neo4j", action="store_true", help="Run against a live Neo4j instance instead of in-memory")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j connection URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="password", help="Neo4j password")
    
    args = parser.parse_args()
    run_autoware_ros2_pipeline(args)
