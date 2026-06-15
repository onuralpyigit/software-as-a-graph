#!/usr/bin/env python3
"""
Worked Example: Validation Pipeline on the Air Traffic Management (ATM) Dataset.
Adheres strictly to the structural and functional specifications of Step 5 Validation.
"""

import argparse
import json
import math
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saag import Client
from saag.infrastructure.memory_repo import MemoryRepository
from saag.infrastructure.neo4j_repo import Neo4jRepository


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


def run_atm_validation(args):
    # 1. Resolve path to atm_system.json
    script_dir = Path(__file__).resolve().parent
    json_path = script_dir.parent / "data" / "scenarios" / "atm_system.json"
    
    if not json_path.exists():
        print(f"Error: ATM System JSON not found at {json_path}")
        sys.exit(1)
        
    print(f"Loading safety-critical ATM topology JSON from: {json_path}")
    with open(json_path, "r") as f:
        topology_data = json.load(f)

    # 2. Initialize repository & client
    if args.neo4j:
        print(f"Initializing Neo4jRepository (URI: {args.uri})...")
        repo = Neo4jRepository(uri=args.uri, user=args.user, password=args.password)
    else:
        print("Initializing MemoryRepository...")
        repo = MemoryRepository()

    try:
        # 3. Save graph to initialize component weights
        print("Saving graph to initialize weights...")
        repo.save_graph(topology_data, clear=True)
        
        # 4. Derive logical dependencies (Step 1 derived rules)
        print("Deriving logical dependencies (G_analysis)...")
        repo.derive_dependencies()
        
        # 5. Create SDK client and run validation
        client = Client(repo=repo)
        print("Running validation pipeline for 'system' layer...")
        val_facade = client.validate(layers=["system"])
        
        # Retrieve layer result
        layer_val = val_facade.layers["system"]
        raw_val = layer_val.raw
        
        # ---------------------------------------------------------------------
        # Table 1: Validation Summary
        # ---------------------------------------------------------------------
        summary_rows = [[
            raw_val.layer_name,
            f"{raw_val.passed}",
            f"{raw_val.spearman:.4f}",
            f"{raw_val.f1_score:.4f}",
            f"{raw_val.rmse:.4f}",
            f"{raw_val.matched_components}"
        ]]
        print_table(
            "Layer Statistical Validation Summary",
            ["Layer Name", "Passed", "Spearman \u03c1", "F1 @ K", "RMSE", "Matched Nodes"],
            summary_rows
        )

        # ---------------------------------------------------------------------
        # Table 2: Unified Validation Gates (G1-G9)
        # ---------------------------------------------------------------------
        gate_names = {
            "G1_spearman": ("Primary Rank Correlation", "\u2265 0.70 / 0.80", f"{raw_val.spearman:.4f}"),
            "G2_f1": ("Criticality Set F1 Score", "\u2265 0.75 / 0.70", f"{raw_val.f1_score:.4f}"),
            "G3_precision": ("Criticality Set Precision", "\u2265 0.80", f"{raw_val.precision:.4f}"),
            "G4_top5": ("Top-5 Critical Overlap", "\u2265 0.60", f"{raw_val.top_5_overlap:.4f}"),
            "G5_predictive_gain": ("Predictive Gain (PG)", "> 0.03", f"{raw_val.predictive_gain:.4f}"),
            "G6_kappa_cta": ("Weighted Kappa \u03ba_CTA", "\u2265 0.70", f"{raw_val.dimensional_validation.get('maintainability', {}).get('weighted_kappa_cta', 0.0):.4f}"),
            "G7_cdcc": ("Cross-Dim Contamination Check", "< 0.30", f"{raw_val.dimensional_validation.get('security', {}).get('cdcc', 1.0):.4f}"),
            "G8_bottleneck_precision": ("Bottleneck Precision BP", "\u2265 0.70", f"{raw_val.dimensional_validation.get('maintainability', {}).get('bottleneck_precision', 0.0):.4f}"),
            "G9_ftr": ("False Top Rate FTR", "\u2264 0.20", f"{raw_val.dimensional_validation.get('security', {}).get('ftr', 1.0):.4f}"),
        }
        
        gate_rows = []
        for gid, (name, threshold, actual) in gate_names.items():
            status = "PASS" if raw_val.gates.get(gid, False) else "FAIL"
            gate_rows.append([gid, name, threshold, actual, status])
            
        print_table(
            "Unified Validation Gates Checklist (G1-G9)",
            ["Gate ID", "Gate Name", "Threshold", "Actual Value", "Status"],
            gate_rows
        )

        # ---------------------------------------------------------------------
        # Table 3: Multi-Dimensional Correlations
        # ---------------------------------------------------------------------
        dim_rows = [
            ["Reliability", "R(v)", "IR(v) (cascade reach)", f"{raw_val.reliability_spearman:.4f}"],
            ["Maintainability", "M(v)", "IM(v) (fragility check)", f"{raw_val.maintainability_spearman:.4f}"],
            ["Availability", "A(v)", "IA(v) (partitioning)", f"{raw_val.availability_spearman:.4f}"],
            ["Vulnerability", "V(v)", "IV(v) (compromise reach)", f"{raw_val.security_spearman:.4f}"],
            ["Composite", "Q*(v)", "I*(v) (composite ground truth)", f"{raw_val.composite_spearman:.4f}"]
        ]
        print_table(
            "Multi-Dimensional Validation (Predictors vs. Sim Ground Truths)",
            ["Dimension", "Predictor", "Ground Truth Target", "Spearman \u03c1"],
            dim_rows
        )

        # ---------------------------------------------------------------------
        # Table 4: System Health Indices
        # ---------------------------------------------------------------------
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
            "System Health and Risk Indices",
            ["Index Name", "Description", "Value"],
            health_rows
        )

        # ---------------------------------------------------------------------
        # Table 5: Node-Type Stratification
        # ---------------------------------------------------------------------
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
            "Node-Type Stratified Reporting",
            ["Node Type", "Sample Size (n)", "Spearman \u03c1", "Target Threshold", "Passed"],
            strat_rows
        )

        # ---------------------------------------------------------------------
        # Table 6: Topic Frequency-Decile Stratification
        # ---------------------------------------------------------------------
        decile_rows = []
        for decile, data in raw_val.frequency_decile_stratified.items():
            decile_rows.append([
                decile,
                f"{data['frequency_range'][0]:.2f} - {data['frequency_range'][1]:.2f} Hz",
                data["n"],
                f"{data['spearman']:.4f}",
                f"{data['p_value']:.6f}"
            ])
        print_table(
            "Topic Frequency-Decile Stratified Reporting",
            ["Decile", "Frequency Range", "Sample size (n)", "Spearman \u03c1", "p-value"],
            decile_rows
        )

        # ---------------------------------------------------------------------
        # Component Predictions vs. Actuals Table (Subset)
        # ---------------------------------------------------------------------
        print("\n=== Component Predictions vs. Actuals (Top 10) ===")
        print("ID | Type        | Predicted | Actual | Classification")
        print("-" * 55)
        for comp in raw_val.comparisons[:10]:
            print(f"{comp.id.ljust(2)} | {comp.type.ljust(11)} | {comp.predicted:.4f}    | {comp.actual:.4f} | {comp.classification}")

        # ---------------------------------------------------------------------
        # Verification Assertions
        # ---------------------------------------------------------------------
        print("\nVerifying Validation Results:")
        assert raw_val.matched_components > 5, f"Expected more than 5 matched components, got {raw_val.matched_components}"
        assert 0.0 < sh["SRI"] < 1.0, f"Expected SRI in (0, 1) bounds, got {sh['SRI']}"
        assert 0.0 <= sh["RCI"] <= 1.0, f"Expected RCI in [0, 1] bounds, got {sh['RCI']}"
        print("  [PASS] Matched components and health indices bounds verified successfully.")

        # ---------------------------------------------------------------------
        # Save output JSON
        # ---------------------------------------------------------------------
        output_path = Path("output/atm_system_validation_report.json")
        val_facade.save(output_path)
        print(f"\nSaved validation report JSON to: {output_path}")
        print("\nValidation pipeline executed and verified successfully on ATM dataset!")

    finally:
        repo.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ATM Validation Pipeline")
    parser.add_argument("--neo4j", action="store_true", help="Run against a live Neo4j instance instead of in-memory")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j connection URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="password", help="Neo4j password")
    
    args = parser.parse_args()
    run_atm_validation(args)
