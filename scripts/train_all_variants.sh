#!/usr/bin/env bash
# scripts/train_all_variants.sh — Block A acceptance command
# ============================================================
# Trains all four model variants on a given scenario and prints per-variant
# Spearman ρ side-by-side.  Acceptance criterion: all four produce non-trivial ρ.
#
# Usage:
#   bash scripts/train_all_variants.sh atm_system
#   bash scripts/train_all_variants.sh atm_system --epochs 100 --seeds 42 123
#
# The first positional argument is the scenario name (without .json extension).
# Additional arguments are forwarded to cli/train_graph.py.

set -euo pipefail

SCENARIO="${1:-atm_system}"
shift || true   # allow no extra args

SCENARIO_JSON="data/scenarios/${SCENARIO}.json"
if [[ ! -f "${SCENARIO_JSON}" ]]; then
    echo "Error: scenario file not found: ${SCENARIO_JSON}" >&2
    exit 1
fi

VARIANTS=("hetero_qos" "homo_unweighted" "homo_scalar" "topology_rmav")
RESULTS=()

echo "================================================================"
echo "  train_all_variants.sh — scenario: ${SCENARIO}"
echo "  Variants: ${VARIANTS[*]}"
echo "================================================================"
echo ""

for VARIANT in "${VARIANTS[@]}"; do
    CKPT_DIR="output/gnn_checkpoints/${SCENARIO}_${VARIANT}"
    echo "── Variant: ${VARIANT} ────────────────────────────────────────"

    if [[ "${VARIANT}" == "topology_rmav" ]]; then
        # RMAV baseline: needs --rmav; if no pre-computed file, skip gracefully
        RMAV_PATH="output/loso_cache/${SCENARIO}/quality_scores.json"
        if [[ -f "${RMAV_PATH}" ]]; then
            PYTHONPATH=. python cli/train_graph.py \
                --structural "output/loso_cache/${SCENARIO}/structural_metrics.json" \
                --simulated  "output/loso_cache/${SCENARIO}/failure_impact.json" \
                --rmav       "${RMAV_PATH}" \
                --variant topology_rmav \
                --output     "output/results/${SCENARIO}_topology_rmav.json" \
                "$@" 2>&1 | tail -5
            RESULTS+=("topology_rmav: see output/results/${SCENARIO}_topology_rmav.json")
        else
            echo "  Skipping topology_rmav (no RMAV cache at ${RMAV_PATH})"
            RESULTS+=("topology_rmav: SKIPPED (no RMAV cache)")
        fi
    else
        # GNN variants: use scenario JSON directly if loso cache not available
        if [[ -d "output/loso_cache/${SCENARIO}" ]]; then
            STRUCTURAL="output/loso_cache/${SCENARIO}/structural_metrics.json"
            SIMULATED="output/loso_cache/${SCENARIO}/failure_impact.json"
        else
            # Fall back to scenario JSON alone (will run neo4j pipeline)
            STRUCTURAL=""
            SIMULATED=""
        fi

        EXTRA_ARGS=""
        if [[ -n "${STRUCTURAL}" ]]; then
            EXTRA_ARGS="--structural ${STRUCTURAL} --simulated ${SIMULATED}"
        fi

        PYTHONPATH=. python cli/train_graph.py \
            ${EXTRA_ARGS} \
            --variant   "${VARIANT}" \
            --checkpoint "${CKPT_DIR}" \
            --output    "output/results/${SCENARIO}_${VARIANT}.json" \
            "$@" 2>&1 | tail -10

        RESULTS+=("${VARIANT}: see output/results/${SCENARIO}_${VARIANT}.json")
    fi
    echo ""
done

echo "================================================================"
echo "  Summary"
echo "================================================================"
for R in "${RESULTS[@]}"; do
    echo "  ${R}"
done
echo ""
echo "Done. Check output/results/ for per-variant JSON results."
