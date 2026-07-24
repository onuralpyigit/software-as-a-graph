#!/usr/bin/env bash
# scripts/populate_loso_cache.sh
# =======================================================================
# Populates output/loso_cache/<scenario>/ for all 8 reference scenarios.
# Each scenario dir needs: topology.json, structural_metrics.json,
#   failure_impact.json, quality_scores.json
#
# Usage:
#   bash scripts/populate_loso_cache.sh
#   bash scripts/populate_loso_cache.sh atm_system av_system   # specific scenarios
#   SCENARIOS="atm_system iot_smart_city_system" bash scripts/populate_loso_cache.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CACHE_DIR="output/loso_cache"
SCENARIOS_DIR="data/scenarios"

ALL_SCENARIOS=(
    atm_system
    av_system
    iot_smart_city_system
    financial_trading_system
    healthcare_system
    hub_and_spoke_system
    microservices_system
    enterprise_system
)

# Allow passing specific scenarios as args
if [ $# -gt 0 ]; then
    TARGETS=("$@")
else
    TARGETS=("${ALL_SCENARIOS[@]}")
fi

echo ""
echo "  ═══════════════════════════════════════════════════════════"
echo "  LOSO Cache Population — ${#TARGETS[@]} scenarios"
echo "  Cache dir: $CACHE_DIR"
echo "  ═══════════════════════════════════════════════════════════"
echo ""

for scenario in "${TARGETS[@]}"; do
    json_path="$SCENARIOS_DIR/${scenario}.json"
    out="$CACHE_DIR/${scenario}"

    if [ ! -f "$json_path" ]; then
        echo "  SKIP: $scenario (no JSON at $json_path)"
        continue
    fi

    # Check if cache is already fully populated
    if [ -f "$out/failure_impact.json" ] && [ -f "$out/quality_scores.json" ]; then
        echo "  EXISTS: $scenario"
        continue
    fi

    echo "  ──────────────────────────────────────────"
    echo "  Processing: $scenario"
    mkdir -p "$out"

    # Step 1: Copy topology
    if [ ! -f "$out/topology.json" ]; then
        cp "$json_path" "$out/topology.json"
        echo "  [1/4] topology.json copied"
    fi

    # Step 2: Import graph
    echo "  [2/4] Importing graph ..."
    PYTHONPATH=. python cli/import_graph.py \
        --input "$out/topology.json" \
        --clear 2>&1 | tail -2 || echo "  (import_graph error — continuing)"

    # Step 3: Structural metrics
    if [ ! -f "$out/structural_metrics.json" ]; then
        echo "  [3/4] Computing structural metrics ..."
        PYTHONPATH=. python cli/analyze_graph.py \
            --layer app \
            --output "$out/structural_metrics.json" 2>&1 | tail -2 || \
            echo "  (analyze_graph error — skipping)"
    else
        echo "  [3/4] structural_metrics.json exists"
    fi

    # Step 4: Fault injection → failure_impact.json
    if [ ! -f "$out/failure_impact.json" ]; then
        echo "  [4/4] Running fault injection ..."
        # Five seeds, not one: the artifact's label_stability block needs at
        # least two to measure test-retest agreement, and the reported rho has
        # no stated ceiling without it.
        # Library is included because app_to_lib cascades at prob 1.0 and yields
        # strong labels; Topic and Node are excluded because the cascade cannot
        # express their failure and they would contribute only spurious zeros.
        PYTHONPATH=. python cli/simulate_graph.py fault-inject \
            --input "$out/topology.json" \
            --output "$out/" \
            --export-json \
            --node-types Application,Broker,Library \
            --seeds 42,123,456,789,2024 2>&1 | tail -3 || \
            echo "  (simulate_graph error — skipping)"
        # Rename if generated with different name
        for f in "$out"/impact_scores*.json "$out"/failure_impact_*.json; do
            [ -f "$f" ] && mv "$f" "$out/failure_impact.json" && break
        done
    else
        echo "  [4/4] failure_impact.json exists"
    fi

    # Step 5: RMAV quality scores
    if [ ! -f "$out/quality_scores.json" ]; then
        echo "  [5/5] Computing RMAV quality scores ..."
        PYTHONPATH=. python cli/predict_graph.py \
            --layer app \
            --output "$out/quality_scores.json" 2>&1 | tail -2 || \
            echo "  (predict_graph error — skipping)"
    else
        echo "  [5/5] quality_scores.json exists"
    fi

    echo "  ✓ $scenario done"
    echo ""
done

echo "  ═══════════════════════════════════════════════════════════"
echo "  Cache population complete."
echo ""
echo "  Status:"
for scenario in "${TARGETS[@]}"; do
    out="$CACHE_DIR/${scenario}"
    fi_ok="✗"; qs_ok="✗"; sm_ok="✗"
    [ -f "$out/failure_impact.json" ] && fi_ok="✓"
    [ -f "$out/quality_scores.json" ] && qs_ok="✓"
    [ -f "$out/structural_metrics.json" ] && sm_ok="✓"
    echo "    $scenario  structural=$sm_ok  simulation=$fi_ok  rmav=$qs_ok"
done
echo ""
