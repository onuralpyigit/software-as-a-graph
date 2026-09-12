#!/usr/bin/env bash
# scripts/populate_loso_cache.sh
# =======================================================================
# Populates output/loso_cache/<scenario>/ for all 12 reference scenarios.
# Each scenario dir needs: topology.json, structural_metrics.json,
#   failure_impact.json, quality_scores.json
#
# Usage:
#   bash scripts/populate_loso_cache.sh
#   bash scripts/populate_loso_cache.sh atm_system av_system   # specific scenarios
#   SCENARIOS="atm_system iot_smart_city_system" bash scripts/populate_loso_cache.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Overridable so a corpus that is not the LOSO corpus can be cached without
# landing in output/loso_cache/, where discover_scenarios() would pick it up as
# an extra fold and silently change every LOSO number.
CACHE_DIR="${CACHE_DIR:-output/loso_cache}"
# The oracle's QoS arm. "ladder" is the published setting. "none" builds the
# label-side control for RQ3.1: I*'s QoS ladder reads reliability and transport
# priority, which are edge-feature dims 9 and 11 -- given only to the -QoS
# predictors. Measuring those arms against QoS-free labels separates "QoS
# features help" from "the label function reads the same QoS". Pair it with a
# CACHE_DIR override so the two label sets never share a directory.
QOS_FACTOR="${QOS_FACTOR:-ladder}"
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
    telecom_ran_system
    industrial_scada_system
    realtime_gaming_system
    logistics_fleet_system
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
        echo "  [1/6] topology.json copied"
    fi

    # Step 2: Import graph
    echo "  [2/6] Importing graph ..."
    PYTHONPATH=. python cli/import_graph.py \
        --input "$out/topology.json" \
        --clear 2>&1 | tail -2 || echo "  (import_graph error — continuing)"

    # Step 3: Structural metrics
    if [ ! -f "$out/structural_metrics.json" ]; then
        echo "  [3/6] Computing structural metrics ..."
        PYTHONPATH=. python cli/analyze_graph.py \
            --layer app \
            --output "$out/structural_metrics.json" 2>&1 | tail -2 || \
            echo "  (analyze_graph error — skipping)"
    else
        echo "  [3/6] structural_metrics.json exists"
    fi

    # Step 4: Fault injection → failure_impact.json
    if [ ! -f "$out/failure_impact.json" ]; then
        echo "  [4/6] Running fault injection ..."
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
            --qos-factor "$QOS_FACTOR" \
            --seeds 42,123,456,789,2024 2>&1 | tail -3 || \
            echo "  (simulate_graph error — skipping)"
        # Rename if generated with different name
        for f in "$out"/impact_scores*.json "$out"/failure_impact_*.json; do
            [ -f "$f" ] && mv "$f" "$out/failure_impact.json" && break
        done
    else
        echo "  [4/6] failure_impact.json exists"
    fi

    # Step 5: Edge criticality -> edge_criticality.json
    # The measured cost of severing a relationship, both endpoints alive. This
    # is what supervises the GNN's edge head; without it networkx_to_hetero_data
    # writes no edge labels at all and L_edge stays inactive (it used to
    # substitute I*(source) x {1.0 if bridge else 0.1}, a structural heuristic
    # standing in for a measurement). Reads topology.json directly, so it needs
    # no live database.
    if [ ! -f "$out/edge_criticality.json" ]; then
        echo "  [5/6] Running edge-removal sweep ..."
        PYTHONPATH=. python cli/simulate_graph.py edge-criticality \
            --input "$out/topology.json" \
            --qos-factor "$QOS_FACTOR" \
            --output "$out/edge_criticality.json" 2>&1 | tail -3 || \
            echo "  (edge-criticality error — skipping)"
    else
        echo "  [5/6] edge_criticality.json exists"
    fi

    # Step 6: RM quality scores
    # --no-antipatterns: the cache consumes RM scores only, and detection is
    # measured separately by reproduce/detection_validation.py. Leaving it on
    # also makes this step hang — DEEP_PIPELINE enumerates every simple path and
    # does not terminate on the larger topologies (see DEFAULT_EXCLUDED_PATTERNS
    # in reproduce/detection_validation.py). It also forces a non-zero exit via
    # the deployment gate, which this script would report as a spurious error.
    if [ ! -f "$out/quality_scores.json" ]; then
        echo "  [6/6] Computing RM quality scores ..."
        PYTHONPATH=. python cli/predict_graph.py \
            --layer app --no-antipatterns \
            --output "$out/quality_scores.json" 2>&1 | tail -2 || \
            echo "  (predict_graph error — skipping)"
    else
        echo "  [6/6] quality_scores.json exists"
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
    echo "    $scenario  structural=$sm_ok  simulation=$fi_ok  rm=$qs_ok"
done
echo ""
