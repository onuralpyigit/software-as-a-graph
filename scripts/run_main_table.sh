#!/usr/bin/env bash
# scripts/run_main_table.sh — Block C: Table 3 training matrix runner
# =====================================================================
# Runs the full 8×4×5 training matrix for Table 3, then renders outputs.
#
# Usage:
#   bash scripts/run_main_table.sh
#   bash scripts/run_main_table.sh --epochs 50 --seeds "42 123"  (fast smoke)
#   bash scripts/run_main_table.sh --resume                       (resume interrupted)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ── Defaults ──────────────────────────────────────────────────────────────────
EPOCHS=300
SEEDS="42,123,456,789,2024"
SCENARIOS_DIR="data/scenarios"
OUTPUT="results/main_table.json"
RESUME=""

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)   EPOCHS="$2";  shift 2 ;;
        --seeds)    SEEDS="$2";   shift 2 ;;
        --output)   OUTPUT="$2";  shift 2 ;;
        --resume)   RESUME="--resume"; shift ;;
        *)          echo "Unknown flag: $1"; exit 1 ;;
    esac
done

echo ""
echo "  ═══════════════════════════════════════════════════════════"
echo "  Block C — Table 3: Main Results Training Matrix"
echo "  Epochs  : $EPOCHS"
echo "  Seeds   : $SEEDS"
echo "  Output  : $OUTPUT"
echo "  Resume  : ${RESUME:-off}"
echo "  ═══════════════════════════════════════════════════════════"
echo ""

# ── Step 0: W1 audit gate ─────────────────────────────────────────────────────
echo "  [0/3] Running W1 QoS pipeline audit ..."
PYTHONPATH=. python -m pytest tests/test_qos_pipeline_audit.py tests/test_baselines.py \
    -q --tb=short 2>&1 | tail -3
echo "  ✓ Gate cleared."
echo ""

# ── Step 1: Training matrix ────────────────────────────────────────────────────
echo "  [1/3] Training matrix (8 scenarios × 4 variants × 5 seeds) ..."
PYTHONPATH=. python tools/middleware26_main_table.py \
    --scenarios-dir "$SCENARIOS_DIR" \
    --output "$OUTPUT" \
    --seeds "$SEEDS" \
    --epochs "$EPOCHS" \
    ${RESUME} \
    --verbose

echo ""
echo "  ✓ Training complete. Raw results: $OUTPUT"
echo ""

# ── Step 2: Render tables ─────────────────────────────────────────────────────
echo "  [2/3] Rendering Table 3 (LaTeX / CSV / Markdown) ..."
PYTHONPATH=. python tools/render_table.py \
    --table3 "$OUTPUT" \
    --output-dir results/

echo ""

# ── Step 3: Summary ────────────────────────────────────────────────────────────
echo "  [3/3] Done."
echo ""
echo "  Outputs:"
for f in results/table3_main_results.tex results/table3_main_results.csv results/table3_main_results.md; do
    if [ -f "$f" ]; then
        echo "    ✓ $f"
    fi
done
echo ""
