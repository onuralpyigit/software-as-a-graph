#!/usr/bin/env bash
# scripts/run_main_table.sh — Block C: Table 3 training matrix runner
# =====================================================================
# Runs the 6×8×N training matrix for Table 3, then renders outputs.
#
# Usage:
#   bash scripts/run_main_table.sh
#   bash scripts/run_main_table.sh --epochs 50 --seeds "42 123"
#   bash scripts/run_main_table.sh --scenarios "atm_system av_system" \
#                                   --variants "hgl hetero_qos"
#   bash scripts/run_main_table.sh --resume
#   bash scripts/run_main_table.sh --epochs 50 -- --dry-run    # pass-through
#
# Flag conventions:
#   • --scenarios / --variants / --seeds  take a single quoted, space-
#     separated string (mirrors how --seeds already worked).
#   • Anything after `--` is forwarded verbatim to the Python tool.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ── Defaults ──────────────────────────────────────────────────────────────────
EPOCHS=300
SEEDS="42 123 456 789 2024"
SCENARIOS=""
VARIANTS=""
OUTPUT="results/main_table.json"
RESUME=""
EXTRA_ARGS=()

# ── Help ──────────────────────────────────────────────────────────────────────
show_help() {
    cat <<'EOF'
Usage: bash scripts/run_main_table.sh [options] [-- python-passthrough-args]

Options:
  --epochs N             Training epochs per cell           (default: 300)
  --seeds   "S1 S2 ..."  Quoted space-separated seed list   (default: 42 123 456 789 2024)
  --scenarios "S1 S2 ..."  Quoted space-separated scenarios (default: all 8)
  --variants  "V1 V2 ..."  Quoted space-separated variants  (default: all 6)
  --output PATH          Output JSON path                   (default: results/main_table.json)
  --resume               Skip cells already in the output JSON
  -h, --help             Show this help and exit
  --                     Pass remaining arguments verbatim to the Python tool

Variant names: topo_baseline, q_topo_baseline, homo_unweighted, homo_scalar,
               hgl, hetero_qos

Examples:
  # Smoke: 2 seeds, 3 scenarios, only the two hetero variants
  bash scripts/run_main_table.sh --epochs 50 \
       --seeds "42 123" \
       --scenarios "atm_system av_system iot_smart_city_system" \
       --variants "hgl hetero_qos"

  # Dry-run via pass-through
  bash scripts/run_main_table.sh -- --dry-run

  # Full 2x3 factorial smoke (§5 acceptance test)
  bash scripts/run_main_table.sh \
       --scenarios "atm_system" \
       --seeds "42 123" \
       --epochs 50 \
       --output results/smoke_6_variants.json
EOF
}

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)      EPOCHS="$2";    shift 2 ;;
        --seeds)       SEEDS="$2";     shift 2 ;;
        --scenarios)   SCENARIOS="$2"; shift 2 ;;
        --variants)    VARIANTS="$2";  shift 2 ;;
        --output)      OUTPUT="$2";    shift 2 ;;
        --resume)      RESUME="--resume"; shift ;;
        -h|--help)     show_help; exit 0 ;;
        --)            shift; EXTRA_ARGS=("$@"); break ;;
        *)             echo "Unknown flag: $1 (use --help, or '--' to pass args to Python)"; exit 1 ;;
    esac
done

# Build optional arg arrays so empty values do not produce stray flags.
SCENARIOS_ARG=();  [[ -n "$SCENARIOS" ]] && SCENARIOS_ARG=(--scenarios $SCENARIOS)
VARIANTS_ARG=();   [[ -n "$VARIANTS"  ]] && VARIANTS_ARG=(--variants  $VARIANTS)
RESUME_ARG=();     [[ -n "$RESUME"    ]] && RESUME_ARG=($RESUME)

echo ""
echo "  ═══════════════════════════════════════════════════════════"
echo "  Block C — Table 3: Main Results Training Matrix"
echo "  Epochs    : $EPOCHS"
echo "  Seeds     : $SEEDS"
echo "  Scenarios : ${SCENARIOS:-<all 8>}"
echo "  Variants  : ${VARIANTS:-<all 6>}"
echo "  Output    : $OUTPUT"
echo "  Resume    : ${RESUME:-off}"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "  Extra     : ${EXTRA_ARGS[*]}"
echo "  ═══════════════════════════════════════════════════════════"
echo ""

# ── Step 0: W1 audit gate ─────────────────────────────────────────────────────
# Show full pytest output (preserves failure detail when the gate breaks).
# `set -e` aborts the script on a non-zero pytest exit.
echo "  [0/3] Running W1 QoS pipeline audit ..."
if ! PYTHONPATH=. python -m pytest \
        tests/test_qos_pipeline_audit.py tests/test_baselines.py \
        -q --tb=short; then
    echo ""
    echo "  ✗ Gate failed — see pytest output above." >&2
    exit 1
fi
echo "  ✓ Gate cleared."
echo ""

# ── Step 1: Training matrix ────────────────────────────────────────────────────
echo "  [1/3] Training matrix ..."
PYTHONPATH=. python tools/middleware26_main_table.py \
    --output "$OUTPUT" \
    --seeds ${SEEDS} \
    --epochs "$EPOCHS" \
    "${SCENARIOS_ARG[@]}" \
    "${VARIANTS_ARG[@]}" \
    "${RESUME_ARG[@]}" \
    "${EXTRA_ARGS[@]}" \
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
for f in results/table3_main_results.tex \
         results/table3_main_results.csv \
         results/table3_main_results.md \
         results/table3_id_metrics.md; do
    [[ -f "$f" ]] && echo "    ✓ $f"
done
echo ""
