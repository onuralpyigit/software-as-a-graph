#!/usr/bin/env bash
# scripts/recalibrate.sh — end-to-end F1 recalibration flow
# =========================================================
# Orchestrates: audit → recalibrate → re-train flagged → re-render
#
# The input JSON is never mutated; results land in <input>_recalibrated.json
# (overridable via --output).  When the recalibration tool cannot recover a
# cell (typically a GAT cell whose checkpoint is missing), the cell is
# flagged with needs_recalibration:true.  Step 3 picks those up via
# scripts/run_main_table.sh --resume and re-trains only them.
#
# Usage:
#   bash scripts/recalibrate.sh                              # default flow
#   bash scripts/recalibrate.sh --topo-only                  # safest subset
#   bash scripts/recalibrate.sh --skip-retrain               # audit+recal only
#   bash scripts/recalibrate.sh -y                           # no prompt
#   bash scripts/recalibrate.sh --input results/foo.json \
#                                --output results/foo_v2.json

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ── Defaults ──────────────────────────────────────────────────────────────────
INPUT="results/main_table.json"
OUTPUT=""
TOPO_ONLY=""
VARIANTS=""
SCENARIOS=""
SKIP_AUDIT=""
SKIP_RETRAIN=""
SKIP_RENDER=""
EPOCHS=300
SEEDS="42 123 456 789 2024"
AUTO_YES=""

# ── Help ──────────────────────────────────────────────────────────────────────
show_help() {
    cat <<'EOF'
Usage: bash scripts/recalibrate.sh [options]

Pipeline:  audit → recalibrate → re-train flagged cells → re-render tables.

Options:
  --input PATH         Existing main_table.json (default: results/main_table.json)
  --output PATH        Recalibrated output JSON (default: <input>_recalibrated.json)
  --topo-only          Only recompute the two structural baselines
                       (always safe — no GAT inference required)
  --variants "A B C"   Restrict recalibration to these variants
  --scenarios "X Y"    Restrict recalibration to these scenarios
  --epochs N           Re-training epochs                       (default: 300)
  --seeds "A B C"      Re-training seeds                        (default: 42 123 456 789 2024)
  --skip-audit         Skip the audit print
  --skip-retrain       Stop after recalibration; do not re-train flagged cells
  --skip-render        Stop before re-rendering
  -y, --yes            Skip the interactive confirmation prompt
  -h, --help           Show this help and exit

Examples:
  # Topo-only (no checkpoint dependency, always works)
  bash scripts/recalibrate.sh --topo-only -y

  # Full recalibration with prompt
  bash scripts/recalibrate.sh \
      --input results/main_table.json \
      --output results/main_table_recalibrated.json

  # Specific scenarios only, auto-yes
  bash scripts/recalibrate.sh \
      --scenarios "atm_system financial_trading_system" -y
EOF
}

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)        INPUT="$2";       shift 2 ;;
        --output)       OUTPUT="$2";      shift 2 ;;
        --topo-only)    TOPO_ONLY=1;      shift ;;
        --variants)     VARIANTS="$2";    shift 2 ;;
        --scenarios)    SCENARIOS="$2";   shift 2 ;;
        --epochs)       EPOCHS="$2";      shift 2 ;;
        --seeds)        SEEDS="$2";       shift 2 ;;
        --skip-audit)   SKIP_AUDIT=1;     shift ;;
        --skip-retrain) SKIP_RETRAIN=1;   shift ;;
        --skip-render)  SKIP_RENDER=1;    shift ;;
        -y|--yes)       AUTO_YES=1;       shift ;;
        -h|--help)      show_help; exit 0 ;;
        *)              echo "Unknown flag: $1 (use --help)" >&2; exit 1 ;;
    esac
done

# ── Validation ────────────────────────────────────────────────────────────────
if [[ ! -f "$INPUT" ]]; then
    echo "Error: input file not found: $INPUT" >&2
    exit 1
fi
if [[ -n "$TOPO_ONLY" && -n "$VARIANTS" ]]; then
    echo "Error: --topo-only and --variants are mutually exclusive." >&2
    exit 1
fi
[[ -z "$OUTPUT" ]] && OUTPUT="${INPUT%.json}_recalibrated.json"

# ── Build arg arrays (so empty values do not emit stray flags) ────────────────
RECAL_VARIANTS=()
if [[ -n "$TOPO_ONLY" ]]; then
    RECAL_VARIANTS=(--variants topo_baseline topo_qos)
elif [[ -n "$VARIANTS" ]]; then
    RECAL_VARIANTS=(--variants $VARIANTS)
fi
RECAL_SCENARIOS=()
[[ -n "$SCENARIOS" ]] && RECAL_SCENARIOS=(--scenarios $SCENARIOS)

# ── Header ────────────────────────────────────────────────────────────────────
echo ""
echo "  ═══════════════════════════════════════════════════════════"
echo "  F1 Recalibration Pipeline"
echo "  Input          : $INPUT"
echo "  Output         : $OUTPUT"
[[ -n "$TOPO_ONLY"     ]] && echo "  Mode           : topo-only"
[[ -n "$VARIANTS"      ]] && echo "  Variants       : $VARIANTS"
[[ -n "$SCENARIOS"     ]] && echo "  Scenarios      : $SCENARIOS"
[[ -n "$SKIP_AUDIT"    ]] && echo "  Audit          : skipped"
[[ -n "$SKIP_RETRAIN"  ]] && echo "  Re-train       : skipped"
[[ -n "$SKIP_RENDER"   ]] && echo "  Re-render      : skipped"
echo "  ═══════════════════════════════════════════════════════════"
echo ""

# ── Step 1: Audit ─────────────────────────────────────────────────────────────
if [[ -z "$SKIP_AUDIT" ]]; then
    echo "  [1/4] Audit ──────────────────────────────────────────────"
    PYTHONPATH=. python tools/recalibrate_main_table.py \
        --input "$INPUT" --audit \
        "${RECAL_VARIANTS[@]}" \
        "${RECAL_SCENARIOS[@]}"
    echo ""
fi

# ── Confirmation prompt ───────────────────────────────────────────────────────
if [[ -z "$AUTO_YES" ]]; then
    read -r -p "  Proceed with recalibration? [y/N] " response
    case "$response" in
        [yY]|[yY][eE][sS]) ;;
        *) echo "  Aborted."; exit 0 ;;
    esac
    echo ""
fi

# ── Step 2: Recalibrate ───────────────────────────────────────────────────────
echo "  [2/4] Recalibrate ────────────────────────────────────────"
PYTHONPATH=. python tools/recalibrate_main_table.py \
    --input  "$INPUT" \
    --output "$OUTPUT" \
    "${RECAL_VARIANTS[@]}" \
    "${RECAL_SCENARIOS[@]}"
echo ""

# ── Step 3: Re-train cells the recalibration could not recover ────────────────
N_FLAGGED=0
if [[ -z "$SKIP_RETRAIN" ]]; then
    # Count cells marked needs_recalibration in the recalibrated JSON.
    N_FLAGGED=$(PYTHONPATH=. python - <<EOF
import json
data = json.loads(open("$OUTPUT").read())
print(sum(1 for c in data.get("cells", []) if c.get("needs_recalibration")))
EOF
)
    if [[ "$N_FLAGGED" -gt 0 ]]; then
        echo "  [3/4] Re-training $N_FLAGGED flagged cells ─────────────────"
        echo "        (scripts/run_main_table.sh --resume; patched done_keys"
        echo "         skips already-calibrated cells, re-trains flagged ones)"
        echo ""
        bash scripts/run_main_table.sh \
            --resume \
            --output "$OUTPUT" \
            --epochs "$EPOCHS" \
            --seeds  "$SEEDS"
        echo ""
    else
        echo "  [3/4] No cells flagged for re-training. Skipping."
        echo ""
    fi
fi

# ── Step 3b: Refresh recalibration metadata after re-training ─────────────────
# The recalibration metadata block written in step 2 reports n_skipped as of
# that moment; step 3 may have recovered some of those cells.  Update the
# audit trail so the JSON tells the complete story.
if [[ "$N_FLAGGED" -gt 0 && -z "$SKIP_RETRAIN" ]]; then
    PYTHONPATH=. python - <<EOF
import json
p = "$OUTPUT"
d = json.loads(open(p).read())
meta = d.get("recalibration", {})
still_flagged = sum(1 for c in d.get("cells", []) if c.get("needs_recalibration"))
recovered = meta.get("n_skipped", 0) - still_flagged
meta["n_retrained_post_recal"] = recovered
meta["n_still_flagged"] = still_flagged
d["recalibration"] = meta
open(p, "w").write(json.dumps(d, indent=2))
print(f"  Audit trail updated: {recovered} cells recovered via re-training; "
      f"{still_flagged} still flagged.")
EOF
    echo ""
fi

# ── Step 4: Re-render tables ──────────────────────────────────────────────────
if [[ -z "$SKIP_RENDER" ]]; then
    echo "  [4/4] Re-render tables ───────────────────────────────────"
    PYTHONPATH=. python tools/render_table.py \
        --table3 "$OUTPUT" \
        --output-dir results/
    echo ""
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo "  ═══════════════════════════════════════════════════════════"
echo "  ✓ Recalibration pipeline complete."
echo "    Recalibrated JSON : $OUTPUT"
[[ -n "$SKIP_RENDER" ]] || echo "    Rendered tables   : results/table3_main_results.{tex,csv,md}"
echo ""
echo "  To promote the recalibrated file to canonical:"
echo "    cp \"$OUTPUT\" \"$INPUT\""
echo "  Or to symlink (keeps both for audit):"
echo "    ln -sf \"\$(basename \"$OUTPUT\")\" \"$INPUT\""
echo "  ═══════════════════════════════════════════════════════════"
