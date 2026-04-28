#!/usr/bin/env bash
# =============================================================================
# run_scenarios.sh — Software-as-a-Graph End-to-End Scenario Runner
#
# Usage:
#   bash cli/run_scenarios.sh [OPTIONS]
#
# Options:
#   --dry-run            Print commands but do not execute them
#   --neo4j-uri URI      Neo4j Bolt URI          (default: bolt://localhost:7687)
#   --neo4j-user USER    Neo4j username           (default: neo4j)
#   --neo4j-password PWD Neo4j password           (default: password)
#   --layers LAYERS      Comma-separated layers   (default: app,infra,mw,system)
#   --smoke              Run only the tiny regression scenario (quick smoke test)
#   --scenario GLOB      Run only matching scenario files (glob pattern)
#
# Per-scenario output:  output/<scenario_name>_results/
# Summary CSV:          output/scenario_summary.csv
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DRY_RUN=""
NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASS="${NEO4J_PASSWORD:-password}"
LAYERS="app,infra,mw,system"
SMOKE_ONLY=""
SCENARIO_PATTERN=""

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)        DRY_RUN="--dry-run"  ; shift ;;
    --neo4j-uri)      NEO4J_URI="$2"       ; shift 2 ;;
    --neo4j-user)     NEO4J_USER="$2"      ; shift 2 ;;
    --neo4j-password) NEO4J_PASS="$2"      ; shift 2 ;;
    --layers)         LAYERS="$2"          ; shift 2 ;;
    --smoke)          SMOKE_ONLY=1         ; shift ;;
    --scenario)       SCENARIO_PATTERN="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INPUT_DIR="$PROJECT_ROOT/input"
OUTPUT_DIR="$PROJECT_ROOT/output"
GNN_DIR="$OUTPUT_DIR/gnn_checkpoints"

# Prefer venv python, fall back to system python3
PYTHON="${PROJECT_ROOT}/software_system_env/bin/python"
[[ -f "$PYTHON" ]] || PYTHON="$(command -v python3)"

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
CYAN='\033[96m'; GREEN='\033[92m'; RED='\033[91m'; YELLOW='\033[93m'
RESET='\033[0m'; BOLD='\033[1m'
hdr()  { printf "\n${CYAN}${BOLD}==============================\n %s\n==============================${RESET}\n" "$*"; }
ok()   { printf "  ${GREEN}✓ %s${RESET}\n" "$*"; }
err()  { printf "  ${RED}✗ %s${RESET}\n" "$*"; }
warn() { printf "  ${YELLOW}⚠ %s${RESET}\n" "$*"; }

# ---------------------------------------------------------------------------
# Build ordered scenario list
#   - scenario_08 (tiny) runs first as a smoke test
#   - remaining scenarios run in numeric order
# ---------------------------------------------------------------------------
build_scenario_list() {
  local smoke="$INPUT_DIR/scenario_08_tiny_regression.yaml"
  local rest=()

  # Collect all scenarios, sorted, excluding the smoke test
  while IFS= read -r f; do
    [[ "$f" == "$smoke" ]] && continue
    rest+=("$f")
  done < <(find "$INPUT_DIR" -maxdepth 1 -name 'scenario_*.yaml' | sort)

  # Apply --smoke or --scenario filter
  if [[ -n "$SMOKE_ONLY" ]]; then
    echo "$smoke"
    return
  fi

  if [[ -n "$SCENARIO_PATTERN" ]]; then
    local matched=()
    [[ "$smoke" == *"$SCENARIO_PATTERN"* ]] && matched+=("$smoke")
    for f in "${rest[@]}"; do
      [[ "$f" == *"$SCENARIO_PATTERN"* ]] && matched+=("$f")
    done
    printf '%s\n' "${matched[@]}"
    return
  fi

  # Default: smoke first, then the rest
  echo "$smoke"
  printf '%s\n' "${rest[@]}"
}

# ---------------------------------------------------------------------------
# Extract a stage timing from a log file (strips ANSI, matches summary table)
# Usage: get_stage_time <log_file> <stage_name>
# Returns the numeric seconds value, or "—" if not found.
# ---------------------------------------------------------------------------
get_stage_time() {
  local log="$1" stage="$2"
  local val
  val=$(sed 's/\x1b\[[0-9;]*m//g' "$log" \
    | grep -iE "^[[:space:]]*[0-9]+[[:space:]]+${stage}" \
    | awk '{print $3}' | sed 's/s//' | tail -1)
  echo "${val:-—}"
}

# ---------------------------------------------------------------------------
# Run a single scenario through run.py
# Usage: run_scenario <config_yaml>
# Returns 0 on success, 1 on failure (or dry-run, always 0).
# ---------------------------------------------------------------------------
run_scenario() {
  local cfg="$1"
  local name result_dir log_file

  name="$(basename "$cfg" .yaml)"
  result_dir="$OUTPUT_DIR/${name}_results"
  log_file="$result_dir/pipeline.log"

  hdr "Scenario: $name"
  mkdir -p "$result_dir"

  # Build command — run.py's --all flag covers the full pipeline
  local cmd=(
    "$PYTHON" "$PROJECT_ROOT/cli/run.py"
    --all
    --config    "$cfg"
    --input     "$OUTPUT_DIR/${name}.json"
    --output-dir "$result_dir"
    --layer     "$LAYERS"
    --clean
    --uri       "$NEO4J_URI"
    --user      "$NEO4J_USER"
    --password  "$NEO4J_PASS"
    --gnn-model "$GNN_DIR"
  )

  echo "  CMD: ${cmd[*]}"

  if [[ -n "$DRY_RUN" ]]; then
    warn "Dry run — skipping execution"
    return 0
  fi

  local t_start=$SECONDS
  if "${cmd[@]}" 2>&1 | tee "$log_file"; then
    ok "Pipeline succeeded for $name"
    local duration=$(( SECONDS - t_start ))

    local gen_t  && gen_t=$(get_stage_time  "$log_file" "Generation")
    local imp_t  && imp_t=$(get_stage_time  "$log_file" "Import")
    local ana_t  && ana_t=$(get_stage_time  "$log_file" "Analysis")
    local pred_t && pred_t=$(get_stage_time "$log_file" "Prediction")
    local sim_t  && sim_t=$(get_stage_time  "$log_file" "Simulation")
    local val_t  && val_t=$(get_stage_time  "$log_file" "Validation")
    local viz_t  && viz_t=$(get_stage_time  "$log_file" "Visual")

    echo "$name,PASS,$duration,$gen_t,$imp_t,$ana_t,$pred_t,$sim_t,$val_t,$viz_t" >> "$SUMMARY_CSV"
    echo "  Duration: ${duration}s"
    return 0
  else
    err "Pipeline FAILED for $name — see $log_file"
    echo "$name,FAIL,,,,,,," >> "$SUMMARY_CSV"
    return 1
  fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"
SUMMARY_CSV="$OUTPUT_DIR/scenario_summary.csv"
echo "scenario,status,duration_s,generate_s,import_s,analyze_s,predict_s,simulate_s,validate_s,visualize_s" > "$SUMMARY_CSV"

hdr "Software-as-a-Graph — Scenario Dataset Pipeline"
echo "  Python : $PYTHON"
echo "  Neo4j  : $NEO4J_URI"
echo "  Layers : $LAYERS"
echo "  Output : $OUTPUT_DIR"
[[ -n "$DRY_RUN" ]] && warn "DRY RUN — commands will be printed but not executed"

PASS=0; FAIL=0

while IFS= read -r scenario; do
  [[ -z "$scenario" ]] && continue
  [[ -f "$scenario" ]] || { warn "File not found, skipping: $scenario"; continue; }

  if run_scenario "$scenario"; then
    PASS=$(( PASS + 1 ))
  else
    FAIL=$(( FAIL + 1 ))
  fi
done < <(build_scenario_list)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
hdr "Summary"
printf "  ${GREEN}PASS: %d${RESET}  ${RED}FAIL: %d${RESET}\n" "$PASS" "$FAIL"
echo "  Summary CSV: $SUMMARY_CSV"
echo ""
if [[ -f "$SUMMARY_CSV" ]]; then
  column -t -s, "$SUMMARY_CSV"
fi
echo ""

[[ "$FAIL" -eq 0 ]] || exit 1
