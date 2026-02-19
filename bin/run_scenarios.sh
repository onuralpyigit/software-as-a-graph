#!/usr/bin/env bash
# =============================================================================
# run_scenarios.sh
# Run every input/scenario_*.yaml through the full 6-stage pipeline.
#
# Usage:
#   bash bin/run_scenarios.sh [--dry-run] [--neo4j-uri bolt://localhost:7687]
#
# Outputs per scenario:  output/<scenario_name>_results/
# Summary CSV:           output/scenario_summary.csv
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DRY_RUN=""
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASS="password"
LAYERS="app,infra,mw"

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)        DRY_RUN="--dry-run" ; shift ;;
    --neo4j-uri)      NEO4J_URI="$2"      ; shift 2 ;;
    --neo4j-user)     NEO4J_USER="$2"     ; shift 2 ;;
    --neo4j-password) NEO4J_PASS="$2"     ; shift 2 ;;
    --layers)         LAYERS="$2"         ; shift 2 ;;
    *) echo "Unknown argument: $1" >&2 ; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INPUT_DIR="$PROJECT_ROOT/input"
OUTPUT_DIR="$PROJECT_ROOT/output"
PYTHON="${PROJECT_ROOT}/software_system_env/bin/python"

# Fallback to system python if venv doesn't exist
if [[ ! -f "$PYTHON" ]]; then
  PYTHON="$(which python3 || which python)"
fi

# ---------------------------------------------------------------------------
# CSV header
# ---------------------------------------------------------------------------
SUMMARY_CSV="$OUTPUT_DIR/scenario_summary.csv"
mkdir -p "$OUTPUT_DIR"
echo "scenario,status,duration_s,generate_s,import_s,analyze_s,simulate_s,validate_s,visualize_s" \
  > "$SUMMARY_CSV"

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
CYAN='\033[96m'; GREEN='\033[92m'; RED='\033[91m'; YELLOW='\033[93m'; RESET='\033[0m'; BOLD='\033[1m'
hdr()  { echo -e "\n${CYAN}${BOLD}==============================${RESET}"; echo -e "${CYAN}${BOLD} $*${RESET}"; echo -e "${CYAN}${BOLD}==============================${RESET}"; }
ok()   { echo -e "  ${GREEN}✓ $*${RESET}"; }
err()  { echo -e "  ${RED}✗ $*${RESET}"; }
warn() { echo -e "  ${YELLOW}⚠ $*${RESET}"; }

# ---------------------------------------------------------------------------
# Run order: 08 (tiny) first as smoke test, then 01-07
# ---------------------------------------------------------------------------
SCENARIO_FILES=(
  "$INPUT_DIR/scenario_08_tiny_regression.yaml"
  "$INPUT_DIR/scenario_01_autonomous_vehicle.yaml"
  "$INPUT_DIR/scenario_02_iot_smart_city.yaml"
  "$INPUT_DIR/scenario_03_financial_trading.yaml"
  "$INPUT_DIR/scenario_04_healthcare.yaml"
  "$INPUT_DIR/scenario_05_hub_and_spoke.yaml"
  "$INPUT_DIR/scenario_06_microservices.yaml"
  "$INPUT_DIR/scenario_07_enterprise_xlarge.yaml"
)

PASS=0; FAIL=0

hdr "Software-as-a-Graph — Scenario Dataset Pipeline"
echo "  Python : $PYTHON"
echo "  Neo4j  : $NEO4J_URI"
echo "  Layers : $LAYERS"
echo "  Output : $OUTPUT_DIR"
[[ -n "$DRY_RUN" ]] && warn "DRY RUN — no commands will be executed"

for CFG in "${SCENARIO_FILES[@]}"; do
  NAME=$(basename "$CFG" .yaml)
  DATA="$OUTPUT_DIR/${NAME}.json"
  RESULT_DIR="$OUTPUT_DIR/${NAME}_results"
  LOG_FILE="$RESULT_DIR/pipeline.log"

  hdr "Scenario: $NAME"

  mkdir -p "$RESULT_DIR"

  T_START=$SECONDS

  # ---- run.py does everything: generate → import → … → visualize ----------
  CMD=(
    "$PYTHON" "$PROJECT_ROOT/bin/run.py"
    --all
    --config  "$CFG"
    --input   "$DATA"
    --output-dir "$RESULT_DIR"
    --layer   "$LAYERS"
    --clean
    --uri     "$NEO4J_URI"
    --user    "$NEO4J_USER"
    --password "$NEO4J_PASS"
  )
  [[ -n "$DRY_RUN" ]] && CMD+=(--dry-run)

  echo "  CMD: ${CMD[*]}"

  STATUS="PASS"
  if [[ -n "$DRY_RUN" ]]; then
    "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
  else
    if "${CMD[@]}" 2>&1 | tee "$LOG_FILE"; then
      ok "Pipeline succeeded for $NAME"
      PASS=$((PASS + 1))
    else
      err "Pipeline FAILED for $NAME — see $LOG_FILE"
      STATUS="FAIL"
      FAIL=$((FAIL + 1))
    fi
  fi

  T_END=$SECONDS
  DURATION=$((T_END - T_START))

  # Extract per-stage timings from log (best-effort grep)
  extract_time() {
    grep -oP "(?<=$1[[:space:]][[:space:]]).*?(?=pass|FAIL)" "$LOG_FILE" 2>/dev/null \
      | grep -oP '[0-9]+\.[0-9]+' | tail -1 || echo "—"
  }

  GEN_T=$(grep -A1 "Stage.*Generation" "$LOG_FILE" 2>/dev/null | grep -oP '[0-9]+\.[0-9]+s' | head -1 || echo "—")
  IMP_T=$(grep -A1 "Stage.*Import"      "$LOG_FILE" 2>/dev/null | grep -oP '[0-9]+\.[0-9]+s' | head -1 || echo "—")
  ANA_T=$(grep -A1 "Stage.*Analysis"    "$LOG_FILE" 2>/dev/null | grep -oP '[0-9]+\.[0-9]+s' | head -1 || echo "—")
  SIM_T=$(grep -A1 "Stage.*Simulation"  "$LOG_FILE" 2>/dev/null | grep -oP '[0-9]+\.[0-9]+s' | head -1 || echo "—")
  VAL_T=$(grep -A1 "Stage.*Validation"  "$LOG_FILE" 2>/dev/null | grep -oP '[0-9]+\.[0-9]+s' | head -1 || echo "—")
  VIZ_T=$(grep -A1 "Stage.*Visual"      "$LOG_FILE" 2>/dev/null | grep -oP '[0-9]+\.[0-9]+s' | head -1 || echo "—")

  echo "$NAME,$STATUS,$DURATION,$GEN_T,$IMP_T,$ANA_T,$SIM_T,$VAL_T,$VIZ_T" >> "$SUMMARY_CSV"

  echo "  Duration: ${DURATION}s"
done

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
hdr "Summary"
echo -e "  ${GREEN}PASS: $PASS${RESET}  ${RED}FAIL: $FAIL${RESET}"
echo "  Summary CSV: $SUMMARY_CSV"
echo ""
if [[ -f "$SUMMARY_CSV" ]]; then
  column -t -s, "$SUMMARY_CSV"
fi
echo ""
