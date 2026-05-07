# reproduce/Makefile
# ===================
# One-command reproducibility bundle for Middleware 2026 submission.
#
# Usage:
#   make setup           — install Python deps
#   make scenarios       — generate the 8 reference scenario JSONs
#   make block0          — W1 QoS pipeline audit (go/no-go gate)
#   make table3          — Table 3: 8×4×5 training matrix
#   make table4          — Table 4: LOSO × 4 variants
#   make figure3         — Figure 3: Gini monotonicity sweep
#   make figure4         — Figure 4: Stratified per-node-type ρ
#   make figure5         — Figure 5: ATM attention subgraph
#   make all             — full end-to-end pipeline
#   make clean           — remove generated outputs (keeps checkpoints)
#   make clean-all       — remove everything including checkpoints

PYTHON     ?= python
PYTHONPATH := .
export PYTHONPATH

SEEDS      ?= 42,123,456,789,2024
EPOCHS     ?= 300
EPOCHS_FAST ?= 50      # for quick smoke-test
SCENARIOS_DIR ?= data/scenarios
RESULTS_DIR   ?= results
OUTPUT_DIR    ?= output

.PHONY: setup scenarios block0 table3 table4 figure3 figure4 figure5 \
        all smoke-test clean clean-all

# ─────────────────────────────────────────────────────────────────────────────
# 0. Environment
# ─────────────────────────────────────────────────────────────────────────────

setup:
	pip install -r requirements.txt
	@echo "✓ Dependencies installed."

# ─────────────────────────────────────────────────────────────────────────────
# 1. Scenario generation (idempotent — skips if files exist)
# ─────────────────────────────────────────────────────────────────────────────

scenarios:
	@mkdir -p $(SCENARIOS_DIR) $(OUTPUT_DIR)/loso_cache
	@for scenario in atm_system av_system iot_smart_city_system \
	                 financial_trading_system healthcare_system \
	                 hub_and_spoke_system microservices_system enterprise_system; do \
	    out=$(SCENARIOS_DIR)/$$scenario.json; \
	    if [ ! -f $$out ]; then \
	        echo "Generating $$scenario ..."; \
	        $(PYTHON) cli/generate_graph.py --config data/scenario_configs/$$scenario.yaml \
	                  --output $$out 2>/dev/null || echo "  SKIP (config not found)"; \
	    else \
	        echo "EXISTS: $$out"; \
	    fi; \
	done
	@echo "✓ Scenarios ready."

# ─────────────────────────────────────────────────────────────────────────────
# Block 0: W1 QoS Pipeline Audit (go/no-go gate — MUST pass before Table 3)
# ─────────────────────────────────────────────────────────────────────────────

block0:
	$(PYTHON) -m pytest tests/test_qos_pipeline_audit.py tests/test_baselines.py \
	          -v --tb=short -q
	@echo "✓ Block 0 PASSED — W1 audit gate cleared."

# ─────────────────────────────────────────────────────────────────────────────
# Block C: Table 3 — 8 × 4 variants × 5 seeds
# ─────────────────────────────────────────────────────────────────────────────

table3: block0
	$(PYTHON) tools/middleware26_main_table.py \
	    --scenarios-dir $(SCENARIOS_DIR) \
	    --output $(RESULTS_DIR)/main_table.json \
	    --seeds $(SEEDS) \
	    --epochs $(EPOCHS) \
	    --resume
	$(PYTHON) tools/render_table.py \
	    --table3 $(RESULTS_DIR)/main_table.json \
	    --output-dir $(RESULTS_DIR)
	@echo "✓ Table 3 ready: $(RESULTS_DIR)/table3_main_results.tex"

# ─────────────────────────────────────────────────────────────────────────────
# Block D: Figure 3 — Gini monotonicity sweep
# ─────────────────────────────────────────────────────────────────────────────

figure3: scenarios
	$(PYTHON) tools/qos_gini_sweep.py \
	    --epochs $(EPOCHS) \
	    --output $(RESULTS_DIR)/gini_sweep.json \
	    --resume
	$(PYTHON) tools/plot_gini_monotonicity.py \
	    --input $(RESULTS_DIR)/gini_sweep.json \
	    --output $(RESULTS_DIR)/figure3_gini_monotonicity
	@echo "✓ Figure 3 ready: $(RESULTS_DIR)/figure3_gini_monotonicity.[png|pdf]"

# ─────────────────────────────────────────────────────────────────────────────
# Block E: Table 4 — LOSO × 4 variants
# ─────────────────────────────────────────────────────────────────────────────

table4:
	$(PYTHON) tools/loso_all_variants.py \
	    --cache-dir $(OUTPUT_DIR)/loso_cache \
	    --epochs $(EPOCHS) \
	    --output $(RESULTS_DIR)/loso_all_variants.json \
	    --resume
	$(PYTHON) tools/render_table.py \
	    --table4 $(RESULTS_DIR)/loso_all_variants.json \
	    --output-dir $(RESULTS_DIR)
	@echo "✓ Table 4 ready: $(RESULTS_DIR)/table4_loso_results.tex"

# ─────────────────────────────────────────────────────────────────────────────
# Block F: Figure 4 — stratified per-node-type ρ
# ─────────────────────────────────────────────────────────────────────────────

figure4:
	$(PYTHON) tools/render_stratified_figure.py \
	    --source auto \
	    --output $(RESULTS_DIR)/figure4_stratified_rho
	@echo "✓ Figure 4 ready: $(RESULTS_DIR)/figure4_stratified_rho.[png|pdf]"

# ─────────────────────────────────────────────────────────────────────────────
# Block G: Figure 5 — ATM attention subgraph
# ─────────────────────────────────────────────────────────────────────────────

figure5:
	$(PYTHON) tools/extract_attention.py \
	    --scenario atm_system \
	    --output-dir $(OUTPUT_DIR)/atm_case_study
	$(PYTHON) tools/render_attention_subgraph.py \
	    --input $(OUTPUT_DIR)/atm_case_study/attention_weights.json \
	    --output $(OUTPUT_DIR)/atm_case_study/attention_subgraph
	@echo "✓ Figure 5 ready: $(OUTPUT_DIR)/atm_case_study/attention_subgraph.[png|pdf]"

# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

all: setup scenarios block0 table3 table4 figure3 figure4 figure5
	@echo ""
	@echo "══════════════════════════════════════════════════════"
	@echo "  Q-HGL Middleware 2026 — Full Pipeline Complete"
	@echo "══════════════════════════════════════════════════════"
	@echo "  Table 3  : $(RESULTS_DIR)/table3_main_results.tex"
	@echo "  Table 4  : $(RESULTS_DIR)/table4_loso_results.tex"
	@echo "  Figure 3 : $(RESULTS_DIR)/figure3_gini_monotonicity.pdf"
	@echo "  Figure 4 : $(RESULTS_DIR)/figure4_stratified_rho.pdf"
	@echo "  Figure 5 : $(OUTPUT_DIR)/atm_case_study/attention_subgraph.pdf"
	@echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Fast smoke-test (50 epochs, 2 seeds, 3 scenarios only)
# ─────────────────────────────────────────────────────────────────────────────

smoke-test: block0
	$(PYTHON) tools/middleware26_main_table.py \
	    --scenarios atm_system av_system iot_smart_city_system \
	    --seeds 42,123 \
	    --epochs $(EPOCHS_FAST) \
	    --output $(RESULTS_DIR)/smoke_main_table.json \
	    --resume
	@echo "✓ Smoke-test complete."

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────

clean:
	rm -rf $(RESULTS_DIR)/*.json $(RESULTS_DIR)/*.tex \
	       $(RESULTS_DIR)/*.csv $(RESULTS_DIR)/*.md \
	       $(RESULTS_DIR)/*.png $(RESULTS_DIR)/*.pdf
	@echo "✓ Results cleaned."

clean-all: clean
	rm -rf $(OUTPUT_DIR)/gnn_checkpoints $(OUTPUT_DIR)/loso \
	       $(OUTPUT_DIR)/atm_case_study data/scenarios/gini_variants
	@echo "✓ Full clean done."
