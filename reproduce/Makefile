# reproduce/Makefile
# ===================
# One-command reproducibility bundle for Middleware 2026 submission.
#
# Usage:
#   make setup           — install Python deps
#   make scenarios       — regenerate the scenario corpus + verify its manifest
#   make cache           — rebuild output/loso_cache from the corpus (needs Neo4j)
#   make block0          — W1 QoS pipeline audit (go/no-go gate)
#   make table3          — Table 3: 8×4×5 training matrix
#   make table4          — Table 4: LOSO × 4 variants
#   make kfold           — Per-domain k-fold × 5 variants (opt-in; primary validation protocol)
#   make convergent-validity — Three-oracle rank agreement (opt-in; expensive)
#   make figure4         — Stratified per-node-type ρ (old Middleware numbering)
#   make figure5         — ATM attention subgraph (old Middleware numbering)
#   make jss-figures     — all 4 figures for the JSS submission (docs/research/jss/latex/figures/)
#   make all             — full end-to-end pipeline
#   make clean           — remove generated outputs (keeps checkpoints)
#   make clean-all       — remove everything including checkpoints

PYTHON     ?= python
PYTHONPATH := .
export PYTHONPATH

# loso_all_variants.py / kfold_all_variants.py take one comma-separated string;
# main_table.py takes nargs="+" ints. Keep both spellings of the same seed set.
comma      := ,
empty      :=
space      := $(empty) $(empty)
SEEDS      ?= 42,123,456,789,2024
SEEDS_LIST  = $(subst $(comma),$(space),$(SEEDS))
EPOCHS     ?= 300
EPOCHS_FAST ?= 50      # for quick smoke-test
SCENARIOS_DIR ?= data/scenarios
RESULTS_DIR   ?= results
OUTPUT_DIR    ?= output

.PHONY: setup scenarios cache block0 table3 table4 kfold convergent-validity \
        topic-weight-sensitivity \
        figure4 figure5 jss-fig1 jss-fig2 jss-fig3 jss-fig4 \
        jss-figures all smoke-test clean clean-all

# ─────────────────────────────────────────────────────────────────────────────
# 0. Environment
# ─────────────────────────────────────────────────────────────────────────────

setup:
	pip install -e ".[all]"
	@echo "✓ Dependencies installed."

# ─────────────────────────────────────────────────────────────────────────────
# 1. Scenario generation (idempotent — skips if files exist)
# ─────────────────────────────────────────────────────────────────────────────

scenarios:
	@mkdir -p $(SCENARIOS_DIR) $(OUTPUT_DIR)/loso_cache
	$(PYTHON) cli/generate_graph.py batch \
	    --input-dir $(SCENARIOS_DIR) --output-dir $(SCENARIOS_DIR) --force
	$(PYTHON) scripts/write_scenario_manifest.py --check
	@echo "✓ Scenarios ready (regenerated from data/scenarios/scenario_*.yaml)."

# Rebuild the ground-truth caches the evaluation harnesses read. Deletes first:
# every step in populate_loso_cache.sh is guarded by "if file missing", so a
# partial cache would otherwise survive and keep feeding stale labels.
# Requires a live Neo4j (import/analyze/predict run against it).
cache: scenarios
	rm -rf $(OUTPUT_DIR)/loso_cache
	bash scripts/populate_loso_cache.sh
	@echo "✓ LOSO caches rebuilt from $(SCENARIOS_DIR)."

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
	$(PYTHON) reproduce/main_table.py \
	    --output $(RESULTS_DIR)/main_table.json \
	    --seeds $(SEEDS_LIST) \
	    --epochs $(EPOCHS) \
	    --resume
	$(PYTHON) reproduce/render_table.py \
	    --table3 $(RESULTS_DIR)/main_table.json \
	    --output-dir $(RESULTS_DIR)
	@echo "✓ Table 3 ready: $(RESULTS_DIR)/table3_main_results.tex"

# ─────────────────────────────────────────────────────────────────────────────
# Block E: Table 4 — LOSO × 4 variants
# ─────────────────────────────────────────────────────────────────────────────

table4:
	$(PYTHON) reproduce/loso_all_variants.py \
	    --cache-dir $(OUTPUT_DIR)/loso_cache \
	    --epochs $(EPOCHS) \
	    --output $(RESULTS_DIR)/loso_all_variants.json \
	    --resume
	$(PYTHON) reproduce/render_table.py \
	    --table4 $(RESULTS_DIR)/loso_all_variants.json \
	    --output-dir $(RESULTS_DIR)
	@echo "✓ Table 4 ready: $(RESULTS_DIR)/table4_loso_results.tex"

# ─────────────────────────────────────────────────────────────────────────────
# Per-domain k-fold — primary in-domain validation protocol (5 variants).
# Opt-in (not part of `all`): trains ~5× more model fits than table4/LOSO for
# the same seed count (scenarios × k folds × seeds vs. scenarios × seeds).
# ─────────────────────────────────────────────────────────────────────────────

kfold:
	$(PYTHON) reproduce/kfold_all_variants.py \
	    --cache-dir $(OUTPUT_DIR)/loso_cache \
	    --epochs $(EPOCHS) \
	    --output $(RESULTS_DIR)/kfold_all_variants.json \
	    --resume
	$(PYTHON) reproduce/render_table.py \
	    --table-kfold $(RESULTS_DIR)/kfold_all_variants.json \
	    --output-dir $(RESULTS_DIR)
	@echo "✓ K-fold ready: $(RESULTS_DIR)/table4_kfold_results.tex"

# ─────────────────────────────────────────────────────────────────────────────
# Three-oracle rank agreement — I*, I_comp, I_dyn (construct-validity check).
# Opt-in (not part of `all`): I_dyn needs one discrete-event simulation run per
# candidate component, so cost scales with corpus size rather than with epochs.
# Use --max-candidates to bound it on the larger scenarios.
# ─────────────────────────────────────────────────────────────────────────────

convergent-validity:
	$(PYTHON) reproduce/convergent_validity.py \
	    --seeds $(SEEDS_LIST) \
	    --output $(RESULTS_DIR)/convergent_validity.json \
	    --verbose
	@echo "✓ Oracle agreement ready: $(RESULTS_DIR)/convergent_validity.json"

# ─────────────────────────────────────────────────────────────────────────────
# Topic-weight (β, α, ψ) sensitivity — the DECLARED outer split of w(t).
# Opt-in (not part of `all`): re-derives the graph and re-runs the closed-form
# scorers once per grid point per scenario. Add --skip-downstream for the
# w(t)-ordering level alone, which is near-instant.
# ─────────────────────────────────────────────────────────────────────────────

topic-weight-sensitivity:
	$(PYTHON) reproduce/topic_weight_sensitivity.py \
	    --output $(RESULTS_DIR)/topic_weight_sensitivity.json
	@echo "✓ Topic-weight sensitivity ready: $(RESULTS_DIR)/topic_weight_sensitivity.json"

# ─────────────────────────────────────────────────────────────────────────────
# Block F: Figure 4 — stratified per-node-type ρ
# ─────────────────────────────────────────────────────────────────────────────

figure4:
	$(PYTHON) reproduce/render_stratified_figure.py \
	    --source auto \
	    --output $(RESULTS_DIR)/figure4_stratified_rho
	@echo "✓ Figure 4 ready: $(RESULTS_DIR)/figure4_stratified_rho.[png|pdf]"

# ─────────────────────────────────────────────────────────────────────────────
# Block G: Figure 5 — ATM attention subgraph
# ─────────────────────────────────────────────────────────────────────────────

figure5:
	$(PYTHON) reproduce/extract_attention.py \
	    --scenario atm_system \
	    --output-dir $(OUTPUT_DIR)/atm_case_study
	$(PYTHON) reproduce/render_attention_subgraph.py \
	    --input $(OUTPUT_DIR)/atm_case_study/attention_weights.json \
	    --output $(OUTPUT_DIR)/atm_case_study/attention_subgraph
	@echo "✓ Figure 5 ready: $(OUTPUT_DIR)/atm_case_study/attention_subgraph.[png|pdf]"

# ─────────────────────────────────────────────────────────────────────────────
# JSS submission figures (docs/research/jss/latex/figures/Figure_1..4).
#
# Since the manuscript was condensed to four figures, printed number, filename
# and draft.md's own caption label all agree -- the old 1,2,6,3,4,5 mismatch is
# gone, because the two floats that caused it (pooled-vs-per-type ρ and the
# propagation-threshold sweep) are no longer in the paper.
#
# The two retired generators are kept, not deleted: their analyses still live
# in docs/research/thesis/material/, and either figure is one command from
# coming back. They are simply not part of `jss-figures` any more.
#   reproduce/render_pooled_vs_pertype_figure.py  (was Figure_4)
#   reproduce/render_threshold_figure.py          (was Figure_6)
# ─────────────────────────────────────────────────────────────────────────────

jss-fig1:
	dot -Tpdf docs/research/jss/latex/figures/src/figure1_pipeline.dot \
	    -o docs/research/jss/latex/figures/Figure_1.pdf
	dot -Tpng -Gdpi=300 docs/research/jss/latex/figures/src/figure1_pipeline.dot \
	    -o docs/research/jss/latex/figures/Figure_1.png
	@echo "✓ Figure 1 (pipeline) ready."

jss-fig2:
	dot -Tpdf docs/research/jss/latex/figures/src/figure2_running_example.dot \
	    -o docs/research/jss/latex/figures/Figure_2.pdf
	dot -Tpng -Gdpi=300 docs/research/jss/latex/figures/src/figure2_running_example.dot \
	    -o docs/research/jss/latex/figures/Figure_2.png
	@echo "✓ Figure 2 (running example) ready."

jss-fig3:
	$(PYTHON) reproduce/extract_attention.py \
	    --scenario atm_system \
	    --output-dir $(OUTPUT_DIR)/atm_case_study
	$(PYTHON) reproduce/render_attention_subgraph.py \
	    --input $(OUTPUT_DIR)/atm_case_study/attention_weights.json \
	    --output docs/research/jss/latex/figures/Figure_3
	@echo "✓ Figure 3 (HGT attention subgraph) ready."

jss-fig4:
	$(PYTHON) reproduce/render_shrinkage_figure.py
	@echo "✓ Figure 4 (AHP shrinkage sensitivity) ready."

jss-figures: jss-fig1 jss-fig2 jss-fig3 jss-fig4
	@echo "✓ All 4 JSS figures ready in docs/research/jss/latex/figures/."

# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

all: setup scenarios block0 table3 table4 figure4 figure5
	@echo ""
	@echo "══════════════════════════════════════════════════════"
	@echo "  Q-HGL Middleware 2026 — Full Pipeline Complete"
	@echo "══════════════════════════════════════════════════════"
	@echo "  Table 3  : $(RESULTS_DIR)/table3_main_results.tex"
	@echo "  Table 4  : $(RESULTS_DIR)/table4_loso_results.tex"
	@echo "  Figure 4 : $(RESULTS_DIR)/figure4_stratified_rho.pdf"
	@echo "  Figure 5 : $(OUTPUT_DIR)/atm_case_study/attention_subgraph.pdf"
	@echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Fast smoke-test (50 epochs, 2 seeds, 3 scenarios only)
# ─────────────────────────────────────────────────────────────────────────────

smoke-test: block0
	$(PYTHON) reproduce/main_table.py \
	    --scenarios atm_system av_system iot_smart_city_system \
	    --seeds 42 123 \
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
	rm -rf $(OUTPUT_DIR)/gnn_checkpoints $(OUTPUT_DIR)/loso $(OUTPUT_DIR)/kfold \
	       $(OUTPUT_DIR)/atm_case_study
	@echo "✓ Full clean done."
