#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "Starting Full Production Run"
echo "=========================================="

cd /home/onuralpyigit/Workspace/SoftwareAsAGraph

echo "[1/4] Running main table (Table 3)..."
PYTHONPATH=. python tools/middleware26_main_table.py \
    --epochs 300 \
    --seeds 42 123 456 789 2024 \
    --output results/main_table.json \
    -v > results/main_table_run.log 2>&1

echo "[2/4] Running LOSO validation (Table 4)..."
PYTHONPATH=. python tools/loso_all_variants.py \
    --epochs 300 \
    --seeds 42 123 456 789 2024 \
    --output results/loso_all_variants.json \
    -v > results/loso_run.log 2>&1

echo "[3/4] Running Gini Sweep (Figure 3)..."
PYTHONPATH=. python tools/qos_gini_sweep.py \
    --epochs 300 \
    -v > results/gini_run.log 2>&1

echo "[4/4] Generating Artifacts (Tables & Figures)..."
PYTHONPATH=. python tools/render_table.py \
    --table3 results/main_table.json \
    --output-dir results/ > results/render_table3.log 2>&1

PYTHONPATH=. python tools/loso_all_variants.py \
    --table-only \
    --output results/loso_all_variants.json > results/render_table4.log 2>&1

PYTHONPATH=. python tools/plot_gini_monotonicity.py \
    --input results/gini_sweep.json \
    --output results/figure3_gini.png > results/render_fig3.log 2>&1

PYTHONPATH=. python tools/render_stratified_figure.py > results/render_fig4.log 2>&1

PYTHONPATH=. python tools/extract_attention.py > results/extract_attention.log 2>&1
PYTHONPATH=. python tools/render_attention_subgraph.py > results/render_fig5.log 2>&1

echo "=========================================="
echo "Production Run Completed"
echo "=========================================="
