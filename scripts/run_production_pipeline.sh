#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "Starting Full Production Run (PARALLEL)"
echo "=========================================="

cd /home/onuralpyigit/Workspace/SoftwareAsAGraph

# Set lower thread counts for parallel PyTorch runs
export OMP_NUM_THREADS=4

echo "[1/3] Launching main table (Table 3) sweep in background..."
PYTHONPATH=. python tools/middleware26_main_table.py \
    --epochs 300 \
    --seeds 42 123 456 789 2024 \
    --output results/main_table.json \
    --resume \
    -v > results/main_table_run.log 2>&1 &
PID1=$!

echo "[2/3] Launching LOSO validation (Table 4) sweep in background..."
PYTHONPATH=. python tools/loso_all_variants.py \
    --epochs 300 \
    --seeds "42,123,456,789,2024" \
    --output results/loso_all_variants.json \
    --resume \
    -v > results/loso_run.log 2>&1 &
PID2=$!

echo "[3/3] Launching Gini Sweep (Figure 3) in background..."
PYTHONPATH=. python tools/qos_gini_sweep.py \
    --epochs 300 \
    --resume \
    -v > results/gini_run.log 2>&1 &
PID3=$!

echo "Waiting for all 3 compute sweeps to finish (PIDs: $PID1 $PID2 $PID3)..."
wait $PID1
wait $PID2
wait $PID3
echo "Compute sweeps completed!"

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
