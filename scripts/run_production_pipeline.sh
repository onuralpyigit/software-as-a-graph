#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "Starting Full Production Run (PARALLEL)"
echo "=========================================="

# Set lower thread counts for parallel PyTorch runs
export OMP_NUM_THREADS=4

echo "[1/2] Launching main table (Table 3) sweep in background..."
PYTHONPATH=. python reproduce/middleware26_main_table.py \
    --epochs 300 \
    --seeds 42 123 456 789 2024 \
    --output results/main_table.json \
    --resume \
    -v > results/main_table_run.log 2>&1 &
PID1=$!

echo "[2/2] Launching LOSO validation (Table 4) sweep in background..."
PYTHONPATH=. python reproduce/loso_all_variants.py \
    --epochs 300 \
    --seeds "42,123,456,789,2024" \
    --output results/loso_all_variants.json \
    --resume \
    -v > results/loso_run.log 2>&1 &
PID2=$!

echo "Waiting for all 2 compute sweeps to finish (PIDs: $PID1 $PID2)..."
wait $PID1
wait $PID2
echo "Compute sweeps completed!"

echo "[3/3] Generating Artifacts (Tables & Figures)..."
PYTHONPATH=. python reproduce/render_table.py \
    --table3 results/main_table.json \
    --output-dir results/ > results/render_table3.log 2>&1

PYTHONPATH=. python reproduce/loso_all_variants.py \
    --table-only \
    --output results/loso_all_variants.json > results/render_table4.log 2>&1

PYTHONPATH=. python reproduce/render_stratified_figure.py > results/render_fig4.log 2>&1

PYTHONPATH=. python reproduce/extract_attention.py > results/extract_attention.log 2>&1
PYTHONPATH=. python reproduce/render_attention_subgraph.py > results/render_fig5.log 2>&1

echo "=========================================="
echo "Production Run Completed"
echo "=========================================="
