#!/usr/bin/env bash
set -eo pipefail

echo "=========================================="
echo "STEP 0: Generate statistical system topology"
echo "=========================================="
PYTHONPATH=. python cli/generate_graph.py \
  --config data/scenarios/scenario_10_atm_system.yaml \
  --output data/system.json

echo "=========================================="
echo "STEP 1: Import structure and calculate weights"
echo "=========================================="
PYTHONPATH=. python cli/import_graph.py \
  --input data/system.json \
  --clear

echo "=========================================="
echo "STEP 2: Execute design-time structural analysis metrics loop"
echo "=========================================="
PYTHONPATH=. python cli/analyze_graph.py \
  --layer system

echo "=========================================="
echo "STEP 4: Run fault-injection cascades"
echo "=========================================="
mkdir -p evaluation
PYTHONPATH=. python cli/simulate_graph.py \
  --layer system \
  --output evaluation/simulation_output.json

echo "=========================================="
echo "STEP 3: Train inductive HGT model"
echo "=========================================="
mkdir -p output/gnn_checkpoints/best_model
PYTHONPATH=. python cli/train_graph.py \
  --layer system \
  --seeds 42 123 456 \
  --output output/gnn_checkpoints/best_model

echo "=========================================="
echo "STEP 5: Perform cross-validation and generate metrics audit"
echo "=========================================="
PYTHONPATH=. python cli/validate_graph.py \
  --input data/system.json \
  --gnn-model output/gnn_checkpoints/best_model

echo "=========================================="
echo "STEP 6: Spin up Next.js visualization layer to host telemetry dashboard"
echo "=========================================="
cd smart
npm run build
