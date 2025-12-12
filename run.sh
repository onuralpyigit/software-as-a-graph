#!/bin/bash

# Step 1: Generate
python generate_graph.py --scale small --scenario iot --output input/system.json

# Step 2: Import
python import_graph.py --input input/system.json --clear

# Step 3: Analyze
python analyze_graph.py --input input/system.json --export-json output/analysis.json

# Step 4: Simulate
python simulate_graph.py --input input/system.json --campaign --export-json output/simulation.json

# Step 5: Visualize
python visualize_graph.py --input input/system.json --output output/dashboard.html --dashboard --analysis output/analysis.json