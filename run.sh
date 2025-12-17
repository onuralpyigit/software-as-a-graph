#!/bin/bash

# Step 1: Generate
python generate_graph.py --scale medium --scenario iot --output input/dataset.json

# Step 2: Import
python import_graph.py --input input/dataset.json --clear --analytics

# Step 3: Analyze
python analyze_graph.py --output-dir output/ --format json html csv --alpha 0.7 --beta 0.3 --gamma 0.0

# Step 4: Simulate
python simulate_graph.py --exhaustive --output-dir output/ --format json html csv

# Step 5: Validate
python validate_graph.py --output-dir output/ --format json html csv

# Step 6: Visualize
python visualize_graph.py --output-dir output/ --full