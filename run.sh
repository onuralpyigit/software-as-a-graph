#!/bin/bash

# Step 1: Generate
python generate_graph.py --scale medium --scenario smart_city --output input/dataset.json

# Step 2: Import
python import_graph.py --input input/dataset.json --clear

# Step 3: Analyze
python analyze_graph.py --neo4j --output-dir output/ --format json html csv

# Step 4: Simulate
python simulate_graph.py --neo4j --exhaustive --output-dir output/ --format json html csv

# Step 5: Visualize
python visualize_graph.py --neo4j --output-dir output/ --full