#!/bin/bash

source ../distributed_system_env/bin/activate

cd ../
python3 generate_graph.py --data input/demo.json > output/graph_modeling.txt
python3 analyze_graph.py > output/graph_analysis.txt

python3 run.py