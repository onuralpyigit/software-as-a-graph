#!/bin/bash

# 1. Generate problematic datasets
echo "Generating problematic patterns..."
cd ..
python generate_graph.py --generate_problematic_dataset

# 2. Test each pattern
cd demo
for pattern in single_point_of_failure god_topic_pattern circular_dependencies chatty_communication hidden_coupling; do
    echo "Testing $pattern..."
    python demo_problem_detection.py \
        --dataset ../output/dataset_${pattern}.json \
        --report ../output/${pattern}_report.json
done

exit 0