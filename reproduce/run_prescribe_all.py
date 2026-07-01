#!/usr/bin/env python3
"""
Batch script to run prescriptive optimization across all seven benchmark scenarios.
Extracts baseline SRI, optimized SRI, delta improvement, and counts of applied refactoring operators.
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saag import Client
from saag.infrastructure.memory_repo import MemoryRepository

SCENARIOS = {
    "av_system.json": "Scenario 01 (Autonomous Vehicle)",
    "iot_smart_city_system.json": "Scenario 02 (IoT Smart City)",
    "financial_trading_system.json": "Scenario 03 (Financial Trading)",
    "healthcare_system.json": "Scenario 04 (Healthcare)",
    "hub_and_spoke_system.json": "Scenario 05 (Hub-and-Spoke)",
    "microservices_system.json": "Scenario 06 (Microservices Mesh)",
    "enterprise_system.json": "Scenario 07 (Hyper-Scale Enterprise)"
}

def main():
    print("| Scenario | Baseline SRI | Mutated SRI | Delta | Splits | Reallocs | Upgrades |")
    print("|----------|:------------:|:-----------:|:-----:|:------:|:--------:|:--------:|")
    
    for filename, name in SCENARIOS.items():
        json_path = Path("data/scenarios") / filename
        if not json_path.exists():
            print(f"Error: {json_path} not found.")
            continue
            
        repo = MemoryRepository()
        with open(json_path) as f:
            data = json.load(f)
        repo.save_graph(data, clear=True)
        repo.derive_dependencies()
        client = Client(repo=repo)
        
        # Analyze system layer
        analysis = client.analyze(layer="system")
        
        # Prescribe mutations
        res = client.prescribe(analysis_result=analysis, layer="system")
        
        policy = res.policy
        splits = len(policy.topic_splits)
        reallocs = len(policy.node_reallocations)
        upgrades = len(policy.qos_upgrades)
        
        print(f"| {name} | {res.original_sri:.4f} | {res.mutated_sri:.4f} | +{res.sri_improvement:.4f} | {splits} | {reallocs} | {upgrades} |")

if __name__ == "__main__":
    main()
