"""
examples/example_antipatterns.py — Automated Architecture Anti-Pattern Detection

This script demonstrates how to integrate Software-as-a-Graph into a CI/CD pipeline
to detect architectural "smells" and block deployments if critical risks are found.

Usage:
    python examples/example_antipatterns.py --topology examples/topologies/ros2_autonomous_vehicle.json
"""
import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to path for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "backend"))

from backend.src.infrastructure.memory_repo import MemoryRepository
from backend.src.analysis.service import AnalysisService
from backend.src.prediction.service import PredictionService
from backend.src.analysis.smells import SmellDetector

def load_json_with_comments(path: str):
    import re
    with open(path, 'r') as f:
        content = f.read()
    content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
    return json.loads(content)

def main():
    parser = argparse.ArgumentParser(description="Architectural Anti-Pattern Guardrail")
    parser.add_argument("--topology", type=str, required=True, help="Path to topology JSON")
    args = parser.parse_args()

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🛡️  Architectural Anti-Pattern Guardrail")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # 1. Load Topology & Construct Model
    print(f"  [1] Loading topology: {os.path.basename(args.topology)}")
    graph_data = load_json_with_comments(args.topology)
    
    repo = MemoryRepository()
    repo.save_graph(graph_data, clear=True)
    
    # 2. Run Analysis & Prediction
    print("  [2] Analyzing system architecture & predicting quality...")
    analysis_service = AnalysisService(repo)
    prediction_service = PredictionService(repo)
    
    # We analyze all primary layers and predict quality to enable smell detection
    layers = ["app", "infra", "mw", "system"]
    layer_results = []
    for layer in layers:
        struct_res = analysis_service.analyze_layer(layer)
        # Prediction populates the .quality field needed by AntiPatternDetector
        quality_res = prediction_service.predict_quality(struct_res.structural)
        struct_res.quality = quality_res
        layer_results.append(struct_res)
    
    # 3. Detect Anti-Patterns (Smells)
    print("  [3] Scanning for architectural anti-patterns...")
    detector = SmellDetector()
    report = detector.detect_all(layer_results)
    
    # 4. Interpret Results
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    detector.print_findings(report)
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # 5. CI/CD Gate Logic
    critical_count = report.by_severity.get("CRITICAL", 0)
    high_count = report.by_severity.get("HIGH", 0)
    
    if critical_count > 0:
        print(f"\n❌ DEPLOYMENT BLOCKED: {critical_count} CRITICAL anti-patterns detected.")
        print("   Architectural risks must be resolved before proceeding.")
        sys.exit(2)
    elif high_count > 3:
        print(f"\n⚠️  WARNING: {high_count} HIGH severity patterns detected.")
        print("   Proceeding, but architectural debt review is recommended.")
        sys.exit(0)
    else:
        print("\n✅ PASSED: No critical architectural risks found.")
        sys.exit(0)

if __name__ == "__main__":
    main()
