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

from src.infrastructure.memory_repo import MemoryRepository
from src.analysis.service import AnalysisService

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
    print("  This script demonstrates how to use the framework's anti-pattern")
    print("  detection engine to enforce architectural guardrails in CI/CD.")

    # 1. Load Topology & Construct Model
    print(f"  [1] Loading topology: {os.path.basename(args.topology)}")
    graph_data = load_json_with_comments(args.topology)
    
    repo = MemoryRepository()
    repo.save_graph(graph_data, clear=True)
    
    # 2. Run Analysis
    # AnalysisService.analyze_layer() is the "one-stop shop" — it computes
    # structural metrics, predicts RMAV quality, and detects anti-patterns.
    print("  [2] Analyzing system architecture & detecting smells...")
    analysis_service = AnalysisService(repo)
    
    # We analyze the full system layer
    result = analysis_service.analyze_layer("system")
    
    # 3. Interpret Results
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    if not result.problems:
        print("✨ No architectural anti-patterns detected.")
    else:
        print(f"🔍 Detected {len(result.problems)} architectural anti-patterns:")
        # Group by severity
        for severity in ["CRITICAL", "HIGH", "MEDIUM"]:
            sev_probs = [p for p in result.problems if p.severity == severity]
            if sev_probs:
                print(f"\n[{severity}] {len(sev_probs)} findings:")
                for p in sev_probs:
                    print(f"  • {p.name:<30} ({p.entity_type}: {p.entity_id})")
                    print(f"    {p.description}")
                    print(f"    Fix: {p.recommendation}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # 4. CI/CD Gate Logic
    # We can use the result.problem_summary or manually count
    critical_count = len([p for p in result.problems if p.severity == "CRITICAL"])
    high_count = len([p for p in result.problems if p.severity == "HIGH"])
    
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
