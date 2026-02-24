"""
Step 1 — Graph Generation
=========================
Demonstrates how to generate a synthetic pub-sub system topology programmatically.

Available built-in scales: tiny, small, medium, large, xlarge
Custom configs can be loaded from a YAML file via load_config().

Run from the project root:
    python examples/example_generation.py
"""
import sys
import json
from pathlib import Path

# Add backend to path (required for all examples)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from src.generation import generate_graph, load_config, SCALE_PRESETS


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def summarise_graph(data: dict, label: str = "") -> None:
    apps      = data.get("applications", [])
    brokers   = data.get("brokers", [])
    topics    = data.get("topics", [])
    nodes     = data.get("nodes", [])
    pub_edges = data.get("publish_edges", [])
    sub_edges = data.get("subscribe_edges", [])

    if label:
        print(f"\n  [{label}]")
    print(f"    Applications : {len(apps)}")
    print(f"    Brokers      : {len(brokers)}")
    print(f"    Topics       : {len(topics)}")
    print(f"    Infra nodes  : {len(nodes)}")
    print(f"    Pub edges    : {len(pub_edges)}")
    print(f"    Sub edges    : {len(sub_edges)}")

    # Show QoS distribution across topics
    if topics:
        qos_counts: dict = {}
        for t in topics:
            qos = t.get("qos_reliability", "UNKNOWN")
            qos_counts[qos] = qos_counts.get(qos, 0) + 1
        print(f"    Topic QoS    : {qos_counts}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    # ── 1. Show available scale presets ───────
    print_section("Available scale presets")
    for name, preset in SCALE_PRESETS.items():
        n_apps = preset.get("num_applications", "?")
        n_nodes = preset.get("num_nodes", "?")
        print(f"  {name:<10}  apps={n_apps}, infra_nodes={n_nodes}")

    # ── 2. Generate a 'tiny' graph (fast, good for smoke-testing) ─────
    print_section("Generating 'tiny' graph (seed=42 for reproducibility)")
    tiny = generate_graph(scale="tiny", seed=42)
    summarise_graph(tiny, "tiny")

    # ── 3. Generate a 'small' graph ───────────
    print_section("Generating 'small' graph (seed=42)")
    small = generate_graph(scale="small", seed=42)
    summarise_graph(small, "small")

    # ── 4. Load a custom YAML config if one exists ────────────────────
    example_yaml = ROOT / "input" / "scenario_01_autonomous_vehicle.yaml"
    if example_yaml.exists():
        print_section(f"Loading custom YAML config: {example_yaml.name}")
        config = load_config(str(example_yaml))
        custom = generate_graph(config=config, seed=99)
        summarise_graph(custom, example_yaml.stem)
    else:
        print_section("Custom YAML config not found — skipping")
        print(f"  (expected at: {example_yaml})")

    # ── 5. Save the 'tiny' graph so subsequent examples can use it ────
    output_path = ROOT / "output" / "example_graph.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(tiny, f, indent=2)

    print_section("Output")
    print(f"  Saved 'tiny' graph → {output_path}")
    print()
    print("  Next step: run  examples/example_import.py")


if __name__ == "__main__":
    main()
