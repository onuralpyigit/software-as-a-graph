#!/usr/bin/env python3
"""
CLI script to generate real-world open-source software topology datasets.
Generates Autoware.universe ROS 2 and Cloud-Native Microservices scenarios.
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saag.adapters.realworld_adapter import RealWorldAdapter


def main():
    scenarios_dir = Path("data/scenarios")
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    # 1. Autoware ROS 2 topology
    autoware_data = RealWorldAdapter.create_autoware_ros2_topology()
    autoware_json_path = scenarios_dir / "scenario_11_realworld_autoware_ros2.json"
    autoware_alias_path = scenarios_dir / "realworld_autoware_ros2.json"
    
    with open(autoware_json_path, "w") as f:
        json.dump(autoware_data, f, indent=2)
    with open(autoware_alias_path, "w") as f:
        json.dump(autoware_data, f, indent=2)
    print(f"Generated {autoware_json_path} (apps: {len(autoware_data['applications'])}, topics: {len(autoware_data['topics'])}, libs: {len(autoware_data['libraries'])})")

    # 2. Cloud Microservices topology
    cloud_data = RealWorldAdapter.create_cloud_microservices_topology()
    cloud_json_path = scenarios_dir / "scenario_12_realworld_cloud_microservices.json"
    cloud_alias_path = scenarios_dir / "realworld_cloud_microservices.json"
    
    with open(cloud_json_path, "w") as f:
        json.dump(cloud_data, f, indent=2)
    with open(cloud_alias_path, "w") as f:
        json.dump(cloud_data, f, indent=2)
    print(f"Generated {cloud_json_path} (apps: {len(cloud_data['applications'])}, topics: {len(cloud_data['topics'])}, libs: {len(cloud_data['libraries'])})")

    # Generate YAML scenario files to support batch scenario discovery
    autoware_yaml = """graph:
  seed: 2026
  domain: autoware_ros2
  scenario: autoware_universe_ros2_autonomous_driving
  counts:
    nodes: 6
    applications: 32
    libraries: 10
    topics: 24
    brokers: 3
"""
    with open(scenarios_dir / "scenario_11_realworld_autoware_ros2.yaml", "w") as f:
        f.write(autoware_yaml)

    cloud_yaml = """graph:
  seed: 2026
  domain: cloud_microservices
  scenario: production_ecommerce_cloud_microservices_mesh
  counts:
    nodes: 6
    applications: 22
    libraries: 8
    topics: 20
    brokers: 4
"""
    with open(scenarios_dir / "scenario_12_realworld_cloud_microservices.yaml", "w") as f:
        f.write(cloud_yaml)

    print("Real-world scenario files and YAML descriptors generated successfully!")


if __name__ == "__main__":
    main()
