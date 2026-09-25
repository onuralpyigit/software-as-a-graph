#!/usr/bin/env python3
"""
Example script to score the thesis draft's §3.6 running example with the RM
explanation layer and print the worked-attribution table of §4.6.

Topology: a1 publishes topic t, which a2 and a3 subscribe to; broker b routes t;
host n runs a1, a2, a3 and b; all three applications use library l. The topic
declares RELIABLE / TRANSIENT_LOCAL / HIGH with a 1 KiB payload at 14 Hz, the
rate at which its weight reproduces the w(t) = 0.596 stated in §3.6.
"""

import sys
from pathlib import Path

# Add project root to sys.path to support direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saag import Client
from saag.infrastructure.memory_repo import MemoryRepository

APPS = ["a1", "a2", "a3"]

TOPOLOGY = {
    "metadata": {"scenario": "thesis_running_example", "generation_mode": "manual", "seed": 42},
    "nodes": [{"id": "n", "name": "host"}],
    "brokers": [{"id": "b", "name": "broker"}],
    "topics": [{
        "id": "t", "name": "t", "size": 1024, "frequency": 14.0,
        "qos": {"reliability": "RELIABLE", "durability": "TRANSIENT_LOCAL", "transport_priority": "HIGH"},
    }],
    "applications": [
        {"id": "a1", "name": "a1", "role": "pub"},
        {"id": "a2", "name": "a2", "role": "sub"},
        {"id": "a3", "name": "a3", "role": "sub"},
    ],
    "libraries": [{"id": "l", "name": "lib"}],
    "relationships": {
        "runs_on": [{"from": c, "to": "n"} for c in APPS + ["b"]],
        "routes": [{"from": "b", "to": "t"}],
        "publishes_to": [{"from": "a1", "to": "t"}],
        "subscribes_to": [{"from": "a2", "to": "t"}, {"from": "a3", "to": "t"}],
        "connects_to": [],
        "uses": [{"from": a, "to": "l"} for a in APPS],
    },
}


def main():
    repo = MemoryRepository()
    try:
        repo.save_graph(TOPOLOGY, clear=True)
        repo.derive_dependencies()
        topic = next(c for c in repo.get_graph_data(include_raw=True).components if c.id == "t")
        print(f"w(t) = {topic.weight:.3f}")

        client = Client(repo=repo)
        quality = client.predict(client.analyze(layer="system")).raw

        print("\nComponent | FT | A | R | M | Q | tiers (FT/A/R/M/Q)")
        for cq in quality.components:
            s, lv = cq.scores, cq.levels
            tiers = "/".join(getattr(lv, d).value for d in
                             ("fault_tolerance", "availability", "reliability", "maintainability", "overall"))
            print(f"{cq.id} | {s.fault_tolerance:.3f} | {s.availability:.3f} | {s.reliability:.3f} | "
                  f"{s.maintainability:.3f} | {s.overall:.3f} | {tiers}")

        print("\nEdge | FT | A | R | M | Q | tier")
        for e in quality.edges:
            s = e.scores
            print(f"{e.source}->{e.target} ({e.dependency_type}) | {s.fault_tolerance:.3f} | "
                  f"{s.availability:.3f} | {s.reliability:.3f} | {s.maintainability:.3f} | "
                  f"{s.overall:.3f} | {e.level.value}")
    finally:
        repo.close()


if __name__ == "__main__":
    main()
