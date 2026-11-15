#!/usr/bin/env python3
"""Prescribe on the ATM tiny scenario, as in examples/run_prescribe_walkthrough.py,
written as JSON for the SoSyM tool paper (Section 7.2).

Usage:
    PYTHONPATH=. python docs/research/sosym/scripts/prescribe_example.py \
        --output results/sosym/prescribe_example.json
"""
import argparse
import collections
import json
import logging
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from saag import Client  # noqa: E402
from saag.analysis.antipattern_detector import CATALOG  # noqa: E402
from saag.infrastructure.memory_repo import MemoryRepository  # noqa: E402
from reproduce._provenance import stamp  # noqa: E402

TOPOLOGY = ROOT / "data/scenarios/atm_system_tiny.json"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "results/sosym/prescribe_example.json")
    args = p.parse_args()
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)

    t0 = time.perf_counter()
    repo = MemoryRepository()
    repo.save_graph(json.loads(TOPOLOGY.read_text()), clear=True)
    repo.derive_dependencies()
    client = Client(repo=repo)
    analysis = client.analyze(layer="system")
    prediction = client.predict(analysis, mode="rm")
    id_of = {spec.name: pid for pid, spec in CATALOG.items()}
    fired = collections.Counter(id_of[pr.name] for pr in (prediction.raw.problems or []))
    result = client.prescribe(analysis, prediction, layer="system").to_dict()
    edits = result["edit_verdicts"]
    data = {
        "scenario": str(TOPOLOGY.relative_to(ROOT)),
        "patterns_fired": dict(fired),
        "n_candidates": len(edits),
        "n_accepted": sum(1 for e in edits if e["accepted"]),
        "n_rejected_worse": sum(1 for e in edits if not e["accepted"] and e["worst_delta"] is not None
                                and e["worst_delta"] < 0),
        "by_kind": {k: {"candidates": sum(1 for e in edits if e["kind"] == k),
                        "accepted": sum(1 for e in edits if e["kind"] == k and e["accepted"])}
                    for k in sorted({e["kind"] for e in edits})},
        "sri": [result["original_sri"], result["mutated_sri"]],
        "seconds": round(time.perf_counter() - t0, 1),
        "provenance": stamp(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2))
    print(json.dumps({k: v for k, v in data.items() if k != "provenance"}, indent=2))


if __name__ == "__main__":
    main()
