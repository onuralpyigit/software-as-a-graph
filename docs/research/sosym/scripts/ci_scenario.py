#!/usr/bin/env python3
"""CI scenario for the SoSyM tool paper (Section 7.5).

Two measurements of the quality gate, both on in-memory models:

1. mini-ATM: the feedback-loop change of examples/run_triage_walkthrough.py,
   gated absolutely and against its merge-base, then with a waiver covering
   one blocking finding.
2. Corpus: for every model of the detection seed sweep below 150 components,
   ``--edits`` random single-subscription additions (seeded), each gated
   absolutely and delta-aware. Reports the exit-code distribution.

Usage:
    PYTHONPATH=. python docs/research/sosym/scripts/ci_scenario.py \
        --output results/sosym/ci_scenario.json
"""
import argparse
import copy
import json
import logging
import random
import sys
import time
import warnings
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from saag import Client  # noqa: E402
from saag.analysis.antipattern_detector import CATALOG  # noqa: E402
from saag.analysis.delta_gate import Waiver, delta_gate, finding_key  # noqa: E402
from saag.infrastructure.memory_repo import MemoryRepository  # noqa: E402
from reproduce._provenance import stamp  # noqa: E402
from reproduce.detection_seed_sweep import SCENARIOS  # noqa: E402
from run_triage_walkthrough import apply_change  # noqa: E402

LAYER = "system"
# DEEP_PIPELINE enumerates every simple source-to-sink path and does not
# terminate on the larger corpus models; excluded here as in the other runs.
PATTERNS = [pid for pid in CATALOG if pid != "DEEP_PIPELINE"]


def findings(topology: dict) -> list:
    repo = MemoryRepository()
    repo.save_graph(topology, clear=True)
    repo.derive_dependencies()
    client = Client(repo=repo)
    diagnosis = client.diagnose(client.analyze(layer=LAYER), active_patterns=PATTERNS)
    return [p.to_dict() for p in (diagnosis.raw.problems or [])]


def absolute_exit(fs: list) -> int:
    sev = {f["severity"] for f in fs}
    return 2 if sev & {"CRITICAL", "HIGH"} else (1 if fs else 0)


def brief(f: dict) -> dict:
    return {"pattern": finding_key(f)[0], "entity_id": f["entity_id"], "severity": f["severity"]}


def mini_atm() -> dict:
    base = json.loads((ROOT / "examples/mini_atm.json").read_text())
    cand = apply_change(base)
    fb, fc = findings(base), findings(cand)
    gate = delta_gate(fb, fc)
    out = {
        "n_base": len(fb), "n_candidate": len(fc),
        "absolute_exit_base": absolute_exit(fb), "absolute_exit_candidate": absolute_exit(fc),
        "delta": {"exit_code": gate.exit_code, "blocking": [brief(f) for f in gate.blocking],
                  "near_fence": [brief(f) for f in gate.near_fence],
                  "new_advisory": [brief(f) for f in gate.new_advisory],
                  "carried": gate.carried, "resolved": [brief(f) for f in gate.resolved]},
    }
    if gate.blocking:
        first = gate.blocking[0]
        key = finding_key(first)
        waiver = Waiver(pattern=key[0], entity=key[2], expires=date.today() + timedelta(days=30),
                        reason="accepted for demo", approved_by="architect")
        waived = delta_gate(fb, fc, waivers=[waiver])
        out["with_waiver"] = {"waiver": {"pattern": key[0], "entity": key[2]},
                              "exit_code": waived.exit_code, "n_blocking": len(waived.blocking),
                              "n_waived": len(waived.waived)}
    return out


def random_subscription(topology: dict, rng: random.Random) -> dict:
    cand = copy.deepcopy(topology)
    subs = {(r["from"], r["to"]) for r in cand["relationships"]["subscribes_to"]}
    apps = [a["id"] for a in cand["applications"]]
    topics = [t["id"] for t in cand["topics"]]
    for _ in range(1000):
        pair = (rng.choice(apps), rng.choice(topics))
        if pair not in subs:
            cand["relationships"]["subscribes_to"].append({"from": pair[0], "to": pair[1]})
            return cand
    raise RuntimeError("no free subscription pair")


def corpus(n_edits: int, max_components: int) -> list:
    rows = []
    for scenario, corpus_tag in SCENARIOS:
        path = ROOT / "data/scenarios" / f"{scenario}.json"
        topo = json.loads(path.read_text())
        n = sum(len(topo.get(k, [])) for k in ("applications", "libraries", "topics", "brokers", "nodes"))
        if n >= max_components:
            continue
        t0 = time.perf_counter()
        fb = findings(topo)
        rng = random.Random(42)
        absolute, delta = Counter(), Counter()
        for _ in range(n_edits):
            fc = findings(random_subscription(topo, rng))
            absolute[absolute_exit(fc)] += 1
            delta[delta_gate(fb, fc).exit_code] += 1
        rows.append({"scenario": scenario, "corpus": corpus_tag, "n_components": n,
                     "n_base_findings": len(fb), "absolute_exit_base": absolute_exit(fb),
                     "edits": n_edits, "absolute": dict(absolute), "delta": dict(delta),
                     "seconds": round(time.perf_counter() - t0, 1)})
        print(rows[-1], flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--edits", type=int, default=10)
    p.add_argument("--max-components", type=int, default=150)
    p.add_argument("--output", type=Path, default=ROOT / "results/sosym/ci_scenario.json")
    args = p.parse_args()
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    data = {"layer": LAYER, "mini_atm": mini_atm(),
            "corpus": corpus(args.edits, args.max_components), "provenance": stamp()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2))
    print(json.dumps(data["mini_atm"], indent=2))


if __name__ == "__main__":
    main()
