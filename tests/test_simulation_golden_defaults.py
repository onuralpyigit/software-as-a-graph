"""
Golden snapshot of the simulation oracles at their default arguments.

Every runtime-simulation option (reactive behaviour, deployment stations, the
fault library, threshold marginalisation, network gating, replica awareness,
...) is opt-in. This test is the guarantee that the defaults still reproduce
the published I*(v) and I_dyn(v) labels and the I_comp(v) and I_M(v) validation
oracles bit for bit, so a new option cannot leak into them.

Regenerate only on purpose, after a deliberate change to the default oracle:

    SAAG_REGEN_GOLDEN=1 pytest tests/test_simulation_golden_defaults.py
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("simpy")

from saag.core.graph_io import load_graph  # noqa: E402
from saag.infrastructure.memory_repo import MemoryRepository  # noqa: E402
from saag.simulation import MessageFlowSimulator  # noqa: E402
from saag.simulation.fault_injector import FaultInjector, RECOMMENDED_SEEDS  # noqa: E402
from saag.simulation.service import SimulationService  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).parent / "golden" / "simulation_defaults.json"

I_STAR_SCENARIOS = ("av_system", "microservices_system")
I_DYN_SCENARIO = "tiny_system"
I_DYN_DURATION = 30.0
EXHAUSTIVE_SCENARIOS = ("atm_system", "microservices_system")


def _graph(name):
    return load_graph(ROOT / "data" / "scenarios" / f"{name}.json")


def _i_star():
    out = {}
    for name in I_STAR_SCENARIOS:
        res = FaultInjector(_graph(name), seeds=RECOMMENDED_SEEDS).run(
            node_types=["Application", "Broker", "Node"])
        out[name] = {n: r.impact_score for n, r in sorted(res.records.items())}
    return out


def _i_dyn():
    g = _graph(I_DYN_SCENARIO)
    apps = sorted(n for n, d in g.nodes(data=True) if d.get("type") == "Application")
    out = {}
    for node in apps[:3]:
        res = MessageFlowSimulator(g, duration=I_DYN_DURATION, fault_node=node, seed=42).run()
        fe = res.fault_event
        out[node] = {
            "i_dyn": fe.i_dyn,
            "delivery_rate_before": fe.delivery_rate_before,
            "delivery_rate_after": fe.delivery_rate_after,
            "total_messages_delivered": res.total_messages_delivered,
            "latency_p95_after": fe.latency_p95_after,
        }
    return out


def _i_comp_and_i_m():
    """
    Run in a child process with PYTHONHASHSEED=0. I_M depends on the hash seed:
    SimulationGraph.get_dependency_edges() orders its arcs by iterating a set of
    topic ids, and the change-propagation BFS is order-sensitive. The snapshot
    pins the code's behaviour at one seed, not a seed-independent value.
    """
    env = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONPATH": str(ROOT)}
    done = subprocess.run([sys.executable, __file__], env=env, cwd=ROOT,
                          capture_output=True, text=True, check=True)
    return json.loads(done.stdout)


def _exhaustive():
    i_comp, i_m = {}, {}
    for name in EXHAUSTIVE_SCENARIOS:
        repo = MemoryRepository()
        repo.save_graph(json.loads((ROOT / "data" / "scenarios" / f"{name}.json").read_text()), clear=True)
        results = sorted(SimulationService(repo).run_failure_simulation_exhaustive(layer="system"),
                         key=lambda r: r.target_id)
        i_comp[name] = {
            r.target_id: {
                "composite_impact": r.impact.composite_impact,
                "reliability_impact": r.impact.reliability_impact,
                "reachability_loss": r.impact.reachability_loss,
                "cascade_count": r.impact.cascade_count,
            }
            for r in results
        }
        i_m[name] = {
            r.target_id: {
                "maintainability_impact": r.impact.maintainability_impact,
                "change_reach": r.impact.change_reach,
                "weighted_change_impact": r.impact.weighted_change_impact,
                "normalized_change_depth": r.impact.normalized_change_depth,
            }
            for r in results
        }
    return {"i_comp": i_comp, "i_m": i_m}


def _compute():
    logging.disable(logging.WARNING)
    try:
        return {"i_star": _i_star(), "i_dyn": _i_dyn(), **_i_comp_and_i_m()}
    finally:
        logging.disable(logging.NOTSET)


@pytest.fixture(scope="module")
def current():
    return _compute()


def test_golden_defaults(current):
    if os.environ.get("SAAG_REGEN_GOLDEN"):
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
    expected = json.loads(GOLDEN.read_text())
    # Round-trip through JSON so float formatting cannot differ from the file.
    assert json.loads(json.dumps(current)) == expected


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    print(json.dumps(_exhaustive()))
