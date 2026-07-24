"""
test_label_determinism.py
─────────────────────────
Guards the reproducibility of `FailureSimulator.simulate_exhaustive`, which
supplies the Validate-stage RMAV oracle.

Before seeding, six cascade branches drew against each edge's own QoS weight
using an unseeded RNG, so identical sweeps disagreed: on healthcare_system the
run-to-run Spearman fell to 0.909 and ~8% of the top-20% set churned between
runs. That put the label noise floor below the rho >= 0.85 gate it was used to
enforce. These tests pin the fix.
"""

import json
from pathlib import Path

import pytest

from saag.infrastructure.memory_repo import MemoryRepository
from saag.simulation.failure_simulator import FailureSimulator
from saag.simulation.graph import SimulationGraph

# healthcare_system is the scenario that actually drifted; a stable scenario
# would pass these tests even with the bug present.
SCENARIO = Path(__file__).resolve().parents[1] / "data" / "scenarios" / "healthcare_system.json"


@pytest.fixture(scope="module")
def graph_data():
    if not SCENARIO.exists():
        pytest.skip(f"scenario fixture missing: {SCENARIO}")
    repo = MemoryRepository()
    repo.save_graph(json.loads(SCENARIO.read_text()), clear=True)
    repo.derive_dependencies()
    return repo.get_graph_data(include_raw=True)


def _sweep(graph_data, **kwargs):
    """Run one exhaustive sweep on a fresh simulator, returning ordered labels."""
    sim = FailureSimulator(SimulationGraph(graph_data))
    return [(r.target_id, r.impact.composite_impact) for r in sim.simulate_exhaustive(layer="system", **kwargs)]


def test_same_seed_is_bit_identical(graph_data):
    """Two sweeps at the same seed must agree exactly, target order included."""
    assert _sweep(graph_data, seed=42) == _sweep(graph_data, seed=42)


def test_default_seed_is_deterministic(graph_data):
    """The default (no seed argument) must be reproducible, not free-running."""
    assert _sweep(graph_data) == _sweep(graph_data)


def test_different_seeds_diverge(graph_data):
    """Guard against 'fixing' determinism by removing the stochasticity itself.

    The cascade model is genuinely probabilistic; seeding must make a run
    repeatable without collapsing it to a single deterministic draw.
    """
    assert _sweep(graph_data, seed=42) != _sweep(graph_data, seed=123)


def test_seed_none_still_runs(graph_data):
    """seed=None restores free-running behaviour for callers that want it."""
    labels = _sweep(graph_data, seed=None)
    assert labels, "expected a non-empty sweep"


def test_component_seed_independent_of_target_set_size(graph_data):
    """A component's seed derives from its id, not its index.

    LOSO folds change which components are swept; a component's label must not
    move just because the sweep around it got bigger or smaller.
    """
    derive = FailureSimulator._derive_seed
    assert derive(42, "AppA") == derive(42, "AppA")
    assert derive(42, "AppA") != derive(42, "AppB")
    assert derive(None, "AppA") is None


def test_derived_seed_is_stable_across_processes():
    """zlib.crc32, not hash(): hash(str) is PYTHONHASHSEED-salted."""
    import subprocess
    import sys

    snippet = (
        "from saag.simulation.failure_simulator import FailureSimulator as F;"
        "print(F._derive_seed(42, 'ICAOMessageLib'))"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True, text=True, check=True,
            env={"PYTHONHASHSEED": str(h), "PYTHONPATH": "."},
        ).stdout.strip()
        for h in (0, 1, 12345)
    }
    assert len(runs) == 1, f"derived seed varies with PYTHONHASHSEED: {runs}"
