"""
test_no_simulation_leakage.py
─────────────────────────────
Guards the feature/label boundary for the GNN criticality models.

Simulation output is the *label*. If any simulation-derived quantity reaches the
feature matrix, reported Spearman rho stops measuring prediction and starts
measuring the model rediscovering its own inputs.

The audit behind these tests is clean today — no key in KEYS_BY_TYPE is
simulation-derived, and blast_radius/cascade_depth exist on StructuralMetrics
but are computed structurally and appear in no feature list. These tests keep it
that way, and close the one live substitution path that would break it.
"""

import ast
import dataclasses
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _simulation_output_names() -> set:
    """Every field name a simulator emits, plus known cascade-shaped proxies."""
    from saag.simulation.models import ImpactMetrics
    from saag.simulation.simulation_results import FaultInjectionRecord

    names = {f.name for f in dataclasses.fields(ImpactMetrics)}
    names |= {f.name for f in dataclasses.fields(FaultInjectionRecord)}
    # Structural quantities that are close proxies for cascade reach. They are
    # legitimately computed without simulation, but must never become features:
    # blast_radius is |descendants|, which is close to what the label measures.
    names |= {"blast_radius", "cascade_depth", "impact_score", "composite_impact"}
    return names


def test_feature_keys_are_not_simulation_derived():
    """No per-type feature key may collide with a simulation output field."""
    from saag.prediction.data_preparation import KEYS_BY_TYPE

    feature_keys = {k for keys in KEYS_BY_TYPE.values() for k in keys}
    overlap = feature_keys & _simulation_output_names()

    assert not overlap, (
        f"Simulation-derived quantities are being used as GNN input features: {sorted(overlap)}. "
        "These are labels, not features — using them inflates every correlation metric."
    )


def test_data_preparation_does_not_import_simulation():
    """Feature engineering must not reach into the simulation package.

    Mirrors tests/test_predict_simulate_separation.py, applied to the module
    that actually builds the feature matrix.
    """
    target = REPO_ROOT / "saag" / "prediction" / "data_preparation.py"
    assert target.exists(), f"File not found: {target}"

    tree = ast.parse(target.read_text(encoding="utf-8"))
    offending = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "simulation" in node.module.split("."):
            offending.append(f"line {node.lineno}: from {node.module} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if "simulation" in alias.name.split("."):
                    offending.append(f"line {node.lineno}: import {alias.name}")

    assert not offending, (
        "data_preparation.py imports from saag.simulation, coupling feature "
        f"construction to the labeler: {offending}"
    )


def _write_sparse_cache(cache_dir: pathlib.Path, n: int = 10) -> set:
    """Write a cache whose simulation labels are all zero but whose RMAV is not.

    This is exactly the shape that used to trigger silent RMAV substitution.
    """
    import json

    cache_dir.mkdir(parents=True, exist_ok=True)
    node_ids = [f"n{i}" for i in range(n)]

    (cache_dir / "failure_impact.json").write_text(json.dumps({
        "schema_version": "2.0",
        "records": {nid: {"impact_score": 0.0, "cascade_depth": 0} for nid in node_ids},
    }))
    (cache_dir / "quality_scores.json").write_text(json.dumps(
        {nid: {"overall": 0.5, "reliability": 0.5} for nid in node_ids}
    ))
    return set(node_ids)


def test_rmav_substitution_raises_by_default(tmp_path):
    """Sparse simulation labels must not be silently swapped for RMAV scores.

    RMAV is computed from the same structural metrics that form the GNN's input
    features, so substituting it makes the labels a function of the features.
    """
    from reproduce.middleware26_main_table import _load_cache_dicts

    nodes = _write_sparse_cache(tmp_path / "sparse_scenario")

    with pytest.raises(ValueError, match="RMAV"):
        _load_cache_dicts(tmp_path / "sparse_scenario", nodes)


def test_rmav_substitution_is_tagged_when_explicitly_allowed(tmp_path):
    """Opting in is permitted, but the result may never be labelled 'Sim'."""
    from reproduce.middleware26_main_table import _load_cache_dicts

    nodes = _write_sparse_cache(tmp_path / "sparse_scenario")

    _, _, _, gt_source = _load_cache_dicts(
        tmp_path / "sparse_scenario", nodes, allow_rmav_substitution=True
    )
    assert gt_source == "RMAV-sub", (
        f"substituted labels reported as {gt_source!r}; they must never be tagged 'Sim'"
    )


def test_dense_simulation_labels_are_untouched(tmp_path):
    """The guard must not fire on healthy caches."""
    import json

    from reproduce.middleware26_main_table import _load_cache_dicts

    cache_dir = tmp_path / "dense_scenario"
    cache_dir.mkdir(parents=True)
    node_ids = [f"n{i}" for i in range(10)]
    (cache_dir / "failure_impact.json").write_text(json.dumps({
        "schema_version": "2.0",
        "records": {nid: {"impact_score": 0.1 + i * 0.05} for i, nid in enumerate(node_ids)},
    }))

    _, sim, _, gt_source = _load_cache_dicts(cache_dir, set(node_ids))
    assert gt_source == "Sim"
    assert len(sim) == len(node_ids)
