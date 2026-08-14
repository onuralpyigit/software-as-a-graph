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

import dataclasses
import pathlib

import pytest


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


#: Graph attributes the impact oracle may consume when building a label. Any of
#: these that is *also* a GNN input feature makes the label a function of a
#: feature — a leak the name-collision check above cannot see, because the
#: feature and the attribute are spelled differently.
_LABEL_INPUT_TOPIC_ATTRS = {
    # Consumed by FailureSimulator._topic_criticality_norm, but only when
    # use_topic_criticality=True — which is off by default for this reason.
    "criticality": "topic_qos_criticality_ord",
}


def test_label_side_topic_attributes_are_not_also_features():
    """Topic.criticality feeds the label only behind an opt-in flag.

    It is simultaneously the Topic feature ``topic_qos_criticality_ord``. If the
    oracle starts consuming it by default, that feature has to leave the feature
    contract or the correlation measures the model rediscovering its own input.
    """
    import inspect

    from saag.prediction.data_preparation import KEYS_BY_TYPE
    from saag.simulation.failure_simulator import FailureSimulator

    default_on = (
        inspect.signature(FailureSimulator.__init__)
        .parameters["use_topic_criticality"]
        .default
    )
    if not default_on:
        return  # opt-in, so no leak

    topic_features = set(KEYS_BY_TYPE.get("Topic", []))
    conflicting = {
        feature for feature in _LABEL_INPUT_TOPIC_ATTRS.values()
        if feature in topic_features
    }
    assert not conflicting, (
        f"use_topic_criticality is on by default while {sorted(conflicting)} is still a "
        "Topic input feature. Remove it from KEYS_BY_TYPE or keep the oracle flag off."
    )


def test_topic_criticality_is_opt_in():
    """Regression: the default oracle must not consume the declared label."""
    import inspect

    from saag.simulation.failure_simulator import FailureSimulator

    assert (
        inspect.signature(FailureSimulator.__init__)
        .parameters["use_topic_criticality"]
        .default
        is False
    )


def _write_sparse_cache(cache_dir: pathlib.Path, n: int = 10) -> set:
    """Write a cache whose simulation labels are all zero but whose RM is not.

    This is exactly the shape that used to trigger silent RM substitution.
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


def test_rm_substitution_raises_by_default(tmp_path):
    """Sparse simulation labels must not be silently swapped for RM scores.

    RM is computed from the same structural metrics that form the GNN's input
    features, so substituting it makes the labels a function of the features.
    """
    from reproduce.main_table import _load_cache_dicts

    nodes = _write_sparse_cache(tmp_path / "sparse_scenario")

    with pytest.raises(ValueError, match="RM"):
        _load_cache_dicts(tmp_path / "sparse_scenario", nodes)


def test_rm_substitution_is_tagged_when_explicitly_allowed(tmp_path):
    """Opting in is permitted, but the result may never be labelled 'Sim'."""
    from reproduce.main_table import _load_cache_dicts

    nodes = _write_sparse_cache(tmp_path / "sparse_scenario")

    _, _, _, gt_source = _load_cache_dicts(
        tmp_path / "sparse_scenario", nodes, allow_rm_substitution=True
    )
    assert gt_source == "RM-sub", (
        f"substituted labels reported as {gt_source!r}; they must never be tagged 'Sim'"
    )


def test_dense_simulation_labels_are_untouched(tmp_path):
    """The guard must not fire on healthy caches."""
    import json

    from reproduce.main_table import _load_cache_dicts

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
