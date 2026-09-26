"""
tests/test_dependency_graph_substrate.py — the DEPENDS_ON projection arms (Amendment 9)
======================================================================================

PREREGISTRATION.md Amendment 9 trains the learners on the Application--Library
DEPENDS_ON projection instead of the native multigraph. What it registered, and
what these tests hold the harness to:

  * the projection is exactly the edge set Amendment 7's InDeg and Reach read, in
    the native node order;
  * node features and labels are bit-identical to the native build -- only the
    edges change -- including the Library infra columns, which need the native
    USES / pub-sub edges the projection no longer has;
  * the QoS-off arm sees no QoS on its edges, and the QoS arms see w(e) in
    ``edge_attr[:, 0]``;
  * a 3-layer GAT's receptive field at an Application is exactly the Application
    plus its dependents within three hops;
  * the corpus has exactly the four (App|Lib, DEPENDS_ON, App|Lib) relations
    HGT-P-QoS's width was derived from.

Fixtures are the committed ``data/scenarios`` files, so no cache is needed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("torch_geometric", reason="torch_geometric not installed")

import torch  # noqa: E402

from saag.evaluation import variant_registry as registry  # noqa: E402

SCENARIOS = Path("data/scenarios")
PROJECTION_RELATIONS = {
    (s, "DEPENDS_ON", d)
    for s in ("Application", "Library") for d in ("Application", "Library")
}


def _bundle(tmp_path: Path, scenario: str):
    """A ScenarioBundle over the committed topology, with deterministic stand-in data."""
    from cli.loso_evaluate import ScenarioBundle
    from saag.core.graph_io import build_graph_from_json
    from saag.prediction.data_preparation import networkx_to_hetero_data

    cache = tmp_path / scenario
    cache.mkdir()
    shutil.copy(SCENARIOS / f"{scenario}.json", cache / "topology.json")
    graph = build_graph_from_json(json.loads((cache / "topology.json").read_text()))
    nodes = sorted(graph.nodes)
    structural = {n: {"in_degree_centrality": (i % 7) / 7.0, "pagerank": (i % 5) / 5.0,
                      "qos_weight": (i % 3) / 3.0}
                  for i, n in enumerate(nodes)}
    simulation = {n: {"composite": (i % 11) / 11.0} for i, n in enumerate(nodes)}
    conv = networkx_to_hetero_data(graph, structural, simulation, {})
    return ScenarioBundle(
        scenario_id=scenario, graph=graph, structural=structural, rm={},
        simulation=simulation, hetero_data=conv.hetero_data,
        n_nodes=graph.number_of_nodes(), n_edges=graph.number_of_edges(),
        n_labelled=conv.num_labelled_nodes, cache_dir=cache,
    )


@pytest.fixture(params=["atm_system", "av_system"])
def bundle(request, tmp_path):
    return _bundle(tmp_path, request.param)


def test_projection_is_the_flow_projection_in_native_order(bundle):
    from cli.loso_evaluate import _dependency_graph
    from saag.prediction.structural_predictor import derive_flow_projection

    g = _dependency_graph(bundle)
    flow = derive_flow_projection(json.loads((bundle.cache_dir / "topology.json").read_text()))
    assert set(g.edges) == set(flow.edges)
    native_order = [n for n, d in bundle.graph.nodes(data=True)
                    if d.get("type") in ("Application", "Library")]
    assert list(g.nodes) == native_order
    for u, v, d in g.edges(data=True):
        assert d["type"] == "DEPENDS_ON"
        assert d["weight"] == d["qos_weight"] == flow[u][v]["qos_weight"]
    assert g.graph["infra_source"] is bundle.graph


def test_dependency_bundle_keeps_native_size_and_labels(bundle):
    from cli.loso_evaluate import _dependency_bundle

    dep = _dependency_bundle(bundle)
    assert dep.n_nodes == bundle.n_nodes            # primary / inner-val selection unchanged
    assert dep.simulation is bundle.simulation
    assert dep.graph is not bundle.graph
    assert _dependency_bundle(None) is None


@pytest.mark.parametrize("use_qos", [False, True])
def test_features_and_labels_match_the_native_build(bundle, use_qos):
    from cli.loso_evaluate import _build_training_hetero, _dependency_bundle

    native = _build_training_hetero(bundle, use_qos, True)
    proj = _build_training_hetero(_dependency_bundle(bundle), use_qos, True)
    for nt in ("Application", "Library"):
        assert torch.equal(native[nt].x, proj[nt].x), nt
        assert torch.equal(native[nt].y, proj[nt].y), nt
    # The Library reach features come from the native graph; without it they
    # would be identically zero on the projection.
    assert proj["Library"].x.abs().sum() > 0
    assert set(proj.node_types) == {"Application", "Library"}
    assert set(proj.edge_types) <= PROJECTION_RELATIONS


def test_qos_reaches_the_edges_only_when_enabled(bundle):
    from cli.loso_evaluate import _build_training_hetero, _dependency_bundle

    dep = _dependency_bundle(bundle)
    off = _build_training_hetero(dep, False, True)
    on = _build_training_hetero(dep, True, True)
    for rel in off.edge_types:
        assert torch.all(off[rel].edge_attr[:, 0] == 1.0)
        assert torch.all(off[rel].edge_attr[:, 9:] == 0.0)
    weights = torch.cat([on[rel].edge_attr[:, 0] for rel in on.edge_types])
    assert weights.min() < 1.0                      # w(e), not the constant 1.0
    assert torch.cat([on[rel].edge_attr[:, 9:] for rel in on.edge_types]).abs().sum() > 0


def test_gat_receptive_field_is_the_three_hop_dependents(bundle):
    from cli.loso_evaluate import _build_training_hetero, _dependency_bundle
    from reproduce.receptive_field_probe import _expected_3hop, gradient_probe
    from saag.prediction.models.baselines import build_baseline

    dep = _dependency_bundle(bundle)
    data = _build_training_hetero(dep, True, False)
    torch.manual_seed(0)
    gat = build_baseline("homo_scalar", hidden_channels=16, num_heads=2,
                         num_layers=3, dropout=0.0, edge_dim=16)
    probe = gradient_probe(gat, data, _expected_3hop(dep))
    assert probe["share_rf_equals_3hop_dependents"] == 1.0
    assert probe["share_of_apps_seeing_other_apps"] > 0


def test_native_masking_still_drops_graph_attributes(bundle):
    """Only a projection's native source survives QoS masking; nothing else changes."""
    from cli.loso_evaluate import _dependency_bundle, _prepare_bundle_graph

    graph, _ = _prepare_bundle_graph(bundle, True)
    assert graph is bundle.graph
    masked, _ = _prepare_bundle_graph(bundle, False)
    assert "infra_source" not in masked.graph
    masked_dep, _ = _prepare_bundle_graph(_dependency_bundle(bundle), False)
    assert masked_dep.graph["infra_source"] is bundle.graph


def test_corpus_projection_has_the_declared_relations():
    """HGT-P-QoS's width (100) was derived from these four relation triples."""
    from reproduce.training_free_suite import FOLDS
    from saag.prediction.structural_predictor import derive_depends_on_edges

    seen_by_scenario = {}
    for sid in FOLDS:
        topo = json.loads((SCENARIOS / f"{sid}.json").read_text())
        kind = {str(a["id"]): "Application" for a in topo.get("applications", [])}
        kind.update({str(lb["id"]): "Library" for lb in topo.get("libraries", [])})
        seen_by_scenario[sid] = {
            (kind[str(e["source"])], "DEPENDS_ON", kind[str(e["target"])])
            for e in derive_depends_on_edges(topo)
            if str(e["source"]) in kind and str(e["target"]) in kind
        }
    for sid, rels in seen_by_scenario.items():
        assert rels <= PROJECTION_RELATIONS, sid
    # The LOSO primary graph carries all four, so no holdout relation is unseen.
    assert seen_by_scenario["enterprise_system"] == PROJECTION_RELATIONS


def test_registry_routes_only_the_amendment9_arms_through_the_projection():
    arms = {"gl_proj_cap", "gl_proj_qos16_cap", "gl_proj_qos16_indeg_prior", "hgl_proj_qos"}
    learned = {v for v in registry.VARIANTS if registry.learns_on_projection(v, "loso")}
    assert learned == arms
    assert registry.learns_on_projection("gl", "in_distribution")   # GAT-S-P
    assert not registry.learns_on_projection("topo_qos", "loso")    # routed by the harness itself
    assert registry.prior_for("gl_proj_qos16_indeg_prior") == "indeg"
    assert registry.prior_for("gl_qos16_prior") == registry.prior_for("hgl_qos_prior") == "topo_qos"
    assert registry.prior_for("gl_proj_qos16_cap") is None
