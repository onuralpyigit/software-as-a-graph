"""
tests/test_attention_and_figures.py
───────────────────────────────────
Tests for:
- reproduce/extract_attention.py: scenario loading, data preparation, attention extraction
- reproduce/render_attention_subgraph.py: attention subgraph rendering
- reproduce/render_results_figure.py: default LOSO path discovery
"""

from pathlib import Path
import json
import pytest

from reproduce.extract_attention import _load_scenario, run_extraction
from reproduce.render_attention_subgraph import _load_attention
from reproduce.render_results_figure import _find_default_loso_path
from saag.prediction.data_preparation import networkx_to_hetero_data


def test_load_scenario_and_data_conversion():
    """Verify _load_scenario properly normalizes metrics and converts to HeteroData without TypeError."""
    from reproduce.extract_attention import _LOSO_CACHE

    cache = _LOSO_CACHE / "atm_system"
    if not cache.exists() or not (cache / "failure_impact.json").exists():
        pytest.skip("output/loso_cache/atm_system not populated")

    g, struct, sim, rm = _load_scenario("atm_system")
    assert g.number_of_nodes() > 0
    assert len(sim) > 0
    assert len(struct) > 0
    # Values inside sim should be dicts of metrics (not top-level metadata)
    sample_key, sample_val = next(iter(sim.items()))
    assert isinstance(sample_val, dict)
    assert "composite" in sample_val

    conv = networkx_to_hetero_data(g, struct, sim, rm)
    assert conv.hetero_data is not None
    assert len(conv.hetero_data.node_types) > 0


def test_load_scenario_synthetic_cache(tmp_path):
    """Verify _load_scenario metric normalization and conversion with a synthetic cache fixture."""
    scenario_cache = tmp_path / "atm_system"
    scenario_cache.mkdir(parents=True)

    (scenario_cache / "failure_impact.json").write_text(json.dumps({
        "records": {
            "atm_client_app": {"impact_score": 0.5, "composite": 0.5},
            "atm_auth_service": {"impact_score": 0.8, "composite": 0.8},
        },
        "metadata": {"version": 2.0},
    }))

    (scenario_cache / "structural_metrics.json").write_text(json.dumps({
        "atm_client_app": {"betweenness": 0.1, "degree": 2},
        "atm_auth_service": {"betweenness": 0.4, "degree": 5},
    }))

    (scenario_cache / "quality_scores.json").write_text(json.dumps({
        "records": {
            "atm_client_app": {"quality_score": 0.3},
        }
    }))

    g, struct, sim, rm = _load_scenario("atm_system", cache_dir=tmp_path)
    assert g.number_of_nodes() > 0
    assert len(sim) > 0
    assert len(struct) > 0
    sample_key, sample_val = next(iter(sim.items()))
    assert isinstance(sample_val, dict)
    assert "composite" in sample_val

    conv = networkx_to_hetero_data(g, struct, sim, rm)
    assert conv.hetero_data is not None
    assert len(conv.hetero_data.node_types) > 0


def test_extract_attention_smoke(tmp_path):
    """Verify run_extraction trains 1 epoch and writes attention_weights.json."""
    out_path = run_extraction(
        scenario="atm_system",
        checkpoint_dir=None,
        output_dir=tmp_path,
        seed=42,
        hidden=16,
        num_heads=2,
        num_layers=2,
        num_epochs=1,
        device="cpu",
    )
    assert out_path.exists()
    data = _load_attention(out_path)
    assert "attention_by_layer" in data
    assert "node_id_map" in data


def test_find_default_loso_path():
    """Verify _find_default_loso_path returns an existing or expected path without raising."""
    p = _find_default_loso_path()
    assert isinstance(p, Path)
    assert "loso_all_variants" in p.name
