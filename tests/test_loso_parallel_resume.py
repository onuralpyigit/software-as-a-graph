"""
test_loso_parallel_resume.py
────────────────────────────
Guards for the three things that make a LOSO sweep restartable and cheap:

  * a fit is reusable only when its *fingerprint* still matches — the
    configuration, the cache contents and the model code — replacing an mtime
    heuristic that, by its own docstring, could not see a code change at all;
  * a run of identical failures abandons the sweep instead of training on (a
    CUDA-only fault in the sibling k-fold harness burned 3.8 h before raising);
  * training-free variants are computed once per fold and replicated across
    seeds, without changing what is reported.
"""

from pathlib import Path

import pytest

from cli.loso_evaluate import (
    SweepAborted,
    _FailFast,
    _FoldPlan,
    _replicate_structural,
    _seed_cfg,
)
from saag.evaluation.fingerprint import CODE_FILES, cache_digest, fit_fingerprint


# ── fingerprints ─────────────────────────────────────────────────────────────

def _cfg(**over):
    base = dict(
        layer="app", epochs=300, lr=3e-4, hidden=64, heads=4, layers=3,
        dropout=0.2, mode="gnn", variant="hgl_qos", eval_population="application",
        weight_decay=1e-4, warmup_T0=None, multitask_weight=0.5,
        rm_consistency_weight=0.0, ranking_weight=0.3, pairwise_ranking_weight=0.1,
        rank_normalize_features=False, rank_normalize_labels=False,
    )
    base.update(over)
    return _seed_cfg(**base)


def test_seed_cfg_rejects_a_missing_parameter():
    """Every parameter that changes a fit must reach the fingerprint."""
    with pytest.raises(TypeError, match="epochs"):
        _seed_cfg(**{k: v for k, v in _cfg().items() if k != "epochs"})


def test_identical_configurations_fingerprint_alike():
    assert fit_fingerprint(_cfg()) == fit_fingerprint(_cfg())


@pytest.mark.parametrize("field,value", [
    ("epochs", 299), ("lr", 1e-3), ("variant", "hgl"), ("layers", 2),
    ("rank_normalize_labels", True), ("eval_population", "labeled"),
])
def test_any_changed_parameter_changes_the_fingerprint(field, value):
    assert fit_fingerprint(_cfg()) != fit_fingerprint(_cfg(**{field: value}))


def test_cache_contents_change_the_fingerprint(tmp_path):
    cache = tmp_path / "cache" / "scenario_a"
    cache.mkdir(parents=True)
    artefact = cache / "topology.json"
    artefact.write_text('{"nodes": []}')
    before = fit_fingerprint(_cfg(), cache_dir=str(tmp_path / "cache"))

    cache_digest.cache_clear()
    artefact.write_text('{"nodes": [1]}')
    after = fit_fingerprint(_cfg(), cache_dir=str(tmp_path / "cache"))
    assert before != after


def test_model_code_is_part_of_the_fingerprint():
    """The hole the mtime heuristic admitted to: code moves, nothing on disk says so."""
    for rel in ("saag/prediction/trainer.py", "saag/prediction/models/core.py"):
        assert rel in CODE_FILES
    root = Path(__file__).resolve().parent.parent
    for rel in CODE_FILES:
        assert (root / rel).exists(), f"{rel} is fingerprinted but does not exist"


# ── fail-fast ────────────────────────────────────────────────────────────────

def test_failures_below_the_streak_do_not_abort():
    guard = _FailFast(streak=3)
    guard.record("boom")
    guard.record("boom")
    guard.record(None)          # a success resets the run
    guard.record("boom")
    guard.record("boom")


def test_a_run_of_failures_aborts_the_sweep():
    guard = _FailFast(streak=3)
    guard.record("cuda device type tensor")
    guard.record("cuda device type tensor")
    with pytest.raises(SweepAborted, match="cuda device type tensor"):
        guard.record("cuda device type tensor")


def test_preflight_reports_a_broken_fit_before_the_sweep(monkeypatch, tmp_path):
    import cli.loso_evaluate as mod

    def _explode(*_a, **_k):
        raise mod.SeedFailed("can't convert cuda:0 device type tensor to numpy")

    monkeypatch.setattr(mod, "_run_seed", _explode)
    plan = _FoldPlan(
        holdout=None, train_set=[], train_ids=[], primary=None, inductives=[],
        val_bundle=None, effective_layers=3, fold_dir=tmp_path / "fold",
    )
    import torch
    with pytest.raises(SweepAborted, match="pre-flight"):
        mod._preflight(plan, _cfg(), torch.device("cpu"), tmp_path)
    assert not (tmp_path / ".preflight").exists(), "probe workspace was left behind"


# ── training-free replication ────────────────────────────────────────────────

def test_structural_replication_copies_values_and_restamps_the_seed():
    original = {
        "spearman_rho": 0.3109, "f1_at_k": 0.4, "seed": 42,
        "per_type_rho": {"Application": 0.31},
        "_full_scores": {"app-1": {"overall": 0.5}},
    }
    clone = _replicate_structural(original, 123)
    assert clone["seed"] == 123
    assert clone["spearman_rho"] == original["spearman_rho"]
    assert clone["_full_scores"] == original["_full_scores"]

    clone["_full_scores"]["app-1"]["overall"] = 9.9
    clone["per_type_rho"]["Application"] = 9.9
    assert original["_full_scores"]["app-1"]["overall"] == 0.5, "shared mutable state"
    assert original["per_type_rho"]["Application"] == 0.31, "shared mutable state"
