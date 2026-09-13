"""
tests/test_kfold_parallel.py
────────────────────────────
Unit tests for parallel execution and resume caching in K-Fold:
- CLI argument parsing in cli/kfold_evaluate.py and reproduce/kfold_all_variants.py
- Fingerprinting and configuration hashing
- Training-free baseline result replication across seeds
- CUDA ProcessPoolExecutor spawn context selection
- Fail-fast sweep abort behavior
- Per-fit shard resume caching
"""

import json
from pathlib import Path
import pytest
import torch

from cli.kfold_evaluate import (
    SweepAborted,
    _FailFast,
    _replicate_structural,
    _seed_cfg,
    _seed_fingerprint,
    _run_kfold_parallel,
    _run_seed,
    parse_args as parse_kfold_args,
)
from reproduce.kfold_all_variants import parse_args as parse_all_variants_args
from saag.evaluation.fingerprint import fit_fingerprint


def _cfg(**over):
    base = dict(
        layer="app", epochs=300, lr=3e-4, hidden=64, heads=4, layers=3,
        dropout=0.2, mode="gnn", variant="hgl_qos", eval_population="application",
        weight_decay=1e-4, warmup_T0=None, multitask_weight=0.5,
        rm_consistency_weight=0.0, ranking_weight=0.3, pairwise_ranking_weight=0.1,
        rank_normalize_features=False, rank_normalize_labels=False,
        qos_injection="pooled",
    )
    base.update(over)
    return _seed_cfg(**base)


# ── CLI Parsing Tests ─────────────────────────────────────────────────────────

def test_kfold_evaluate_cli_args(monkeypatch):
    test_args = [
        "kfold_evaluate.py",
        "--jobs", "4",
        "--torch-threads", "2",
        "--resume",
        "--variant", "gl",
    ]
    monkeypatch.setattr("sys.argv", test_args)
    args = parse_kfold_args()
    assert args.jobs == 4
    assert args.torch_threads == 2
    assert args.resume is True
    assert args.variant == "gl"


def test_kfold_all_variants_cli_args(monkeypatch):
    test_args = [
        "kfold_all_variants.py",
        "--jobs", "3",
        "--torch-threads", "1",
        "--resume",
        "--skip", "atm,av",
    ]
    monkeypatch.setattr("sys.argv", test_args)
    args = parse_all_variants_args()
    assert args.jobs == 3
    assert args.torch_threads == 1
    assert args.resume is True
    assert args.skip == "atm,av"


# ── Fingerprint & Replication Tests ──────────────────────────────────────────

def test_kfold_seed_cfg_rejects_missing_parameter():
    with pytest.raises(TypeError, match="epochs"):
        _seed_cfg(**{k: v for k, v in _cfg().items() if k != "epochs"})


def test_kfold_fingerprint_changes_on_parameter_delta():
    base_fp = fit_fingerprint(_cfg())
    mod_fp = fit_fingerprint(_cfg(epochs=250))
    assert base_fp != mod_fp


def test_kfold_structural_replication():
    original = {
        "spearman_rho": 0.456,
        "f1_at_k": 0.5,
        "seed": 42,
        "per_type_rho": {"Application": 0.45},
        "_full_scores": {"app-1": {"overall": 0.7}},
    }
    cloned = _replicate_structural(original, 123)
    assert cloned["seed"] == 123
    assert cloned["spearman_rho"] == original["spearman_rho"]
    cloned["_full_scores"]["app-1"]["overall"] = 1.0
    assert original["_full_scores"]["app-1"]["overall"] == 0.7


# ── Fail-Fast Sweep Abort Tests ──────────────────────────────────────────────

def test_fail_fast_aborts_after_streak():
    guard = _FailFast(streak=3)
    guard.record("CUDA out of memory")
    guard.record("CUDA out of memory")
    with pytest.raises(SweepAborted, match="CUDA out of memory"):
        guard.record("CUDA out of memory")


# ── Parallel Runner Multiprocessing Context ───────────────────────────────────

def test_kfold_parallel_uses_spawn_on_cuda(monkeypatch, tmp_path):
    captured_ctx = []

    class DummyExecutor:
        def __init__(self, *args, **kwargs):
            captured_ctx.append(kwargs.get("mp_context"))
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, *args, **kwargs):
            raise SweepAborted("aborted for test")

    import concurrent.futures
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", DummyExecutor)

    fake_bundle = type("Bundle", (), {"scenario_id": "test_scenario", "n_labelled": 10})()
    try:
        _run_kfold_parallel(
            bundles=[fake_bundle], k=2, seeds=[42], cfg=_cfg(),
            target_device=torch.device("cuda"), workdir=tmp_path, jobs=2,
            resume=False, cache_dir=tmp_path, skip=[], torch_threads=1,
            expected_ids=["test_scenario"],
        )
    except SweepAborted:
        pass

    assert len(captured_ctx) == 1
    assert captured_ctx[0] is not None
    assert captured_ctx[0].get_start_method() == "spawn"


# ── Resume Shard Caching Tests ────────────────────────────────────────────────

def test_run_seed_resume_reuses_cached_shard(tmp_path):
    from cli.kfold_evaluate import _seed_metrics

    bundle = type("Bundle", (), {"scenario_id": "test_scenario", "n_labelled": 10})()
    fold_dir = tmp_path / "fold_0"
    seed_dir = fold_dir / "seed_42"
    seed_dir.mkdir(parents=True)
    cfg = _cfg()
    fp = _seed_fingerprint(bundle.scenario_id, 2, 0, 42, cfg, torch.device("cpu"), cache_dir=tmp_path)

    cached_metrics = {
        "seed": 42,
        "spearman_rho": 0.777,
        "f1_at_k": 0.888,
        "precision_at_k": 0.8,
        "recall_at_k": 0.9,
        "ndcg_10": 0.85,
        "rmse": 0.1,
        "mae": 0.05,
        "n": 10,
        "prediction_mode": "gnn",
        "per_type_rho": {},
    }
    shard_payload = {
        "fingerprint": fp,
        "metrics": cached_metrics,
    }
    (seed_dir / "seed_result.json").write_text(json.dumps(shard_payload))

    # Calling _seed_metrics with resume=True should read from seed_result.json
    # without needing actual model training or data.
    res = _seed_metrics(
        bundle=bundle,
        k=2,
        fold_idx=0,
        seed=42,
        cfg=cfg,
        target_device=torch.device("cpu"),
        fold_dir=fold_dir,
        resume=True,
        cache_dir=tmp_path,
    )
    assert res == cached_metrics
    assert res["spearman_rho"] == 0.777
