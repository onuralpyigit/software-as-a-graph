#!/usr/bin/env python3
"""
test_validate_predictions.py
────────────────────────────
Verification fixtures for the validation harness. These do not test the SaG
pipeline; they test that the HARNESS correctly recovers known statistical
structure, by constructing data with a known answer:

  T1  Simpson's paradox: each node type is a strong positive (Q,I) cluster, but
      the type centroids are arranged so pooling washes the signal out. Expect
      pooled |ρ| small, every per-type ρ strongly positive — mirroring the
      reported pooled ρ≈0.075 vs per-type 0.63–0.90.

  T2  Library blast-radius gap: library nodes have high I, low Q. Expect them to
      surface as the top rank-displacement outliers.

  T3  Precision@k and convergent validity compute the arithmetic correctly on a
      hand-checkable arrangement.
"""

import numpy as np

from cli.validate_graph import (
    GroundTruthSource, Prediction, align, build_report, precision_at_k,
    rank_displacement, stratified,
)


def _make_simpson_fixture():
    """4 types, each a positive cluster; centroids flat in I across types so the
    pooled correlation is near zero while per-type is strongly positive."""
    rng = np.random.default_rng(7)
    preds, gt = {}, {}
    # All type centroids sit at I≈0.5 but at different Q bands — pooling flat.
    type_q_band = {"Application": 0.8, "Broker": 0.6, "Topic": 0.4, "Node": 0.2}
    for ntype, q_center in type_q_band.items():
        for j in range(8):
            nid = f"{ntype}_{j}"
            # within-type: I increases monotonically with a small local Q spread
            local = (j - 3.5) / 10.0            # -0.35 .. +0.35
            q = q_center + local * 0.15 + rng.normal(0, 0.005)
            i = 0.5 + local * 0.6 + rng.normal(0, 0.01)  # strong positive slope
            preds[nid] = Prediction(nid, ntype, float(q))
            gt[nid] = float(i)
    return preds, gt


def test_simpson_paradox_recovered():
    preds, gt = _make_simpson_fixture()
    rows = align(preds, gt)

    from cli.validate_graph import _safe_corr
    pooled = _safe_corr([r[2] for r in rows], [r[3] for r in rows])
    strat = stratified(rows)

    # Pooled signal is washed out ...
    assert abs(pooled.rho) < 0.30, f"pooled ρ should be small, got {pooled.rho}"
    # ... while every per-type correlation is strong and positive.
    for ntype, c in strat.items():
        assert c.reportable, ntype
        assert c.rho > 0.70, f"{ntype} per-type ρ should be strong, got {c.rho}"
    print(f"T1 ok: pooled ρ={pooled.rho:+.3f}, "
          f"per-type ρ={{{', '.join(f'{t}:{c.rho:+.2f}' for t,c in strat.items())}}}")


def test_library_blast_radius_is_outlier():
    preds, gt = _make_simpson_fixture()
    # Inject library nodes: high I (≈0.97), modest Q (≈0.48) — the gap.
    for j in range(3):
        nid = f"ICAOMessageLib_{j}"
        preds[nid] = Prediction(nid, "Library", 0.48)
        gt[nid] = 0.97
    rows = align(preds, gt)

    disp = rank_displacement(rows, top=5)
    flagged = {d[0] for d in disp if d[2] <= -2}
    assert any(n.startswith("ICAOMessageLib") for n in flagged), \
        f"library nodes should be flagged as blind spots, got {flagged}"
    print(f"T2 ok: blind-spot outliers = {sorted(flagged)}")


def test_precision_at_k_and_convergent():
    # Hand-checkable: Q and I rank the same top-3 — P@3 = 1.0; swap one — 2/3.
    preds = {f"n{i}": Prediction(f"n{i}", "Application", q)
             for i, q in enumerate([0.9, 0.8, 0.7, 0.2, 0.1])}
    gt = {f"n{i}": v for i, v in enumerate([0.9, 0.8, 0.7, 0.2, 0.1])}
    rows = align(preds, gt)
    assert precision_at_k(rows, 3) == 1.0
    assert precision_at_k(rows, 5) == 1.0

    gt2 = dict(gt); gt2["n2"], gt2["n3"] = 0.2, 0.7  # push n2 out of top-3
    rows2 = align(preds, gt2)
    assert abs(precision_at_k(rows2, 3) - 2 / 3) < 1e-9

    # Convergent validity path runs and reports a finite ρ for two sources.
    sa = GroundTruthSource("cascade", gt)
    sb = GroundTruthSource("latency", gt2)
    text, blob = build_report(preds, [sa, sb])
    assert "CONVERGENT VALIDITY" in text
    assert blob["convergent_validity"]
    print("T3 ok: P@3=1.0 then 0.67; convergent-validity block emitted")


def test_multi_seed_spread_reported():
    preds, base = _make_simpson_fixture()
    rng = np.random.default_rng(1)
    per_seed = [{k: v + rng.normal(0, 0.02) for k, v in base.items()}
                for _ in range(5)]
    src = GroundTruthSource("cascade", per_seed=per_seed)
    text, blob = build_report(preds, [src])
    assert "MULTI-SEED" in text
    assert blob["sources"]["cascade"]["multi_seed"]["seeds"] == 5
    print("T4 ok: multi-seed mean ± std reported across 5 seeds")


if __name__ == "__main__":
    test_simpson_paradox_recovered()
    test_library_blast_radius_is_outlier()
    test_precision_at_k_and_convergent()
    test_multi_seed_spread_reported()
    print("\nALL PASS")
